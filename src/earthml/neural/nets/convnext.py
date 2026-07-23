# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from .. import EarthMLLightningModule
from .. import resolve_loss
from .utils import GeoPadConv2d


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
    ) -> None:
        super().__init__()

        self.dwconv = GeoPadConv2d(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            longitude_padding=longitude_padding,
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(dim)
            )
            if layer_scale_init_value > 0
            else None
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = residual + self.drop_path(x)
        return x

class ConvNeXt(EarthMLLightningModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        n_channels (int): Number of input image channels. Default: 3
        out_channels (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(
        self,
        loss: str,
        learning_rate: float,
        weight_decay: float,
        norm: str | None, # only LayerNorm supported
        supervised: bool,
        loss_params: dict[str, dict[str, Any]] | None = None,
        n_channels: int = 3,
        n_classes: int = 1000, 
        zero_init_output: bool = False,
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0., 
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        loss_params = loss_params or {"loss": {}, "net": {}}

        super().__init__(
            **loss_params.get("net", {}),
            optimizer_lr=learning_rate,
            weight_decay=weight_decay,
        )

        if len(depths) != 4 or len(dims) != 4:
            raise ValueError(
                "ConvNeXt currently requires exactly four encoder stages."
            )

        self.loss_name = loss
        self.loss = resolve_loss(loss, loss_params["loss"])
        self.supervised = supervised

        needs_var = loss == "GaussianNLLFromLogits"

        self.n_channels = n_channels
        self.n_classes = n_classes

        out_channels = 2 * n_classes if needs_var else n_classes

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(
                n_channels,
                dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm(
                dims[0],
                eps=1e-6,
                data_format="channels_first",
            ),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(
                    dims[i],
                    eps=1e-6,
                    data_format="channels_first",
                ),
                nn.Conv2d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value,
                    longitude_padding=longitude_padding,
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(
            dims[-1],
            eps=1e-6,
            data_format="channels_first",
        ) # final norm layer

        self.up_layers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        for i in range(3, 0, -1):
            self.up_layers.append(
                nn.Conv2d(
                    dims[i],
                    dims[i - 1],
                    kernel_size=1,
                )
            )

            self.decoder_stages.append(
                nn.Sequential(
                    nn.Conv2d(
                        2 * dims[i - 1],
                        dims[i - 1],
                        kernel_size=1,
                    ),
                    Block(
                        dim=dims[i - 1],
                        drop_path=0.0,
                        layer_scale_init_value=layer_scale_init_value,
                        longitude_padding=longitude_padding,
                    ),
                    Block(
                        dim=dims[i - 1],
                        drop_path=0.0,
                        layer_scale_init_value=layer_scale_init_value,
                        longitude_padding=longitude_padding,
                    ),
                )
            )

        self.output_conv = nn.Conv2d(
            dims[0],
            out_channels,
            kernel_size=1,
        )

        self.apply(self._init_weights)

        if zero_init_output:
            nn.init.zeros_(self.output_conv.weight)
            if self.output_conv.bias is not None:
                nn.init.zeros_(self.output_conv.bias)


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        features = []

        for downsample, stage in zip(
            self.downsample_layers,
            self.stages,
            strict=True,
        ):
            x = downsample(x)
            x = stage(x)
            features.append(x)

        return features


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        if min(input_size) < 32:
            raise ValueError(
                "ConvNeXt requires input height and width of at least 32, "
                f"got {input_size}"
            )

        features = self.forward_features(x)
        x = self.norm(features[-1])

        for decoder_index, encoder_index in enumerate(
            range(len(features) - 2, -1, -1)
        ):
            skip = features[encoder_index]

            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            x = self.up_layers[decoder_index](x)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_stages[decoder_index](x)

        x = self.output_conv(x)

        return F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
