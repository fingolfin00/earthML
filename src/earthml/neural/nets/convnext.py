from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from .. import EarthMLLightningModule
from .. import resolve_loss
from .utils import GeoPadConv2d


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
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
        longitude_padding: Literal[
            "circular",
            "replicate",
            "zero",
        ] = "zero",
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

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim)
            )
        else:
            self.gamma = None

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)

        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


class SpatialTransformerBlock(nn.Module):
    """Transformer encoder block operating on flattened spatial features."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.norm1 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape

        # NCHW -> N(HW)C
        x = x.flatten(2).transpose(1, 2)

        normalized = self.norm1(x)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        x = x + self.drop_path(attention_output)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # N(HW)C -> NCHW
        x = x.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return x


class ConvNeXtTransformerUNet(EarthMLLightningModule):
    """Encoder-decoder ConvNeXt model for dense spatial prediction.

    This model adapts the ConvNeXt architecture from "A ConvNet for the
    2020s" for pixel-wise regression or prediction tasks. It consists of:

    - a configurable ConvNeXt encoder;
    - skip connections from every encoder resolution;
    - a configurable decoder that progressively restores spatial resolution;
    - a final 1x1 convolution producing one or more output fields.

    The encoder starts with a stride-2 convolution. Each subsequent
    encoder stage also begins with a stride-2 convolution. Consequently,
    the total encoder downsampling factor is::

        2 ** num_stages

    Decoder stages are ordered from the deepest, lowest-resolution stage to
    the shallowest, highest-resolution stage. For example::

        dims = (64, 128, 256, 512)
        decoder_depths = (2, 3, 4)

    gives decoder transitions::

        512 -> 256, using 2 ConvNeXt blocks
        256 -> 128, using 3 ConvNeXt blocks
        128 -> 64,  using 4 ConvNeXt blocks

    At each decoder stage, the current representation is bilinearly
    interpolated to the corresponding skip-connection resolution, projected
    to the skip width, concatenated with the skip features, and processed by
    the configured number of ConvNeXt blocks.

    Parameters
    ----------
    loss
        Name of the loss function resolved by :func:`resolve_loss`.
    learning_rate
        Learning rate passed to the optimizer configuration.
    weight_decay
        Weight-decay coefficient passed to the optimizer configuration.
    norm
        Retained for compatibility with the common model interface. This
        architecture uses LayerNorm internally.
    supervised
        Whether the model is used in supervised mode.
    loss_params
        Optional nested configuration containing ``"loss"`` and ``"net"``
        keyword arguments.
    n_channels
        Number of input channels.
    n_classes
        Number of predicted output fields. For probabilistic losses requiring
        both a prediction and variance parameter, the final channel count is
        doubled.
    zero_init_output
        If True, initialize the final output convolution weights and biases
        to zero.
    longitude_padding
        Longitude padding used by the depthwise convolutions inside ConvNeXt
        blocks. Supported values are ``"circular"``, ``"replicate"``, and
        ``"zero"``.
    encoder_depths
        Number of ConvNeXt blocks in each encoder stage. Its length must equal
        the length of ``dims``.
    decoder_depths
        Number of ConvNeXt blocks in each decoder stage, ordered from deepest
        to shallowest. Its length must be ``len(dims) - 1``.
    dims
        Channel width of each encoder stage. The sequence length determines
        the number of encoder stages.
    drop_path_rate
        Maximum stochastic-depth probability in the encoder. Drop-path rates
        increase linearly from zero to this value across encoder blocks.
        Decoder blocks currently use no stochastic depth.
    layer_scale_init_value
        Initial value of the learnable layer-scale parameter in each ConvNeXt
        block. Set to a non-positive value to disable layer scaling.

    Raises
    ------
    ValueError
        If fewer than two stages are configured, tuple lengths are
        inconsistent, any depth is less than one, any stage width is not
        positive, or the input spatial dimensions are smaller than the total
        downsampling factor.

    Notes
    -----
    The output is interpolated back to the exact input height and width.
    Spatial dimensions therefore do not need to be divisible by the total
    downsampling factor, but both dimensions must be at least that large.
    """
    def __init__(
        self,
        loss: str,
        learning_rate: float,
        weight_decay: float,
        norm: str | None,
        supervised: bool,
        loss_params: dict[str, dict[str, Any]] | None = None,
        n_channels: int = 3,
        n_classes: int = 1,
        zero_init_output: bool = False,
        longitude_padding: Literal[
            "circular",
            "replicate",
            "zero",
        ] = "zero",
        encoder_depths: tuple[int, ...] = (3, 3, 9, 3),
        decoder_depths: tuple[int, ...] = (3, 3, 3),
        dims: tuple[int, ...] = (64, 128, 256, 512),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.0,
        transformer_depth: int = 1,
        refinement_depth: int = 2,
    ) -> None:
        loss_params = loss_params or {}

        super().__init__(
            **loss_params.get("net", {}),
            optimizer_lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.loss_name = loss
        self.loss = resolve_loss(
            loss,
            loss_params.get("loss", {}),
        )

        num_stages = len(dims)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

        if num_stages < 2:
            raise ValueError(
                f"At least two stages are required, got {num_stages}"
            )

        if len(encoder_depths) != num_stages:
            raise ValueError(
                "encoder_depths and dims must have the same length"
            )

        if len(decoder_depths) != num_stages - 1:
            raise ValueError(
                "decoder_depths must have len(dims) - 1 entries"
            )

        if any(depth < 1 for depth in encoder_depths):
            raise ValueError(
                "All encoder depths must be at least 1"
            )

        if any(depth < 1 for depth in decoder_depths):
            raise ValueError(
                "All decoder depths must be at least 1"
            )

        if any(dim < 1 for dim in dims):
            raise ValueError(
                "All dimensions must be positive"
            )

        if transformer_depth < 0:
            raise ValueError(
                "transformer_depth must be >= 0"
            )

        if refinement_depth < 0:
            raise ValueError(
                "refinement_depth must be >= 0"
            )

        if not 0.0 <= drop_path_rate <= 1.0:
            raise ValueError(
                "drop_path_rate must be between 0 and 1"
            )

        if transformer_depth > 0 and dims[-1] % transformer_heads != 0:
            raise ValueError(
                f"Bottleneck dimension {dims[-1]} must be divisible "
                f"by transformer_heads={transformer_heads}"
            )

        self.supervised = supervised
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.total_downsampling_factor = 2 ** num_stages

        needs_var = loss == "GaussianNLLFromLogits"
        out_channels = (
            2 * n_classes
            if needs_var
            else n_classes
        )

        # ------------------------------------------------------------------
        # Encoder
        # ------------------------------------------------------------------

        self.downsample_layers = nn.ModuleList()

        self.downsample_layers.append(
            nn.Sequential(
                GeoPadConv2d(
                    n_channels,
                    dims[0],
                    kernel_size=3,
                    stride=2,
                    longitude_padding=longitude_padding,
                ),
                LayerNorm(
                    dims[0],
                    eps=1e-6,
                    data_format="channels_first",
                ),
            )
        )

        for i in range(num_stages - 1):
            self.downsample_layers.append(
                nn.Sequential(
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
            )

        # Progressive stochastic depth across the encoder.
        encoder_drop_rates = torch.linspace(
            0.0,
            drop_path_rate,
            sum(encoder_depths),
        ).tolist()

        self.encoder_stages = nn.ModuleList()

        block_index = 0

        for dim, depth in zip(
            dims,
            encoder_depths,
            strict=True,
        ):
            blocks = []

            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dim,
                        drop_path=encoder_drop_rates[block_index],
                        layer_scale_init_value=layer_scale_init_value,
                        longitude_padding=longitude_padding,
                    )
                )

                block_index += 1

            self.encoder_stages.append(
                nn.Sequential(*blocks)
            )

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------

        self.bottleneck = nn.Sequential(
            *[
                SpatialTransformerBlock(
                    dim=dims[-1],
                    num_heads=transformer_heads,
                    mlp_ratio=transformer_mlp_ratio,
                    dropout=transformer_dropout,
                    drop_path=drop_path_rate,
                )
                for _ in range(transformer_depth)
            ]
        )

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------

        self.up_layers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        for decoder_depth, encoder_index in zip(
            decoder_depths,
            range(num_stages - 1, 0, -1),
            strict=True,
        ):
            input_dim = dims[encoder_index]
            output_dim = dims[encoder_index - 1]

            self.up_layers.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=1,
                )
            )

            blocks = [
                nn.Conv2d(
                    2 * output_dim,
                    output_dim,
                    kernel_size=1,
                )
            ]

            blocks.extend(
                ConvNeXtBlock(
                    dim=output_dim,
                    drop_path=0.0,
                    layer_scale_init_value=layer_scale_init_value,
                    longitude_padding=longitude_padding,
                )
                for _ in range(decoder_depth)
            )

            self.decoder_stages.append(
                nn.Sequential(*blocks)
            )

        # ------------------------------------------------------------------
        # Full-resolution refinement
        # ------------------------------------------------------------------

        self.refinement = nn.Sequential(
            *[
                ConvNeXtBlock(
                    dim=dims[0],
                    drop_path=0.0,
                    layer_scale_init_value=layer_scale_init_value,
                    longitude_padding=longitude_padding,
                )
                for _ in range(refinement_depth)
            ]
        )

        # ------------------------------------------------------------------
        # Prediction head
        # ------------------------------------------------------------------

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

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:
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
            self.encoder_stages,
            strict=True,
        ):
            x = downsample(x)
            x = stage(x)

            features.append(x)

        return features

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_size = x.shape[-2:]

        if min(input_size) < self.total_downsampling_factor:
            raise ValueError(
                "Input height and width must be at least "
                f"{self.total_downsampling_factor}, "
                f"got {input_size}"
            )

        features = self.forward_features(x)

        # Bottleneck
        x = features[-1]
        x = self.bottleneck(x)

        # Decoder
        for projection, stage, skip in zip(
            self.up_layers,
            self.decoder_stages,
            reversed(features[:-1]),
            strict=True,
        ):
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            x = projection(x)
            x = torch.cat((x, skip), dim=1)
            x = stage(x)

        # Full-resolution reconstruction
        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        x = self.refinement(x)

        return self.output_conv(x)


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
