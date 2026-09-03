'''
Description: UNet architecture with CBAM. This script is derived from  SmaAT-UNet:
    https://github.com/HansBambel/SmaAt-UNet/tree/master/models
'''
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from ...logging import get_logger
from .. import EarthMLLightningModule
from .. import resolve_loss
from .utils import make_norm, GeoPadConv2d

logger = get_logger(__name__)


DEFAULT_DEBUG_NUMERICS = False

# DEBUG
# import traceback
# _old_view = torch.Tensor.view

# def _debug_view(self, *shape):
#     print(">>> Tensor.view called with shape:", shape)
#     traceback.print_stack(limit=15)
#     return _old_view(self, *shape)

# torch.Tensor.view = _debug_view

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous().flatten(1)

class ChannelAttention(nn.Module):
    def __init__(
        self,
        input_channels: int,
        reduction_ratio: int = 16,
    ) -> None:
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x).contiguous()
        max_values = self.max_pool(x).contiguous()
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale

class SpatialAttention(nn.Module):
    def __init__(
        self,
        kernel_size: int = 7,
        norm: str | None = "BatchNorm2d",
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
    ) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"

        self.conv = GeoPadConv2d(
            2,
            1,
            kernel_size=kernel_size,
            bias=False,
            longitude_padding=longitude_padding,
        )
        self.norm = make_norm(norm, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.norm(out)
        return x * torch.sigmoid(out)


class CBAM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        norm: str | None = "BatchNorm2d",
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
    ) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(
            input_channels,
            reduction_ratio=reduction_ratio,
        )
        self.spatial_att = SpatialAttention(
            kernel_size=kernel_size,
            norm=norm,
            longitude_padding=longitude_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_size: int,
        kernels_per_layer: int = 1,
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
    ) -> None:
        super().__init__()

        depthwise_channels = in_channels * kernels_per_layer

        self.depthwise = GeoPadConv2d(
            in_channels,
            depthwise_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=True,
            longitude_padding=longitude_padding,
        )

        self.pointwise = nn.Conv2d(
            depthwise_channels,
            output_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

# class DoubleConvDS(CustomLightningModule):
class DoubleConvDS(L.LightningModule):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        kernels_per_layer: int = 1,
        norm: str | None = "BatchNorm2d",
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # print("DoubleConvDS params:")
        # print(f" in_channels: {in_channels}")
        # print(f" mid_channels: {mid_channels}")
        # print(f" out_channels: {out_channels}")
        # print(f" kernels_per_layer: {kernels_per_layer}")
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                longitude_padding=longitude_padding,
            ),
            make_norm(norm, mid_channels),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                longitude_padding=longitude_padding,
            ),
            make_norm(norm, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels_per_layer: int = 1,
        norm: str | None = "BatchNorm2d",
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
    ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(
                in_channels,
                out_channels,
                kernels_per_layer=kernels_per_layer,
                norm=norm,
                longitude_padding=longitude_padding,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        kernels_per_layer: int = 1,
        norm: str | None = "BatchNorm2d",
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
    ) -> None:
        super().__init__()

        self.bilinear = bilinear

        if bilinear:
            self.up = None
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
                norm=norm,
                longitude_padding=longitude_padding,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                kernels_per_layer=kernels_per_layer,
                norm=norm,
                longitude_padding=longitude_padding,
            )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        if self.bilinear:
            x1 = F.interpolate(
                x1,
                size=x2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            x1 = self.up(x1)

            if x1.shape[-2:] != x2.shape[-2:]:
                x1 = F.interpolate(
                    x1,
                    size=x2.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SmaAt_UNet(EarthMLLightningModule):
    def __init__(
        self,
        loss: str,
        learning_rate: float,
        weight_decay: float,
        norm: str | None,
        supervised: bool,
        loss_params: dict[str, dict[str, Any]] | None = None,
        n_channels: int = 1,
        n_classes: int = 1,
        zero_init_output: bool = False,
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
        kernels_per_layer: int = 2,
        bilinear: bool = True,
        reduction_ratio: int = 16,
        base_channels: int = 64,
        depth: int = 5, # total encoder levels, including input block
    ) -> None:
        if depth < 2:
            raise ValueError(f"depth must be at least 2, got {depth}")

        loss_params = loss_params or {"loss": {}, "net": {}}

        super().__init__(
            **loss_params.get("net", {}),
            optimizer_lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.loss_name = loss
        self.loss = resolve_loss(loss, loss_params["loss"])
        self.supervised = supervised

        needs_var = (
            "GaussianNLL" in loss
            or "GaussianNLLFromLogits" in loss
        )

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.debug_numerics = DEFAULT_DEBUG_NUMERICS
        self.bilinear = bilinear
        self.depth = depth

        out_channels = 2 * n_classes if needs_var else n_classes
        factor = 2 if bilinear else 1

        # Nominal U-Net widths before the bilinear bottleneck adjustment.
        channels = [
            base_channels * (2**level)
            for level in range(depth)
        ]

        self.inc = DoubleConvDS(
            n_channels,
            channels[0],
            kernels_per_layer=kernels_per_layer,
            norm=norm,
            longitude_padding=longitude_padding,
        )

        self.encoder_cbam = nn.ModuleList([
            CBAM(
                channels[0],
                reduction_ratio=reduction_ratio,
                norm=norm,
                longitude_padding=longitude_padding,
            )
        ])

        # Encoder
        self.downs = nn.ModuleList()

        for level in range(1, depth):
            in_ch = channels[level - 1]
            out_ch = channels[level]

            if level == depth - 1:
                out_ch //= factor

            self.downs.append(
                DownDS(
                    in_ch,
                    out_ch,
                    kernels_per_layer=kernels_per_layer,
                    norm=norm,
                    longitude_padding=longitude_padding,
                )
            )

            self.encoder_cbam.append(
                CBAM(
                    out_ch,
                    reduction_ratio=reduction_ratio,
                    norm=norm,
                    longitude_padding=longitude_padding,
                )
            )

        # Decoder
        self.ups = nn.ModuleList()

        for level in reversed(range(depth - 1)):
            out_ch = (
                channels[level]
                if level == 0
                else channels[level] // factor
            )

            self.ups.append(
                UpDS(
                    channels[level + 1],
                    out_ch,
                    bilinear,
                    kernels_per_layer=kernels_per_layer,
                    norm=norm,
                    longitude_padding=longitude_padding,
                )
            )

        self.outc = OutConv(
            channels[0],
            out_channels,
        )

        if zero_init_output:
            # Weight and bias init to zero
            nn.init.zeros_(self.outc.conv.weight)

            if self.outc.conv.bias is not None:
                nn.init.zeros_(self.outc.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug_numerics:
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError(
                    "Input tensor contains NaN/Inf values: "
                    f"shape={tuple(x.shape)}"
                )

            for name, param in self.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    raise ValueError(
                        f"Model parameter {name!r} contains NaN/Inf values"
                    )

        # Raw encoder representation
        x = self.inc(x)

        # Attention is used for the skip, but not for the next down block
        skips = [self.encoder_cbam[0](x)]

        for level, down in enumerate(self.downs, start=1):
            x = down(x)
            skips.append(self.encoder_cbam[level](x))

        # The deepest attention output becomes the decoder input
        x = skips.pop()

        for up in self.ups:
            x = up(x, skips.pop())

        return self.outc(x)
