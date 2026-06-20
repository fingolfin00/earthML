'''
Description: UNet architecture with CBAM. This script is derived from  SmaAT-UNet:
    https://github.com/HansBambel/SmaAt-UNet/tree/master/models
'''
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from ...logging import get_logger
from .. import EarthMLLightningModule
from .. import resolve_loss
from .utils import make_norm

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
    ) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
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
    ) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size, norm=norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class DepthwiseSeparableConv(EarthMLLightningModule):
    def __init__(
        self,
        in_channels: int,
        output_channels: int,
        kernel_size: int,
        padding: int = 0,
        kernels_per_layer: int = 1,
    ) -> None:
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        # print("DepthwiseSeparableConv params:")
        # print(f" kernel_size: {kernel_size}")
        # print(f" padding: {padding}")
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"Depthwise input: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        # # print(x)        
        x = self.depthwise(x)
        # print(f"Depthwise output: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        # print(f"Pointwise input: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        x = self.pointwise(x)
        # print(f"Pointwise output: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        return x

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
                # extra_logger=extra_logger,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            make_norm(norm, mid_channels),
            nn.LeakyReLU(inplace=True),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                # extra_logger=extra_logger,
                kernels_per_layer=kernels_per_layer,
                padding=1,
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
    ) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(
                in_channels,
                out_channels,
                kernels_per_layer=kernels_per_layer,
                norm=norm,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpDS(EarthMLLightningModule):
    """Upscaling then double conv"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        kernels_per_layer: int = 1,
        norm: str | None = "BatchNorm2d",
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
                norm=norm,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                kernels_per_layer=kernels_per_layer,
                norm=norm,
            )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        # print(f"UpDS.forward - x1 (from deeper layer): shape={x1.shape}, dtype={x1.dtype}, device={x1.device}")
        # print(f"UpDS.forward - x2 (skip connection): shape={x2.shape}, dtype={x2.dtype}, device={x2.device}")

        x1 = self.up(x1)
        # print(f"UpDS.forward - x1 after upsample: shape={x1.shape}, dtype={x1.dtype}, device={x1.device}")
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX < 0 or diffY < 0:
            logger.warning("UpDS.forward - Negative padding needed! x1:%s, x2:%s", x1.shape, x2.shape)
            # This indicates an issue with dimensions being larger than expected,
            # which could lead to problematic padding arguments.
            # You might want to raise an error or adjust logic here.

        # print(f"UpDS.forward - padding: [{diffX // 2}, {diffX - diffX // 2}, {diffY // 2}, {diffY - diffY // 2}]")
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # print(f"UpDS.forward - x1 after padding: shape={x1.shape}, dtype={x1.dtype}, device={x1.device}")

        x = torch.cat([x2, x1], dim=1)
        # print(f"UpDS.forward - x after concatenation: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        
        return self.conv(x) # This calls DoubleConvDS, which in turn calls DepthwiseSeparableConv

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
        kernels_per_layer: int = 2,
        bilinear: bool = True,
        reduction_ratio: int = 16,
        base_channels: int = 64,
    ) -> None:
        loss_params = loss_params or {"loss": {}, "net": {}}

        super().__init__(
            **loss_params.get("net", {}),
            optimizer_lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.loss_name = loss
        self.loss = resolve_loss(loss, loss_params["loss"])
        self.supervised = supervised

        needs_var: bool = ("GaussianNLL" in loss) or ("GaussianNLLFromLogits" in loss)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.debug_numerics = DEFAULT_DEBUG_NUMERICS
        self.bilinear = bilinear

        out_channels: int = (2 * self.n_classes) if needs_var else self.n_classes
        factor: int = 2 if self.bilinear else 1
        b: int = base_channels

        self.inc = DoubleConvDS(self.n_channels, b, kernels_per_layer=kernels_per_layer, norm=norm)
        self.cbam1 = CBAM(b, reduction_ratio=reduction_ratio, norm=norm)

        self.down1 = DownDS(b, 2 * b, kernels_per_layer=kernels_per_layer, norm=norm)
        self.cbam2 = CBAM(2 * b, reduction_ratio=reduction_ratio, norm=norm)

        self.down2 = DownDS(2 * b, 4 * b, kernels_per_layer=kernels_per_layer, norm=norm)
        self.cbam3 = CBAM(4 * b, reduction_ratio=reduction_ratio, norm=norm)

        self.down3 = DownDS(4 * b, 8 * b, kernels_per_layer=kernels_per_layer, norm=norm)
        self.cbam4 = CBAM(8 * b, reduction_ratio=reduction_ratio, norm=norm)

        self.down4 = DownDS(8 * b, 16 * b // factor, kernels_per_layer=kernels_per_layer, norm=norm)
        self.cbam5 = CBAM(16 * b // factor, reduction_ratio=reduction_ratio, norm=norm)

        self.up1 = UpDS(16 * b, 8 * b // factor, self.bilinear, kernels_per_layer=kernels_per_layer, norm=norm)
        self.up2 = UpDS(8 * b, 4 * b // factor, self.bilinear, kernels_per_layer=kernels_per_layer, norm=norm)
        self.up3 = UpDS(4 * b, 2 * b // factor, self.bilinear, kernels_per_layer=kernels_per_layer, norm=norm)
        self.up4 = UpDS(2 * b, b, self.bilinear, kernels_per_layer=kernels_per_layer, norm=norm)

        self.outc = OutConv(b, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.debug_numerics:
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError(f"Input tensor contains NaN/Inf values: shape={tuple(x.shape)}")

            for name, param in self.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    raise ValueError(f"Model parameter '{name}' contains NaN/Inf values")
        # logger.debug("SmaAt_UNet forward input shape: %s", tuple(x.shape))
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4Att = self.cbam4(x4)
        x5 = self.down4(x4)
        x5Att = self.cbam5(x5)
        # print(f"Before up1: x5Att shape: {x5Att.shape}, x4Att shape: {x4Att.shape}")
        x = self.up1(x5Att, x4Att)
        # print(f"After up1: x shape: {x.shape}, x3Att shape: {x3Att.shape}")
        x = self.up2(x, x3Att)
        # print(f"After up2: x shape: {x.shape}, x2Att shape: {x2Att.shape}")
        x = self.up3(x, x2Att)
        # print(f"After up3: x shape: {x.shape}, x1Att shape: {x1Att.shape}")
        x = self.up4(x, x1Att)
        # print(f"After up4: x shape: {x.shape}")
        logits = self.outc(x)
        return logits
