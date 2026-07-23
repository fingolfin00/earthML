from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


def make_norm(
    norm: str | None,
    num_channels: int
) -> nn.Identity | nn.BatchNorm2d | nn.InstanceNorm2d | nn.GroupNorm:
    if norm is None:
        return nn.Identity()

    norm_l = norm.lower()

    if norm_l == "batchnorm2d":
        return nn.BatchNorm2d(num_channels)

    if norm_l == "instancenorm2d":
        return nn.InstanceNorm2d(num_channels, affine=True)

    if norm_l == "groupnorm":
        # choose largest valid group count up to 32
        for g in (32, 16, 8, 4, 2, 1):
            if num_channels % g == 0:
                return nn.GroupNorm(g, num_channels)
        return nn.GroupNorm(1, num_channels)

    raise ValueError(f"Unsupported norm={norm}")


class GeoPadConv2d(nn.Module):
    """
    Conv2d with configurable longitude padding and replicate latitude padding.

    Input shape:
        (N, C, latitude, longitude)

    Longitude padding:
        - "circular": periodic longitude
        - "replicate": repeat boundary values
        - "zero": constant zero padding

    Latitude padding is always "replicate".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        longitude_padding: Literal["circular", "replicate", "zero"] = "zero",
    ) -> None:
        super().__init__()

        # Validation
        if longitude_padding not in {
            "circular",
            "replicate",
            "zero",
        }:
            raise ValueError(
                "longitude_padding must be one of "
                "'circular', 'replicate', or 'zero', "
                f"got {longitude_padding!r}"
            )

        if kernel_size <= 0:
            raise ValueError(
                f"kernel_size must be positive, got {kernel_size}"
            )

        if stride <= 0:
            raise ValueError(
                f"stride must be positive, got {stride}"
            )

        self.pad_left = (kernel_size - 1) // 2
        self.pad_right = kernel_size // 2

        self.pad_top = (kernel_size - 1) // 2
        self.pad_bottom = kernel_size // 2

        self.longitude_padding = longitude_padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Longitude padding: left and right.
        if self.longitude_padding == "zero":
            x = F.pad(
                x,
                (self.pad_left, self.pad_right, 0, 0),
                mode="constant",
                value=0.0,
            )
        else:
            x = F.pad(
                x,
                (self.pad_left, self.pad_right, 0, 0),
                mode=self.longitude_padding,
            )

        # Latitude padding: top and bottom are never periodic.
        x = F.pad(
            x,
            (0, 0, self.pad_top, self.pad_bottom),
            mode="replicate",
        )

        return self.conv(x)
