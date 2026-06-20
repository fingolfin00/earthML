from torch import nn


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
