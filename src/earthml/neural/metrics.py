import math

import torch
import torch.nn.functional as F
from torchmetrics import Metric

# ==========================================================
# Internal functional utilities
# ==========================================================

Dim = int | tuple[int, ...] | None


def latitude_weights(
    x: torch.Tensor,
    latitudes: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine-latitude weights broadcastable to a (N, C, H, W) tensor.

    Parameters
    ----------
    x
        Tensor shaped (N, C, H, W).
    latitudes
        Latitude coordinate shaped (H,).
    """
    if x.ndim != 4:
        raise ValueError(
            "latitude_weights expects x shaped (N, C, H, W)."
        )

    h = x.shape[-2]

    if latitudes.numel() != h:
        raise ValueError(
            f"latitudes length {latitudes.numel()} != H {h}"
        )

    weights = torch.cos(
        torch.deg2rad(
            latitudes.to(
                device=x.device,
                dtype=x.dtype,
            )
        )
    )

    return weights.view(1, 1, h, 1)


def expand_mask_to(
    x: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Return a boolean mask broadcasted to x.shape. True = valid.

    Supported behavior:
    - mask=None -> all True
    - exact/broadcastable masks are accepted directly
    - lower-rank masks are aligned to trailing dims by default
    - special handling for common layouts like:
        x:    (N,C,H,W), mask: (N,H,W)    -> (N,1,H,W)
        x:    (N,C,T,H,W), mask: (N,T,H,W)-> (N,1,T,H,W)

    Raises:
        ValueError if the mask cannot be broadcast to x.shape.
    """
    if mask is None:
        return torch.ones_like(x, dtype=torch.bool)

    mask = mask.to(device=x.device)
    if mask.dtype != torch.bool:
        mask = mask != 0

    # Fast path: already broadcastable as-is
    try:
        return torch.broadcast_to(mask, x.shape)
    except RuntimeError:
        pass

    # Common case:
    # x=(N,C,H,W), mask=(N,H,W) -> insert singleton channel dim
    # x=(N,C,T,H,W), mask=(N,T,H,W) -> insert singleton channel dim
    if x.ndim >= 3 and mask.ndim == x.ndim - 1 and mask.shape[0] == x.shape[0]:
        candidate = mask.unsqueeze(1)
        try:
            return torch.broadcast_to(candidate, x.shape)
        except RuntimeError:
            pass

    # General fallback: align mask to trailing dimensions
    # e.g. (H,W) -> (1,1,H,W), (T,H,W) -> (1,1,T,H,W)
    if mask.ndim < x.ndim:
        candidate = mask.reshape((1,) * (x.ndim - mask.ndim) + tuple(mask.shape))
        try:
            return torch.broadcast_to(candidate, x.shape)
        except RuntimeError:
            pass

    raise ValueError(
        f"mask with shape {tuple(mask.shape)} is not broadcastable to target shape {tuple(x.shape)}"
    )


def _masked_weighted_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    dim: Dim = None,
) -> torch.Tensor:
    """
    Mean over valid elements, optionally with additional weights.

    ``mask`` defines valid elements.
    ``weights`` defines relative importance of valid elements.
    """
    m = expand_mask_to(
        x,
        mask,
    ).to(dtype=x.dtype)

    if weights is not None:
        weights = torch.broadcast_to(
            weights.to(
                device=x.device,
                dtype=x.dtype,
            ),
            x.shape,
        )

        m = m * weights

    numerator = (
        x * m
    ).sum(dim=dim)

    denominator = m.sum(dim=dim)

    return numerator / denominator


def _restore_reduced_dims(
    x: torch.Tensor,
    dim: Dim,
    ndim: int,
) -> torch.Tensor:
    """
    Restore dimensions removed by a reduction so the result can
    broadcast against the original tensor.
    """
    if dim is None:
        return x

    dims = (
        (dim,)
        if isinstance(dim, int)
        else dim
    )

    dims = tuple(
        d if d >= 0 else ndim + d
        for d in dims
    )

    for d in sorted(dims):
        x = x.unsqueeze(d)

    return x


def _masked_bias(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_weighted_mean(
        preds - target,
        mask,
        weights=weights,
        dim=dim,
    )


def _masked_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_weighted_mean(
        (preds - target).square(),
        mask,
        weights=weights,
        dim=dim,
    )


def _masked_mae(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_weighted_mean(
        (preds - target).abs(),
        mask,
        weights=weights,
        dim=dim,
    )


def _masked_crmse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    dim: Dim = None,
) -> torch.Tensor:
    error = preds - target

    bias = _masked_weighted_mean(
        error,
        mask,
        weights=weights,
        dim=dim,
    )

    bias = _restore_reduced_dims(
        bias,
        dim,
        error.ndim,
    )

    return torch.sqrt(
        _masked_weighted_mean(
            (error - bias).square(),
            mask,
            weights=weights,
            dim=dim,
        )
    )


def _masked_std(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    dim: Dim = None,
) -> torch.Tensor:
    mean = _masked_weighted_mean(
        x,
        mask,
        weights=weights,
        dim=dim,
    )

    mean = _restore_reduced_dims(
        mean,
        dim,
        x.ndim,
    )

    variance = _masked_weighted_mean(
        (x - mean).square(),
        mask,
        weights=weights,
        dim=dim,
    )

    return torch.sqrt(variance)


# ==========================================================
# Functional masked metrics
# ==========================================================

def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_weighted_mean(
        x,
        mask,
        dim=dim,
    )


def masked_bias(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_bias(
        preds,
        target,
        mask,
        dim=dim,
    )


def masked_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_mse(
        preds,
        target,
        mask,
        dim=dim,
    )


def masked_rmse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return torch.sqrt(
        _masked_mse(
            preds,
            target,
            mask,
            dim=dim,
        )
    )


def masked_mae(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_mae(
        preds,
        target,
        mask,
        dim=dim,
    )


def masked_crmse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_crmse(
        preds,
        target,
        mask,
        dim=dim,
    )


def masked_std(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_std(
        x,
        mask,
        dim=dim,
    )


def masked_std_ratio(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return (
        _masked_std(
            preds,
            mask,
            dim=dim,
        )
        / _masked_std(
            target,
            mask,
            dim=dim,
        )
    )


def masked_temporal_corr(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dim: int | tuple[int, ...] = 0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Masked correlation over the requested dimension(s).

    With dim=0 and inputs (N, C, H, W), returns correlation
    independently at every (C, H, W) location.
    """
    m = expand_mask_to(
        preds,
        mask,
    ).to(dtype=preds.dtype)

    count = m.sum(dim=dim)

    pred_mean = masked_mean(
        preds,
        m,
        dim=dim,
    )

    target_mean = masked_mean(
        target,
        m,
        dim=dim,
    )

    pred_mean = _restore_reduced_dims(
        pred_mean,
        dim,
        preds.ndim,
    )

    target_mean = _restore_reduced_dims(
        target_mean,
        dim,
        target.ndim,
    )

    pred_anom = (
        preds - pred_mean
    ) * m

    target_anom = (
        target - target_mean
    ) * m

    covariance = (
        pred_anom
        * target_anom
    ).sum(dim=dim)

    pred_var = (
        pred_anom.square()
    ).sum(dim=dim)

    target_var = (
        target_anom.square()
    ).sum(dim=dim)

    corr = covariance / torch.sqrt(
        pred_var * target_var
    ).clamp_min(eps)

    valid = (
        (count > 1)
        & (pred_var > eps)
        & (target_var > eps)
    )

    return corr.masked_fill(
        ~valid,
        torch.nan,
    )


# ==========================================================
# Functional geographic metrics
# ==========================================================

def masked_geo_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_weighted_mean(
        x,
        mask,
        weights=latitude_weights(
            x,
            latitudes,
        ),
        dim=dim,
    )


def masked_geo_bias(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_bias(
        preds,
        target,
        mask,
        weights=latitude_weights(
            preds,
            latitudes,
        ),
        dim=dim,
    )


def masked_geo_mse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_mse(
        preds,
        target,
        mask,
        weights=latitude_weights(
            preds,
            latitudes,
        ),
        dim=dim,
    )


def masked_geo_rmse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return torch.sqrt(
        masked_geo_mse(
            preds,
            target,
            mask,
            latitudes,
            dim=dim,
        )
    )


def masked_geo_mae(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_mae(
        preds,
        target,
        mask,
        weights=latitude_weights(
            preds,
            latitudes,
        ),
        dim=dim,
    )


def masked_geo_crmse(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_crmse(
        preds,
        target,
        mask,
        weights=latitude_weights(
            preds,
            latitudes,
        ),
        dim=dim,
    )


def masked_geo_std(
    x: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    return _masked_std(
        x,
        mask,
        weights=latitude_weights(
            x,
            latitudes,
        ),
        dim=dim,
    )


def masked_geo_std_ratio(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    dim: Dim = None,
) -> torch.Tensor:
    weights = latitude_weights(
        preds,
        latitudes,
    )

    return (
        _masked_std(
            preds,
            mask,
            weights=weights,
            dim=dim,
        )
        / _masked_std(
            target,
            mask,
            weights=weights,
            dim=dim,
        )
    )


# ==========================================================
# Spatial patch metrics
# ==========================================================

def spatial_patch_mse(
    error: torch.Tensor,
    mask: torch.Tensor,
    *,
    patch_size: int,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute spatial patch MSE.

    Returns
    -------
    spatial_patch_mse
        Shape (Hp, Wp).

    valid_spatial_patches
        Boolean mask with shape (Hp, Wp).
    """
    mask_f = expand_mask_to(
        error,
        mask,
    ).to(dtype=error.dtype)

    pooled_sq_err = F.avg_pool2d(
        error.square() * mask_f,
        kernel_size=patch_size,
        stride=patch_size,
        ceil_mode=True,
        count_include_pad=False,
    )

    patch_valid_fraction = F.avg_pool2d(
        mask_f,
        kernel_size=patch_size,
        stride=patch_size,
        ceil_mode=True,
        count_include_pad=False,
    )

    patch_mse = (
        pooled_sq_err
        / patch_valid_fraction.clamp_min(eps)
    )

    patch_valid = (
        patch_valid_fraction > 0
    )

    patch_error_sum = (
        patch_mse
        * patch_valid
    ).sum(dim=(0, 1))

    patch_count = patch_valid.sum(
        dim=(0, 1)
    )

    spatial_patch_mse = (
        patch_error_sum
        / patch_count.clamp_min(1)
    )

    valid_spatial_patches = (
        patch_count > 0
    )

    return (
        spatial_patch_mse,
        valid_spatial_patches,
    )


def spatial_cvar_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    *,
    patch_size: int,
    cvar_fraction: float,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    patch_mse, valid_spatial_patches = (
        spatial_patch_mse(
            y_pred - y_true,
            mask,
            patch_size=patch_size,
            eps=eps,
        )
    )

    patch_losses = patch_mse[
        valid_spatial_patches
    ]

    if patch_losses.numel() == 0:
        raise ValueError(
            "No valid spatial patches."
        )

    num_worst = max(
        1,
        math.ceil(
            cvar_fraction
            * patch_losses.numel()
        ),
    )

    worst_patch_losses, worst_indices = (
        torch.topk(
            patch_losses,
            k=num_worst,
        )
    )

    valid_locations = torch.nonzero(
        valid_spatial_patches,
        as_tuple=False,
    )

    worst_patch_locations = (
        valid_locations[
            worst_indices
        ]
    )

    return (
        worst_patch_losses.mean(),
        worst_patch_locations,
        worst_patch_losses,
        patch_mse,
    )


# ==========================================================
# Stateful TorchMetric base classes
# ==========================================================

class _MaskedMetric(Metric):
    """
    Base class supporting both ordinary and cosine-latitude-weighted
    masked metrics.

    If latitudes=None, effective weights are simply the validity mask.

    If latitudes are supplied, effective weights are:

        mask * cos(latitude)
    """

    def __init__(
        self,
        *,
        latitudes: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if latitudes is None:
            self.register_buffer(
                "latitude_weights",
                None,
            )
        else:
            weights = torch.cos(
                torch.deg2rad(
                    torch.as_tensor(
                        latitudes,
                        dtype=torch.float32,
                    )
                )
            )

            self.register_buffer(
                "latitude_weights",
                weights,
            )

    def _weights(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = expand_mask_to(
            x,
            mask,
        ).to(dtype=x.dtype)

        if self.latitude_weights is None:
            return weights

        if x.ndim != 4:
            raise ValueError(
                "Geographic metrics expect tensors shaped "
                "(N, C, H, W)."
            )

        h = x.shape[-2]

        if self.latitude_weights.numel() != h:
            raise ValueError(
                f"latitudes length "
                f"{self.latitude_weights.numel()} != H {h}"
            )

        geo_weights = (
            self.latitude_weights
            .to(
                device=x.device,
                dtype=x.dtype,
            )
            .view(1, 1, h, 1)
        )

        return (
            weights
            * geo_weights
        )


class _MaskedErrorMetric(_MaskedMetric):
    """
    Base class for metrics expressible as a weighted mean of an
    elementwise prediction error function.
    """

    is_differentiable = False

    sum_value: torch.Tensor
    weight_sum: torch.Tensor

    def __init__(
        self,
        *,
        latitudes: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
        )

        self.add_state(
            "sum_value",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "weight_sum",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def _value(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape, "
                f"got {preds.shape} and {target.shape}"
            )

        weights = self._weights(
            preds,
            mask,
        )

        self.sum_value += (
            self._value(
                preds,
                target,
            )
            * weights
        ).sum()

        self.weight_sum += (
            weights.sum()
        )

    def compute(self) -> torch.Tensor:
        return (
            self.sum_value
            / self.weight_sum
        )


# ==========================================================
# Bias
# ==========================================================

class MaskedBias(_MaskedErrorMetric):
    higher_is_better = None

    def _value(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return preds - target


class MaskedGeoBias(MaskedBias):
    def __init__(
        self,
        latitudes: torch.Tensor,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
        )


# ==========================================================
# MSE / RMSE
# ==========================================================

class MaskedMSE(_MaskedErrorMetric):
    higher_is_better = False

    def _value(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (
            preds - target
        ).square()


class MaskedGeoMSE(MaskedMSE):
    def __init__(
        self,
        latitudes: torch.Tensor,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
        )


class MaskedRMSE(MaskedMSE):
    higher_is_better = False

    def compute(self) -> torch.Tensor:
        return torch.sqrt(
            super().compute()
        )


class MaskedGeoRMSE(MaskedGeoMSE):
    higher_is_better = False

    def compute(self) -> torch.Tensor:
        return torch.sqrt(
            super().compute()
        )


# ==========================================================
# MAE
# ==========================================================

class MaskedMAE(_MaskedErrorMetric):
    higher_is_better = False

    def _value(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (
            preds - target
        ).abs()


class MaskedGeoMAE(MaskedMAE):
    def __init__(
        self,
        latitudes: torch.Tensor,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
        )


# ==========================================================
# Standard-deviation ratio
# ==========================================================

class _MaskedStdRatio(_MaskedMetric):
    is_differentiable = False
    higher_is_better = None

    sum_pred: torch.Tensor
    sum_target: torch.Tensor
    sum_pred_sq: torch.Tensor
    sum_target_sq: torch.Tensor
    weight_sum: torch.Tensor

    def __init__(
        self,
        *,
        latitudes: torch.Tensor | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
        )

        self.eps = eps

        self.add_state(
            "sum_pred",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sum_target",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sum_pred_sq",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sum_target_sq",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "weight_sum",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape, "
                f"got {preds.shape} and {target.shape}"
            )

        weights = self._weights(
            preds,
            mask,
        )

        self.sum_pred += (
            preds * weights
        ).sum()

        self.sum_target += (
            target * weights
        ).sum()

        self.sum_pred_sq += (
            preds.square()
            * weights
        ).sum()

        self.sum_target_sq += (
            target.square()
            * weights
        ).sum()

        self.weight_sum += (
            weights.sum()
        )

    def compute(self) -> torch.Tensor:
        pred_mean = (
            self.sum_pred
            / self.weight_sum
        )

        target_mean = (
            self.sum_target
            / self.weight_sum
        )

        pred_var = (
            self.sum_pred_sq
            / self.weight_sum
            - pred_mean.square()
        )

        target_var = (
            self.sum_target_sq
            / self.weight_sum
            - target_mean.square()
        )

        pred_std = torch.sqrt(
            pred_var.clamp_min(0.0)
        )

        target_std = torch.sqrt(
            target_var.clamp_min(0.0)
        )

        return (
            pred_std
            / target_std.clamp_min(
                self.eps
            )
        )


class MaskedStdRatio(_MaskedStdRatio):
    def __init__(
        self,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            eps=eps,
        )


class MaskedGeoStdRatio(_MaskedStdRatio):
    def __init__(
        self,
        latitudes: torch.Tensor,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
            eps=eps,
        )


# ==========================================================
# Spatial correlation
# ==========================================================

class _MaskedSpatialCorr(_MaskedMetric):
    """
    Mean spatial correlation.

    For inputs shaped (N, C, H, W), computes one spatial
    correlation over H x W for every sample/channel pair,
    then averages all valid correlations.

    When latitudes are supplied, the spatial correlation uses
    cosine-latitude weights.
    """

    is_differentiable = False
    higher_is_better = True

    sum_corr: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        *,
        latitudes: torch.Tensor | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
        )

        self.eps = eps

        self.add_state(
            "sum_corr",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "count",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape, "
                f"got {preds.shape} and {target.shape}"
            )

        if preds.ndim != 4:
            raise ValueError(
                "MaskedSpatialCorr expects tensors shaped "
                "(N, C, H, W)."
            )

        valid_mask = expand_mask_to(
            preds,
            mask,
        )

        weights = self._weights(
            preds,
            valid_mask,
        )

        n, c = preds.shape[:2]

        x = preds.reshape(
            n,
            c,
            -1,
        )

        y = target.reshape(
            n,
            c,
            -1,
        )

        w = weights.reshape(
            n,
            c,
            -1,
        )

        m = valid_mask.reshape(
            n,
            c,
            -1,
        )

        valid_count = m.sum(
            dim=-1,
        )

        weight_sum = w.sum(
            dim=-1,
        )

        denom = weight_sum.clamp_min(
            self.eps
        )

        x_mean = (
            (x * w).sum(dim=-1)
            / denom
        )

        y_mean = (
            (y * w).sum(dim=-1)
            / denom
        )

        x_centered = (
            x
            - x_mean.unsqueeze(-1)
        )

        y_centered = (
            y
            - y_mean.unsqueeze(-1)
        )

        covariance = (
            x_centered
            * y_centered
            * w
        ).sum(dim=-1)

        x_variance = (
            x_centered.square()
            * w
        ).sum(dim=-1)

        y_variance = (
            y_centered.square()
            * w
        ).sum(dim=-1)

        valid = (
            (valid_count > 1)
            & (weight_sum > self.eps)
            & (x_variance > self.eps)
            & (y_variance > self.eps)
        )

        corr = covariance / torch.sqrt(
            x_variance
            * y_variance
        ).clamp_min(self.eps)

        corr = corr[
            valid
        ]

        self.sum_corr += corr.sum()

        self.count += corr.numel()

    def compute(self) -> torch.Tensor:
        return (
            self.sum_corr
            / self.count
        )


class MaskedSpatialCorr(_MaskedSpatialCorr):
    def __init__(
        self,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            eps=eps,
        )


class MaskedGeoSpatialCorr(_MaskedSpatialCorr):
    def __init__(
        self,
        latitudes: torch.Tensor,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
            eps=eps,
        )


# ==========================================================
# Temporal correlation
# ==========================================================

class _MaskedTemporalCorr(_MaskedMetric):
    """
    Mean temporal correlation across valid channel/grid cells.

    Statistics are accumulated over the sample dimension N.

    For every (C, H, W) location:

        corr_n(pred, target)

    is computed over all updates.

    The ordinary version then takes an unweighted mean across
    valid (C, H, W) locations.

    The geographic version takes a cosine-latitude-weighted mean
    across valid (C, H, W) locations.
    """

    is_differentiable = False
    higher_is_better = True

    count: torch.Tensor
    sum_pred: torch.Tensor
    sum_target: torch.Tensor
    sum_pred_sq: torch.Tensor
    sum_target_sq: torch.Tensor
    sum_cross: torch.Tensor

    def __init__(
        self,
        *,
        latitudes: torch.Tensor | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
        )

        self.eps = eps

        # Shape (C, H, W) is not known until the first update.
        self.add_state(
            "count",
            default=torch.tensor([]),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sum_pred",
            default=torch.tensor([]),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sum_target",
            default=torch.tensor([]),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sum_pred_sq",
            default=torch.tensor([]),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sum_target_sq",
            default=torch.tensor([]),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "sum_cross",
            default=torch.tensor([]),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must have the same shape, "
                f"got {preds.shape} and {target.shape}"
            )

        if preds.ndim != 4:
            raise ValueError(
                "MaskedTemporalCorr expects tensors shaped "
                "(N, C, H, W)."
            )

        m = expand_mask_to(
            preds,
            mask,
        ).to(dtype=preds.dtype)

        count = m.sum(
            dim=0
        )

        sum_pred = (
            preds * m
        ).sum(dim=0)

        sum_target = (
            target * m
        ).sum(dim=0)

        sum_pred_sq = (
            preds.square()
            * m
        ).sum(dim=0)

        sum_target_sq = (
            target.square()
            * m
        ).sum(dim=0)

        sum_cross = (
            preds
            * target
            * m
        ).sum(dim=0)

        if self.count.numel() == 0:
            self.count = torch.zeros_like(
                count
            )

            self.sum_pred = torch.zeros_like(
                sum_pred
            )

            self.sum_target = torch.zeros_like(
                sum_target
            )

            self.sum_pred_sq = torch.zeros_like(
                sum_pred_sq
            )

            self.sum_target_sq = torch.zeros_like(
                sum_target_sq
            )

            self.sum_cross = torch.zeros_like(
                sum_cross
            )

        self.count += count
        self.sum_pred += sum_pred
        self.sum_target += sum_target
        self.sum_pred_sq += sum_pred_sq
        self.sum_target_sq += sum_target_sq
        self.sum_cross += sum_cross

    def compute(self) -> torch.Tensor:
        count = self.count

        pred_mean = (
            self.sum_pred
            / count
        )

        target_mean = (
            self.sum_target
            / count
        )

        covariance = (
            self.sum_cross
            - count
            * pred_mean
            * target_mean
        )

        pred_var = (
            self.sum_pred_sq
            - count
            * pred_mean.square()
        )

        target_var = (
            self.sum_target_sq
            - count
            * target_mean.square()
        )

        valid = (
            (count > 1)
            & (pred_var > self.eps)
            & (target_var > self.eps)
        )

        corr = covariance / torch.sqrt(
            pred_var
            * target_var
        ).clamp_min(self.eps)

        if self.latitude_weights is None:
            return corr[
                valid
            ].mean()

        # corr shape: (C, H, W)
        _, h, _ = corr.shape

        if self.latitude_weights.numel() != h:
            raise ValueError(
                f"latitudes length "
                f"{self.latitude_weights.numel()} != H {h}"
            )

        geo_weights = (
            self.latitude_weights
            .to(
                device=corr.device,
                dtype=corr.dtype,
            )
            .view(1, h, 1)
            .expand_as(corr)
        )

        weights = geo_weights[
            valid
        ]

        values = corr[
            valid
        ]

        return (
            (values * weights).sum()
            / weights.sum()
        )


class MaskedTemporalCorr(_MaskedTemporalCorr):
    def __init__(
        self,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            eps=eps,
        )


class MaskedGeoTemporalCorr(_MaskedTemporalCorr):
    def __init__(
        self,
        latitudes: torch.Tensor,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            latitudes=latitudes,
            eps=eps,
        )
