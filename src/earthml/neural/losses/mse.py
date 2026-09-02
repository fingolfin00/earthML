from typing import Literal, Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from ...logging import get_logger
from .utils import _expand_mask_to, _masked_mean_var


logger = get_logger(__name__)


def _spatial_patch_mse(
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
    Compute geographical patch MSE.

    Returns:
        spatial_patch_mse:
            Shape (Hp, Wp).

        valid_spatial_patches:
            Boolean mask with shape (Hp, Wp).
    """
    mask_f = mask.to(
        device=error.device,
        dtype=error.dtype,
    )

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

    patch_valid = patch_valid_fraction > 0

    patch_error_sum = (
        patch_mse * patch_valid
    ).sum(dim=(0, 1))

    patch_count = patch_valid.sum(dim=(0, 1))

    spatial_patch_mse = (
        patch_error_sum
        / patch_count.clamp_min(1)
    )

    valid_spatial_patches = patch_count > 0

    return (
        spatial_patch_mse,
        valid_spatial_patches,
    )


def _spatial_cvar_mse(
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
    spatial_patch_mse, valid_spatial_patches = (
        _spatial_patch_mse(
            y_pred - y_true,
            mask,
            patch_size=patch_size,
            eps=eps,
        )
    )

    patch_losses = spatial_patch_mse[
        valid_spatial_patches
    ]

    if patch_losses.numel() == 0:
        raise ValueError(
            "No valid spatial patches"
        )

    num_worst = max(
        1,
        math.ceil(
            cvar_fraction
            * patch_losses.numel()
        ),
    )

    worst_patch_losses, worst_indices = torch.topk(
        patch_losses,
        k=num_worst,
    )

    valid_locations = torch.nonzero(
        valid_spatial_patches,
        as_tuple=False,
    )

    worst_patch_locations = valid_locations[
        worst_indices
    ]

    return (
        worst_patch_losses.mean(),
        worst_patch_locations,
        worst_patch_losses,
        spatial_patch_mse,
    )


class SpatialCVaRDiagnostics(nn.Module):
    """
    Common spatial robustness diagnostics independent of training loss.
    """

    def __init__(
        self,
        latitudes: torch.Tensor,
        patch_size: int = 10,
        cvar_fraction: float = 0.2,
        relative_floor_fraction: float = 0.05,
        relative_degradation_thresholds: tuple[float, ...] = (
            0.0,
            0.01,
            0.05,
            0.10,
        ),
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if patch_size < 1:
            raise ValueError("patch_size must be positive")

        if not 0.0 < cvar_fraction <= 1.0:
            raise ValueError(
                "cvar_fraction must be in (0, 1]"
            )

        if relative_floor_fraction < 0.0:
            raise ValueError(
                "relative_floor_fraction must be non-negative"
            )

        if any(
            threshold < 0.0
            for threshold in relative_degradation_thresholds
        ):
            raise ValueError(
                "relative degradation thresholds must be non-negative"
            )

        self.patch_size = int(patch_size)
        self.cvar_fraction = float(cvar_fraction)
        self.relative_floor_fraction = float(
            relative_floor_fraction
        )
        self.relative_degradation_thresholds = tuple(
            float(threshold)
            for threshold in relative_degradation_thresholds
        )
        self.eps = float(eps)

        self.global_loss = GeoMaskedMSELoss(
            latitudes=latitudes,
            eps=eps,
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        mask_b = _expand_mask_to(
            y_true,
            mask,
        ).to(
            device=y_true.device,
            dtype=torch.bool,
        )

        global_geo_mse = self.global_loss(
            y_pred,
            y_true,
            mask=mask_b,
        )

        (
            cvar_mse,
            worst_patch_locations,
            worst_patch_losses,
            spatial_patch_mse,
        ) = _spatial_cvar_mse(
            y_pred,
            y_true,
            mask_b,
            patch_size=self.patch_size,
            cvar_fraction=self.cvar_fraction,
            eps=self.eps,
        )

        model_patch_mse, model_valid = _spatial_patch_mse(
            y_pred - y_true,
            mask_b,
            patch_size=self.patch_size,
            eps=self.eps,
        )

        baseline_patch_mse, baseline_valid = _spatial_patch_mse(
            -y_true,
            mask_b,
            patch_size=self.patch_size,
            eps=self.eps,
        )

        valid_patches = model_valid & baseline_valid

        if not valid_patches.any():
            raise ValueError(
                "SpatialCVaRDiagnostics: no valid spatial patches"
            )

        patch_degradation = (
            model_patch_mse
            - baseline_patch_mse
        )

        valid_degradation = patch_degradation[
            valid_patches
        ]

        degraded_patches = (
            (patch_degradation > 0.0)
            & valid_patches
        )

        mean_patch_degradation = (
            valid_degradation.mean()
        )

        mean_positive_patch_degradation = (
            valid_degradation
            .clamp_min(0.0)
            .mean()
        )

        if degraded_patches.any():
            mean_degraded_patch_degradation = (
                patch_degradation[
                    degraded_patches
                ].mean()
            )
        else:
            mean_degraded_patch_degradation = (
                patch_degradation.sum() * 0.0
            )

        degraded_patch_fraction = (
            degraded_patches.sum()
            / valid_patches.sum().clamp_min(1)
        )

        max_patch_degradation = (
            valid_degradation.max()
        )

        # Relative degradation diagnostics.
        mean_baseline_patch_mse = (
            baseline_patch_mse[
                valid_patches
            ].mean()
        )

        baseline_floor = (
            self.relative_floor_fraction
            * mean_baseline_patch_mse
        ).clamp_min(self.eps)

        baseline_scale = (
            baseline_patch_mse
            .clamp_min(baseline_floor)
        )

        relative_patch_degradation = (
            patch_degradation
            / baseline_scale
        )

        result = {
            "global_geo_mse": global_geo_mse,
            "cvar_mse": cvar_mse,
            "cvar_ratio": (
                cvar_mse
                / global_geo_mse.clamp_min(self.eps)
            ),
            "worst_patch_locations": worst_patch_locations,
            "worst_patch_losses": worst_patch_losses,
            "spatial_patch_mse": spatial_patch_mse,

            # Absolute degradation diagnostics.
            "mean_patch_degradation": (
                mean_patch_degradation
            ),
            "mean_degraded_patch_degradation": (
                mean_degraded_patch_degradation
            ),
            "mean_positive_patch_degradation": (
                mean_positive_patch_degradation
            ),
            "degraded_patch_fraction": (
                degraded_patch_fraction
            ),
            "max_patch_degradation": (
                max_patch_degradation
            ),
            "patch_degradation": patch_degradation,
            "degraded_patches": degraded_patches,

            # Relative degradation diagnostics.
            "relative_patch_degradation": (
                relative_patch_degradation
            ),
            "mean_baseline_patch_mse": (
                mean_baseline_patch_mse
            ),
            "baseline_floor": baseline_floor,
        }

        valid_count = valid_patches.sum().clamp_min(1)

        for threshold in self.relative_degradation_thresholds:
            degraded = (
                (relative_patch_degradation > threshold)
                & valid_patches
            )

            fraction = (
                degraded.sum()
                / valid_count
            )

            threshold_pct = round(
                100 * threshold
            )

            result[
                f"relative_degraded_patch_fraction_gt_{threshold_pct}pct"
            ] = fraction

        return result


# -------------------------
# MaskedMSELoss
# -------------------------
class MaskedMSELoss(nn.Module):
    """
    MSE loss that respects a boolean mask. Returns sum(sq_err) / (number_of_valid_elements).
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if y_pred.shape != y_true.shape:
            raise ValueError(f"y_pred ({y_pred.shape}) and y_true ({y_true.shape}) must have the same shape")

        # Broadcast mask to y_true shape; ensure boolean on same device
        mask_b = _expand_mask_to(y_true, mask).to(device=y_true.device, dtype=torch.bool)

        sq_err = (y_pred - y_true) ** 2
        sq_err = sq_err * mask_b.to(dtype=sq_err.dtype, device=sq_err.device)  # zero invalid

        # valid_count as float on same device/dtype as sq_err to avoid dtype/device surprises
        valid_count = mask_b.to(dtype=sq_err.dtype, device=sq_err.device).sum()
        if valid_count.item() == 0: # sync to CPU to check if zero
            raise ValueError("No valid pixels in mask")
        valid_count = valid_count.clamp_min(self.eps)
        return sq_err.sum() / valid_count


# -------------------------
# GeoMSELoss
# -------------------------
class GeoMSELoss(nn.Module):
    """
    Spatially weighted MSE loss that ignores boolean mask.
    latitudes: 1D tensor length H (degrees)
    """
    def __init__(self, latitudes: torch.Tensor, eps: float = 1e-12):
        super().__init__()
        weights = torch.cos(torch.deg2rad(latitudes))
        self.register_buffer("weights", weights)  # 1D tensor length H
        logger.debug("GeoMSELoss eps=%s", eps)
        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        y_pred / y_true: (N,C,H,W)
        mask: ignored, kept to respect signature of custom losses
        """
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape")

        sq_err = (y_pred - y_true) ** 2

        # build latitude weight shape and broadcast
        lat = self.weights.to(dtype=y_true.dtype, device=y_true.device)
        # assume 4D (N,C,H,W)
        H = y_true.shape[2]
        if lat.numel() != H:
            raise ValueError(f"latitudes length {lat.numel()} != H {H}")
        w = lat.view(1, 1, H, 1).expand_as(y_true).to(dtype=sq_err.dtype)

        # compute weighted mean over all pixels
        numerator = (sq_err * w).sum()
        denom = w.sum()
        if denom.item() == 0:
            raise ValueError("GeoMSELoss: zero valid weighted elements")
        denom = denom.clamp_min(self.eps)
        return numerator / denom


# -------------------------
# GeoMaskedMSELoss
# -------------------------
class GeoMaskedMSELoss(nn.Module):
    """
    Spatially weighted MSE loss that respects a boolean mask.
    latitudes: 1D tensor length H (degrees)
    """
    def __init__(
        self,
        latitudes: torch.Tensor,
        eps: float = 1e-12,
        persistent: bool = True,
    ):
        super().__init__()

        weights = torch.cos(
            torch.deg2rad(latitudes)
        )

        self.register_buffer(
            "weights",
            weights,
            persistent=persistent,
        )

        self.eps = float(eps)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        y_pred / y_true: (N,C,H,W)
        mask: broadcastable to y_true (True=valid)
        """
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape")

        mask_b = _expand_mask_to(y_true, mask).to(device=y_true.device, dtype=torch.bool)  # boolean same shape as y_true
        sq_err = (y_pred - y_true) ** 2
        # sq_err = sq_err * mask_b.to(dtype=sq_err.dtype, device=sq_err.device)  # redundant

        # build latitude weight shape and broadcast
        lat = self.weights.to(dtype=y_true.dtype, device=y_true.device)
        # assume 4D (N,C,H,W)
        H = y_true.shape[2]
        if lat.numel() != H:
            raise ValueError(f"latitudes length {lat.numel()} != H {H}")
        w = lat.view(1, 1, H, 1).expand_as(y_true).to(dtype=sq_err.dtype)

        # zero-out weights at masked positions, then compute weighted mean over valid pixels
        w_masked = w * mask_b.to(dtype=w.dtype, device=w.device)
        numerator = (sq_err * w_masked).sum()
        denom = w_masked.sum()
        if denom.item() == 0:
            raise ValueError("GeoMaskedMSELoss: mask has zero valid (weighted) elements")
        denom = denom.clamp_min(self.eps)

        return numerator / denom


# -------------------------
# VarNormMaskMSELoss
# -------------------------
class VarNormMaskMSELoss(nn.Module):
    """
    MSE normalized by variance with mask support.
    - spatial: variance at each spatial cell across (N,C) -> shape (1,1,H,W)
    - channel: per-channel variance across all non-channel dims -> shape (1,C,1,1)
    - geochannel, temporal, geotemporal supported (masked)
    """
    def __init__(
        self,
        eps: float = 1e-12,
        variance_type: Literal["channel", "geochannel", "spatial", "temporal", "geotemporal"] = "channel",
        latitudes: Optional[torch.Tensor] = None,
        relative_floor_frac: float = 1e-3,
        min_valid_count: int = 1,
    ):
        super().__init__()
        self.eps = float(eps)
        self.variance_type = variance_type
        self.relative_floor_frac = float(relative_floor_frac)
        self.min_valid_count = int(min_valid_count)

        if variance_type in ("geochannel", "geotemporal"):
            if latitudes is None:
                raise ValueError(f"latitudes must be provided for variance_type='{variance_type}'")
            if not isinstance(latitudes, torch.Tensor):
                latitudes = torch.tensor(latitudes, dtype=torch.float32)
            self.register_buffer("latitudes", latitudes)
        else:
            self.register_buffer("latitudes", None)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        var_field: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape")

        # mask expanded to full shape
        mask_b = _expand_mask_to(y_true, mask).to(device=y_true.device, dtype=torch.bool)

        sq_err = (y_pred - y_true) ** 2
        sq_err = sq_err * mask_b.to(dtype=sq_err.dtype, device=sq_err.device)  # zero invalid

        x = y_true
        ndim = x.ndim
        if ndim not in (4, 5):
            raise ValueError("y_true must be 4D (N,C,H,W) or 5D (N,C,T,H,W)")

        # If var_field provided, trust it (but clamp) — validate broadcastability
        if var_field is not None:
            var_field = var_field.to(device=y_true.device, dtype=y_true.dtype)
            try:
                torch.broadcast_shapes(var_field.shape, sq_err.shape)
            except RuntimeError as e:
                raise ValueError("var_field must be broadcastable to y_true/target shape") from e
            denom = var_field.clamp_min(self.eps)
        else:
            # choose reduction dims common to many types:
            # For "channel" we reduce over all except dim=1 (channel)
            # For "spatial" we reduce over (0,1) to keep spatial dims
            if self.variance_type == "channel":
                reduce_dims = tuple(d for d in range(x.ndim) if d != 1)
                _, var, count = _masked_mean_var(x, mask_b, reduce_dims, unbiased=False, keepdim=True, eps=self.eps)
                denom = var.clamp_min(self.eps)

            elif self.variance_type == "geochannel":
                # weighted per-channel variance along lat dimension
                lat = self.latitudes
                if lat is None:
                    raise ValueError("latitudes buffer missing for 'geochannel'")
                lat = lat.to(dtype=x.dtype, device=x.device)
                lat_axis = 2 if ndim == 4 else 3
                H = x.shape[lat_axis]
                if lat.numel() != H:
                    raise ValueError(f"latitudes length {lat.numel()} does not match H={H}")

                # build w with shape broadcastable to x
                if ndim == 4:
                    w = torch.cos(torch.deg2rad(lat)).view(1,1,H,1).expand_as(x).to(dtype=x.dtype)
                else:
                    w = torch.cos(torch.deg2rad(lat)).view(1,1,1,H,1).expand_as(x).to(dtype=x.dtype)
                # zero weights at invalid positions
                w = w * mask_b.to(dtype=w.dtype, device=w.device)
                # compute weighted mean/var per-channel over all dims except channel
                reduce_dims = tuple(d for d in range(x.ndim) if d != 1)
                numerator = (x * w).sum(dim=reduce_dims, keepdim=True)
                wsum = w.sum(dim=reduce_dims, keepdim=True)
                if (wsum == 0).any():
                    raise ValueError("geochannel: no valid weighted elements for some channels/positions")
                wsum = wsum.clamp_min(self.eps)
                mean_w = numerator / wsum
                var_w = (((x - mean_w) ** 2) * w).sum(dim=reduce_dims, keepdim=True) / wsum
                var_w = var_w.clamp_min(0.0)
                denom = var_w.clamp_min(self.eps)

            elif self.variance_type == "spatial":
                # reduce over batch and channel -> keep spatial dims (and time if present)
                reduce_dims = (0, 1)
                _, var, count = _masked_mean_var(x, mask_b, reduce_dims, unbiased=False, keepdim=True, eps=self.eps)
                # put a relative floor based on mean variance to avoid tiny denom at a few pixels
                mean_var = var.mean().item()
                floor_val = max(self.eps, self.relative_floor_frac * mean_var)
                # if count < min_valid_count, use a larger floor so that poorly-sampled cells are downweighted
                small_count_mask = (count < self.min_valid_count)
                denom = var.clone()
                denom = denom.clamp_min(floor_val)
                if small_count_mask.any():
                    big_floor_t = torch.tensor(max(floor_val, 10.0 * self.eps), device=denom.device, dtype=denom.dtype)
                    denom = torch.where(small_count_mask, torch.maximum(denom, big_floor_t), denom)

            elif self.variance_type == "temporal":
                # per-channel variance across batch + time + spatial dims
                if ndim != 5:
                    # fallback: channel variance
                    reduce_dims = tuple(d for d in range(x.ndim) if d != 1)
                    _, var, _ = _masked_mean_var(x, mask_b, reduce_dims, unbiased=False, keepdim=True, eps=self.eps)
                else:
                    # reduce over batch and time and spatial dims -> keep channel
                    reduce_dims = (0, 2, 3, 4)
                    _, var, count = _masked_mean_var(x, mask_b, reduce_dims, unbiased=False, keepdim=True, eps=self.eps)
                denom = var.clamp_min(self.eps)

            elif self.variance_type == "geotemporal":
                # weighted in latitude across time & batch; requires 5D
                if ndim != 5:
                    raise ValueError("'geotemporal' requires a 5D input (N,C,T,H,W)")
                lat = self.latitudes
                if lat is None:
                    raise ValueError("latitudes buffer missing for 'geotemporal'")
                lat = lat.to(dtype=x.dtype, device=x.device)
                H = x.shape[3]
                if lat.numel() != H:
                    raise ValueError(f"latitudes length {lat.numel()} does not match H={H}")

                w = torch.cos(torch.deg2rad(lat)).view(1,1,1,H,1).expand_as(x).to(dtype=x.dtype)
                # zero weights at masked positions
                w = w * mask_b.to(dtype=w.dtype, device=w.device)
                reduce_dims = tuple(d for d in range(x.ndim) if d != 1)
                numerator = (x * w).sum(dim=reduce_dims, keepdim=True)
                wsum = w.sum(dim=reduce_dims, keepdim=True)
                if (wsum == 0).any():
                    raise ValueError("geotemporal: no valid weighted elements for some channels/positions")
                wsum = wsum.clamp_min(self.eps)
                mean_w = numerator / wsum
                var_w = (((x - mean_w) ** 2) * w).sum(dim=reduce_dims, keepdim=True) / wsum
                denom = var_w.clamp_min(self.eps)

            else:
                raise ValueError(f"Unknown variance_type: {self.variance_type}")

        # final masked average: sum((sq_err / denom) over valid pixels) / number_of_valid_pixels
        # denom should broadcast to sq_err shape
        # count valid pixels overall:
        valid_count = mask_b.to(dtype=sq_err.dtype, device=sq_err.device).sum()
        if valid_count.item() == 0:
            raise ValueError("VarNormMaskMSELoss: mask has zero valid elements")
        valid_count = valid_count.clamp_min(self.eps)
        loss = (sq_err / denom).sum() / valid_count
        return loss


# -------------------------
# HeteroBiasCorrectionLoss
# -------------------------
class HeteroBiasCorrectionLoss(nn.Module):
    """
    Loss = VarNormMaskMSELoss + lambda_identity * identity-preserving term.

    Uses VarNormMaskMSELoss for the variance-normalized MSE term (same masking and var_field logic),
    and keeps the same identity-preserving logic as before.
    """
    def __init__(
        self,
        lambda_identity: float = 0.1,
        bias_scale: float = 0.5,
        eps: float = 1e-6,
        # pass-through knobs for VarNormMaskMSELoss behavior
        variance_type: Literal["channel", "geochannel", "spatial", "temporal", "geotemporal"] = "channel",
        latitudes: Optional[torch.Tensor] = None,
        relative_floor_frac: float = 1e-3,
        min_valid_count: int = 1,
    ):
        super().__init__()
        self.lambda_identity = float(lambda_identity)
        self.bias_scale = float(bias_scale)
        self.eps = float(eps)

        self.var_mse = VarNormMaskMSELoss(
            eps=eps,
            variance_type=variance_type,
            latitudes=latitudes,
            relative_floor_frac=relative_floor_frac,
            min_valid_count=min_valid_count,
        )

        self.loss_components = None

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x_input: torch.Tensor,
        var_field: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape")
        try:
            _ = (y_true - x_input)
        except RuntimeError:
            raise ValueError("x_input must be broadcastable to y_true")

        # Variance-normalized MSE term (delegated VarianceNormalizedMSELoss, includes mask validity checks)
        var_norm_mse = self.var_mse(y_pred=y_pred, y_true=y_true, var_field=var_field, mask=mask)

        # Identity-preserving term
        mask_b = _expand_mask_to(y_true, mask).to(device=y_true.device, dtype=torch.bool)

        true_bias = y_true - x_input
        bias_mag = true_bias.abs()

        w_id = torch.exp(-bias_mag / self.bias_scale)
        w_id = w_id * mask_b.to(dtype=w_id.dtype, device=w_id.device)

        id_err = (y_pred - x_input) ** 2
        w_id_sum = w_id.sum().clamp_min(self.eps)
        identity_loss = (w_id * id_err).sum() / w_id_sum

        total_loss = var_norm_mse + self.lambda_identity * identity_loss

        self.loss_components = {
            "var_norm_mse": var_norm_mse.detach().cpu(),
            "identity_loss": identity_loss.detach().cpu(),
        }
        return total_loss


# -------------------------
# GaussianNLLFromLogits
# -------------------------
class GaussianNLLFromLogits(nn.Module):
    """
    Computes Gaussian NLL from logits (mu, raw_var) with optional mask:
      - if pred is (mu, var) pair, var must be positive already
      - if pred is tensor with 2*C channels, split into mu and raw_var -> var = softplus(raw_var)+eps
    Masking: compute elementwise NLL, zero invalid positions, then average over valid pixels.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if isinstance(pred, (tuple, list)):
            mu, var = pred
        else:
            mu, raw_var = torch.chunk(pred, 2, dim=1)
            var = F.softplus(raw_var) + self.eps

        if mu.shape != target.shape:
            raise ValueError("mu and target must have same shape")

        mask_b = _expand_mask_to(target, mask).to(device=target.device, dtype=torch.bool)
        # Gaussian NLL per element (no reduction):
        # 0.5 * (log(2*pi*var) + (target-mu)^2 / var)
        var = var.clamp_min(self.eps)
        diff2 = (target - mu) ** 2
        nll_elem = 0.5 * (torch.log(2.0 * math.pi * var) + diff2 / var)
        nll_elem = nll_elem * mask_b.to(dtype=nll_elem.dtype, device=nll_elem.device)
        valid_count = mask_b.to(dtype=nll_elem.dtype, device=nll_elem.device).sum()
        if valid_count.item() == 0:
            raise ValueError("GaussianNLLFromLogits: mask has zero valid elements")
        valid_count = valid_count.clamp_min(self.eps)
        loss = nll_elem.sum() / valid_count
        return loss


class GeoMaskedMSEMultiScaleLoss(nn.Module):
    """
    Latitude-weighted masked pixel MSE plus multiscale bias penalties.

    The loss contains:

        pixel_loss
        + lambda_multiscale * sample_multiscale_loss
        + lambda_batch_mean * batch_mean_multiscale_loss

    sample_multiscale_loss:
        Penalizes spatially pooled signed errors independently for every sample.

    batch_mean_multiscale_loss:
        Groups samples by month, computes their mean signed error, and penalizes
        coherent spatial bias patterns at several spatial scales.

    Args:
        latitudes:
            Latitude coordinate with shape (H,).

        pool_kernel_sizes:
            Spatial pooling window sizes. For example, (5, 11, 21) penalizes
            increasingly broad error structures.

        scale_weights:
            Optional weight for each pooling scale. If omitted, all scales
            receive equal weight.

        lambda_multiscale:
            Weight applied to the per-sample multiscale loss.

        lambda_batch_mean:
            Weight applied to the month-grouped batch-mean multiscale loss.

        pool_stride:
            Pooling stride. If None, each scale uses stride equal to its
            kernel size. A smaller fixed stride gives overlapping windows.

        eps:
            Numerical stability constant.
    """
    def __init__(
        self,
        latitudes: torch.Tensor,
        pool_kernel_sizes: tuple[int, ...] = (5, 11, 21),
        scale_weights: tuple[float, ...] | None = None,
        lambda_multiscale: float = 0.1,
        lambda_batch_mean: float = 0.1,
        lambda_identity: float = 0.1,
        bias_scale: float = 0.5,
        pool_stride: int | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if not pool_kernel_sizes:
            raise ValueError(
                "pool_kernel_sizes must contain at least one scale"
            )

        if any(kernel_size < 1 for kernel_size in pool_kernel_sizes):
            raise ValueError(
                "All pool kernel sizes must be positive"
            )

        if pool_stride is not None and pool_stride < 1:
            raise ValueError(
                "pool_stride must be positive or None"
            )

        if scale_weights is None:
            scale_weights = tuple(
                1.0 for _ in pool_kernel_sizes
            )

        if len(scale_weights) != len(pool_kernel_sizes):
            raise ValueError(
                "scale_weights and pool_kernel_sizes must have "
                "the same length"
            )

        if any(weight < 0 for weight in scale_weights):
            raise ValueError(
                "scale_weights must be non-negative"
            )

        weight_sum = sum(scale_weights)

        if weight_sum <= 0:
            raise ValueError(
                "At least one scale weight must be positive"
            )

        self.pool_kernel_sizes = tuple(
            int(size) for size in pool_kernel_sizes
        )
        self.pool_stride = pool_stride
        self.lambda_multiscale = float(lambda_multiscale)
        self.lambda_batch_mean = float(lambda_batch_mean)
        self.lambda_identity = float(lambda_identity)
        self.bias_scale = float(bias_scale)
        self.eps = float(eps)

        latitudes = torch.as_tensor(
            latitudes,
            dtype=torch.float32,
        )

        self.register_buffer(
            "latitude_weights",
            torch.cos(torch.deg2rad(latitudes)),
        )

        normalized_scale_weights = torch.tensor(
            scale_weights,
            dtype=torch.float32,
        )
        normalized_scale_weights /= normalized_scale_weights.sum()

        self.register_buffer(
            "scale_weights",
            normalized_scale_weights,
        )

        self.pixel_loss = GeoMaskedMSELoss(
            latitudes=latitudes,
            eps=eps,
        )

        self.loss_components: dict[str, torch.Tensor] | None = None

    def _stride_for_scale(
        self,
        kernel_size: int,
    ) -> int:
        if self.pool_stride is None:
            return kernel_size

        return self.pool_stride

    def _masked_pool(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        kernel_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mask-aware spatial mean pooling.

        Returns:
            pooled_x:
                Mean value over valid cells in each window.

            pooled_valid_fraction:
                Fraction of valid cells in each window.
        """
        mask_f = mask.to(
            device=x.device,
            dtype=x.dtype,
        )

        stride = self._stride_for_scale(kernel_size)

        pooled_sum = F.avg_pool2d(
            x * mask_f,
            kernel_size=kernel_size,
            stride=stride,
            ceil_mode=True,
            count_include_pad=False,
        )

        pooled_valid_fraction = F.avg_pool2d(
            mask_f,
            kernel_size=kernel_size,
            stride=stride,
            ceil_mode=True,
            count_include_pad=False,
        )

        pooled_x = (
            pooled_sum
            / pooled_valid_fraction.clamp_min(self.eps)
        )

        return pooled_x, pooled_valid_fraction

    def _pool_geo_weights(
        self,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        kernel_size: int,
    ) -> torch.Tensor:
        """
        Pool latitude-area weights using the same geometry as the error.
        """
        lat_weights = self.latitude_weights.to(
            device=device,
            dtype=dtype,
        )

        if lat_weights.numel() != height:
            raise ValueError(
                f"Latitude count {lat_weights.numel()} "
                f"does not match tensor height {height}"
            )

        geo_weights = lat_weights.view(
            1,
            1,
            height,
            1,
        ).expand(
            1,
            1,
            height,
            width,
        )

        stride = self._stride_for_scale(kernel_size)

        return F.avg_pool2d(
            geo_weights,
            kernel_size=kernel_size,
            stride=stride,
            ceil_mode=True,
            count_include_pad=False,
        )

    def _single_scale_loss(
        self,
        error: torch.Tensor,
        mask: torch.Tensor,
        *,
        kernel_size: int,
    ) -> torch.Tensor:
        """
        Penalize pooled signed error for one spatial scale.
        """
        pooled_error, pooled_valid_fraction = self._masked_pool(
            error,
            mask,
            kernel_size=kernel_size,
        )

        _, _, height, width = error.shape

        pooled_geo_weights = self._pool_geo_weights(
            height=height,
            width=width,
            device=error.device,
            dtype=error.dtype,
            kernel_size=kernel_size,
        )

        weights = pooled_geo_weights * pooled_valid_fraction

        numerator = (pooled_error.square() * weights).sum()
        denominator = weights.sum().clamp_min(self.eps)

        return numerator / denominator

    def _multiscale_loss(
        self,
        error: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Compute weighted loss over all configured pooling scales.
        """
        scale_losses = [
            self._single_scale_loss(
                error,
                mask,
                kernel_size=kernel_size,
            )
            for kernel_size in self.pool_kernel_sizes
        ]

        stacked_losses = torch.stack(scale_losses)

        scale_weights = self.scale_weights.to(
            device=stacked_losses.device,
            dtype=stacked_losses.dtype,
        )

        total = (stacked_losses * scale_weights).sum()

        return total, scale_losses

    def _batch_mean_error(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the masked mean signed error across the batch.
        """
        mask_f = mask.to(
            device=y_pred.device,
            dtype=y_pred.dtype,
        )

        error = y_pred - y_true

        valid_count = mask_f.sum(
            dim=0,
            keepdim=True,
        )

        mean_error = (
            error * mask_f
        ).sum(
            dim=0,
            keepdim=True,
        ) / valid_count.clamp_min(1.0)

        mean_mask = valid_count > 0

        return mean_error, mean_mask

    def _month_grouped_batch_mean_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: torch.Tensor,
        months: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize multiscale patterns in the mean error of each month group.
        """
        month_losses: list[torch.Tensor] = []

        for month in months.unique():
            selected = months == month

            if selected.sum().item() < 2:
                continue

            mean_error, mean_mask = self._batch_mean_error(
                y_pred=y_pred[selected],
                y_true=y_true[selected],
                mask=mask[selected],
            )

            month_loss, _ = self._multiscale_loss(
                mean_error,
                mean_mask,
            )

            month_losses.append(month_loss)

        if not month_losses:
            return y_pred.sum() * 0.0

        return torch.stack(month_losses).mean()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        x_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        months: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "y_pred and y_true must have the same shape"
            )

        if y_pred.ndim != 4:
            raise ValueError(
                "Expected tensors with shape (N, C, H, W)"
            )

        mask_b = _expand_mask_to(
            y_true,
            mask,
        ).to(
            device=y_true.device,
            dtype=torch.bool,
        )

        try:
            _ = y_true - x_input
        except RuntimeError as e:
            raise ValueError(
                "x_input must be broadcastable to y_true"
            ) from e

        pixel_loss = self.pixel_loss(
            y_pred,
            y_true,
            mask=mask_b,
        )

        error = y_pred - y_true

        multiscale_loss, sample_scale_losses = (
            self._multiscale_loss(
                error,
                mask_b,
            )
        )

        if months is None:
            batch_mean_loss = y_pred.sum() * 0.0
        else:
            months = months.to(
                device=y_pred.device,
                dtype=torch.long,
            )

            if months.ndim != 1:
                raise ValueError(
                    "months must be a 1D tensor"
                )

            if months.shape[0] != y_pred.shape[0]:
                raise ValueError(
                    "months length must match batch size"
                )

            batch_mean_loss = (
                self._month_grouped_batch_mean_loss(
                    y_pred=y_pred,
                    y_true=y_true,
                    mask=mask_b,
                    months=months,
                )
            )

        weighted_multiscale_loss = self.lambda_multiscale * multiscale_loss

        weighted_batch_mean_loss = self.lambda_batch_mean * batch_mean_loss

        # Identity-preserving bias-correction term
        true_bias = y_true - x_input
        bias_mag = true_bias.abs()

        # High weight where the true correction is small
        w_identity = torch.exp(
            -bias_mag / self.bias_scale
        )

        # Apply validity mask
        w_identity = (
            w_identity
            * mask_b.to(
                dtype=w_identity.dtype,
                device=w_identity.device,
            )
        )

        _, _, height, _ = y_true.shape

        lat_weights = self.latitude_weights.to(
            device=y_true.device,
            dtype=y_true.dtype,
        ).view(1, 1, height, 1)

        w_identity = (
            torch.exp(-bias_mag / self.bias_scale)
            * lat_weights
            * mask_b.to(dtype=y_true.dtype)
        )

        identity_loss = (
            w_identity * (y_pred - x_input).square()
        ).sum() / w_identity.sum().clamp_min(self.eps)

        weighted_identity_loss = self.lambda_identity * identity_loss

        total_loss = (
            pixel_loss
            + weighted_multiscale_loss
            + weighted_batch_mean_loss
            + weighted_identity_loss
        )

        components: dict[str, torch.Tensor] = {
            "pixel_loss": pixel_loss.detach(),
            "multiscale_loss": multiscale_loss.detach(),
            "batch_mean_loss": batch_mean_loss.detach(),
            "identity_loss": identity_loss.detach(),
            "weighted_multiscale_loss": (
                weighted_multiscale_loss.detach()
            ),
            "weighted_batch_mean_loss": (
                weighted_batch_mean_loss.detach()
            ),
            "weighted_identity_loss": (
                weighted_identity_loss.detach()
            ),
        }

        for kernel_size, scale_loss in zip(
            self.pool_kernel_sizes,
            sample_scale_losses,
        ):
            components[
                f"scale_{kernel_size}_loss"
            ] = scale_loss.detach()

        self.loss_components = components

        return total_loss


class SpatialCVaRMSELoss(nn.Module):
    """
    Latitude-weighted global MSE plus a spatial CVaR penalty.

    The spatial field is divided into non-overlapping patches.
    MSE is computed independently for each patch, and the mean
    of the worst fraction of patch losses is added to the global loss.

    This reduces compensation between well-performing and poorly-performing
    geographical regions.

    Loss:

        global_geo_mse + lambda_cvar * spatial_cvar_mse

    Args:
        latitudes:
            Latitude coordinate with shape (H,).

        patch_size:
            Spatial patch size in grid cells.

        cvar_fraction:
            Fraction of worst patches included in the CVaR term.
            For example, 0.2 means the worst 20% of patches.

        lambda_cvar:
            Weight applied to the spatial CVaR term.

        eps:
            Numerical stability constant.
    """
    def __init__(
        self,
        latitudes: torch.Tensor,
        patch_size: int = 10,
        cvar_fraction: float = 0.2,
        lambda_cvar: float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if patch_size < 1:
            raise ValueError("patch_size must be positive")

        if not 0.0 < cvar_fraction <= 1.0:
            raise ValueError(
                "cvar_fraction must be in (0, 1]"
            )

        self.patch_size = int(patch_size)
        self.cvar_fraction = float(cvar_fraction)
        self.lambda_cvar = float(lambda_cvar)
        self.eps = float(eps)

        self.global_loss = GeoMaskedMSELoss(
            latitudes=latitudes,
            eps=eps,
            persistent=False,
        )

        self.loss_components = None

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask_b = _expand_mask_to(
            y_true,
            mask,
        ).to(
            device=y_true.device,
            dtype=torch.bool,
        )

        global_loss = self.global_loss(
            y_pred,
            y_true,
            mask=mask_b,
        )

        cvar_loss, _, _, _ = _spatial_cvar_mse(
            y_pred,
            y_true,
            mask_b,
            patch_size=self.patch_size,
            cvar_fraction=self.cvar_fraction,
            eps=self.eps,
        )

        weighted_cvar_loss = (
            self.lambda_cvar * cvar_loss
        )

        total_loss = (
            global_loss
            + weighted_cvar_loss
        )

        self.loss_components = {
            "global_loss": global_loss.detach(),
            "cvar_loss": cvar_loss.detach(),
            "weighted_cvar_loss": (
                weighted_cvar_loss.detach()
            ),
        }

        return total_loss


class SpatialDegradationMSELoss(nn.Module):
    """
    Latitude-weighted global MSE plus a penalty on the worst
    relative spatial degradations with respect to a zero-residual baseline.

    The spatial field is divided into non-overlapping patches.

    This loss assumes that predicting zero corresponds to applying
    no correction to the baseline forecast. It is therefore intended
    for residual correction targets.

    For each geographical patch:

        relative_degradation =
            (model_patch_mse - baseline_patch_mse)
            / baseline_scale

    The worst fraction of patches according to relative degradation
    is selected, and only positive degradation within that tail is penalized.

    Final loss:

        global_geo_mse
            + lambda_degradation
            * worst_relative_degradation_loss

    Args:
        latitudes:
            Latitude coordinate with shape (H,).

        patch_size:
            Spatial patch size in grid cells.

        lambda_degradation:
            Weight applied to the relative degradation penalty.

        degradation_fraction:
            Fraction of spatial patches included in the worst-degradation tail.
            For example, 0.2 means the worst 20% of patches.

        relative_floor_fraction:
            Minimum denominator used for relative degradation,
            expressed as a fraction of mean baseline patch MSE.

        eps:
            Numerical stability constant.
    """

    def __init__(
        self,
        latitudes: torch.Tensor,
        patch_size: int = 10,
        lambda_degradation: float = 0.5,
        degradation_fraction: float = 0.2,
        relative_floor_fraction: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if patch_size < 1:
            raise ValueError(
                "patch_size must be positive"
            )

        if lambda_degradation < 0:
            raise ValueError(
                "lambda_degradation must be non-negative"
            )

        if not 0.0 < degradation_fraction <= 1.0:
            raise ValueError(
                "degradation_fraction must be in (0, 1]"
            )

        if relative_floor_fraction < 0:
            raise ValueError(
                "relative_floor_fraction must be non-negative"
            )

        self.patch_size = int(patch_size)
        self.lambda_degradation = float(
            lambda_degradation
        )
        self.degradation_fraction = float(
            degradation_fraction
        )
        self.relative_floor_fraction = float(
            relative_floor_fraction
        )
        self.eps = float(eps)

        self.global_loss = GeoMaskedMSELoss(
            latitudes=latitudes,
            eps=eps,
        )

        self.loss_components: (
            dict[str, torch.Tensor] | None
        ) = None

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "y_pred and y_true must have the same shape"
            )

        if y_pred.ndim != 4:
            raise ValueError(
                "Expected tensors with shape (N, C, H, W)"
            )

        mask_b = _expand_mask_to(
            y_true,
            mask,
        ).to(
            device=y_true.device,
            dtype=torch.bool,
        )

        global_loss = self.global_loss(
            y_pred,
            y_true,
            mask=mask_b,
        )

        model_patch_mse, model_valid = (
            _spatial_patch_mse(
                y_pred - y_true,
                mask_b,
                patch_size=self.patch_size,
                eps=self.eps,
            )
        )

        baseline_patch_mse, baseline_valid = (
            _spatial_patch_mse(
                -y_true,
                mask_b,
                patch_size=self.patch_size,
                eps=self.eps,
            )
        )

        valid_patches = (
            model_valid
            & baseline_valid
        )

        if not valid_patches.any():
            raise ValueError(
                "SpatialDegradationMSELoss: "
                "no valid spatial patches"
            )

        mean_baseline_patch_mse = (
            baseline_patch_mse[
                valid_patches
            ].mean()
        )

        baseline_floor = (
            self.relative_floor_fraction
            * mean_baseline_patch_mse
        ).clamp_min(self.eps)

        baseline_scale = (
            baseline_patch_mse
            .clamp_min(baseline_floor)
        )

        relative_patch_degradation = (
            model_patch_mse
            - baseline_patch_mse
        ) / baseline_scale

        valid_relative_degradation = (
            relative_patch_degradation[
                valid_patches
            ]
        )

        num_worst = max(
            1,
            math.ceil(
                self.degradation_fraction
                * valid_relative_degradation.numel()
            ),
        )

        worst_relative_degradation = torch.topk(
            valid_relative_degradation,
            k=num_worst,
        ).values

        worst_positive_degradation = (
            worst_relative_degradation
            .clamp_min(0.0)
        )

        mean_worst_relative_degradation = (
            worst_positive_degradation.mean()
        )

        weighted_degradation_loss = (
            self.lambda_degradation
            * mean_worst_relative_degradation
        )

        total_loss = (
            global_loss
            + weighted_degradation_loss
        )

        degraded_patches = (
            (relative_patch_degradation > 0)
            & valid_patches
        )

        degraded_patch_fraction = (
            degraded_patches.sum()
            / valid_patches.sum().clamp_min(1)
        )

        loss_ratio = (
            weighted_degradation_loss
            / global_loss.detach().clamp_min(self.eps)
        )

        self.loss_components = {
            "global_loss": global_loss.detach(),

            "mean_worst_relative_degradation": (
                mean_worst_relative_degradation.detach()
            ),

            "weighted_degradation_loss": (
                weighted_degradation_loss.detach()
            ),

            "loss_ratio": (
                loss_ratio.detach()
            ),

            "degraded_patch_fraction": (
                degraded_patch_fraction.detach()
            ),

            "mean_baseline_patch_mse": (
                mean_baseline_patch_mse.detach()
            ),

            "baseline_floor": (
                baseline_floor.detach()
            ),
        }

        return total_loss
