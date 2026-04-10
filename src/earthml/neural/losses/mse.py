from typing import Literal, Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from ...logging import get_logger
from .utils import _expand_mask_to, _masked_mean_var


logger = get_logger(__name__)


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
    def __init__(self, latitudes: torch.Tensor, eps: float = 1e-12):
        super().__init__()
        weights = torch.cos(torch.deg2rad(latitudes))
        self.register_buffer("weights", weights)  # 1D tensor length H
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
