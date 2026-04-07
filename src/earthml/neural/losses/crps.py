from typing import Literal, Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from .utils import _expand_mask_to


# -------------------------
# EmpiricalCRPSLoss
# -------------------------
class EmpiricalCRPSLoss(nn.Module):
    """
    Empirical CRPS for ensemble predictions when realizations are packed into
    an existing tensor dimension and y_true repeats the same target across realizations.

    Typical use:
      - channel-packed ensemble:
          y_pred, y_true: (N, C*R, H, W)
          packed_dim=1, num_realizations=R
          internally reshaped to (N, C, R, H, W)

      - time-packed ensemble:
          y_pred, y_true: (N, C, T*R, H, W)
          packed_dim=2, num_realizations=R
          internally reshaped to (N, C, T, R, H, W)

    Assumptions:
      - Along packed_dim, realizations are contiguous with R as the fastest sub-index.
        This matches your dataset's reshape(..., C*R, ...) / reshape(..., T*R, ...) convention.
      - y_true is duplicated across realizations along the packed dimension.
      - mask is duplicated the same way.

    CRPS at each target location:
        mean_r |x_r - y| - 0.5 * mean_{r,s} |x_r - x_s|

    Final reduction:
        masked mean over valid target elements,
        optionally latitude-weighted like GeoWeightedMSELoss.
    """
    def __init__(
        self,
        num_realizations: int,
        packed_dim: int = 1,
        latitudes: Optional[torch.Tensor] = None,
        eps: float = 1e-12,
        fair: bool = False,
    ):
        super().__init__()
        if num_realizations < 1:
            raise ValueError("num_realizations must be >= 1")
        self.num_realizations = int(num_realizations)
        self.packed_dim = int(packed_dim)
        self.eps = float(eps)
        self.fair = bool(fair)

        if latitudes is not None and not isinstance(latitudes, torch.Tensor):
            latitudes = torch.tensor(latitudes, dtype=torch.float32)
        self.register_buffer("latitudes", latitudes)

    def _canonical_dim(self, ndim: int, dim: int) -> int:
        if dim < 0:
            dim = ndim + dim
        if dim < 0 or dim >= ndim:
            raise ValueError(f"Invalid dim={dim} for tensor with ndim={ndim}")
        return dim

    def _unpack_realizations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split packed_dim of size base * R into (..., base, R, ...),
        with R as the fastest sub-index.
        """
        pdim = self._canonical_dim(x.ndim, self.packed_dim)
        D = x.shape[pdim]
        R = self.num_realizations

        if D % R != 0:
            raise ValueError(
                f"Size of packed_dim ({D}) must be divisible by num_realizations ({R})"
            )

        base = D // R
        new_shape = x.shape[:pdim] + (base, R) + x.shape[pdim + 1:]
        return x.reshape(new_shape)

    def _build_lat_weights(self, y_base: torch.Tensor) -> torch.Tensor:
        """
        Build latitude weights broadcastable to y_base.
        Follows your existing convention:
          - 4D: (N,C,H,W)   -> lat axis = 2
          - 5D: (N,C,T,H,W) -> lat axis = 3
        """
        if self.latitudes is None:
            return torch.ones_like(y_base, dtype=y_base.dtype, device=y_base.device)

        if y_base.ndim not in (4, 5):
            raise ValueError(
                "Latitude weighting supports y_base with ndim 4 or 5: "
                "(N,C,H,W) or (N,C,T,H,W)"
            )

        lat = self.latitudes.to(dtype=y_base.dtype, device=y_base.device)
        lat_axis = 2 if y_base.ndim == 4 else 3
        H = y_base.shape[lat_axis]
        if lat.numel() != H:
            raise ValueError(f"latitudes length {lat.numel()} != H {H}")

        w_lat = torch.cos(torch.deg2rad(lat))
        if y_base.ndim == 4:
            w = w_lat.view(1, 1, H, 1)
        else:
            w = w_lat.view(1, 1, 1, H, 1)

        return w.expand_as(y_base)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred ({y_pred.shape}) and y_true ({y_true.shape}) must have the same shape"
            )

        mask_b = _expand_mask_to(y_true, mask).to(device=y_true.device, dtype=torch.bool)

        # Unpack packed ensemble dimension:
        # e.g. (N,C*R,H,W) -> (N,C,R,H,W)
        #      (N,C,T*R,H,W) -> (N,C,T,R,H,W)
        pred_g = self._unpack_realizations(y_pred)
        true_g = self._unpack_realizations(y_true)
        mask_g = self._unpack_realizations(mask_b)

        pdim = self._canonical_dim(y_pred.ndim, self.packed_dim)
        rdim = pdim + 1  # after unpacking, realization dim is inserted right after packed_dim's base dim

        # Recover the single target from repeated y_true.
        # Mask-aware mean across realization dim.
        mask_f = mask_g.to(dtype=true_g.dtype, device=true_g.device)
        true_count = mask_f.sum(dim=rdim)
        true_sum = (true_g * mask_f).sum(dim=rdim)
        y_base = true_sum / true_count.clamp_min(self.eps)   # shape without R
        valid_base = true_count > 0                         # boolean mask without R

        # First term: mean_r |x_r - y|
        term_obs = torch.abs(pred_g - y_base.unsqueeze(rdim)).mean(dim=rdim)

        # Second term: ensemble spread correction
        # spread = sum_{i<j} |x_i - x_j|
        # unfair: spread / R^2
        # fair:   spread / (R * (R - 1))
        R = pred_g.shape[rdim]
        if R < 2:
            raise ValueError("Fair/empirical CRPS requires at least 2 realizations")

        spread = torch.zeros_like(term_obs)
        for i in range(R - 1):
            spread = spread + torch.abs(
                pred_g.select(rdim, i).unsqueeze(rdim) - pred_g.narrow(rdim, i + 1, R - i - 1)
            ).sum(dim=rdim)

        if self.fair:
            term_ens = spread / (R * (R - 1))
        else:
            term_ens = spread / (R * R)

        crps_pointwise = term_obs - term_ens  # shape = y_base.shape

        # Final weighting / reduction
        valid_base_f = valid_base.to(dtype=crps_pointwise.dtype, device=crps_pointwise.device)
        w = self._build_lat_weights(y_base).to(dtype=crps_pointwise.dtype, device=crps_pointwise.device)
        w_masked = w * valid_base_f

        denom = w_masked.sum()
        if denom.item() == 0:
            raise ValueError("EmpiricalCRPSLoss: mask has zero valid (weighted) elements")
        denom = denom.clamp_min(self.eps)

        return (crps_pointwise * w_masked).sum() / denom
