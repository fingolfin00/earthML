from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F

class GeoWeightedMSELoss (nn.Module):
    """
    Spatially weighted MSE loss.
    """
    def __init__(self, latitudes: torch.Tensor):
        super().__init__()
        weights = torch.cos(torch.deg2rad(latitudes))
        self.register_buffer("weights", weights) # will move with model and be on same device/dtype as inputs

    def forward(
        self,
        y_pred: torch.Tensor,        # corrected forecast
        y_true: torch.Tensor,        # analysis
    ):
        sq_err = (y_pred - y_true) ** 2 # shape NCHW

        # Reshape for broadcasting over H
        w = self.weights.view(1, 1, sq_err.shape[2], 1)

        # Expand to NCHW and guard against division by zero
        return (sq_err * w).sum() / w.expand_as(sq_err).sum().clamp_min(1e-12)

class VarianceNormalizedMSELoss (nn.Module):
    """
    MSE loss normalized by variance of the target field.

    Supported variance types:
      - "channel"      : per-channel variance across batch and all non-channel dims
      - "geochannel"   : same as "channel" but latitude-weighted (requires `latitudes`)
      - "spatial"      : per-spatial variance across batch and channel -> shape (1,1,H,W)
      - "temporal"     : per-channel variance across time & batch (requires time dim)
      - "geotemporal"  : temporal variance weighted by latitude (requires time dim & latitudes)

    Input shapes supported:
      - 4D: (N, C, H, W)
      - 5D: (N, C, T, H, W)
    """
    def __init__ (
        self,
        eps: float = 1e-12,
        variance_type: Literal["channel", "geochannel", "spatial", "temporal", "geotemporal"] = "channel",
        latitudes: Optional[torch.Tensor] = None,  # 1D tensor of latitudes (degrees) length == H
    ):
        super().__init__()
        self.eps = float(eps)
        self.variance_type = variance_type

        if variance_type in ("geochannel", "geotemporal"):
            if latitudes is None:
                raise ValueError(f"latitudes must be provided for variance_type='{variance_type}'")
            # keep raw lat array; converted to module device/dtype in forward
            if not isinstance(latitudes, torch.Tensor):
                latitudes = torch.tensor(latitudes, dtype=torch.float32)
            self.register_buffer("latitudes", latitudes)
        else:
            self.register_buffer("latitudes", None)

    def _dims_except_channel (self, x: torch.Tensor):
        return tuple(d for d in range(x.ndim) if d != 1)

    def forward (
        self,
        y_pred: torch.Tensor,        # corrected forecast
        y_true: torch.Tensor,        # analysis (target)
        var_field: Optional[torch.Tensor] = None,
    ):
        """
        y_true / y_pred: shape (N, C, H, W) or (N, C, T, H, W)
        var_field: optional externally provided variance; must be broadcastable to sq_err
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_pred and y_true must have the same shape")

        sq_err = (y_pred - y_true) ** 2

        if var_field is not None:
            # Use provided variance (assumed already appropriate shape), clamp to avoid 0
            denom = var_field.clamp_min(self.eps)
        else:
            x = y_true
            ndim = x.ndim
            if ndim not in (4, 5):
                raise ValueError("y_true must be 4D (N,C,H,W) or 5D (N,C,T,H,W)")

            # Sum over all dims except channel (1)
            sum_dims = self._dims_except_channel(x)

            if self.variance_type == "channel":
                # Per-channel variance across batch and all non-channel dims
                # Mean per-channel:
                mean = x.mean(dim=sum_dims, keepdim=True)
                est_var = ((x - mean) ** 2).mean(dim=sum_dims, keepdim=True)
                denom = est_var.clamp_min(self.eps)

            elif self.variance_type == "geochannel":
                # Weighted per-channel variance with latitudes (weights along H axis)
                lat = self.latitudes
                if lat is None:
                    raise ValueError("latitudes buffer missing for 'geochannel'")
                # Ensure lat on same device/dtype as x
                lat = lat.to(dtype=x.dtype, device=x.device)

                # Find latitude axis index (H). For 4D: idx=2, for 5D: idx=3
                lat_axis = 2 if ndim == 4 else 3
                H = x.shape[lat_axis]
                if lat.numel() != H:
                    raise ValueError(f"latitudes length {lat.numel()} does not match H={H}")

                # Build weight tensor broadcastable to x shape
                # Start with shape (H,)
                w = torch.cos(torch.deg2rad(lat)).view([1, 1] + ([H] if ndim == 4 else [1, H]) + ([1] if ndim == 4 else [1]))
                # The above view yields:
                #  - 4D: (1,1,H,1)
                #  - 5D: (1,1,1,H,1)  -> later expand to x shape

                w = w.expand_as(x)

                # Weighted mean per-channel
                numerator = (x * w).sum(dim=sum_dims, keepdim=True)
                denom_w = w.sum(dim=sum_dims, keepdim=True).clamp_min(1e-12)
                mean_w = numerator / denom_w

                # Weighted variance per-channel
                est_var = (((x - mean_w) ** 2) * w).sum(dim=sum_dims, keepdim=True) / denom_w
                denom = est_var.clamp_min(self.eps)

            elif self.variance_type == "spatial":
                # Variance per-spatial location across batch and channel.
                # Result shape: (1,1,H,W) for 4D, (1,1,T,H,W) for 5D
                # Sum dims: batch (0) and channel (1)
                if ndim == 4:
                    est_var = x.var(dim=(0,1), unbiased=False, keepdim=True)  # shape (1,1,H,W)
                else:
                    # 5D: (N,C,T,H,W) -> variance over N and C -> shape (1,1,1,H,W)
                    est_var = x.var(dim=(0,1,2), unbiased=False, keepdim=True)
                denom = est_var.clamp_min(self.eps)

            elif self.variance_type == "temporal":
                # Per-channel variance across time and batch. Requires time dim (5D).
                if ndim != 5:
                    # Fallback: treat as channel variance if no time dim exists
                    mean = x.mean(dim=sum_dims, keepdim=True)
                    est_var = ((x - mean) ** 2).mean(dim=sum_dims, keepdim=True)
                else:
                    # Sum over batch and time and spatial dims except channel:
                    sum_dims_temp = tuple(d for d in range(x.ndim) if d not in (1,))  # sum all except channel
                    mean = x.mean(dim=sum_dims_temp, keepdim=True)
                    est_var = ((x - mean) ** 2).mean(dim=sum_dims_temp, keepdim=True)
                denom = est_var.clamp_min(self.eps)

            elif self.variance_type == "geotemporal":
                # Weighted in latitude and across time & batch; requires 5D input
                if ndim != 5:
                    raise ValueError("'geotemporal' requires a 5D input (N,C,T,H,W)")
                lat = self.latitudes
                if lat is None:
                    raise ValueError("latitudes buffer missing for 'geotemporal'")
                lat = lat.to(dtype=x.dtype, device=x.device)
                lat_axis = 3  # (N,C,T,H,W) -> H is axis 3
                H = x.shape[lat_axis]
                if lat.numel() != H:
                    raise ValueError(f"latitudes length {lat.numel()} does not match H={H}")

                # Create weight shape (1,1,1,H,1) and expand
                w = torch.cos(torch.deg2rad(lat)).view(1, 1, 1, H, 1).expand_as(x)

                sum_dims_geo = tuple(d for d in range(x.ndim) if d != 1)  # all except channel
                numerator = (x * w).sum(dim=sum_dims_geo, keepdim=True)
                denom_w = w.sum(dim=sum_dims_geo, keepdim=True).clamp_min(1e-12)
                mean_w = numerator / denom_w
                est_var = (((x - mean_w) ** 2) * w).sum(dim=sum_dims_geo, keepdim=True) / denom_w
                denom = est_var.clamp_min(self.eps)

            else:
                raise ValueError(f"Unknown variance_type: {self.variance_type}")

        # Final normalized MSE
        var_norm_mse = (sq_err / denom).mean()
        return var_norm_mse

class HeteroBiasCorrectionLoss (nn.Module):
    """
    Loss for bias correction with:
    - variance-normalized MSE
    - identity-preserving term when true bias is small
    """
    def __init__(
        self,
        lambda_identity: float = 0.1,
        bias_scale: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.bias_scale = bias_scale
        self.eps = eps
        self.loss_components = None

    def forward(
        self,
        y_pred: torch.Tensor,        # corrected forecast
        y_true: torch.Tensor,        # analysis
        x_input: torch.Tensor,       # raw forecast
        var_field: torch.Tensor | None = None,
    ):
        # Variance-normalized MSE term
        sq_err = (y_pred - y_true) ** 2

        if var_field is not None:
            denom = var_field + self.eps
        else:
            # Per-channel variance estimated from batch
            est_var = y_true.var(dim=(0, 2, 3), keepdim=True)
            denom = est_var + self.eps

        var_norm_mse = (sq_err / denom).mean()

        # Identity-preserving term
        # True bias between raw input and target
        true_bias = y_true - x_input
        bias_mag = true_bias.abs()

        # Weight strong where true bias is small
        w_id = torch.exp(-bias_mag / self.bias_scale)

        # Penalize changing the input when its bias is small
        id_err = (y_pred - x_input) ** 2
        identity_loss = (w_id * id_err).mean()

        total_loss = var_norm_mse + self.lambda_identity * identity_loss

        self.loss_components = {
            "var_norm_mse": var_norm_mse.detach(),
            "identity_loss": identity_loss.detach(),
        }
        return total_loss

class GaussianNLLFromLogits(nn.Module):
    """
    Expects pred to be either:
      - tuple/list: (mu, var)  where var is variance (positive)
      - tensor:     [B, 2*C, H, W] where channels are (mu, raw_var)
    Converts raw_var -> var via softplus, then applies GaussianNLLLoss.
    """
    def __init__(
        self,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.nll = nn.GaussianNLLLoss(eps=eps, reduction=reduction)

    def forward(self, pred, target):
        if isinstance(pred, (tuple, list)):
            mu, var = pred
        else:
            mu, raw_var = torch.chunk(pred, 2, dim=1)
            var = F.softplus(raw_var) + self.eps

        return self.nll(mu, target, var)
