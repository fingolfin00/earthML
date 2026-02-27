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
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        y_pred: torch.Tensor,        # corrected forecast
        y_true: torch.Tensor,        # analysis
        var_field: torch.Tensor | None = None,
    ):
        sq_err = (y_pred - y_true) ** 2

        if var_field is not None:
            denom = var_field + self.eps
        else:
            # per-channel variance estimated from batch
            est_var = y_true.var(dim=(0, 2, 3), keepdim=True)
            denom = est_var + self.eps

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
