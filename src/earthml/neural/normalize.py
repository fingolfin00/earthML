from typing import Optional

import joblib, torch

from ..logging import get_logger


logger = get_logger(__name__)


class Normalize:
    """
    Torch normalizer class
    """
    def __init__(self, mean=None, std=None):
        """
        mean/std expected shapes after fit: (1, C, 1, 1)
        """
        self.mean = mean
        self.std = std

    @staticmethod
    def _masked_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        metric_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-12,
        per_channel_mean: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        pred, target: (N, C, H, W)

        Masks (all optional, bool or 0/1, shape (N,C,H,W)):
        - pred_mask:   used for pred mean/std
        - target_mask: used for target mean/std
        - metric_mask: used for error/R2/corr computations

        If a mask is None, it defaults as follows:
        - pred_mask   -> metric_mask if provided else all-ones
        - target_mask -> metric_mask if provided else all-ones
        - metric_mask -> (pred_mask & target_mask) if both provided else whichever is provided else all-ones

        Returns dict with:
        pred_mean, pred_std, target_mean, target_std, mae, rmse, r2, corr
        Shapes:
        (1, C, 1, 1) if per_channel_mean=True
        (1, 1, 1, 1) otherwise
        """

        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: {pred.shape} vs {target.shape}")
        if pred.ndim != 4:
            raise ValueError("Expected pred/target shape (N,C,H,W)")

        device = pred.device
        calc_dtype = (
            torch.float32
            if pred.dtype in (torch.float16, torch.bfloat16)
            else pred.dtype
        )
        pred = pred.to(dtype=calc_dtype)
        target = target.to(dtype=calc_dtype)
        dtype = pred.dtype

        def _as_bool_mask(m: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if m is None:
                return None
            if m.shape != pred.shape:
                raise ValueError(f"Mask shape mismatch: expected {pred.shape}, got {m.shape}")
            return m.to(device=device).bool()

        pred_mask_b = _as_bool_mask(pred_mask)
        target_mask_b = _as_bool_mask(target_mask)
        metric_mask_b = _as_bool_mask(metric_mask)

        if metric_mask_b is None:
            if pred_mask_b is not None and target_mask_b is not None:
                metric_mask_b = pred_mask_b & target_mask_b
            elif pred_mask_b is not None:
                metric_mask_b = pred_mask_b
            elif target_mask_b is not None:
                metric_mask_b = target_mask_b
            else:
                metric_mask_b = torch.ones_like(pred, dtype=torch.bool, device=device)

        if pred_mask_b is None:
            pred_mask_b = metric_mask_b
        if target_mask_b is None:
            target_mask_b = metric_mask_b

        reduce_dims = (0, 2, 3) if per_channel_mean else (0, 1, 2, 3)

        def _masked_mean_std(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            m = m.bool()
            count = m.sum(dim=reduce_dims, keepdim=True).clamp_min(1).to(dtype=dtype)
            s = (x * m).sum(dim=reduce_dims, keepdim=True)
            mu = s / count
            var = ((x - mu) ** 2 * m).sum(dim=reduce_dims, keepdim=True) / count
            sd = torch.sqrt(var + eps)
            return mu, sd

        pred_mean, pred_std = _masked_mean_std(pred, pred_mask_b)
        target_mean, target_std = _masked_mean_std(target, target_mask_b)

        # Metrics computed on metric_mask
        m = metric_mask_b
        count_m = m.sum(dim=reduce_dims, keepdim=True).clamp_min(1).to(dtype=dtype)

        diff = (pred - target) * m
        mae = diff.abs().sum(dim=reduce_dims, keepdim=True) / count_m
        mse = (diff ** 2).sum(dim=reduce_dims, keepdim=True) / count_m
        rmse = torch.sqrt(mse + eps)

        # R² uses target mean over the metric mask (standard definition for masked set)
        target_mean_m, _ = _masked_mean_std(target, m)
        ss_res = (diff ** 2).sum(dim=reduce_dims, keepdim=True)
        ss_tot = ((target - target_mean_m) ** 2 * m).sum(dim=reduce_dims, keepdim=True)
        r2 = 1 - ss_res / (ss_tot + eps)

        # Corr over metric mask
        pred_mean_m, _ = _masked_mean_std(pred, m)
        pred_centered = (pred - pred_mean_m) * m
        target_centered = (target - target_mean_m) * m

        cov = (pred_centered * target_centered).sum(dim=reduce_dims, keepdim=True) / count_m
        pred_var = (pred_centered ** 2).sum(dim=reduce_dims, keepdim=True) / count_m
        target_var = (target_centered ** 2).sum(dim=reduce_dims, keepdim=True) / count_m
        corr = cov / (torch.sqrt(pred_var * target_var) + eps)

        return {
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "corr": corr,
        }

    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "Normalize":
        return joblib.load(filepath)

    def fit(self, dataset, filepath: str | None = None, dim: str = "x", eps: float = 1e-12):
        """
        dataset.<dim>      : (N,C,H,W)
        dataset.<dim>_mask : (N,C,H,W)
        """
        data = getattr(dataset, dim)
        mask = getattr(dataset, f"{dim}_mask")

        stats = self._masked_metrics(
            pred=data,
            target=data,
            pred_mask=mask,
            target_mask=mask,
            metric_mask=mask,
            eps=eps,
            per_channel_mean=True,  # or pass your existing setting
        )

        self.mean = stats["target_mean"]
        self.std = stats["target_std"]

        if filepath is not None:
            self.save(filepath)
        return self

    def _check_filepath(self, filepath: str) -> None:
        if self.fitted():
            return

        loaded = self.load(filepath)
        self.mean = loaded.mean
        self.std = loaded.std

    def _broadcast_params(self, t: torch.Tensor):
        """
        Return mean/std broadcastable to t without mutating state.
        """
        if not self.fitted():
            raise ValueError("Transform not fitted (mean/std are None)")

        if t.ndim == 4:      # (N,C,H,W)
            mean = self.mean
            std = self.std
        elif t.ndim == 3:    # (C,H,W)
            # expect stored (1,C,1,1) -> (C,1,1)
            mean = self.mean.squeeze(0)
            std = self.std.squeeze(0)
        else:
            raise ValueError(f"Unsupported tensor shape {t.shape}")

        if t.ndim == 3 and t.shape[0] != mean.shape[0]:
            raise ValueError(f"Channel mismatch: tensor has C={t.shape[0]}, normalizer has C={mean.shape[0]}")

        return mean.to(device=t.device, dtype=t.dtype), std.to(device=t.device, dtype=t.dtype)

    def __call__(self, t: torch.Tensor, eps: float = 1e-12, **kwargs) -> torch.Tensor:
        mean, std = self._broadcast_params(t)
        return (t - mean) / (std + eps)

    def inverse_tensor(
        self,
        t: torch.Tensor,
        months: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean, std = self._broadcast_params(t)
        return t * std + mean


class MonthlyNormalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean  # (12,C,1,1)
        self.std = std    # (12,C,1,1)

    def fit(self, dataset, dim: str = "x", eps: float = 1e-12):
        data = getattr(dataset, dim)
        mask = getattr(dataset, f"{dim}_mask")
        months = dataset.months

        means = []
        stds = []

        for month in range(1, 13):
            sel = months == month

            if not torch.any(sel):
                raise ValueError(f"No samples available for month {month}")

            x_m = data[sel]
            m_m = mask[sel]

            stats = Normalize._masked_metrics(
                pred=x_m,
                target=x_m,
                pred_mask=m_m,
                target_mask=m_m,
                metric_mask=m_m,
                eps=eps,
                per_channel_mean=True,
            )

            means.append(stats["target_mean"].squeeze(0))
            stds.append(stats["target_std"].squeeze(0))

        self.mean = torch.stack(means, dim=0)
        self.std = torch.stack(stds, dim=0)
        return self

    def __call__(self, t: torch.Tensor, month: int, eps: float = 1e-12):
        month_idx = month - 1
        mean = self.mean[month_idx].to(t.device, t.dtype)
        std = self.std[month_idx].to(t.device, t.dtype)
        return (t - mean) / (std + eps)

    def inverse_tensor(self, t: torch.Tensor, months: torch.Tensor, eps: float = 1e-12):
        out = torch.empty_like(t)

        for i in range(t.shape[0]):
            month_idx = int(months[i].item()) - 1
            mean = self.mean[month_idx].to(t.device, t.dtype)
            std = self.std[month_idx].to(t.device, t.dtype)
            out[i] = t[i] * (std + eps) + mean

        return out
