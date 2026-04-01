import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric
from torchmetrics.image import SpatialCorrelationCoefficient


class MaskedMAE(Metric):
    is_differentiable = False
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_abs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        m = mask.to(dtype=preds.dtype, device=preds.device)
        self.sum_abs += (preds.sub(target).abs() * m).sum()
        self.count += m.sum()

    def compute(self):
        return self.sum_abs / self.count # NaN if count is zero


class MaskedRMSE(Metric):
    is_differentiable = False
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        m = mask.to(dtype=preds.dtype, device=preds.device)
        self.sum_sq += ((preds - target) ** 2 * m).sum()
        self.count += m.sum()

    def compute(self):
        return torch.sqrt(self.sum_sq / self.count) # NaN if count is zero


class MaskedSpatialCorr(Metric):
    is_differentiable = False
    higher_is_better = True

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.add_state("sum_corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # preds/target: (N,C,H,W), mask broadcastable to that
        m = mask.to(dtype=preds.dtype, device=preds.device)
        N, C = preds.shape[:2]
        x = preds.reshape(N, C, -1)
        y = target.reshape(N, C, -1)
        mm = m.reshape(N, C, -1)

        count = mm.sum(dim=-1)  # (N,C)
        valid = count > 1

        # masked means
        denom = count.clamp_min(1.0)
        mx = (x * mm).sum(dim=-1) / denom
        my = (y * mm).sum(dim=-1) / denom

        xc = (x - mx.unsqueeze(-1)) * mm
        yc = (y - my.unsqueeze(-1)) * mm

        cov = (xc * yc).sum(dim=-1)
        vx = (xc * xc).sum(dim=-1)
        vy = (yc * yc).sum(dim=-1)

        corr = cov / (torch.sqrt(vx * vy).clamp_min(self.eps))
        corr = corr[valid]

        self.sum_corr += corr.sum()
        self.num += torch.tensor(corr.numel(), device=self.sum_corr.device, dtype=self.sum_corr.dtype)

    def compute(self):
        return self.sum_corr / self.num # NaN if count is zero
