from typing import Literal, Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F


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
