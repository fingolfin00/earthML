from typing import Tuple

import torch


def _masked_stats(
    x: torch.Tensor,
    mask: torch.Tensor,
    reduce_dims: Tuple[int, ...],
    keepdim: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute masked sum, sumsq, and count over reduce_dims.
    mask must be the same shape as x.
    Returns (sum, sumsq, count) with keepdim behavior.
    """
    m = mask.to(dtype=x.dtype, device=x.device)
    s = (x * m).sum(dim=reduce_dims, keepdim=keepdim)
    ssq = ((x * m) ** 2).sum(dim=reduce_dims, keepdim=keepdim)
    count = m.sum(dim=reduce_dims, keepdim=keepdim)
    return s, ssq, count


def masked_mean_var(
    x: torch.Tensor,
    mask: torch.Tensor,
    reduce_dims: Tuple[int, ...],
    unbiased: bool = False,
    keepdim: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (mean, var, count) computed with mask applied.
    Population variance by default (unbiased=False). var >= 0.
    """
    s, ssq, count = _masked_stats(x, mask, reduce_dims, keepdim=keepdim)
    safe_count = count.clamp_min(eps)
    mean = s / safe_count
    var = (ssq / safe_count) - mean * mean
    var = var.clamp_min(0.0)
    if unbiased:
        # Bessel correction only when count > 1
        corr = torch.where(safe_count > 1.0, safe_count / (safe_count - 1.0), torch.ones_like(safe_count))
        var = var * corr
    return mean, var, count
