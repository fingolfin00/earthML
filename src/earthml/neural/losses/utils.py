from typing import Optional, Tuple

import torch


def _expand_mask_to(
    x: torch.Tensor,
    mask: Optional[torch.Tensor]
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

def _masked_stats(
    x: torch.Tensor,
    mask: torch.Tensor,
    reduce_dims: Tuple[int, ...],
    keepdim: bool = True
) -> Tuple[torch.Tensor]:
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

def _masked_mean_var(
    x: torch.Tensor,
    mask: torch.Tensor,
    reduce_dims: Tuple[int, ...],
    unbiased: bool = False,
    keepdim: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor]:
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
