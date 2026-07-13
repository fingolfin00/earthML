from typing import Any, Optional, Callable
import importlib, inspect

import torch
from torch import nn

from .losses import get_loss_class

def resolve_loss(
    name: str,
    params: dict | None = None,
) -> nn.Module:
    params = params or {}

    try:
        return get_loss_class(name)(**params)
    except KeyError:
        pass

    if "." not in name:
        loss_cls = getattr(nn, name, None)

        if loss_cls is None:
            raise ValueError(
                f"Loss {name!r} not found in the custom registry or torch.nn"
            )

        return loss_cls(**params)

    module_path, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    loss_cls = getattr(module, class_name)

    loss_instance = loss_cls(**params)

    if not isinstance(loss_instance, nn.Module):
        raise TypeError(
            f"Resolved loss {name!r} is not an nn.Module: "
            f"{type(loss_instance).__name__}"
        )

    return loss_instance


def call_loss(
    loss_fn: Callable,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calls loss_fn with supported kwargs only.
    Supports losses with signatures:
      - loss(y_pred, y_true)
      - loss(y_pred, y_true, mask=...)
      - loss(y_pred, y_true, x_input=..., mask=...)
    """
    sig = inspect.signature(loss_fn.forward if hasattr(loss_fn, "forward") else loss_fn)
    params = sig.parameters

    kwargs: dict[str, Any] = {}

    # only pass mask if accepted
    if mask is not None and "mask" in params:
        kwargs["mask"] = mask

    # Only pass the input tensor when the loss explicitly asks for it.
    if x0 is not None and "x_input" in params:
        kwargs["x_input"] = x0

    return loss_fn(y_pred, y_true, **kwargs)
