from typing import Any, Optional, Callable
import importlib, inspect

import torch
from torch import nn

from .losses import get_loss_class

def resolve_loss(
    name: str,
    params: dict,
):
    """
    Try to resolve loss with thiw priority order:
    - first try our custom losses
    - then if there's no "." in name try to import from torch.nn
    - last resort import directly the module as is with importlib, if "." is name
    """
    # if one of our custom losses
    try:
        return get_loss_class(name)(**params)
    except KeyError:
        loss_import_name = name

    # simple name -> try torch.nn
    if "." not in loss_import_name:
        if hasattr(nn, loss_import_name):
            return getattr(nn, loss_import_name)(**params)
        raise ValueError(f"Loss '{loss_import_name}' not found in torch.nn")

    # dotted path -> dynamic import
    module_path, class_name = loss_import_name.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(**params)


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
