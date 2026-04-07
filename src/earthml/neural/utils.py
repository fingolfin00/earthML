from typing import Any, Optional, Callable
import importlib, inspect

import torch
from torch import nn

from ..losses.registry import get_loss_class

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
      - loss(y_pred, y_true, x_input, mask=...)  (your custom)
    """
    sig = inspect.signature(loss_fn.forward if hasattr(loss_fn, "forward") else loss_fn)
    params = sig.parameters

    kwargs: dict[str, Any] = {}

    # only pass mask if accepted
    if mask is not None and "mask" in params:
        kwargs["mask"] = mask

    # pass x_input if requested and accepted (you call it x_input in your custom)
    if x0 is not None:
        if "x_input" in params:
            kwargs["x_input"] = x0
            return loss_fn(y_pred, y_true, **kwargs)
        # some custom losses might take it positionally, but your code uses positional (mu,y,x0,...)
        # If forward has 3rd positional param and you want to use it, do it explicitly:
        positional = [p for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(positional) >= 3:
            return loss_fn(y_pred, y_true, x0, **kwargs)

    return loss_fn(y_pred, y_true, **kwargs)
