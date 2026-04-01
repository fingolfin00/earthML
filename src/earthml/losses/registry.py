import importlib
from typing import Any, Type


_LOSSES: dict[str, str] = {
    "MSELoss"                   : "torch.nn.MSELoss",
    "MaskedMSELoss"             : "earthml.losses.mse:MaskedMSELoss",
    "GeoMSELoss"                : "earthml.losses.mse:GeoMSELoss",
    "GeoMaskedMSELoss"          : "earthml.losses.mse:GeoMaskedMSELoss",
    "VarNormMaskMSELoss"        : "earthml.losses.mse:VarNormMaskMSELoss",
    "HeteroBiasCorrectionLoss"  : "earthml.losses.mse:HeteroBiasCorrectionLoss",
    "EmpiricalCRPSLoss"         : "earthml.losses.crps:EmpiricalCRPSLoss",
    "GaussianNLLFromLogits"     : "earthml.losses.nll:GaussianNLLFromLogits", # TODO not working
}


def get_loss_class(name: str) -> Type[Any]:
    try:
        target = _LOSSES[name]
    except KeyError as e:
        raise KeyError(f"Unknown net '{name}'. Known: {sorted(_LOSSES)}") from e

    module_path, class_name = target.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def build_loss(name: str, *args, **kwargs) -> Any:
    cls = get_loss_class(name)
    return cls(*args, **kwargs)

def list_losses() -> list[str]:
    return sorted(_LOSSES.keys())
