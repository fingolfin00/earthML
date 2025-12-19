import importlib
from typing import Any, Type

_NETS: dict[str, str] = {
    "SmaAt_UNet": "earthml.nets.smaatunet:SmaAt_UNet",
    "WeatherUNet": "earthml.nets.weatherunet:WeatherUNet",
    "GNN": "earthml.nets.gnn:GNN",
}

def get_net_class(name: str) -> Type[Any]:
    try:
        target = _NETS[name]
    except KeyError as e:
        raise KeyError(f"Unknown net '{name}'. Known: {sorted(_NETS)}") from e

    module_path, class_name = target.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def build_net(name: str, *args, **kwargs) -> Any:
    cls = get_net_class(name)
    return cls(*args, **kwargs)

def list_nets() -> list[str]:
    return sorted(_NETS.keys())
