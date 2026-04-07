from .base import BaseSource
from .dataclasses import SourceConfig, RegridConfig, DataSource
from .registry import get_source_class, build_source
from .registry import _SOURCES as _REGISTRY_SOURCES

from .earthkit import EarthkitSourceConfig
from .juno_local import JunoLocalSourceConfig, JunoLocalSourceFileNameConfig
from .xarray_local import XarrayLocalSourceConfig
from .copernicusmarine import CopernicusmarineSourceConfig


__all__ = [
    "BaseSource",
    "get_source_class",
    "build_source",
    "SourceConfig",
    "RegridConfig",
    "DataSource",
    "EarthkitSourceConfig",
    "JunoLocalSourceConfig",
    "JunoLocalSourceFileNameConfig",
    "XarrayLocalSourceConfig",
    "CopernicusmarineSourceConfig",
]


_SOURCES = {"SumSource": (".combinators", "SumSource")}
for target in _REGISTRY_SOURCES.values():
    module_path, class_name = target.split(":")
    _SOURCES[class_name] = (f".{module_path.rsplit('.', 1)[1]}", class_name)

def __getattr__(name: str):
    if name in _SOURCES:
        import importlib
        mod_name, attr = _SOURCES[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value  # cache
        __all__.append(name)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
