from ._runtime import configure_sources_runtime
from .base import BaseSource
from .dataclasses import SourceConfig, SourceConfigContainer, RegridConfig, DataSource
from .registry import get_source_class, build_source
from .registry import _SOURCES as _REGISTRY_SOURCES
from .registry import _SOURCE_CONFIGS as _REGISTRY_SOURCE_CONFIGS
from .utils import save_zarr

configure_sources_runtime()

__all__ = [
    "BaseSource",
    "get_source_class",
    "build_source",
    "SourceConfig",
    "SourceConfigContainer",
    "RegridConfig",
    "DataSource",
    # source dataclasses
    "EarthkitSource",
    "JunoLocalSource",
    "XarrayLocalSource",
    "CopernicusmarineSource",
    # source config
    "EarthkitSourceConfig",
    "JunoLocalSourceConfig",
    "JunoLocalSourceFileNameConfig",
    "JunoLocalSourcePathConfig",
    "XarrayLocalSourceConfig",
    "CopernicusmarineSourceConfig",
    # utils
    "save_zarr"
]


_SOURCES = {"SumSource": (".combinators", "SumSource")}
for target in _REGISTRY_SOURCES.values():
    module_path, class_name = target.split(":")
    _SOURCES[class_name] = (f".{module_path.rsplit('.', 1)[1]}", class_name)
for target in _REGISTRY_SOURCE_CONFIGS.values():
    module_path, class_name = target.split(":")
    _SOURCES[class_name] = (f".{module_path.rsplit('.', 1)[1]}", class_name)
_SOURCES["JunoLocalSourceFileNameConfig"] = (".juno_local", "JunoLocalSourceFileNameConfig")
_SOURCES["JunoLocalSourcePathConfig"] = (".juno_local", "JunoLocalSourcePathConfig")

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
