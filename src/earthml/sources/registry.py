from typing import Any
import importlib

from .dataclasses import DataSource

_SOURCES = {
    "xarray-local": "earthml.sources.xarray_local:XarrayLocalSource",
    "juno-local": "earthml.sources.juno_local:JunoLocalSource",
    "earthkit": "earthml.sources.earthkit:EarthkitSource",
}

_SOURCE_CONFIGS = {
    "xarray-local": "earthml.sources.xarray_local:XarrayLocalSourceConfig",
    "juno-local": "earthml.sources.juno_local:JunoLocalSourceConfig",
    "earthkit": "earthml.sources.earthkit:EarthkitSourceConfig",
}


def _import_target(target: str):
    mod, cls = target.split(":")
    return getattr(importlib.import_module(mod), cls)


# Source building
def get_source_class(name: str):
    return _import_target(_SOURCES[name])

def build_source(name: str, params: dict[str, DataSource | Any]): # TODO use full list of supported config dataclasses, not Any
    cls = get_source_class(name)
    return cls(**params)

def list_sources() -> list[str]:
    return sorted(_SOURCES.keys())


# Source config building
def get_source_config_class(name: str):
    target = _SOURCE_CONFIGS.get(name)
    if target is None:
        return None
    return _import_target(target)

def build_source_config(name: str, **kwargs): # TODO unused, but may be useful
    config_cls = get_source_config_class(name)
    if config_cls is None:
        raise KeyError(f"Source '{name}' does not define a config dataclass")
    return config_cls(**kwargs)
