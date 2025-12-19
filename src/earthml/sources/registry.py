import importlib

_SOURCES = {
    "xarray-local": "earthml.sources.xarray_local:XarrayLocalSource",
    "juno-local": "earthml.sources.juno_local:JunoLocalSource",
    "earthkit": "earthml.sources.earthkit:EarthkitSource",
}

def get_source_class (name: str):
    target = _SOURCES[name]
    mod, cls = target.split(":")
    return getattr(importlib.import_module(mod), cls)

def build_source (name: str, **kwargs):
    cls = get_source_class(name)
    return cls(**kwargs)

def list_sources () -> list[str]:
    return sorted(_SOURCES.keys())
