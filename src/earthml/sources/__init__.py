from .base import BaseSource
from .registry import get_source_class

__all__ = ["BaseSource", "get_source_class"]

_LAZY = {
    "SumSource": (".combinators", "SumSource"),
    "XarrayLocalSource": (".xarray_local", "XarrayLocalSource"),
    "JunoLocalSource": (".juno_local", "JunoLocalSource"),
    "EarthkitSource": (".earthkit", "EarthkitSource"),
}

def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name, __name__)
        value = getattr(mod, attr)
        globals()[name] = value  # cache
        __all__.append(name)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
