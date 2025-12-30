from typing import Callable, Dict, Type
from importlib import metadata

from .base import BaseLauncher

_LAUNCHERS: Dict[str, Type[BaseLauncher]] = {}

def register_launcher (name: str):
    """Decorator generator"""
    def deco (cls: Type[BaseLauncher]):
        _LAUNCHERS[name] = cls
        cls.launcher_name = name
        return cls
    return deco

def load_entrypoint_launchers (group: str = "earthml.launchers") -> None:
    """Allow other packages to register launchers without changing earthml."""
    eps = metadata.entry_points(group=group)
    for ep in eps:
        cls = ep.load()
        if issubclass(cls, BaseLauncher):
            _LAUNCHERS[ep.name] = cls

def available_launchers () -> list[str]:
    return sorted(_LAUNCHERS.keys())

def build_launcher (name: str, **kwargs) -> BaseLauncher:
    if name not in _LAUNCHERS:
        load_entrypoint_launchers()
    if name not in _LAUNCHERS:
        raise KeyError(f"Unknown launcher '{name}'. Available: {available_launchers()}")
    return _LAUNCHERS[name](**kwargs)
