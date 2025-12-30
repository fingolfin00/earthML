from typing import Callable, Dict
from ..dataclasses import ProviderSpec

_PROVIDERS: Dict[str, Callable[..., dict]] = {}

def register_provider (name: str):
    def deco (fn: Callable[..., dict]):
        _PROVIDERS[name] = fn
        return fn
    return deco

def build_provider (name: str, **kwargs) -> ProviderSpec:
    if name not in _PROVIDERS:
        raise KeyError(f"Unknown provider '{name}'. Available: {sorted(_PROVIDERS)}")
    return _PROVIDERS[name](**kwargs)

def available_providers () -> list[str]:
    return sorted(_PROVIDERS)
