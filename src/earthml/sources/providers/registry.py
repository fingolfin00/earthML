from typing import Callable, Dict

from .dataclasses import Provider
from .. import SourceConfigContainer


_PROVIDERS: Dict[str, Callable[..., dict]] = {}
JUNO_LOCAL_PROVIDER_NAMES = (
    "atmo.juno.ecmwf.analysis.6hourly",
    "atmo.juno.ecmwf.forecast.hourly",
    "ocean.juno.cmcc.hindcast",
    "ocean.juno.gopaf.forecast.daily.atlantic",
)


def register_source_config_provider(name: str):
    def deco (fn: Callable[..., dict]):
        _PROVIDERS[name] = fn
        return fn
    return deco

def build_source_config_provider(provider: Provider) -> SourceConfigContainer:
    if provider.name not in _PROVIDERS:
        raise KeyError(f"Unknown provider '{provider.name}'. Available: {available_source_config_providers()}")
    return _PROVIDERS[provider.name](**provider.params)

def available_source_config_providers() -> list[str]:
    return sorted(_PROVIDERS)
