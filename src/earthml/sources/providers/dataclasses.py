from dataclasses import dataclass

from ..dataclasses import DataSource, SourceConfig


@dataclass
class ProviderSpec:
    source: str          # e.g. "earthkit" or "juno-local"
    params: dict
