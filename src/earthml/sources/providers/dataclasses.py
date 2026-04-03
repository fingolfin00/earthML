from dataclasses import dataclass

from ..dataclasses import DataSource, SourceConfig


@dataclass
class Provider:
    source: str          # e.g. "ocean.juno.cmcc.hindcast"
    params: dict
