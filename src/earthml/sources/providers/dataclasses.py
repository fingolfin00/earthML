from dataclasses import dataclass


@dataclass
class Provider:
    source: str          # e.g. "ocean.juno.cmcc.hindcast"
    params: dict
