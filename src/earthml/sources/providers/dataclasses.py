from dataclasses import dataclass


@dataclass
class Provider:
    name: str          # e.g. "ocean.juno.cmcc.hindcast"
    params: dict
