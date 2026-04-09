from dataclasses import dataclass, asdict, astuple
from datetime import datetime
from typing import List, Dict, Optional, Literal


@dataclass
class Region:
    name    : str
    lon     : tuple
    lat     : tuple


@dataclass
class Leadtime:
    name    : str # variable name if applicable
    unit    : Literal["hours", "days", "months"]
    value   : int


@dataclass
class Variable:
    name    : str
    longname: Optional[str] = None
    unit    : Optional[str] = None
    levhpa  : Optional[int] = None # level in hPa
    levm    : Optional[int] = None # level in meter
    leadtime: Optional[Leadtime] = None

    def __post_init__(self):
        if self.longname is None:
            self.longname = self.name


@dataclass
class TimeRange:
    start   : datetime
    end     : datetime
    freq    : str
    shifted : Optional[Dict] = None # TODO maybe we can remove? Why was shifted introduced?

    def __add__(self, other: "TimeRange") -> "TimeRange":
        if not isinstance(other, TimeRange):
            return NotImplemented

        if self.freq != other.freq:
            raise ValueError(f"Cannot add TimeRange with different freq: {self.freq} vs {other.freq}")
        if self.shifted != other.shifted:
            raise ValueError(f"Cannot add TimeRange with different 'shifted' values: {self.shifted} vs {other.shifted}")

        return TimeRange(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            freq=self.freq,
            shifted=self.shifted,
        )


@dataclass
class DataSelection:
    variable: Variable | List[Variable]
    region  : Region
    period  : TimeRange


@dataclass
class Dims:
    time: str
    latitude: str
    longitude: str
    realization: str | None
    leadtime: str | None

    def items(self):
        return ((k, v) for k, v in asdict(self).items() if v is not None)

    def __iter__(self):
        return (v for v in astuple(self) if v is not None)

    def __getitem__(self, index):
        return tuple(v for v in astuple(self) if v is not None)[index]

    def __len__(self):
        return sum(1 for v in astuple(self) if v is not None)
