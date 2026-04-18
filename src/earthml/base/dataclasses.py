from dataclasses import dataclass, asdict, astuple
from datetime import datetime
from typing import Dict, Optional, Literal, TypeAlias

import pandas as pd


@dataclass(frozen=True)
class Region:
    name    : str
    lon     : tuple
    lat     : tuple


LeadtimeUnit: TypeAlias = Literal["hours", "days", "months"]
@dataclass(frozen=True)
class Leadtime:
    name    : str # variable name if applicable
    unit    : LeadtimeUnit
    value   : int

    def to_timedelta(
        self,
        month_length_days: int = 30,
    ) -> pd.Timedelta:
        """
        Convert a Leadtime to Pandas timedelta representation.

        Calendar months are not supported by ``pd.to_timedelta``. For leadtime-axis
        matching we therefore keep the convention that one conceptual month maps to
        30 days unless a caller provides a different approximation explicitly.
        """
        if self.unit == "months":
            return pd.Timedelta(days=self.value * month_length_days)
        return pd.to_timedelta(f"{self.value} {self.unit}")


@dataclass(frozen=True)
class Variable:
    name    : str
    longname: Optional[str] = None
    unit    : Optional[str] = None
    levhpa  : Optional[int] = None # level in hPa
    levm    : Optional[int] = None # level in meter
    leadtime: Optional[Leadtime] = None

    def __post_init__(self):
        if self.longname is None:
            object.__setattr__(self, "longname", self.name)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class DataSelection:
    variable: Variable | tuple[Variable]
    region  : Region
    period  : TimeRange


@dataclass(frozen=True)
class Dims:
    time: str
    latitude: str
    longitude: str
    level: str | None
    realization: str | None
    leadtime: str | None
    """
    items and __len__ exclude None dims
    """
    def items(self):
        return ((k, v) for k, v in asdict(self).items() if v is not None)

    def __iter__(self):
        return (v for v in astuple(self))

    def __getitem__(self, index):
        return astuple(self)[index]

    def __len__(self):
        return sum(1 for v in astuple(self) if v is not None)
