from dataclasses import dataclass, asdict, astuple
from datetime import datetime
from typing import Dict, Optional, Any, Literal, cast
from enum import StrEnum

import pandas as pd


TargetMode = Literal["analysis", "residual", "anomaly", "anomaly_residual", "anomaly_residual_realization"]


@dataclass(frozen=True)
class Region:
    name    : str
    lon     : tuple
    lat     : tuple


class LeadtimeUnit(StrEnum):
    HOURS  = "hours"
    DAYS   = "days"
    MONTHS = "months"
    YEARS  = "years"

@dataclass(frozen=True)
class Leadtime:
    name: str
    unit: LeadtimeUnit
    value: int

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
    
    def to_offset(self) -> pd.Timedelta | pd.DateOffset:
        if self.unit == LeadtimeUnit.MONTHS:
            return pd.DateOffset(months=self.value)

        return pd.to_timedelta(
            self.value,
            unit=cast(Any, {
                LeadtimeUnit.HOURS: "h",
                LeadtimeUnit.DAYS: "D",
            }[self.unit]),
        )

    def shift_time(self, time: pd.Timestamp) -> pd.Timestamp:
        return time - self.to_offset()


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
    time: str | None
    latitude: str | None
    longitude: str | None
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


class VariableName(StrEnum):
    T2M = "t2m"
    MSL = "msl"
    MSLP = "mslp"
    D2M = "d2m"
    U10 = "u10"
    V10 = "v10"
    TP = "tp"
    TCC = "tcc"
    SST = "sst"
    SSS = "sss"
    SSH = "ssh"
    T14D = "t14d"

class RegionName(StrEnum):
    CONUS = "ConUS"
    EUROPE = "Europe"
    WORLD = "World"

class RunDims(StrEnum):
    LEADTIME = "leadtime"
    TRAIN_PERIOD = "train_period"
    REGION = "region"
    LOSS = "loss"
    VARIANT = "variant"

class ClimPeriod(StrEnum):
    DAYOFYEAR = "dayofyear"
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    DAYOFYEAR_HOUR = "dayofyear_hour"
    DAY_HOUR = "day_hour"
    MONTH_HOUR = "month_hour"

class TimeDimBasis(StrEnum):
    TARGET = "target_time"
    REFERENCE = "reference_time"
