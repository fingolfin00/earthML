from dataclasses import dataclass, field, is_dataclass
from typing import Any, TYPE_CHECKING, TypeAlias
from collections.abc import Mapping, Sequence, Set

from datetime import datetime
import cftime
import pandas as pd
import xarray as xr

from ..base import DataSelection


if TYPE_CHECKING:
    from .earthkit import EarthkitSourceConfig
    from .juno_local import JunoLocalSourceConfig
    from .xarray_local import XarrayLocalSourceConfig
    from .copernicusmarine import CopernicusmarineSourceConfig

    SourceConfig: TypeAlias = (
        JunoLocalSourceConfig
        | CopernicusmarineSourceConfig
        | EarthkitSourceConfig
        | XarrayLocalSourceConfig
    )
else:
    SourceConfig = Any



@dataclass
class DataSource:
    source          : str
    data_selection  : DataSelection

    def __post_init__(self):
        if not self.source:
            raise ValueError("source must not be empty")
        if self.data_selection is None:
            raise ValueError("data_selection must not be None")
        if self.data_selection.period is None:
            raise ValueError("data_selection.period must not be None")

    def __add__(self, other: "DataSource") -> "DataSource":
        if not isinstance(other, DataSource):
            return NotImplemented

        # Check that selection (except period) is compatible
        if self.data_selection.region != other.data_selection.region:
            raise ValueError("Cannot add DataSource with different regions")

        if self.data_selection.variable != other.data_selection.variable:
            raise ValueError("Cannot add DataSource with different variables")

        # Combine periods via TimeRange.__add__
        combined_period = self.data_selection.period + other.data_selection.period

        # Build new DataSelection with combined period
        combined_selection = DataSelection(
            variable=self.data_selection.variable,
            region=self.data_selection.region,
            period=combined_period,
        )

        # Combined source naming convention
        combined_source_name = (
            self.source if self.source == other.source
            else f"{self.source}+{other.source}"
        )

        return DataSource(
            source=combined_source_name,
            data_selection=combined_selection,
        )

    def __radd__(self, other: "DataSource") -> "DataSource":
        # Allows sum([ds1, ds2, ds3]) to work
        if other == 0:
            return self
        return self.__add__(other)


SampleValues = (
    Mapping[str, Any]
    | pd.DatetimeIndex
    | xr.CFTimeIndex
    | Sequence[datetime]
    | Sequence[cftime.datetime]
    | Sequence[pd.Timedelta]
)

@dataclass
class Sample:
    samples : SampleValues = field(default_factory=dict)
    missed  : Set[datetime] = field(default_factory=set)
    extra   : dict = field(default_factory=dict)

@dataclass
class RegridConfig:
    regrid_resolution   : float | tuple[float, float] | None = None # tuple is for different regrid resolution between lat and lon
    regrid_vars         : list[str] = field(default_factory=list)

@dataclass
class SourceConfigContainer:
    source: str
    config: SourceConfig

    def __post_init__(self) -> None: # TODO verify this check makes sense
        from .registry import get_source_config_class, list_sources

        available_sources = list_sources()
        if self.source not in available_sources:
            raise ValueError(
                f"Unknown source '{self.source}'. Available sources: {available_sources}"
            )

        expected_config_cls = get_source_config_class(self.source)
        if expected_config_cls is None:
            return

        if not is_dataclass(self.config):
            raise TypeError(
                f"Config for source '{self.source}' must be a dataclass instance, "
                f"got {type(self.config).__name__}"
            )

        if not isinstance(self.config, expected_config_cls):
            raise TypeError(
                f"Config for source '{self.source}' must be an instance of "
                f"{expected_config_cls.__name__}, got {type(self.config).__name__}"
            )
