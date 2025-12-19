from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Set, Dict, Optional, Callable, Tuple
import xarray as xr
# Local imports
from .logging import Logger

@dataclass
class Region:
    name: str
    lon: tuple
    lat: tuple

@dataclass
class Leadtime:
    name: str
    unit: str
    value: int

@dataclass
class Variable:
    name:     str
    longname: Optional[str] = None
    unit:     Optional[str] = None
    levhpa:   Optional[int] = None # level in hPa
    levm:     Optional[int] = None # level in meter
    leadtime: Optional[Leadtime] = None

    def __post_init__(self):
        if self.longname is None:
            self.longname = self.name

@dataclass
class TimeRange:
    start: datetime
    end: datetime
    freq: str
    shifted: Optional[Dict] = None

    def __add__(self, other: "TimeRange") -> "TimeRange":
        if not isinstance(other, TimeRange):
            return NotImplemented

        # You can be strict or relaxed here. This is the strict version:
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
    region: Region
    period: TimeRange

@dataclass
class DataSource:
    source: str
    data_selection: DataSelection

    def __add__(self, other: "DataSource") -> "DataSource":
        if not isinstance(other, DataSource):
            return NotImplemented

        # 1. Check that selection (except period) is compatible
        if self.data_selection.region != other.data_selection.region:
            raise ValueError("Cannot add DataSource with different regions")

        if self.data_selection.variable != other.data_selection.variable:
            raise ValueError("Cannot add DataSource with different variables")

        # 2. Combine periods via TimeRange.__add__
        combined_period = self.data_selection.period + other.data_selection.period

        # 3. Build new DataSelection with combined period
        combined_selection = DataSelection(
            variable=self.data_selection.variable,
            region=self.data_selection.region,
            period=combined_period,
        )

        # 4. Decide how to name the combined source
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

@dataclass
class Sample:
    samples: dict | List[datetime] = field(default_factory=dict)
    missed: Set[datetime] = field(default_factory=set)
    extra: dict = None

@dataclass
class ExperimentDataset:
    role: str
    datasource: DataSource | List[DataSource]
    source_params: dict | List[dict] = None
    save: bool = False

PreprocessFn = Callable[[xr.Dataset, xr.Dataset], Tuple[xr.Dataset, xr.Dataset]]
@dataclass
class ExperimentConfig:
    # Globals
    name: str
    work_path: str
    # NN
    seed: int
    net: str
    extra_net_args: dict
    # Hyperparameters
    learning_rate: float
    batch_size: int
    epochs: int
    loss: str
    loss_params: dict
    norm_strategy: str
    supervised: bool
    train_percent: float
    earlystopping_patience: int
    accumulate_grad_batches: int
    # Dataset
    train: ExperimentDataset | List[ExperimentDataset]
    test: ExperimentDataset | List[ExperimentDataset]
    # Optional
    torch_preprocess_fn: Optional[PreprocessFn] = None # called after Xarray dataset loading, before torch dataset generation
