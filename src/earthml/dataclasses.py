from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Set
# Local imports
from .logging import Logger

@dataclass
class Region:
    name: str
    lon: tuple
    lat: tuple

@dataclass
class Variable:
    name: str
    unit: str
    levhpa: int = None

@dataclass
class TimeRange:
    start: datetime
    end: datetime
    freq: str

@dataclass
class DataSelection:
    variable: Variable | List[Variable]
    region: Region
    period: TimeRange

@dataclass
class DataSource:
    source: str
    data_selection: DataSelection

@dataclass
class Sample:
    samples: dict | List[datetime] = field(default_factory=dict)
    missed: Set[datetime] = field(default_factory=set)
    extra: dict = None

@dataclass
class ExperimentDataset:
    role: str
    datasource: DataSource
    source_params: dict = None

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
    norm_strategy: str
    supervised: bool
    train_percent: float
    earlystopping_patience: int
    accumulate_grad_batches: int
    # Dataset
    lead_time: str
    train: ExperimentDataset | List[ExperimentDataset]
    test: ExperimentDataset | List[ExperimentDataset]
