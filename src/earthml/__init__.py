from importlib.metadata import version as _version

__version__ = _version("earthml")

from .base import DataSelection, TimeRange, Leadtime
from .sources import DataSource
from .misc import Dask, Table

from .neural.losses import build_loss

from .experiments.mlbc import MLBCNeuralNet, MLBCExperimentLauncherConfig, MLBCExperimentLauncher, load_exp

from .metrics import get_runs_and_metrics, metrics_to_df

from .accessors.earthml import EarthMLAccessor as _EarthMLAccessor # triggers accessor __init__

__all__ = [
    "__version__",
    # types
    "DataSource",
    "DataSelection",
    "TimeRange",
    "Leadtime",
    # misc
    "Dask",
    "Table",
    # loss
    "build_loss",
    # experiment
    "MLBCNeuralNet",
    "MLBCExperimentLauncherConfig",
    "MLBCExperimentLauncher",
    "load_exp",
    # metrics
    "get_runs_and_metrics",
    "metrics_to_df",
]
