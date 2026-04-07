from importlib.metadata import version as _version

__version__ = _version("earthml")

from .base import DataSelection, TimeRange, Leadtime
from .sources import DataSource

from .neural.losses import build_loss

from .experiments.mlbc import MLBCNeuralNet, MLBCExperimentLauncherConfig, MLBCExperimentLauncher

from .accessors.earthml import EarthMLAccessor as _EarthMLAccessor # triggers accessor __init__

__all__ = [
    "__version__",
    "DataSource",
    "DataSelection",
    "TimeRange",
    "Leadtime",
    "MLBCNeuralNet",
    "MLBCExperimentLauncherConfig",
    "MLBCExperimentLauncher",
    "build_loss",
]
