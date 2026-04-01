from importlib.metadata import version as _version

__version__ = _version("earthml")

from .base.dataclasses import DataSelection, TimeRange, Leadtime
from .sources.dataclasses import DataSource

from .losses.registry import build_loss

from .experiments.mlbc.dataclasses import MLBCNeuralNet, MLBCExperimentLauncherConfig
from .experiments.mlbc.launcher import MLBCExperimentLauncher

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
