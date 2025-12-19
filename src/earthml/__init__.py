from importlib.metadata import version as _version

__version__ = _version("earthml")

from .dataclasses import DataSource, DataSelection, ExperimentConfig, ExperimentDataset

__all__ = [
    "__version__",
    "DataSource",
    "DataSelection",
    "ExperimentConfig",
    "ExperimentDataset",
]
