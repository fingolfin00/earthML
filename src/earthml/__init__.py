from importlib.metadata import version as _version
import importlib

# Static import for autocomplete
from .accessors import earthml as _earthml_accessor

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .base import DataSelection, TimeRange, Leadtime, Region
    from .sources import DataSource
    from .misc import (
        Dask,
        Table,
        PiBRdY,
        WRdY,
        SeqPiBRdY,
        SeqBPi,
        SeqWRdY,
    )
    from .neural.losses import build_loss
    from .neural.dataset import XarrayDataset
    from .neural.nets import build_net
    from .neural.normalize import Normalize, MonthlyNormalize
    from .neural.module import SplitDataModule
    from .experiments.mlbc import (
        MLBCNeuralNet,
        MLBCExperimentLauncherConfig,
        MLBCExperimentLauncher,
        load_exp,
        load_all_exp_from_folder,
    )
    from .metrics import get_metrics, metrics_to_df_single_region


__version__ = _version("earthml")

__all__ = [
    "__version__",
    # types
    "DataSource",
    "DataSelection",
    "TimeRange",
    "Leadtime",
    "Region",
    # misc
    "Dask",
    "Table",
    # colormaps
    "PiBRdY",
    "WRdY",
    "SeqPiBRdY",
    "SeqBPi",
    "SeqWRdY",
    # neural
    "build_loss",
    "build_net",
    "XarrayDataset",
    "Normalize",
    "MonthlyNormalize",
    "SplitDataModule",
    # experiment
    "MLBCNeuralNet",
    "MLBCExperimentLauncherConfig",
    "MLBCExperimentLauncher",
    "load_exp",
    "load_all_exp_from_folder",
    # metrics
    "get_metrics",
    "metrics_to_df_single_region",
]

# Lazy dynamical imports
_EXPORTS = {
    "DataSelection": (".base", "DataSelection"),
    "TimeRange": (".base", "TimeRange"),
    "Leadtime": (".base", "Leadtime"),
    "Region": (".base", "Region"),
    "DataSource": (".sources", "DataSource"),
    "Dask": (".misc", "Dask"),
    "Table": (".misc", "Table"),
    "PiBRdY": (".misc", "PiBRdY"),
    "WRdY": (".misc", "WRdY"),
    "SeqPiBRdY": (".misc", "SeqPiBRdY"),
    "SeqBPi": (".misc", "SeqBPi"),
    "SeqWRdY": (".misc", "SeqWRdY"),
    "build_loss": (".neural.losses", "build_loss"),
    "build_net": (".neural.nets", "build_net"),
    "XarrayDataset": (".neural.dataset", "XarrayDataset"),
    "Normalize": (".neural.normalize", "Normalize"),
    "MonthlyNormalize": (".neural.normalize", "MonthlyNormalize"),
    "SplitDataModule": (".neural.module", "SplitDataModule"),
    "MLBCNeuralNet": (".experiments.mlbc", "MLBCNeuralNet"),
    "MLBCExperimentLauncherConfig": (".experiments.mlbc", "MLBCExperimentLauncherConfig"),
    "MLBCExperimentLauncher": (".experiments.mlbc", "MLBCExperimentLauncher"),
    "load_exp": (".experiments.mlbc", "load_exp"),
    "load_all_exp_from_folder": (".experiments.mlbc", "load_all_exp_from_folder"),
    "get_metrics": (".metrics", "get_metrics"),
    "metrics_to_df_single_region": (".metrics", "metrics_to_df_single_region"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
