from importlib.metadata import version as _version
import importlib

# Static import for autocomplete
from .accessors import earthml as _earthml_accessor

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .base import (
        DataSelection,
        TimeRange,
        Leadtime,
        Region,
        Settings,

        open_zarr,
        open_zarr_var,
        open_nc,
        open_nc_var,

        get_experiment_configs,
        get_and_subset_datasets,

        aggregate_leadtime_ds,
        aggregate_leadtime_da,
        aggregate_leadtime_da_dayweighted,

        ensure_time_coord,
    )

    from .misc import (
        Dask,
        Table,
    )

    from .plots import (
        PiBRdY,
        WRdY,
        SeqBYRd,
        SeqBPi,
        SeqPiBRdY,
        SeqWRdY,
    )

    from .metrics import (
        get_metrics,
        calculate_climatology,
        calculate_save_and_subset_climatologies,
    )

    from .neural.losses import build_loss
    from .neural.dataset import XarrayDataset
    from .neural.nets import build_net
    from .neural.normalize import Normalize, MonthlyNormalize
    from .neural.module import SplitDataModule

__version__ = _version("earthml")

__all__ = [
    "__version__",
    # types
    "DataSelection",
    "TimeRange",
    "Leadtime",
    "Region",
    "Settings",
    # misc
    "Dask",
    "Table",
    # colormaps
    "PiBRdY",
    "WRdY",
    "SeqBYRd",
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
    # metrics
    "get_metrics",
    # utils
    "get_and_subset_datasets",
    "get_experiment_configs",
    "open_zarr",
    "open_zarr_var",
    "open_nc",
    "open_nc_var",
    "ensure_time_coord",
    # agggregation
    "aggregate_leadtime_ds",
    "aggregate_leadtime_da",
    "aggregate_leadtime_da_dayweighted",
    # climatology
    "calculate_climatology",
    "calculate_save_and_subset_climatologies",
]

# Lazy dynamical imports
_EXPORTS = {
    "DataSelection": (".base", "DataSelection"),
    "TimeRange": (".base", "TimeRange"),
    "Leadtime": (".base", "Leadtime"),
    "Region": (".base", "Region"),
    "Settings": (".base", "Settings"),

    "open_zarr": (".base", "open_zarr"),
    "open_zarr_var": (".base", "open_zarr_var"),
    "open_nc": (".base", "open_nc"),
    "open_nc_var": (".base", "open_nc_var"),

    "get_and_subset_datasets": (".base", "get_and_subset_datasets"),
    "get_experiment_configs": (".base", "get_experiment_configs"),

    "aggregate_leadtime_ds": (".base", "aggregate_leadtime_ds"),
    "aggregate_leadtime_da": (".base", "aggregate_leadtime_da"),
    "aggregate_leadtime_da_dayweighted": (".base", "aggregate_leadtime_da_dayweighted"),

    "ensure_time_coord": (".base", "ensure_time_coord"),

    "Dask": (".misc", "Dask"),
    "Table": (".misc", "Table"),

    "PiBRdY": (".plots", "PiBRdY"),
    "WRdY": (".plots", "WRdY"),
    "WRdY": (".plots", "WRdY"),
    "SeqBYRd": (".plots", "SeqBYRd"),
    "SeqBPi": (".plots", "SeqBPi"),
    "SeqWRdY": (".plots", "SeqWRdY"),

    "build_loss": (".neural.losses", "build_loss"),
    "build_net": (".neural.nets", "build_net"),
    "XarrayDataset": (".neural.dataset", "XarrayDataset"),
    "Normalize": (".neural.normalize", "Normalize"),
    "MonthlyNormalize": (".neural.normalize", "MonthlyNormalize"),
    "SplitDataModule": (".neural.module", "SplitDataModule"),

    "get_metrics": (".metrics", "get_metrics"),

    "calculate_climatology": (".metrics", "calculate_climatology"),
    "calculate_save_and_subset_climatologies": (".metrics", "calculate_save_and_subset_climatologies"),
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
