from importlib.metadata import version as _version
import importlib

__version__ = _version("earthml")

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
    "load_all_exp_from_folder",
    # metrics
    "get_runs_and_metrics",
    "metrics_to_df",
]

_EXPORTS = {
    "DataSelection": (".base", "DataSelection"),
    "TimeRange": (".base", "TimeRange"),
    "Leadtime": (".base", "Leadtime"),
    "DataSource": (".sources", "DataSource"),
    "Dask": (".misc", "Dask"),
    "Table": (".misc", "Table"),
    "build_loss": (".neural.losses", "build_loss"),
    "MLBCNeuralNet": (".experiments.mlbc", "MLBCNeuralNet"),
    "MLBCExperimentLauncherConfig": (".experiments.mlbc", "MLBCExperimentLauncherConfig"),
    "MLBCExperimentLauncher": (".experiments.mlbc", "MLBCExperimentLauncher"),
    "load_exp": (".experiments.mlbc", "load_exp"),
    "load_all_exp_from_folder": (".experiments.mlbc", "load_all_exp_from_folder"),
    "get_runs_and_metrics": (".metrics", "get_runs_and_metrics"),
    "metrics_to_df": (".metrics", "metrics_to_df"),
}


def __getattr__(name: str):
    if name == "_EarthMLAccessor":
        module = importlib.import_module(".accessors.earthml", __name__)
        value = getattr(module, "EarthMLAccessor")
        globals()[name] = value
        return value

    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


importlib.import_module(".accessors.earthml", __name__)
