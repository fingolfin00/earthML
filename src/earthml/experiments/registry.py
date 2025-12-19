import importlib
from dataclasses import dataclass
from typing import Type, Any

_EXPERIMENTS: dict[str, str] = {
    "ML-forecast-correction": "earthml.experiments.mlfc:ExperimentMLFC",
}

def get_experiment_class(name: str) -> Type[Any]:
    """
    Return the experiment class for a given experiment name.

    Keeps imports lazy so importing earthml doesn't pull in torch/lightning.
    """
    try:
        target = _EXPERIMENTS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown experiment '{name}'. Known: {sorted(_EXPERIMENTS)}"
        ) from e

    module_path, class_name = target.split(":")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def build_experiment(name: str, *args, **kwargs) -> Any:
    """
    Convenience factory: instantiate the experiment.
    Example: build_experiment(config.experiment_name, config)
    """
    cls = get_experiment_class(name)
    return cls(*args, **kwargs)


def list_experiments() -> list[str]:
    """Return available experiment names."""
    return sorted(_EXPERIMENTS.keys())
