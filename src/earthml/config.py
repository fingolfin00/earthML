from itertools import product
import tomllib
from pathlib import Path
# Local imports
from .dataclasses import RunConfig

class Combo:
    def __init__ (self, toml_path: str):
        with open(toml_path, "rb") as f:
            self.cfg = tomllib.load(f)
        self.name = self.cfg["global"]["combo_name"]
        self.combo_folder = Path(self.cfg["global"]["work_root_path"]).joinpath(self.name)
        self.combo_folder.mkdir(parents=True, exist_ok=True)
        self.runs = self._generate_multirun()

    @staticmethod
    def _remove_unnecessary_keys (d: dict) -> dict:
        """
        Remove keys whose dict values do not share any value with
        the non-dict values of d.
        """
        non_dict_values = [v for v in d.values() if not isinstance(v, dict)]

        result = {}
        for k, v in d.items():
            if not isinstance(v, dict):
                # keep scalar/list values as they are
                result[k] = v
            else:
                # keep only sub-dicts that contain relevant values
                for subkey, subval in v.items():
                    if isinstance(subval, dict):
                        if subkey in non_dict_values:
                            result[k] = subval
                    else:
                        result[k] = v
        return result

    def _expand_combinations (self, d: dict) -> list:
        """Return a list of dicts for all combinations of list-valued items in d."""
        keys, values = zip(*[
            (k, v if isinstance(v, list) else [v])
            for k, v in d.items()
        ])
        return [self._remove_unnecessary_keys(dict(zip(keys, combo))) for combo in product(*values)]

    @staticmethod
    def _generate_dict_description (d: dict) -> str:
        return '_'.join(f'{str(key)}={(str(value))}' for key, value in d.items())

    def _generate_multirun (self) -> list[RunConfig]:
        """Return a list of RunConfig dataclasses (one per run)."""
        global_settings = self.cfg["global"]
        data_settings   = self.cfg["data"]
        hyper = self.cfg["run_ensemble"]["hyper"]
        experiments = self.cfg["run_ensemble"]["experiment"]

        runs: list[RunConfig] = []

        # Expand all combinations of hyperparameters
        hyper_combos = self._expand_combinations(hyper)
        for hyper in hyper_combos:
            for exp in experiments:
                # Expand experiment-level parameters
                exp_combos = self._expand_combinations(exp)
                for combo in exp_combos:
                    run_name = self._generate_dict_description(combo) + self._generate_dict_description(hyper)
                    # Combine into one config
                    run = RunConfig(
                        name=run_name,
                        global_settings=global_settings,
                        # logfile=
                        net=combo["net"],
                        learning_rate=hyper["learning_rate"],
                        batch_size=hyper["batch_size"],
                        epochs=hyper["epochs"],
                        loss=hyper["loss"],
                        norm_strategy=hyper["norm_strategy"],
                        supervised=hyper["supervised"],
                        train_percent=hyper["train_percent"],
                        earlystopping_patience=hyper["earlystopping_patience"],
                        variable=combo["variable"],
                        variable_settings=data_settings["variable"][combo["variable"]],
                        region=combo["region"],
                        region_settings=data_settings["region"][combo["region"]],
                        lead_time=combo["lead_time"],
                        source=combo["source"],
                        source_settings=data_settings["source"][combo["source"]],
                        train=combo["train"],
                        test=combo["test"],
                    )
                    runs.append(run)

        return runs
