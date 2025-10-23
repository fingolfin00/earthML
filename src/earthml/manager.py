from dataclasses import dataclass, asdict
from itertools import product
import tomllib, os
from pathlib import Path
from rich.table import Table
# Local imports
from .logging import Logger

@dataclass
class RunConfig:
    # Globals
    name: str
    global_settings: dict
    # Log
    # logfile: Path
    # NN
    net: str
    # Hyperparameters
    learning_rate: float
    batch_size: int
    epochs: int
    loss: str
    norm_strategy: str
    supervised: bool
    train_percent: float
    earlystopping_patience: int
    # Dataset Parameters
    region: dict
    variable: str
    region: str
    region_settings: dict
    lead_time: str
    source: str
    source_settings: dict
    # Time ranges
    train: dict
    test: dict

class Combo:
    def __init__ (self, toml_path: str):
        with open(toml_path, "rb") as f:
            self.cfg = tomllib.load(f)
        self.name = self.cfg["global"]["combo_name"]
        self.combo_folder = os.path.join(self.cfg["global"]["work_root_path"], self.name)
        os.makedirs(self.combo_folder, exist_ok=True)
        self.runs = self.generate_multirun()

    @staticmethod
    def remove_unnecessary_keys (d: dict) -> dict:
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

    def expand_combinations (self, d: dict):
        """Return a list of dicts for all combinations of list-valued items in d."""
        keys, values = zip(*[
            (k, v if isinstance(v, list) else [v])
            for k, v in d.items()
        ])
        return [self.remove_unnecessary_keys(dict(zip(keys, combo))) for combo in product(*values)]

    @staticmethod
    def generate_dict_description (d: dict) -> str:
        return '_'.join(f'{key}={value}' for key, value in d.items())

    def generate_multirun (self) -> list[RunConfig]:
        """Return a list of RunConfig dataclasses (one per run)."""
        global_settings = self.cfg["global"]
        data_settings   = self.cfg["data"]
        hyper = self.cfg["run_ensemble"]["hyper"]
        experiments = self.cfg["run_ensemble"]["experiment"]

        runs: list[RunConfig] = []

        # Expand all combinations of hyperparameters
        hyper_combos = self.expand_combinations(hyper)
        for hyper in hyper_combos:
            for exp in experiments:
                # Expand experiment-level parameters
                exp_combos = self.expand_combinations(exp)
                for combo in exp_combos:
                    run_name = self.generate_dict_description(combo) + self.generate_dict_description(hyper)
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

class Launcher:
    def __init__ (self, tomlfn="earthml.toml"):
        self.combo = Combo(tomlfn)
        self.logger = Logger(
            os.path.join(self.combo.combo_folder, "combo.log"),
            log_level=self.combo.cfg["global"]["log_level"]
        ).logger
        self.logger.debug(f"EarthML configuration file: {os.path.abspath(tomlfn)}")
        self.logger.info(f'Init combo "{self.combo.name}" with {len(self.combo.runs)} runs')

    def generate_runs_rich_table (self) -> list[Table]:
        tables = []
        for r in self.combo.runs:
            table = Table(title=f"Run Configuration")
            table.add_column("Parameter", justify="left", style="cyan")
            table.add_column("Value", justify="left", style="green")
            for key, value in asdict(r).items():
                table.add_row(key, str(value))
            tables.append(table)
        return tables
