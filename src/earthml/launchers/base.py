# earthml/launchers/base.py
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Literal

from ..experiments import build_experiment
from ..dataclasses import ExperimentConfig

RunMode = Literal["dryrun", "train", "test", "train_test"]

@dataclass
class BaseLauncher(ABC):
    """Example launcher"""
    exp_root_folder: str
    exp_suffix: str = ""
    seed: int = 42

    launcher_name: str = "base"

    @abstractmethod
    def build_config (self) -> ExperimentConfig:
        pass

    def build_experiment (self):
        cfg = self.build_config()
        return build_experiment("experiment", config=cfg)

    def run (self, mode: RunMode = "train_test"):
        if mode in ("dryrun", "train", "test", "train_test"):
            exp = self.build()
        if mode in ("train", "train_test"):
            exp.train()
        if mode in ("test", "train_test"):
            exp.test()
        return exp
