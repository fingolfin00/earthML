from typing import Literal
from dataclasses import dataclass, field, fields, asdict

from pathlib import Path
import hashlib, json

import pandas as pd

from .definitions import LeadtimeUnit


TrainerPrecision = Literal[
    64,
    32,
    16,
    "64-true",
    "32-true",
    "16-true",
    "bf16-true",
    "16-mixed",
    "bf16-mixed",
    "transformer-engine",
    "transformer-engine-float16",
    "transformer-engine-bfloat16",
]

HASH_IGNORE = {
    "root_dir",
    "data_root_dir",
    "exp_root_dir",
    "plot_root_dir",
    # "max_epochs",
    # "early_stopping_patience",
    "torch_workers",
    # "trainer_precision",
}


@dataclass(frozen=True)
class Settings:
    root_dir: Path | None = Path.home() / "ML" / "seasonal"

    data_root_dir: Path | None = None
    exp_root_dir: Path | None = None
    plot_root_dir: Path | None = None

    lead_period_offset: int = -1

    var_file_fc: str = "mslp"
    var_file_an: str = "mslp"
    var_fc: str = "mslp"
    var_an: str = "mslp"
    model_fc: str = "sps4_atmo"
    model_an: str = "era5"

    leadtimes: list[int | float] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    leadtime_unit: LeadtimeUnit = LeadtimeUnit.MONTHS

    region_name: str = "World"
    region: dict[str, tuple[int | float, int | float]] | None = None

    train_start: str = "1993-01-01"
    train_end: str = "2020-12-01"
    test_start: str = "2021-01-01"
    test_end: str = "2022-12-01"

    target_mode: Literal["analysis", "residual", "anomaly", "anomaly_residual"] = "analysis"

    seed: int = 42

    realization_as_channel: bool = False
    output_realizations: str = "deterministic"
    split_strategy: Literal["time", "random"] = "time"
    pretrain_norm: Literal["full", "monthly"] = "full"

    net_name: str = "SmaAt_UNet"
    loss_name: str = "MSELoss"

    init_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50

    target_realization_avg: bool = False
    fill_nan_value: float = 0.0
    torch_mask: Literal["target", "input", "both"] = "target"

    training_norm: str = "GroupNorm"

    reduction_ratio: int = 16
    kernels_per_layer: int = 1
    base_channels: int = 32

    train_fraction: float = 0.85
    accumulate_grad_batches: int = 2
    early_stopping_patience: int = 10

    torch_workers: int = 8
    trainer_precision: TrainerPrecision = "16-mixed"

    # ---------------------------
    # Derived paths
    # ---------------------------

    @property
    def data_dir(self) -> Path:
        if self.data_root_dir is not None:
            return self.data_root_dir
        return self.root_dir / "data"

    @property
    def download_dir(self) -> Path:
        return self.data_dir / "download"

    @property
    def input_dir(self) -> Path:
        return self.data_dir / "input"

    @property
    def input_clim_dir(self) -> Path:
        return self.data_dir / "climatology"

    @property
    def exp_dir(self) -> Path:
        base = (
            self.exp_root_dir
            if self.exp_root_dir is not None
            else self.root_dir / "experiments"
        )
        return base / self.output_name

    @property
    def output_dir(self) -> Path:
        return self.exp_dir / "output"

    @property
    def output_clim_dir(self) -> Path:
        return self.exp_dir / "climatology"

    @property
    def plot_dir(self) -> Path:
        return (
            self.exp_dir / "plots"
            if self.plot_root_dir is None
            else self.plot_root_dir / self.output_name
        )

    @property
    def config_path(self) -> Path:
        return self.exp_dir / "config.json"

    @property
    def input_fc(self) -> Path:
        return self._find_dataset(self.model_fc, self.var_file_fc)

    @property
    def input_an(self) -> Path:
        return self._find_dataset(self.model_an, self.var_file_an)

    @property
    def input_mlfc_test(self) -> Path:
        return self.output_dir / "test_corrected.zarr"

    @property
    def input_mlfc_train(self) -> Path:
        return self.output_dir / "train_corrected.zarr"

    @property
    def input_fc_clim(self) -> Path:
        return self.input_clim_dir / f"{self.model_fc}_{self.var_fc}_train_clim.zarr"

    @property
    def input_an_clim(self) -> Path:
        return self.input_clim_dir / f"{self.model_an}_{self.var_an}_train_clim.zarr"

    @property
    def input_mlfc_clim(self) -> Path:
        return self.output_clim_dir / f"train_corrected_clim.zarr"

    # ---------------------------
    # Derived names
    # ---------------------------

    @property
    def output_name(self) -> str:
        return (
            f"{self.target_mode}_"
            f"{self.var_fc}_"
            f"{self.var_an}_"
            f"{self.region_name}_"
            f"lead{self.lead_period_offset:+d}_"
            f"pretrain_{self.pretrain_norm}_norm_"
            f"{self.net_name.lower()}_"
            f"{self.loss_name.lower()}_"
            f"c{self.base_channels}_"
            f"bs{self.batch_size}_"
            f"lr{self.init_learning_rate:.0e}_"
            f"{self.training_norm.lower()}"
        )

    @property
    def config_hash(self) -> str:
        config = asdict(self)

        for key in HASH_IGNORE:
            config.pop(key, None)

        payload = json.dumps(
            config,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha1(payload.encode()).hexdigest()[:6]

    @property
    def seasonal_leadtime_windows(self) -> dict:
        return {
            "0-1-2": [1, 2, 3],
            "1-2-3": [2, 3, 4],
            "2-3-4": [3, 4, 5],
            "3-4-5": [4, 5, 6],
        }

    # ---------------------------
    # Helpers
    # ---------------------------

    def make_dirs(self) -> None:
        for directory in (
            self.exp_dir,
            self.download_dir,
            self.input_dir,
            self.output_dir,
            self.output_clim_dir,
            self.plot_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def save_config(self) -> None:
        config = asdict(self)
        config["config_hash"] = self.config_hash

        with open(self.config_path, "w") as f:
            json.dump(
                config,
                f,
                indent=4,
                sort_keys=True,
                default=str,
            )

    def _find_dataset(
        self,
        model: str,
        var: str,
    ) -> Path:
        base = self.input_dir / f"{model}_{var}"

        for ext in (".zarr", ".nc"):
            path = base.with_suffix(ext)
            if path.exists():
                return path

        raise FileNotFoundError(f"No dataset found for {base} (.zarr or .nc)")


    @classmethod
    def from_json(cls, path: Path, **overrides) -> "Settings":
        with open(path) as f:
            config = json.load(f)

        config.pop("config_hash", None)

        path_fields = {
            "root_dir",
            "data_root_dir",
            "exp_root_dir",
            "plot_root_dir",
        }

        for name in path_fields:
            if config.get(name) is not None:
                config[name] = Path(config[name])

        config.update(overrides)

        return cls(**config)

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    def equals(
        self,
        other: "Settings",
        *,
        ignore: set[str] | None = None,
    ) -> bool:
        ignore = ignore or set()

        unknown = ignore - self.field_names()
        if unknown:
            raise ValueError(f"Unknown fields: {unknown}")

        a = asdict(self)
        b = asdict(other)

        for field in ignore:
            a.pop(field, None)
            b.pop(field, None)

        return a == b


    def comparison_key(
        self,
        *,
        ignore: set[str] | None = None,
    ) -> str:
        ignore = ignore or set()

        unknown = ignore - self.field_names()
        if unknown:
            raise ValueError(f"Unknown fields: {unknown}")

        config = asdict(self)

        for field in ignore:
            config.pop(field, None)

        return json.dumps(config, sort_keys=True, default=str)


    def __post_init__(self):
        if self.root_dir is None:
            missing = [
                name
                for name in ("data_root_dir", "exp_root_dir", "plot_root_dir")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "When root_dir is None, the following must be provided: "
                    + ", ".join(missing)
                )

        if pd.Timestamp(self.train_end) >= pd.Timestamp(self.test_start):
            raise ValueError("Train and test periods overlap.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.accumulate_grad_batches <= 0:
            raise ValueError("accumulate_grad_batches must be positive.")

        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")

        if not 0 < self.train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1].")

        if self.pretrain_norm not in {"full", "monthly"}:
            raise ValueError("pretrain_norm must be 'full' or 'monthly'.")

        if self.leadtime_unit not in list(LeadtimeUnit):
            raise ValueError(f"leadtime_unit must be one of {list(LeadtimeUnit)}.")

        if not self.leadtimes:
            raise ValueError("leadtimes cannot be empty.")
