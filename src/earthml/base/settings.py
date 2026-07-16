from typing import Literal, get_args
from dataclasses import dataclass, field, fields, asdict

from pathlib import Path
import hashlib, json

import math
import pandas as pd

from .definitions import LeadtimeUnit, ClimPeriod, TargetMode
from ..neural import (
    SplitStrategy,
    NormalizationMode,
)


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

    extra_suffix_folder: str = ""

    lead_period_offset: int = -1

    var_file_fc: str = "mslp"
    var_file_an: str = "mslp"
    var_fc: str = "mslp"
    var_an: str = "mslp"
    model_fc: str = "sps4_atmo"
    model_an: str = "era5"

    leadtimes: list[int | float] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    leadtime_unit: LeadtimeUnit = LeadtimeUnit.MONTHS
    seasonal_window_size: int = 3

    region_name: str = "World"
    region: dict[str, tuple[int | float, int | float]] | None = None

    train_start: str = "1993-01-01"
    train_end: str = "2016-12-01"
    val_start: str = "2017-01-01"
    val_end: str = "2020-12-01"
    test_start: str = "2021-01-01"
    test_end: str = "2022-12-01"

    target_mode: TargetMode = "analysis"
    clim_period: ClimPeriod = ClimPeriod.MONTH

    seed: int = 42

    channel_representation: Literal["variable", "realization", "init_period"] = "variable"
    init_period_dim: ClimPeriod = ClimPeriod.MONTH
    output_realizations: Literal["deterministic", "ensemble"] = "deterministic" # used only if channel_representation=="realization"
    split_strategy: SplitStrategy = "time"
    normalization: Literal["full", "monthly"] = "full"
    normalization_mode: NormalizationMode = "channel"

    seasonal_encoding: bool = False
    ensemble_encoding: bool = False

    net_name: str = "SmaAt_UNet"
    loss_name: str = "MSELoss"
    target_scale_degrees: int | float = 15.0 # only for GeoMaskedMSELowFreqLoss

    init_learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50

    target_realization_avg: bool = False
    fill_nan_value: float = 0.0
    torch_mask: Literal["target", "input", "both"] = "target"

    training_norm: str = "GroupNorm"

    depth: int = 5 # total encoder levels, including input block
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
    def input_mlfc_train(self) -> Path:
        return self.output_dir / "train_corrected.zarr"

    @property
    def input_mlfc_val(self) -> Path:
        return self.output_dir / "val_corrected.zarr"

    @property
    def input_mlfc_test(self) -> Path:
        return self.output_dir / "test_corrected.zarr"

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
        parts = [
            self.var_fc,
            self.var_an,
            self.region_name,
            str(self.target_mode),
        ]

        if self.target_mode in ("anomaly", "anomaly_residual", "anomaly_residual_realization"):
            parts.append(f"climperiod_{self.clim_period}")

        if self.seasonal_encoding == True and self.channel_representation!="init_period":
            parts.append("seasonal_encoding")
        if self.ensemble_encoding == True:
            parts.append("ens_encoding")

        parts += [
            f"{self.channel_representation}_as_c",
            f"lead{self.lead_period_offset:+d}",
            f"{self.normalization}_{self.normalization_mode}_norm",
            self.net_name.lower(),
            self.loss_name.lower(),
            f"depth{self.depth}",
            f"bc{self.base_channels}",
            f"bs{self.batch_size}",
            f"lr{self.init_learning_rate:.0e}",
            self.training_norm.lower(),
            self.extra_suffix_folder,
        ]

        return "_".join(filter(None, parts))

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
    def seasonal_leadtime_windows(self) -> dict[str, list[int | float]]:
        n = self.seasonal_window_size

        return {
            "-".join(
                str(lead + self.lead_period_offset)
                for lead in window
            ): list(window)
            for window in (
                self.leadtimes[i:i + n]
                for i in range(len(self.leadtimes) - n + 1)
            )
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


def __post_init__(self) -> None:
    # ---------------------------
    # Paths
    # ---------------------------

    if self.root_dir is None:
        missing = [
            name
            for name in (
                "data_root_dir",
                "exp_root_dir",
                "plot_root_dir",
            )
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError(
                "When root_dir is None, the following must be provided: "
                + ", ".join(missing)
            )

    for name in (
        "root_dir",
        "data_root_dir",
        "exp_root_dir",
        "plot_root_dir",
    ):
        value = getattr(self, name)
        if value is not None and not isinstance(value, Path):
            raise TypeError(
                f"{name} must be a pathlib.Path or None, "
                f"got {type(value).__name__}."
            )

    # ---------------------------
    # Strings and identifiers
    # ---------------------------

    for name in (
        "var_file_fc",
        "var_file_an",
        "var_fc",
        "var_an",
        "model_fc",
        "model_an",
        "region_name",
        "net_name",
        "loss_name",
        "training_norm",
    ):
        value = getattr(self, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    if not isinstance(self.extra_suffix_folder, str):
        raise TypeError("extra_suffix_folder must be a string.")

    # ---------------------------
    # Date ranges
    # ---------------------------

    date_names = (
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    )

    dates: dict[str, pd.Timestamp] = {}

    for name in date_names:
        value = getattr(self, name)

        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a valid datetime-like value, got {value!r}."
            ) from exc

        if pd.isna(timestamp):
            raise ValueError(f"{name} cannot be NaT.")

        dates[name] = timestamp

    if dates["train_start"] > dates["train_end"]:
        raise ValueError("train_start must be before or equal to train_end.")

    if dates["val_start"] > dates["val_end"]:
        raise ValueError("val_start must be before or equal to val_end.")

    if dates["test_start"] > dates["test_end"]:
        raise ValueError("test_start must be before or equal to test_end.")

    if dates["train_end"] >= dates["val_start"]:
        raise ValueError(
            "Training and validation periods overlap: "
            "train_end must be earlier than val_start."
        )

    if dates["val_end"] >= dates["test_start"]:
        raise ValueError(
            "Validation and test periods overlap: "
            "val_end must be earlier than test_start."
        )

    # ---------------------------
    # Literal and enum settings
    # ---------------------------

    def check_literal(name: str, literal_type: object) -> None:
        value = getattr(self, name)
        allowed = get_args(literal_type)

        if value not in allowed:
            raise ValueError(
                f"{name} must be one of {allowed}, got {value!r}."
            )

    check_literal("target_mode", TargetMode)
    check_literal("output_realizations", Literal["deterministic", "ensemble"])
    check_literal("split_strategy", SplitStrategy)
    check_literal("normalization", Literal["full", "monthly"])
    check_literal("normalization_mode", NormalizationMode)
    check_literal("torch_mask", Literal["target", "input", "both"])
    check_literal("trainer_precision", TrainerPrecision)

    if not isinstance(self.leadtime_unit, LeadtimeUnit):
        try:
            LeadtimeUnit(self.leadtime_unit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"leadtime_unit must be one of "
                f"{tuple(member.value for member in LeadtimeUnit)}, "
                f"got {self.leadtime_unit!r}."
            ) from exc

    if not isinstance(self.clim_period, ClimPeriod):
        try:
            ClimPeriod(self.clim_period)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"clim_period must be one of "
                f"{tuple(member.value for member in ClimPeriod)}, "
                f"got {self.clim_period!r}."
            ) from exc

    # ---------------------------
    # Lead times and region
    # ---------------------------

    if not isinstance(self.lead_period_offset, int):
        raise TypeError("lead_period_offset must be an integer.")

    if not isinstance(self.leadtimes, list):
        raise TypeError("leadtimes must be a list.")

    if not self.leadtimes:
        raise ValueError("leadtimes cannot be empty.")

    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in self.leadtimes
    ):
        raise TypeError(
            "Every leadtime must be an int or float, excluding bool."
        )

    if any(not math.isfinite(float(value)) for value in self.leadtimes):
        raise ValueError("leadtimes cannot contain NaN values.")

    if len(set(self.leadtimes)) != len(self.leadtimes):
        raise ValueError("leadtimes cannot contain duplicate values.")

    if not isinstance(self.seasonal_window_size, int):
        raise TypeError("seasonal_window_size must be an integer.")

    if self.seasonal_window_size < 1:
        raise ValueError("seasonal_window_size must be at least 1.")

    if self.seasonal_window_size > len(self.leadtimes):
        raise ValueError(
            "seasonal_window_size cannot exceed the number of leadtimes."
        )

    if self.region is not None:
        if not isinstance(self.region, dict):
            raise TypeError("region must be a dictionary or None.")

        allowed_region_keys = {"latitude", "longitude"}
        unknown_keys = set(self.region) - allowed_region_keys

        if unknown_keys:
            raise ValueError(
                f"Unknown region coordinates: {sorted(unknown_keys)}. "
                f"Expected only {sorted(allowed_region_keys)}."
            )

        for coordinate, bounds in self.region.items():
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    for value in bounds
                )
            ):
                raise TypeError(
                    f"region[{coordinate!r}] must be a tuple of two numbers."
                )

            lower, upper = bounds

            if lower >= upper:
                raise ValueError(
                    f"region[{coordinate!r}] lower bound must be "
                    f"smaller than its upper bound."
                )

            if coordinate == "latitude" and not (
                -90 <= lower <= 90 and -90 <= upper <= 90
            ):
                raise ValueError(
                    "Latitude bounds must lie within [-90, 90]."
                )

            if coordinate == "longitude" and not (
                -360 <= lower <= 360 and -360 <= upper <= 360
            ):
                raise ValueError(
                    "Longitude bounds must lie within [-360, 360]."
                )

    # ---------------------------
    # Boolean options
    # ---------------------------

    for name in (
        "seasonal_encoding",
        "ensemble_encoding",
        "target_realization_avg",
    ):
        if not isinstance(getattr(self, name), bool):
            raise TypeError(f"{name} must be a bool.")

    if (
        self.output_realizations == "ensemble"
        and self.channel_representation=="realization"
    ):
        raise ValueError(
            "output_realizations='ensemble' requires "
            "channel_representation='realization."
        )

    # ---------------------------
    # Numeric training settings
    # ---------------------------

    positive_int_fields = (
        "seed",
        "batch_size",
        "max_epochs",
        "depth",
        "reduction_ratio",
        "kernels_per_layer",
        "base_channels",
        "accumulate_grad_batches",
        "early_stopping_patience",
    )

    for name in positive_int_fields:
        value = getattr(self, name)

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if name == "seed":
            if value < 0:
                raise ValueError("seed must be non-negative.")
        elif value <= 0:
            raise ValueError(f"{name} must be positive.")

    if (
        isinstance(self.torch_workers, bool)
        or not isinstance(self.torch_workers, int)
    ):
        raise TypeError("torch_workers must be an integer.")

    if self.torch_workers < 0:
        raise ValueError("torch_workers must be non-negative.")

    positive_float_fields = (
        "init_learning_rate",
        "weight_decay",
    )

    for name in positive_float_fields:
        value = getattr(self, name)

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric.")

        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")

        if name == "weight_decay":
            if value < 0:
                raise ValueError("weight_decay must be non-negative.")
        elif value <= 0:
            raise ValueError(f"{name} must be positive.")

    if (
        isinstance(self.train_fraction, bool)
        or not isinstance(self.train_fraction, (int, float))
    ):
        raise TypeError("train_fraction must be numeric.")

    if not 0 < self.train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1].")

    if (
        isinstance(self.fill_nan_value, bool)
        or not isinstance(self.fill_nan_value, (int, float))
    ):
        raise TypeError("fill_nan_value must be numeric.")

    if not math.isfinite(float(self.fill_nan_value)):
        raise ValueError("fill_nan_value must be finite.")

    # ---------------------------
    # Network consistency
    # ---------------------------

    if self.depth < 2:
        raise ValueError("depth must be at least 2.")
