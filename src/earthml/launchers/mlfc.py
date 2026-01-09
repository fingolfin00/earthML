"""
ML Forecast Correction (MLFC) launcher
"""

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Literal, Protocol

from rich import print

from .. import catalog
from ..dataclasses import (
    Leadtime,
    Variable,
    Region,
    TimeRange,
    DataSelection,
    DataSource,
    ExperimentDataset,
    ExperimentConfig,
)
from ..experiments import build_experiment
from .runtime import Runtime
import earthml.providers
from ..providers.registry import build_provider
from ..providers.base import merge


RunMode = Literal["train", "test", "train_test", "dryrun"]


# -------------------------
# Interface (contract)
# -------------------------

class MLFCDomainScenario (Protocol):
    name: str

    def needs_ca_bundle (self) -> bool: ...
    def net_channels (self) -> tuple[int, int]: ...
    def build_train_datasets (self) -> list[ExperimentDataset]: ...
    def build_test_datasets (self) -> list[ExperimentDataset]: ...
    def make_experiment_name (self) -> str: ...
    @property
    def torch_preprocess_fun (self) -> Callable | None: ...


# -------------------------
# Common base scenario
# -------------------------

def _as_list (x):
    return x if isinstance(x, list) else [x]

def _n_channels (var_or_list) -> int:
    return len(var_or_list) if isinstance(var_or_list, list) else 1

def _var_names (var_or_list) -> str:
    return "".join(v.name for v in _as_list(var_or_list))


@dataclass
class MLFCScenario:
    """
    Shared scenario logic:
    - catalog creation
    - var/region selection
    - naming
    - channel counting
    - "needs_ca_bundle" heuristic
    - build train and test datasets
    """
    name: str

    # leadtime variable in dataset
    leadtime_var_name: str
    leadtime_var_value: int
    leadtime_var_unit: Literal["hours", "days", "months"]

    # actual leadtime
    leadtime_value: int
    leadtime_unit: Literal["hours", "days", "months"]

    # catalog selection knobs
    var_fc_key: str
    var_an_key: str
    region_key: str

    # periods
    train_period: TimeRange | dict[str, TimeRange | list[TimeRange]]
    test_period: TimeRange | dict[str, TimeRange | list[TimeRange]]

    # providers (names + kwargs)
    input_provider: str | dict[str, str | list[str]]
    target_provider: str | dict[str, str | list[str]]
    input_provider_kwargs: dict[str, Any] | dict[str, dict[str, Any] | list[dict[str, Any]]] = field(default_factory=dict)
    target_provider_kwargs: dict[str, Any] | dict[str, dict[str, Any] | list[dict[str, Any]]] = field(default_factory=dict)

    # save datasets
    save_train: bool = True
    save_test: bool = True

    torch_preprocess_fn: Callable | None = None

    def _cat (self) -> SimpleNamespace:
        return catalog.make_catalog(leadtime_var=self.leadtime_var_name, leadtime=self.leadtime_var_value, leadtime_unit=self.leadtime_var_unit)

    def var_fc (self) -> Variable:
        return getattr(self._cat().var, self.var_fc_key)

    def var_an (self) -> Variable:
        return getattr(self._cat().var, self.var_an_key)

    def region (self) -> Region:
        return getattr(self._cat().region, self.region_key)

    def needs_ca_bundle (self) -> bool:
        # Tune this rule if you prefer a more explicit flag
        return ("earthkit" in self.input_provider) or ("earthkit" in self.target_provider)

    def net_channels (self) -> tuple[int, int]:
        c = _n_channels(self.var_fc())
        return c, c

    @property
    def torch_preprocess_fun (self) -> Callable | None:
        return self.torch_preprocess_fn

    @staticmethod
    def _get_period_extrema (period: TimeRange | dict[str, TimeRange | list[TimeRange]], default_type: str = "input") -> tuple[datetime]:
        if isinstance(period, dict):
            return (
                period[default_type][0].start if isinstance(period[default_type], list) else period[default_type].start,
                period[default_type][-1].end if isinstance(period[default_type], list) else period[default_type].end
            )
        else:
            return period.start, period.end

    def make_experiment_name (self) -> str:
        train_start, train_end = self._get_period_extrema(self.train_period)
        test_start, test_end = self._get_period_extrema(self.test_period)
        return (
            f"exp_{self.name}_{_var_names(self.var_fc())}-{_var_names(self.var_an())}"
            f"_{self.leadtime_value}{self.leadtime_unit}_{self.region().name}"
            f"_{train_start:%Y%m%d}-{train_end:%Y%m%d}"
            f"_{test_start:%Y%m%d}-{test_end:%Y%m%d}"
            f"_{self.input_provider}_{self.target_provider}"
        )

    @staticmethod
    def _generate_datasources_and_params_lists (
            var: Variable, region: Region, leadtime: Leadtime,
            period_type: str, periods: TimeRange | list[TimeRange],
            provider_names: str | dict[str, str | list[str]],
            provider_kwargs: dict[str, Any] | dict[str, dict[str, Any] | list[dict[str, Any]]]
        ) -> tuple[list[DataSource], list[dict[str, Any]]]:

        (leadtime_value, leadtime_unit) = (leadtime.value, leadtime.unit) if leadtime else (0, "hours")

        if isinstance(provider_names, dict):
            provider_names = provider_names[period_type]
        if period_type in list(provider_kwargs.keys()):
            provider_kwargs = provider_kwargs[period_type]

        if isinstance(periods, list):
            datasources: list[DataSource] = []
            params: list[dict[str, Any]] = []

            provider_names = provider_names if isinstance(provider_names, list) else len(periods)*[provider_names]
            provider_kwargs = provider_kwargs if isinstance(provider_kwargs, list) else len(periods)*[provider_kwargs]

            for period, provider_name, kwargs in zip(periods, provider_names, provider_kwargs):
                provider = build_provider(provider_name, **merge(kwargs, dict(var_name=var.name, leadtime_value=leadtime_value, leadtime_unit=leadtime_unit)))
                datasources.append(DataSource(source=provider.source, data_selection=DataSelection(var, region, period)))
                params.append(provider.params)
        else:
            assert type(provider_names) == str and type(provider_kwargs) == dict, f"Mismatch between periods and provider {provider_names}, please check"
            provider = build_provider(provider_names, **merge(provider_kwargs, dict(var_name=var.name, leadtime_value=leadtime_value, leadtime_unit=leadtime_unit)))
            datasources = DataSource(source=provider.source, data_selection=DataSelection(var, region, periods))
            params = provider.params
        
        return datasources, params

    def _build_datasets (self, period_type: str, period: TimeRange | dict[str, TimeRange | list[TimeRange]]) -> list[ExperimentDataset]:
        # Input: either single period or segmented periods
        periods_input = period["input"] if isinstance(period, dict) else period
        input_datasources, input_params = self._generate_datasources_and_params_lists(self.var_fc(), self.region(), Leadtime("leadtime", self.leadtime_unit, self.leadtime_value), period_type, periods_input, self.input_provider, self.input_provider_kwargs)

        # Target: either single period or segmented periods
        periods_target = period["target"] if isinstance(period, dict) else period
        target_datasources, target_params = self._generate_datasources_and_params_lists(self.var_an(), self.region(), None, period_type, periods_target, self.target_provider, self.target_provider_kwargs)

        return [
            ExperimentDataset(role="input", save=self.save_train, datasource=input_datasources, source_params=input_params),
            ExperimentDataset(role="target", save=self.save_train, datasource=target_datasources, source_params=target_params),
        ]

    def build_train_datasets (self) -> list[ExperimentDataset]:
        return self._build_datasets("train", self.train_period)

    def build_test_datasets (self) -> list[ExperimentDataset]:
        return self._build_datasets("test", self.test_period)

# -------------------------
# MLFC launcher (common ML setup)
# -------------------------

@dataclass
class MLFCRunner:
    """
    Common runner for ML-forecast-correction experiments.

    - scenario provides datasets + naming + net channels + optional preprocess hook
    - runner provides hparams, net/loss, runtime, experiment build/run
    """
    scenario: MLFCDomainScenario

    # output
    exp_root_folder: str
    exp_suffix: str = ""

    # common ML knobs
    seed: int = 42
    net: str = "SmaAt_UNet"

    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    loss: str = "MSELoss"
    loss_params: dict[str, dict[str, Any]] = field(default_factory=lambda: {"net": {}, "loss": {}})

    norm_strategy: str = "BatchNorm2d"
    supervised: bool = True
    train_percent: float = 0.9
    earlystopping_patience: int = 30
    accumulate_grad_batches: int = 2

    # runtime
    dask_workers: int | None = None

    def build_config (self) -> ExperimentConfig:
        exp_name = self.scenario.make_experiment_name()
        exp_path = Path(self.exp_root_folder) / f"{exp_name}{self.exp_suffix}"
        print(f"Experiment path: {exp_path}")

        n_channels, n_classes = self.scenario.net_channels()

        cfg = ExperimentConfig(
            name=exp_name,
            work_path=exp_path,
            seed=self.seed,
            net=self.net,
            extra_net_args=dict(n_channels=n_channels, n_classes=n_classes),

            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            epochs=self.epochs,

            loss=self.loss,
            loss_params=self.loss_params,
            norm_strategy=self.norm_strategy,
            supervised=self.supervised,

            train_percent=self.train_percent,
            earlystopping_patience=self.earlystopping_patience,
            accumulate_grad_batches=self.accumulate_grad_batches,

            train=self.scenario.build_train_datasets(),
            test=self.scenario.build_test_datasets(),

            torch_preprocess_fn=self.scenario.torch_preprocess_fun,
        )
        return cfg

    def build (self):
        cfg = self.build_config()
        # TODO make experiment id configurable in the future
        return build_experiment("ML-forecast-correction", config=cfg)

    def run (self, mode: RunMode = "train_test"):
        rt = Runtime(dask_workers=self.dask_workers, needs_ca_bundle=self.scenario.needs_ca_bundle())
        d = rt.start()
        if mode in ("dryrun", "train", "test", "train_test"):
            exp = self.build()
        if mode in ("train", "train_test"):
            exp.train()
        if mode in ("test", "train_test"):
            exp.test()
        d.close()
        return exp
