from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Sequence

from rich import print

from .utils import _var_names
from .catalog import make_catalog
from .dataclasses import MLBCExperimentDataset, MLBCExperimentLauncherConfig, MLBCExperimentDatasetRole

from ...base import (
    Leadtime,
    Variable,
    Region,
    TimeRange,
    DataSelection,
)
from ...sources import DataSource
from ...sources.providers import build_provider


@dataclass
class MLBCDatasetGenerator:
    """
    MLBC dataset generation:
    - catalog creation
    - var/region selection
    - naming
    - channel counting
    - "needs_ca_bundle" heuristic
    - build train and test datasets
    """
    experiment              : MLBCExperimentLauncherConfig
    # leadtime variable in dataset (days in seasonal)
    leadtime_var_input      : Leadtime
    leadtime_var_target     : Leadtime
    # actual leadtime (months for seasonal)
    leadtime_input          : Leadtime
    leadtime_target         : Leadtime
    # catalog selection knobs
    var_input_key           : str
    var_target_key          : str
    region_key              : str
    # periods
    train_period            : TimeRange | dict[MLBCExperimentDatasetRole, TimeRange | Sequence[TimeRange]]
    test_period             : TimeRange | dict[MLBCExperimentDatasetRole, TimeRange | Sequence[TimeRange]]
    # providers (names + args)
    input_provider          : str | dict[MLBCExperimentDatasetRole, str | Sequence[str]]
    target_provider         : str | dict[MLBCExperimentDatasetRole, str | Sequence[str]]
    input_provider_args     : dict[str, Any] | dict[MLBCExperimentDatasetRole, dict[str, Any] | Sequence[dict[str, Any]]] = field(default_factory=dict)
    target_provider_args    : dict[str, Any] | dict[MLBCExperimentDatasetRole, dict[str, Any] | Sequence[dict[str, Any]]] = field(default_factory=dict)
    # save datasets
    save_train              : bool = True
    save_test               : bool = True


    def __post_init__(self):
        self.train = self.build_train_datasets()
        self.test = self.build_test_datasets()

    @property
    def cat(self) -> SimpleNamespace:
        return make_catalog(
            leadtime_input=self.leadtime_var_input,
            leadtime_target=self.leadtime_var_target,
        )

    @property
    def var_input(self) -> Variable:
        return getattr(self.cat.var, self.var_input_key)

    @property
    def var_target(self) -> Variable:
        # if analysis in same dataset as forecast, analysis leadtime (e.g 15 days for monthly forecasts) is specified in catalog
        return getattr(self.cat.var, self.var_target_key)

    @property
    def region(self) -> Region:
        return getattr(self.cat.region, self.region_key)


    @staticmethod
    def _get_period_extrema(
        period: TimeRange | dict[str, TimeRange | Sequence[TimeRange]],
        default_type: str = "input"
    ) -> tuple[datetime]:
        if isinstance(period, dict):
            return (
                period[default_type][0].start if isinstance(period[default_type], list) else period[default_type].start,
                period[default_type][-1].end if isinstance(period[default_type], list) else period[default_type].end
            )
        elif isinstance(period, Sequence):
            return period[0].start, period[-1].start
        else:
            return period.start, period.end

    @property
    def experiment_run_name(self) -> str:
        train_start, train_end = self._get_period_extrema(self.train_period)
        test_start, test_end = self._get_period_extrema(self.test_period)
        return (
            f"exp_{self.experiment.type}_{self.experiment.name}"
            f"_{_var_names(self.var_target_key)}-{_var_names(self.var_input_key)}"
            f"_{str(self.leadtime_input.value)}{self.leadtime_input.unit}_{self.region.name}"
            f"_{train_start:%Y%m%d}-{train_end:%Y%m%d}"
            f"_{test_start:%Y%m%d}-{test_end:%Y%m%d}"
            f"{self.experiment.suffix}"
        )

    @property
    def experiment_path(self) -> Path:
        return Path(self.experiment.root_path) / Path(self.experiment_run_name)


    @staticmethod
    def _generate_datasources_and_config_lists(
        var: Variable, region: Region, leadtime: Leadtime,
        period_type: str, periods: TimeRange | Sequence[TimeRange],
        provider_names: str | dict[str, str | Sequence[str]],
        provider_args: dict[str, Any] | dict[str, dict[str, Any] | Sequence[dict[str, Any]]]
    ) -> tuple[Sequence[DataSource], Sequence[dict[str, Any]]]: # Use SourceConfig not Any

        (leadtime_value, leadtime_unit) = (leadtime.value, leadtime.unit) if leadtime else (0, "hours")

        if isinstance(provider_names, dict):
            provider_names = provider_names[period_type]
        if period_type in list(provider_args.keys()):
            provider_args = provider_args[period_type]

        if isinstance(periods, list):
            datasources: Sequence[DataSource] = []
            configs: Sequence[Any] = [] # TODO not Any but actual source config dataclasses

            provider_names = provider_names if isinstance(provider_names, list) else len(periods)*[provider_names]
            provider_args = provider_args if isinstance(provider_args, list) else len(periods)*[provider_args]

            for period, provider_name, args in zip(periods, provider_names, provider_args):
                # print(f"Building datasource for period {period.start} -> {period.end} with provider {provider_name} and args {args}")
                args_updated = args | dict(var_name=var.name, leadtime_value=leadtime_value, leadtime_unit=leadtime_unit)
                provider = build_provider(
                    provider_name,
                    **args_updated,
                )
                datasources.append(DataSource(source=provider.source, data_selection=DataSelection(var, region, period)))
                configs.append(provider.config)
        else: # TODO check this branch, not sure it works
            assert type(provider_names) == str and type(provider_args) == dict, f"Mismatch between periods and provider {provider_names}, please check..."
            args_updates = provider_args | dict(var_name=var.name, leadtime_value=leadtime_value, leadtime_unit=leadtime_unit)
            provider = build_provider(
                provider_names,
                args_updates,
            )
            datasources = DataSource(source=provider.source, data_selection=DataSelection(var, region, periods))
            configs = provider.config

        # print(f"Built datasources: {datasources} with configs: {configs}")
        return datasources, configs

    def _build_datasets(
        self,
        period_type: str,
        period: TimeRange | dict[str, TimeRange | Sequence[TimeRange]],
        save: bool,
    ) -> Sequence[MLBCExperimentDataset]:
        # Input: either single period or segmented periods
        periods_input = period["input"] if isinstance(period, dict) else period
        input_datasources, input_configs = self._generate_datasources_and_config_lists(
            self.var_input,
            self.region,
            self.leadtime_input,
            period_type,
            periods_input,
            self.input_provider,
            self.input_provider_args
        )

        # Target: either single period or segmented periods
        periods_target = period["target"] if isinstance(period, dict) else period
        target_datasources, target_configs = self._generate_datasources_and_config_lists(
            self.var_target,
            self.region,
            self.leadtime_target,
            period_type,
            periods_target,
            self.target_provider,
            self.target_provider_args
        )

        return [
            MLBCExperimentDataset(
                role="input",
                save=save,
                datasource=input_datasources,
                source_configs=input_configs
            ),
            MLBCExperimentDataset(
                role="target",
                save=save,
                datasource=target_datasources,
                source_configs=target_configs
            ),
        ]


    # Entry points
    def build_train_datasets(self) -> Sequence[MLBCExperimentDataset]:
        return self._build_datasets(
            "train",
            self.train_period,
            self.save_train,
        )

    def build_test_datasets(self) -> Sequence[MLBCExperimentDataset]:
        return self._build_datasets(
            "test",
            self.test_period,
            self.save_test,
        )
