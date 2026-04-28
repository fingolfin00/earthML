from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from rich import print

from .utils import _var_names
from .catalog import make_catalog
from .dataclasses import MLBCExperimentDataset, MLBCExperimentLauncherConfig
from .registry import MLBCExperimentDatasetRole, MLBCExperimentMode

from ...base import (
    Leadtime,
    Variable,
    Region,
    TimeRange,
    DataSelection,
)
from ...sources import DataSource, SourceConfig
from ...sources.providers import build_source_config_provider, Provider


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
    train_period            : TimeRange | dict[MLBCExperimentDatasetRole, TimeRange | Sequence[TimeRange]] # we support dict, see _get_train_periods in launcher.py
    test_period             : TimeRange | dict[MLBCExperimentDatasetRole, TimeRange | Sequence[TimeRange]]
    # providers (names + args)
    providers               : dict[MLBCExperimentDatasetRole, dict[MLBCExperimentMode, Provider | Sequence[Provider]]]
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
            return period[0].start, period[-1].end
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
            f"{self.experiment.run_name_suffix}"
        )

    @property
    def experiment_path(self) -> Path:
        return Path(self.experiment.root_path) / Path(self.experiment_run_name)


    @staticmethod
    def _generate_datasources_and_config_lists(
        var: Variable,
        region: Region,
        leadtime: Leadtime,
        provider_mode: MLBCExperimentMode, # test or train
        periods: TimeRange | Sequence[TimeRange], # can be more than one to support split periods
        providers: Provider | Sequence[Provider] | dict[MLBCExperimentMode, Provider | Sequence[Provider]],
    ) -> tuple[Sequence[DataSource], Sequence[dict[str, SourceConfig]]]:

        (leadtime_value, leadtime_unit) = (leadtime.value, leadtime.unit) if leadtime else (0, "hours")

        periods = periods if isinstance(periods, Sequence) else [periods]

        if isinstance(providers, dict):
            if provider_mode in list(providers.keys()):
                providers = providers[provider_mode]
            else:
                raise ValueError(f"Wrong Provider mode {provider_mode}. Pick one {MLBCExperimentMode}") # TODO is this ok?

        datasources: Sequence[DataSource] = []
        configs: Sequence[SourceConfig] = []

        providers = (
            list(providers)
            if isinstance(providers, Sequence) and not isinstance(providers, Provider)
            else len(periods) * [providers]
        )

        for period, provider in zip(periods, providers, strict=True):
            provider_with_runtime_params = Provider(
                name=provider.name,
                params=provider.params | dict(
                    var_name=var.name,
                    leadtime_value=leadtime_value,
                    leadtime_unit=leadtime_unit,
                ),
            )
            source_config = build_source_config_provider(provider_with_runtime_params)
            datasources.append(DataSource(source=source_config.source, data_selection=DataSelection(var, region, period)))
            configs.append(source_config.config)

        # print(f"Built datasources: {datasources} with configs: {configs}")
        return datasources, configs

    def _build_datasets(
        self,
        mode: MLBCExperimentMode,
        period: TimeRange | dict[MLBCExperimentDatasetRole, TimeRange | Sequence[TimeRange]],
        save: bool,
    ) -> Sequence[MLBCExperimentDataset]:
        # Input: either single period or segmented periods
        periods_input = period[MLBCExperimentDatasetRole.INPUT] if isinstance(period, dict) else period
        input_datasources, input_configs = self._generate_datasources_and_config_lists(
            self.var_input,
            self.region,
            self.leadtime_input,
            mode,
            periods_input,
            self.providers[MLBCExperimentDatasetRole.INPUT],
        )

        # Target: either single period or segmented periods
        periods_target = period[MLBCExperimentDatasetRole.TARGET] if isinstance(period, dict) else period
        target_datasources, target_configs = self._generate_datasources_and_config_lists(
            self.var_target,
            self.region,
            self.leadtime_target,
            mode,
            periods_target,
            self.providers[MLBCExperimentDatasetRole.TARGET],
        )

        return [
            MLBCExperimentDataset(
                role=MLBCExperimentDatasetRole.INPUT,
                save=save,
                datasource=input_datasources,
                source_configs=input_configs
            ),
            MLBCExperimentDataset(
                role=MLBCExperimentDatasetRole.TARGET,
                save=save,
                datasource=target_datasources,
                source_configs=target_configs
            ),
        ]


    # Entry points
    def build_train_datasets(self) -> Sequence[MLBCExperimentDataset]:
        return self._build_datasets(
            MLBCExperimentMode.TRAIN,
            self.train_period,
            self.save_train,
        )

    def build_test_datasets(self) -> Sequence[MLBCExperimentDataset]:
        return self._build_datasets(
            MLBCExperimentMode.TEST,
            self.test_period,
            self.save_test,
        )
