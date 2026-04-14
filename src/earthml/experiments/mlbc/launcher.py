from dataclasses import dataclass
from typing import Sequence, Callable, Any, TypeAlias
from itertools import product
import multiprocessing

from pathlib import Path

import traceback

from dateutil.relativedelta import relativedelta
from datetime import datetime

from ...misc import Table
from ...base import Leadtime, TimeRange
from ...logging import (
    add_experiment_file_handler,
    get_logger,
    log_renderable,
    remove_experiment_file_handler,
)
from ...sources.providers import Provider
from ...sources.providers.registry import JUNO_LOCAL_PROVIDER_NAMES

from .dataset import MLBCDatasetGenerator
from .dataclasses import MLBCExperimentLauncherConfig, MLBCNeuralNet, MLBCExperimentConfig
from .registry import MLBCExperimentDatasetRole, MLBCRunMode, MLBCExperimentName, MLBCExperimentMode, MLBCExperimentType
from .train_test import MLBCExperiment
from .utils import half_train_periods_days, halved_windows_split_by_cutoff
from .runtime import Runtime
from .conversion import celsius_to_kelvin
from .registry import available_exp_types, available_exps, available_runmodes


ProviderKwargs: TypeAlias = dict[str, Any]
ProviderKwargsSlot: TypeAlias = ProviderKwargs | Sequence[ProviderKwargs]
ProviderKwargsByMode: TypeAlias = dict[MLBCExperimentMode, ProviderKwargsSlot | Sequence[ProviderKwargsSlot]]


@dataclass
class MLBCExperimentLauncher:
    # Exp setup
    experiment                  : MLBCExperimentLauncherConfig # type, name, root folder, suffix
    run_mode                    : MLBCRunMode
    max_retries                 : int
    # Exp settings
    variables_input             : str | Sequence[str]
    variables_target            : str | Sequence[str]
    leadtimes                   : Leadtime | Sequence[Leadtime]
    regions                     : str | Sequence[str]
    train_periods               : TimeRange | Sequence[TimeRange]
    test_periods                : TimeRange | Sequence[TimeRange]
    net                         : MLBCNeuralNet
    # Optional
    dask_workers                : int | None = None
    memory_limit                : str = "auto"
    juno_file_open_workers      : int | None = None
    only_longest_train_period   : bool = True
    # Providers args
    earthkit_cache_dir          : str | Path = Path("/tmp/.earthkit_data")
    regrid_resolution           : float | None = None
    # Source providers
    providers                   : dict[MLBCExperimentDatasetRole, str | Sequence[str]] | None = None
    providers_kwargs            : dict[MLBCExperimentDatasetRole, ProviderKwargsByMode | None] | None = None
    # Experiment
    inpaint_nan                 : bool = False
    anomaly                     : bool = False
    per_epoch_resplit           : bool = False
    torch_preprocess_fn         : Callable | None = None
    target_realization_avg      : bool = False  # whether to average over target realizations when loading target data to torch
    realization_as_channel      : bool = False  # whether to use realization a channel dimension
    force_retrain               : bool = False  # whether to ignore previous training artifacts and restart from scratch


    def __post_init__(self):
        """
        Calculate lead times, variables, source providers and train periods
        """
        # Check experiment validity
        if self.run_mode not in available_runmodes():
            raise ValueError(f"MLBC experiment only supports the following run modes: {available_runmodes()}")
        if self.experiment.type not in available_exp_types():
            raise ValueError(f"MLBC experiment {self.experiment.type} not supported, choose between {available_exp_types()}")
        if self.experiment.name not in available_exps():
            raise ValueError(f"MLBC experiment {self.experiment.name} not supported, choose between {available_exps()}")

        # Init variables as list
        self.leadtimes          = [self.leadtimes] if isinstance(self.leadtimes, Leadtime) else self.leadtimes
        self.variables_input    = [self.variables_input] if isinstance(self.variables_input, str) else self.variables_input
        self.variables_target   = [self.variables_target] if isinstance(self.variables_target, str) else self.variables_target
        self.variables = [
            {"input": var_in, "target": var_tg} for var_in, var_tg in zip(
                self.variables_input,
                self.variables_target,
                strict=True
            )
        ]
        self.regions            = [self.regions] if isinstance(self.regions, str) else self.regions

        # Set var categories
        self.vars_cloud_cds          = {
            "ocean": ("ssh", "sss", "t14d", "t17d", "t20d", "t26d", "t28d", "mld01", "mld03", "das300", "dat300", "sit"), # https://cds.climate.copernicus.eu/datasets/seasonal-monthly-ocean
            "atmo" : ("sst", "t2m", "d2m", "msl", "tp", "u10", "v10"), # https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels
        }

        # Lead time const and experiment type check
        self.leadtime_var_unit = "days"
        if self.experiment.type == MLBCExperimentType.SEASONAL:
            # self.all_leadtimes       = (1, 2, 3, 4, 5, 6) # months
            self.all_leadtimes_vars  = {
                "ocean" : {1: 15, 2: 45, 3: 75, 4: 105, 5: 135, 6: 165},
                "atmo"  : {1: 30, 2: 60, 3: 90, 4: 120, 5: 150, 6: 180},  # atmo seasonal forecast has end of month leadtimes
            }
        if self.experiment.type == MLBCExperimentType.WEATHER:
            # leadtimes_weather_days = (0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9) # all possible days
            # leadtimes_weather_hours = (0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9) # all possible hours
            # self.all_leadtimes       = leadtimes_weather_days # days
            self.all_leadtimes_vars  = {
                "ocean" : {1: 24, 2: 48, 3: 72, 4: 96, 5: 120, 6: 144, 7: 168, 8: 192, 9: 216},
                "atmo"  : {0.5: 12, 1: 24, 2: 48, 3: 72},
            }

        self.train_periods = self._get_train_periods()
        self.test_periods = self.test_periods if isinstance(self.test_periods, Sequence) else [self.test_periods]

        if self.juno_file_open_workers is None:
            self.juno_file_open_workers = multiprocessing.cpu_count()

        # If realization_as_channel is true use provided input and output channel dims, else infer from number of variables
        n_channels, n_classes = (self.net.n_channels, self.net.n_classes) if self.realization_as_channel else (len(self.variables), len(self.variables))
        # Determine output realizations from requested n_channels and n_classes
        if n_classes==1:
            self.output_realizations = "deterministic"
        else:
            if n_classes==n_channels:
                self.output_realizations = "ensemble"
            else:
                raise ValueError(f"Unsupported combo input n_channels={n_channels}, output n_classes={n_classes}")


    @staticmethod
    def _format_time_range(period: TimeRange | Sequence[TimeRange]) -> str:
        if isinstance(period, Sequence):
            parts = []
            for tp in period:
                parts.append(f"{tp.start:%Y-%m-%d}..{tp.end:%Y-%m-%d} ({tp.freq})")
            return " | ".join(parts)
        return f"{period.start:%Y-%m-%d}..{period.end:%Y-%m-%d} ({period.freq})"

    @staticmethod
    def _format_train_period_groups(period_groups: list[dict[MLBCExperimentDatasetRole, TimeRange | Sequence[TimeRange]]]) -> str:
        lines = []
        for idx, period_d in enumerate(period_groups, start=1):
            input_period = MLBCExperimentLauncher._format_time_range(period_d["input"])
            target_period = MLBCExperimentLauncher._format_time_range(period_d["target"])
            lines.append(f"{idx}: input={input_period}; target={target_period}")
        return "\n".join(lines)

    def _summary_table_data(self, exp_gen_cfgs: Sequence[dict]) -> dict[str, dict[str, Any]]:
        return {
            "experiment.type": self.experiment.type,
            "experiment.name": self.experiment.name,
            "experiment.suffix": self.experiment.suffix,
            "experiment.root_path": str(self.experiment.root_path),
            "run.mode": self.run_mode,
            "run.planned_runs": len(exp_gen_cfgs),
            "run.max_retries": self.max_retries,
            "data.variables_input": list(self.variables_input),
            "data.variables_target": list(self.variables_target),
            "data.regions": list(self.regions),
            "data.leadtimes": [f"{lt.value} {lt.unit}" for lt in self.leadtimes],
            "data.train_period_groups": self._format_train_period_groups(self.train_periods),
            "data.test_periods": self._format_time_range(self.test_periods),
            "runtime.dask_workers": self.dask_workers,
            "runtime.juno_file_open_workers": self.juno_file_open_workers,
            "runtime.regrid_resolution": self.regrid_resolution,
            "runtime.earthkit_cache_dir": str(self.earthkit_cache_dir),
            "runtime.only_longest_train_period": self.only_longest_train_period,
            "options.anomaly": self.anomaly,
            "options.inpaint_nan": self.inpaint_nan,
            "options.per_epoch_resplit": self.per_epoch_resplit,
            "options.target_realization_avg": self.target_realization_avg,
            "options.realization_as_channel": self.realization_as_channel,
            "options.output_realizations": self.output_realizations,
            "options.force_retrain": self.force_retrain,
            "options.torch_preprocess_fn": getattr(self.torch_preprocess_fn, "__name__", None) if self.torch_preprocess_fn else None,
            "net.name": self.net.name,
            "net.loss": self.net.loss,
            "net.batch_size": self.net.batch_size,
            "net.epochs": self.net.epochs,
            "net.learning_rate": self.net.learning_rate,
            "net.accumulate_grad_batches": self.net.accumulate_grad_batches,
            "net.train_percent": self.net.train_percent,
            "net.earlystopping_patience": self.net.earlystopping_patience,
            "net.n_channels": self.net.n_channels,
            "net.n_classes": self.net.n_classes,
        }


    def _handle_sequence_of_train_periods(
        self,
        train_periods: TimeRange | Sequence[TimeRange]
    ) -> list[list[TimeRange]]:
        if isinstance(train_periods, Sequence):
            inner_train_periods = train_periods
        else:
            inner_train_periods = [train_periods]

        if self.only_longest_train_period:
            return [inner_train_periods]
        else:
            if isinstance(train_periods, Sequence):
                start = train_periods[0].start
                end = train_periods[-1].end
                assert any([train_periods[0].freq==tp.freq for tp in train_periods]), f"All train periods freq must be the same. First freq {train_periods[0].freq}"
                cutoff = train_periods[0].end
                full_tp = TimeRange(start, end, train_periods[0].freq)

                return halved_windows_split_by_cutoff(
                    full_tp,
                    cutoff,
                    min_months=end.month-start.month,
                    anchor="end",
                    month_start=True,
                ) # return a list of list of TimeRange
            else:
                return [half_train_periods_days(
                    train_periods, # single TimeRange
                    min_months=12,
                    anchor="end",
                    month_start=True # TODO check what this does, month_start flag may be useless
                )]

    def _get_train_periods(self) -> list[dict[MLBCExperimentDatasetRole, TimeRange]]:
        """
        Run in post_init
        """
        # Period const
        self.cutoff_oras5_consolidated = datetime(2014, 12, 31) # cutoff date between consolidated and operational ORAS5 datasets
        delta_train = relativedelta(self.train_periods.end, self.train_periods.start)
        months_train = delta_train.years * 12 + delta_train.months

        # Train input periods
        train_periods_input = self._handle_sequence_of_train_periods(self.train_periods)

        # Train target periods
        if self.experiment.name in (
            MLBCExperimentName.JUNO_CMCC__ORAS5,
            MLBCExperimentName.CDS_CMCC__ORAS5,
        ):
            #TODO issue if train_period is only one month, check
            train_periods_target = halved_windows_split_by_cutoff(
                self.train_periods,
                self.cutoff_oras5_consolidated,
                min_months=(months_train if self.only_longest_train_period else 12),
                anchor="end",
                month_start=True,
            ) # return a list of list of TimeRange
        else:
            train_periods_target = self._handle_sequence_of_train_periods(self.train_periods) # no need for ORAS5 splitting

        # Periods sanity check # TODO can be removed?
        # if len(train_periods_input) != len(train_periods_target):
        #     raise ValueError(f"Number of calculated input train periods ({len(train_periods_input)}) != target train periods ({len(train_periods_target)})")
        
        return [
            {
                "input" : tp_in,
                "target": tp_tg,
            } for tp_in, tp_tg in zip(train_periods_input, train_periods_target, strict=True)
        ]


    def _get_experiment_vars(
        self,
        var_input: str,
        var_target: str
    ) -> Sequence[str]:
        """
        Run in mainloop
        """
        if self.experiment.name == MLBCExperimentName.JUNO_CMCC__ORAS5: # ocean seasonal: local Juno + ORAS5
            var_catalog_keys = (f"{var_input}_juno_fc", f"{var_target}_oras5_an")
        if self.experiment.name == MLBCExperimentName.CDS_CMCC__ORAS5: # ocean seasonal: SPS4 + ORAS5
            var_catalog_keys = (f"{var_input}_cds_fc", f"{var_target}_oras5_an")
        if self.experiment.name == MLBCExperimentName.CDS_CMCC__ERA5: # atmo seasonal
            var_catalog_keys = (f"{var_input}_cds_fc", f"{var_target}_era5_seasonal_an")

        # Common (weather and seasonal) Juno experiment variable naming convention
        if self.experiment.name in (
            MLBCExperimentName.JUNO_CMCC__JUNO_CMCC,
            MLBCExperimentName.JUNO_ECMWF__JUNO_ECMWF,
        ): # local Juno atmo weather or ocean (not really available)
            var_catalog_keys = (f"{var_input}_juno_fc", f"{var_target}_juno_an")

        if self.experiment.name in (
            MLBCExperimentName.JUNO_CMCC__CMEMS,
            MLBCExperimentName.CMEMS__CMEMS,
        ): # ocean weather: CMEMS global ocean fc and an
            var_catalog_keys = (f"{var_input}_gopaf_fc", f"{var_target}_gopaf_an")

        if self.experiment.name == MLBCExperimentName.JUNO_ECMWF__ERA5: # atmo weather: local Juno forecasts + ERA5 reanalysis
            raise NotImplementedError(f"Experiment {self.experiment.name} not yet implemented.")
        return var_catalog_keys


    def _get_source_providers(
        self,
        var_input: str,
        var_target: str,
        periods: dict[MLBCExperimentDatasetRole, TimeRange | Sequence[TimeRange]] | TimeRange | Sequence[TimeRange],
        mode: MLBCExperimentMode,
    ) -> dict[MLBCExperimentDatasetRole, Sequence[Provider]]:
        """
        Run in mainloop
        """
        target_provider_list, target_provider_args_list = [], []
        input_provider_list, input_provider_args_list = [], []

        if isinstance(periods, dict):
            input_period = periods[MLBCExperimentDatasetRole.INPUT]
            target_period = periods[MLBCExperimentDatasetRole.TARGET]
        else:
            input_period = periods
            target_period = periods

        input_period = [input_period] if isinstance(input_period, TimeRange) else input_period
        target_period = [target_period] if isinstance(target_period, TimeRange) else target_period

        def _validate_provider_name_list(
            provider_names: Sequence[str | Sequence[str]],
            role_periods: Sequence[TimeRange],
            role: MLBCExperimentDatasetRole,
        ) -> None:
            if len(provider_names) != len(role_periods):
                raise ValueError(
                    f"Autogenerated {role} providers for mode={mode} do not match target periods: "
                    f"{len(provider_names)} providers for {len(role_periods)} periods"
                )
            for idx, provider_name in enumerate(provider_names):
                if isinstance(provider_name, Sequence) and not isinstance(provider_name, str):
                    if len(provider_name) == 0:
                        raise ValueError(
                            f"Autogenerated {role} providers at index {idx} for mode={mode} is an empty sequence"
                        )
                elif not isinstance(provider_name, str):
                    raise TypeError(
                        f"Autogenerated {role} provider at index {idx} for mode={mode} must be a string or sequence of strings, "
                        f"got {type(provider_name)}"
                    )

        def _infer_provider_names(
            start: datetime,
            end: datetime,
        ) -> tuple[str, str | Sequence[str]]:
            # TODO and make more robust
            # Seasonal
            if self.experiment.type == MLBCExperimentType.SEASONAL:
                if self.experiment.name == MLBCExperimentName.JUNO_CMCC__ORAS5:
                    input_provider = "ocean.juno.cmcc.hindcast"
                    if start <= self.cutoff_oras5_consolidated < end: # duplicate to acommodate two provider args, see below
                        target_provider = [
                            "ocean.earthkit.oras5.reanalysis.monthly", # TODO extend to other target ds than ORAS5
                            "ocean.earthkit.oras5.reanalysis.monthly",
                        ]
                    else:
                        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

                if self.experiment.name == MLBCExperimentName.JUNO_CMCC__JUNO_CMCC:
                    input_provider = "ocean.juno.cmcc.hindcast"
                    target_provider = "ocean.juno.cmcc.hindcast"

                if self.experiment.name == MLBCExperimentName.CDS_CMCC__ORAS5:
                    input_provider = (
                        "atmo.earthkit.cmcc.hindcast.monthly"
                        if var_input in self.vars_cloud_cds["atmo"]
                        else "ocean.earthkit.cmcc.hindcast.monthly"
                    )
                    if start <= self.cutoff_oras5_consolidated < end: # duplicate to acommodate two provider args, see below
                        target_provider = [
                            "ocean.earthkit.oras5.reanalysis.monthly",
                            "ocean.earthkit.oras5.reanalysis.monthly",
                        ]
                    else:
                        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

                if self.experiment.name == MLBCExperimentName.CDS_CMCC__ERA5:
                    input_provider = ( # TODO refactor with above?
                        "atmo.earthkit.cmcc.hindcast.monthly"
                        if var_input in self.vars_cloud_cds["atmo"]
                        else "ocean.earthkit.cmcc.hindcast.monthly"
                    )
                    target_provider = "atmo.earthkit.era5.reanalysis.monthly"

            # Weather
            if self.experiment.type == MLBCExperimentType.WEATHER:
                if self.experiment.name == MLBCExperimentName.JUNO_ECMWF__JUNO_ECMWF: # local Juno atmo weather
                    input_provider = "atmo.juno.ecmwf.forecast.hourly"
                    target_provider = "atmo.juno.ecmwf.analysis.6hourly"

                if self.experiment.name == MLBCExperimentName.CMEMS__CMEMS: # ocean weather cloud (cannot work, no past forecasts)
                    input_provider = "ocean.copernicusmarine.gopaf.forecast.hourly"
                    target_provider = "ocean.copernicusmarine.gopaf.analysis.hourly"

                if self.experiment.name == MLBCExperimentName.JUNO_CMCC__CMEMS: # ocean weather Juno local + cloud for analysis (daily)
                    input_provider = "ocean.juno.gopaf.forecast.daily"
                    target_provider = "ocean.copernicusmarine.gopaf.analysis.daily"

            return input_provider, target_provider

        # Automatic providers generation based on experiment type and name
        if self.providers is None:
            for tp in input_period:
                input_provider, _ = _infer_provider_names(tp.start, tp.end)
                input_provider_list.append(input_provider)

            for tp in target_period:
                _, target_provider = _infer_provider_names(tp.start, tp.end)
                target_provider_list.append(target_provider)

            _validate_provider_name_list(input_provider_list, input_period, MLBCExperimentDatasetRole.INPUT)
            _validate_provider_name_list(target_provider_list, target_period, MLBCExperimentDatasetRole.TARGET)

        else: # providers provided by user, their responsability they are consistent with train periods
            raw_target_provider = self.providers[MLBCExperimentDatasetRole.TARGET]
            raw_input_provider = self.providers[MLBCExperimentDatasetRole.INPUT]
            target_provider_list = (
                list(raw_target_provider)
                if isinstance(raw_target_provider, Sequence) and not isinstance(raw_target_provider, str)
                else [raw_target_provider]
            )
            input_provider_list = (
                list(raw_input_provider)
                if isinstance(raw_input_provider, Sequence) and not isinstance(raw_input_provider, str)
                else [raw_input_provider]
            )

            if len(target_provider_list) == 1 and len(target_period) > 1:
                target_provider_list = len(target_period) * target_provider_list
            if len(input_provider_list) == 1 and len(input_period) > 1:
                input_provider_list = len(input_period) * input_provider_list

        # Provider args generation
        for tp in input_period:
            # Common provider args
            provider_args_common = dict(regrid_resolution=float(self.regrid_resolution)) if self.regrid_resolution is not None else {}
            if self.experiment.name == MLBCExperimentName.CDS_CMCC__ORAS5:
                provider_args = provider_args_common | dict(earthkit_cache_dir=self.earthkit_cache_dir, split_month=12)
            else:
                provider_args = provider_args_common
            input_provider_args_list.append(provider_args)

        for tp in target_period:
            start, end = tp.start, tp.end
            # Common provider args
            provider_args_common = dict(regrid_resolution=float(self.regrid_resolution)) if self.regrid_resolution is not None else {}

            # Target provider args

            # ORAS5 args
            target_provider_args_oras5_common = dict(
                earthkit_cache_dir=self.earthkit_cache_dir,
                # vertical_resolution="single_level",
                convert_unit={"sst": (celsius_to_kelvin, "K")} if var_target == "sst" else None,
            )
            target_provider_args_oras5_consolidated = target_provider_args_oras5_common | dict(
                product_type="consolidated",
            )
            target_provider_args_oras5_operational = target_provider_args_oras5_common | dict(
                product_type="operational",
            )

            # Combine input and target args if necessary (only for ORAS5 currently)
            if self.experiment.name == MLBCExperimentName.CDS_CMCC__ORAS5:
                # Check if target period in ORAS5 consolidated or operational datasets, or both
                if start <= self.cutoff_oras5_consolidated < end:
                    target_provider_args = [
                        provider_args_common | target_provider_args_oras5_consolidated,
                        provider_args_common | target_provider_args_oras5_operational,
                    ]
                elif end <= self.cutoff_oras5_consolidated:
                    target_provider_args = provider_args_common | target_provider_args_oras5_consolidated
                else:
                    target_provider_args = provider_args_common | target_provider_args_oras5_operational
            else:
                target_provider_args = provider_args_common

            target_provider_args_list.append(target_provider_args)

        # Update provider args with user's kwargs
        # print(self.providers_kwargs)
        if self.providers_kwargs:
            target_mode_kwargs = self.providers_kwargs.get(MLBCExperimentDatasetRole.TARGET)
            if target_mode_kwargs is not None and target_mode_kwargs.get(mode) is not None:
                raw_target_kwargs = target_mode_kwargs[mode]
                user_target_kwargs = (
                    list(raw_target_kwargs)
                    if isinstance(raw_target_kwargs, Sequence) and not isinstance(raw_target_kwargs, dict)
                    else [raw_target_kwargs]
                )
                assert len(target_provider_args_list) == len(user_target_kwargs), (
                    f"User provided kwargs {user_target_kwargs} for target [{mode}] source provider(s) not compatible"
                )
                for i, ar in enumerate(target_provider_args_list):
                    if isinstance(ar, list):
                        if isinstance(user_target_kwargs[i], dict):
                            target_provider_args_list[i] = [a | user_target_kwargs[i] for a in ar]
                        else:
                            assert len(ar) == len(user_target_kwargs[i]), (
                                f"User provided kwargs {user_target_kwargs[i]} for target [{mode}] split provider(s) not compatible"
                            )
                            target_provider_args_list[i] = [a | u for a, u in zip(ar, user_target_kwargs[i], strict=True)]
                    else:
                        target_provider_args_list[i] = ar | user_target_kwargs[i]

            input_mode_kwargs = self.providers_kwargs.get(MLBCExperimentDatasetRole.INPUT)
            if input_mode_kwargs is not None and input_mode_kwargs.get(mode) is not None:
                raw_input_kwargs = input_mode_kwargs[mode]
                user_input_kwargs = (
                    list(raw_input_kwargs)
                    if isinstance(raw_input_kwargs, Sequence) and not isinstance(raw_input_kwargs, dict)
                    else [raw_input_kwargs]
                )
                assert len(input_provider_args_list) == len(user_input_kwargs), (
                    f"User provided kwargs {user_input_kwargs} for input [{mode}] source provider(s) not compatible"
                )
                for i, ar in enumerate(input_provider_args_list):
                    input_provider_args_list[i] = ar | user_input_kwargs[i]

        input_providers: list[Provider] = []
        target_providers: list[Provider] = []

        assert len(input_provider_list) == len(input_provider_args_list), (
            f"Input providers/kwargs mismatch for mode={mode}: {len(input_provider_list)} != {len(input_provider_args_list)}"
        )
        assert len(target_provider_list) == len(target_provider_args_list), (
            f"Target providers/kwargs mismatch for mode={mode}: {len(target_provider_list)} != {len(target_provider_args_list)}"
        )

        for provider_name, provider_args in zip(input_provider_list, input_provider_args_list, strict=True):
            if provider_name in JUNO_LOCAL_PROVIDER_NAMES:
                provider_args = provider_args | {"file_open_workers": self.juno_file_open_workers}
            input_providers.append(Provider(name=provider_name, params=provider_args))

        for provider_name, provider_args in zip(target_provider_list, target_provider_args_list, strict=True):
            if isinstance(provider_name, Sequence) and not isinstance(provider_name, str):
                assert isinstance(provider_args, list), (
                    f"Target provider {provider_name} requires a list of kwargs, got {type(provider_args)}"
                )
                target_providers.extend(
                    Provider(
                        name=name,
                        params=(
                            args | {"file_open_workers": self.juno_file_open_workers}
                            if name in JUNO_LOCAL_PROVIDER_NAMES
                            else args
                        ),
                    )
                    for name, args in zip(provider_name, provider_args, strict=True)
                )
            else:
                if provider_name in JUNO_LOCAL_PROVIDER_NAMES:
                    provider_args = provider_args | {"file_open_workers": self.juno_file_open_workers}
                target_providers.append(Provider(name=provider_name, params=provider_args))

        return {
            MLBCExperimentDatasetRole.INPUT: input_providers,
            MLBCExperimentDatasetRole.TARGET: target_providers,
        }


    @staticmethod
    def _needs_ca_bundle(
        providers: dict[MLBCExperimentDatasetRole, dict[MLBCExperimentMode, Provider | Sequence[Provider]]],
    ) -> bool:
        """Heuristic to decide if setting SSL certificate env variable. See runtime.py"""
        def _contains_earthkit(provider: Provider | Sequence[Provider]) -> bool:
            provider_list = (
                list(provider)
                if isinstance(provider, Sequence) and not isinstance(provider, Provider)
                else [provider]
            )
            return any("earthkit" in p.name for p in provider_list)

        return any(
            _contains_earthkit(provider)
            for role_providers in providers.values()
            for provider in role_providers.values()
        )

    def _gen_exp_configs(self):
        exp_configs: list[dict] = []
        needs_ca_bundles: list[bool] = []

        for (lt_idx, leadtime), variable, region, train_period_d in product(
            enumerate(self.leadtimes),
            self.variables,
            self.regions,
            self.train_periods
        ):
            var_input, var_target = variable["input"], variable["target"]
            var_input_key, var_target_key = self._get_experiment_vars(var_input, var_target)

            if self.experiment.name == MLBCExperimentName.CDS_CMCC__ORAS5: # ocean seasonal experiment, except SST for forecasts
                if var_input in self.vars_cloud_cds["ocean"]:
                    try:
                        leadtime_var_value = self.all_leadtimes_vars["ocean"][leadtime.value]
                    except KeyError as exc:
                        raise ValueError(f"Unsupported ocean seasonal leadtime: {leadtime.value} {leadtime.unit}") from exc
                    leadtime_var_input = Leadtime(leadtime.name, self.leadtime_var_unit, leadtime_var_value)
                elif var_input in self.vars_cloud_cds["atmo"]:
                    try:
                        leadtime_var_value = self.all_leadtimes_vars["atmo"][leadtime.value]
                    except KeyError as exc:
                        raise ValueError(f"Unsupported atmo seasonal leadtime: {leadtime.value} {leadtime.unit}") from exc
                    leadtime_var_input = Leadtime(leadtime.name, self.leadtime_var_unit, leadtime_var_value)
                else:
                    raise ValueError(f"Input variable {var_input} for MLBC experiment {self.experiment.name} not supported")

            if self.experiment.name in (
                MLBCExperimentName.JUNO_ECMWF__JUNO_ECMWF,
                MLBCExperimentName.CDS_CMCC__ERA5,
            ): # atmo experiments
                try:
                    leadtime_var_value = self.all_leadtimes_vars["atmo"][leadtime.value]
                except KeyError as exc:
                    raise ValueError(f"Unsupported atmo seasonal leadtime: {leadtime.value} {leadtime.unit}") from exc
                leadtime_var_input = Leadtime(leadtime.name, self.leadtime_var_unit, leadtime_var_value)

            if self.experiment.name in (
                MLBCExperimentName.JUNO_CMCC__CMEMS,
                MLBCExperimentName.CMEMS__CMEMS,
            ): # ocean weather experiment from cloud (cmems_cmems) cannot work currently (no past forecasts)
                leadtime_var_input = Leadtime(leadtime.name, self.leadtime_var_unit, self.all_leadtimes_vars["ocean"][leadtime.value])

            # TODO most experiments are not implemented, we need to make this more general maybe?

            leadtime_target = Leadtime(leadtime.name, leadtime.unit, 0) # TODO maybe make it configurable by user?

            train_providers = self._get_source_providers(var_input, var_target, train_period_d, MLBCExperimentMode.TRAIN)
            test_providers = self._get_source_providers(var_input, var_target, self.test_periods, MLBCExperimentMode.TEST)
            providers_d = {
                MLBCExperimentDatasetRole.INPUT: {
                    MLBCExperimentMode.TRAIN: train_providers[MLBCExperimentDatasetRole.INPUT],
                    MLBCExperimentMode.TEST: test_providers[MLBCExperimentDatasetRole.INPUT],
                },
                MLBCExperimentDatasetRole.TARGET: {
                    MLBCExperimentMode.TRAIN: train_providers[MLBCExperimentDatasetRole.TARGET],
                    MLBCExperimentMode.TEST: test_providers[MLBCExperimentDatasetRole.TARGET],
                },
            }

            exp_configs.append(dict(
                experiment=self.experiment,
                leadtime_var_input=leadtime_var_input,
                leadtime_var_target=leadtime_target,
                leadtime_input=leadtime,
                leadtime_target=leadtime_target,
                var_input_key=var_input_key,
                var_target_key=var_target_key,
                region_key=region,
                train_period=train_period_d,    # list of dictionaries of TimeRanges, keys: ["input", "target"]
                test_period=self.test_periods,   # one-dim list of TimeRange
                providers=providers_d, # dict [train, test] of dicts [input, target] of providers [Provider]
                save_train=True,
                save_test=True,
            ))
            needs_ca_bundles.append(self._needs_ca_bundle(providers_d))

        return exp_configs, any(needs_ca_bundles)


    def get_exp_data(
        self,
        exp_gen_config: dict,
    ) -> tuple[str, MLBCDatasetGenerator, MLBCExperimentConfig]:
        dataset = MLBCDatasetGenerator(**exp_gen_config)

        exp_cfg = MLBCExperimentConfig(
            name=self.experiment.name,
            work_path=dataset.experiment_path,
            net=self.net,
            train_dataset=dataset.train,
            test_dataset=dataset.test,
            anomaly=self.anomaly,
            inpaint_nan=self.inpaint_nan,
            per_epoch_resplit=self.per_epoch_resplit,
            torch_preprocess_fn=self.torch_preprocess_fn,
            target_realization_avg=self.target_realization_avg,
            realization_as_channel=self.realization_as_channel,
            output_realizations=self.output_realizations,
        )

        return dataset.experiment_run_name, dataset, exp_cfg


    def _run_exp(
        self,
        mode: MLBCRunMode,
        dataset: MLBCDatasetGenerator,
        exp: MLBCExperiment,
    ):
        run_name = dataset.experiment_run_name
        if mode not in ("prepare", "train", "test", "train_test", "train_test_on_train"):
            raise ValueError(f"Invalid run mode {mode}")
        if mode in ("train", "train_test", "train_test_on_train"):
            exp.train(force_retrain=self.force_retrain)
        if mode in ("test", "train_test", "train_test_on_train"):
            exp.test()
        if mode == "train_test_on_train":
            exp.test_on_train()
        return run_name, dataset, exp


    def run(self):
        failed_runs: list[tuple[int, str, str]] = []
        exp_gen_cfgs, needs_ca_bundle = self._gen_exp_configs()
        logger = get_logger()
        log_renderable(
            Table(self._summary_table_data(exp_gen_cfgs), title="Launcher Summary", twocols=True).table,
            logger=logger,
        )
        for run_idx, exp_cfg in enumerate(exp_gen_cfgs):
            success = False
            run_name = None
            e_last = None
            for attempt in range(self.max_retries):
                runtime = Runtime(
                    dask_workers=self.dask_workers,
                    memory_limit=self.memory_limit,
                    needs_ca_bundle=needs_ca_bundle,
                )
                dask_runtime = runtime.start()
                log_file = None
                try:
                    run_name, dataset, exp_cfg_obj = self.get_exp_data(exp_cfg)
                    log_file = add_experiment_file_handler(
                        logger,
                        dataset.experiment_path / "logs" / "experiment.log",
                    )
                    exp = MLBCExperiment(config=exp_cfg_obj)

                    logger.info(
                        "Run %s/%s, attempt %s/%s: %s",
                        run_idx + 1,
                        len(exp_gen_cfgs),
                        attempt + 1,
                        self.max_retries,
                        run_name,
                    )
                    logger.info("Experiment path: %s", dataset.experiment_path)

                    run_name, dataset, exp = self._run_exp(
                        mode=self.run_mode,
                        dataset=dataset,
                        exp=exp,
                    )
                    logger.info("Run completed successfully: %s", run_name)
                    success = True
                    break
                except (RuntimeError, OSError) as e:
                    e_last = e
                    failed_name = run_name if run_name else f"run_{run_idx}"
                    if attempt + 1 == self.max_retries:
                        logger.exception("Run failed permanently: %s", failed_name)
                    else:
                        logger.warning(
                            "Run failed and will be retried (%s/%s): %s: %s",
                            attempt + 1,
                            self.max_retries,
                            type(e).__name__,
                            e,
                        )
                    if attempt + 1 == self.max_retries: # print traceback on last retry only
                        traceback.print_exc()
                finally:
                    if log_file is not None:
                        remove_experiment_file_handler(logger, log_file)
                    dask_runtime.close()
            if not success:
                failed_run_name = run_name if run_name else run_idx
                if e_last is None:
                    failed_runs.append((failed_run_name, "UnknownError", "No exception captured"))
                else:
                    failed_runs.append((failed_run_name, type(e_last).__name__, repr(e_last)))
                continue

        for failed_run_name, exc_type, err in failed_runs:
            logger.error("FAIL %s", failed_run_name)
            logger.error("%s: %s", exc_type, err)
