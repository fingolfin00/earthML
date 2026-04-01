from dataclasses import dataclass
from typing import Sequence, Literal, Callable
from itertools import product

from pathlib import Path

import traceback

from rich import print

from dateutil.relativedelta import relativedelta
from datetime import datetime

from ...misc.dask import Dask
from ...base.dataclasses import Leadtime, TimeRange
from .dataset import MLBCDatasetGenerator
from .dataclasses import MLBCExperimentLauncherConfig, MLBCNeuralNet, MLBCExperimentDatasetRole, MLBCRunMode, MLBCExperimentConfig
from .train_test import MLBCExperiment
from .utils import half_train_periods_days, halved_windows_split_by_cutoff
from .runtime import Runtime
from .conversion import celsius_to_kelvin


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
    train_period                : TimeRange
    test_period                 : TimeRange
    net                         : MLBCNeuralNet
    # Optional
    dask_workers                : int | None = None
    only_longest_train_period   : bool = True
    # Providers args
    earthkit_cache_dir          : str | Path = Path("/tmp/.earthkit_data")
    regrid_resolution           : float | None = None
    # Experiment
    inpaint_nan                 : bool = False
    anomaly                     : bool = False
    per_epoch_resplit           : bool = False
    torch_preprocess_fn         : Callable | None = None
    target_realization_avg      : bool = False  # whether to average over target realizations when loading target data to torch
    realization_as_channel      : bool = False  # whether to use realization a channel dimension


    def __post_init__(self):
        """
        Calculate lead times, variables, source providers and train periods
        """
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
        self.vars_cloud_oras5        = ("sst", "ssh") # TODO implement or remove, only for juno experiments
        self.vars_cloud_cds          = {
            "ocean": ("ssh", "sss", "t14d", "t17d", "t20d", "t26d", "t28d", "mld01", "mld03", "das300", "dat300", "sit"), # https://cds.climate.copernicus.eu/datasets/seasonal-monthly-ocean
            "atmo" : ("sst",), # https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels
        }

        # Lead time const
        self.leadtime_var_unit = "days"
        if self.experiment.type == "seasonal":
            self.all_leadtimes       = (1, 2, 3, 4, 5, 6) # months
            self.all_leadtimes_vars  = {
                "ocean" : (15, 45, 75, 105, 135, 165),
                "atmo"  : (30, 60, 90, 120, 150, 180),  # atmo seasonal forecast has end of month leadtimes
            }
        elif self.experiment.type == "weather":
            leadtimes_weather_days = (12, 24, 48, 72)
            self.all_leadtimes       = leadtimes_weather_days # days
            self.all_leadtimes_vars  = {
                "ocean" : leadtimes_weather_days,
                "atmo"  : leadtimes_weather_days,
            }
        else:
            raise ValueError(f"Experiment {self.experiment.type} not supported, choose between ['seasonal', 'weather']")
        
        # 6) leadtime conventions + target provider kwargs common
        # if experiment == "juno-cmcc_juno-cmcc":
        #     target_provider_args_common = provider_args_common | dict(
        #         file_path_var_prefix="00_ocean_6hr_surface_",  # WRONG
        #     )

        #     all_leadtimes = all_leadtime_hours_sst
        #     all_leadtime_multiple = all_leadtime_hours_sst
        #     leadtime_var_an_value = 0
        #     leadtime_var_unit = "hours"
        #     leadtime_unit = "hours"
        # else:

        # all_leadtimes_months = tuple(leadtime_d)
        # leadtime_var_an_value = 15
        # leadtime_var_unit = "days"
        # leadtime_unit = "months"

        self.train_periods_list = self._get_train_periods()
        self.test_periods_list = [self.test_period]

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


    def _get_train_periods(self) -> list[dict[MLBCExperimentDatasetRole, TimeRange]]:
        """
        Run in post_init
        """
        # Period const
        self.cutoff_oras5_consolidated = datetime(2014, 12, 31) # cutoff date between consolidated and operational ORAS5 datasets
        delta_train = relativedelta(self.train_period.end, self.train_period.start)
        months_train = delta_train.years * 12 + delta_train.months

        train_periods_input = (
            [[self.train_period]]
            if self.only_longest_train_period
            else [half_train_periods_days(
                self.train_period,
                min_months=12,
                anchor="end",
                month_start=True # TODO check what this does, month_start flag may be useless
            )]
        )

        if self.experiment.name == "juno-cmcc_juno-cmcc":
            train_periods_target = train_periods_input # no need for splitting
        else:
            #TODO issue if train_period is only one month, check
            train_periods_target = halved_windows_split_by_cutoff(
                self.train_period,
                self.cutoff_oras5_consolidated,
                min_months=(months_train if self.only_longest_train_period else 12),
                anchor="end",
                month_start=True,
            ) # return a list of list of TimeRange

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
        if self.experiment.type == "seasonal":
            if self.experiment.name == "juno-cmcc_oras5":
                var_catalog_keys = (f"{var_input}_juno_fc", f"{var_target}_oras5_an")
            elif self.experiment.name == "juno-cmcc_juno-cmcc":
                var_catalog_keys = (f"{var_input}_juno_fc", f"{var_target}_juno_an")
            elif self.experiment.name == "cds-cmcc_oras5":
                var_catalog_keys = (f"{var_input}_cds_fc", f"{var_target}_oras5_an")
            else:
                raise ValueError(f"Experiment {self.experiment.name} for type {self.experiment.type} not supported.")
        elif self.experiment.type == "weather": # TODO implement!
            raise NotImplementedError(f"Experiment 'weather' not yet implemented")
        else:
            raise ValueError(f"Not supported experiment type {self.experiment.type}. Choose between ['seasonal', 'weather]")
        return var_catalog_keys


    def _get_source_providers(
        self,
        var_input: str,
        var_target: str,
        target_period: TimeRange | Sequence[TimeRange],
    ) -> dict[MLBCExperimentDatasetRole, dict[Literal["name", "args"]]]:
        """
        Run in mainloop
        """
        target_period = [target_period] if isinstance(target_period, TimeRange) else target_period

        target_provider_list, target_provider_args_list = [], []
        input_provider_list, input_provider_args_list = [], []

        for tp in target_period:
            start, end = tp.start, tp.end

            if self.experiment.type == "seasonal":
                if self.experiment.name == "juno-cmcc_oras5":
                    input_provider = "ocean.juno.cmcc.hindcast"
                    if start <= self.cutoff_oras5_consolidated < end: # duplicate to acommodate two provider args, see below
                        target_provider = [
                            "ocean.earthkit.oras5.reanalysis.monthly", # TODO extend to other target ds than ORAS5
                            "ocean.earthkit.oras5.reanalysis.monthly",
                        ]
                    else:
                        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"
                elif self.experiment.name == "juno-cmcc_juno-cmcc":
                    input_provider = "ocean.juno.cmcc.hindcast"
                    target_provider = "ocean.juno.cmcc.hindcast"
                elif self.experiment.name == "cds-cmcc_oras5":
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
                else:
                    raise ValueError(f"Experiment {self.experiment.name} for type {self.experiment.type} not supported.")

            elif self.experiment.type == "weather": # TODO implement!
                input_provider, target_provider = None, None

            else:
                raise ValueError(f"Not supported experiment type {self.experiment.type}. Choose between ['seasonal', 'weather]")
            
            # Common provider kwargs
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

            # Combine target args if necessary
            if self.experiment.name == "cds-cmcc_oras5":
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

            # Input provider args
            input_provider_args = (
                provider_args_common | dict(earthkit_cache_dir=self.earthkit_cache_dir, split_month=12)
                if self.experiment.name == "cds-cmcc_oras5"
                else provider_args_common
            )

            target_provider_list.append(target_provider)
            target_provider_args_list.append(target_provider_args)

            input_provider_list.append(input_provider)
            input_provider_args_list.append(input_provider_args)

        return {
            "input": {
                "name": input_provider_list,
                "args": input_provider_args_list,
            },
            "target":{
                "name": target_provider_list,
                "args": target_provider_args_list,
            },
        }


    @staticmethod
    def _needs_ca_bundle(input_provider, target_provider) -> bool:
        """Heuristic to decide if setting SSL certificate env variable. See runtime.py"""
        def _contains_earthkit(provider: str | dict[MLBCExperimentDatasetRole, str | Sequence[str]]) -> bool:
            if isinstance(provider, str):
                return "earthkit" in provider
            return any(
                "earthkit" in provider_name
                for provider_value in provider.values()
                for provider_name in (provider_value if isinstance(provider_value, Sequence) else [provider_value])
            )

        return _contains_earthkit(input_provider) or _contains_earthkit(target_provider)

    def _gen_exp_configs(self):
        exp_configs: list[dict] = []
        needs_ca_bundles: list[bool] = []

        for (lt_idx, leadtime), variable, region, train_period_d in product(
            enumerate(self.leadtimes),
            self.variables,
            self.regions,
            self.train_periods_list
        ):
            var_input, var_target = variable["input"], variable["target"]
            var_input_key, var_target_key = self._get_experiment_vars(var_input, var_target)

            if self.experiment.type == "seasonal":
                if self.experiment.name == "cds-cmcc_oras5":
                    if var_input in self.vars_cloud_cds["ocean"]:
                        leadtime_var_input = Leadtime(leadtime.name, self.leadtime_var_unit, self.all_leadtimes_vars["ocean"][lt_idx])
                    elif var_input in self.vars_cloud_cds["atmo"]:
                        leadtime_var_input = Leadtime(leadtime.name, self.leadtime_var_unit, self.all_leadtimes_vars["atmo"][lt_idx])
                    else:
                        raise ValueError(f"Input variable {var_input:!r} not supported")
                else:
                    raise NotImplementedError(f"Seasonal experiment {self.experiment.name} not yet implemented")
            elif self.experiment.type == "weather":
                raise NotImplementedError(f"Experiment 'weather' not yet implemented")
            else:
                raise ValueError(f"Not supported experiment type {self.experiment.type}. Choose between ['seasonal', 'weather]")

            leadtime_target = Leadtime(leadtime.name, leadtime.unit, 0) # TODO maybe make it configurable by user?

            # TODO absolutely refactor this is insane
            train_providers_and_args_d = self._get_source_providers(var_input, var_target, train_period_d["target"])
            test_providers_and_args_d = self._get_source_providers(var_input, var_target, self.test_period)
            providers_d = {
                "input": {
                    "train": train_providers_and_args_d["input"]["name"],
                    "test" : test_providers_and_args_d["input"]["name"],
                },
                "target": {
                    "train": train_providers_and_args_d["target"]["name"],
                    "test" : test_providers_and_args_d["target"]["name"],
                },
            }
            providers_args_d = {
                "input": {
                    "train": train_providers_and_args_d["input"]["args"],
                    "test" : test_providers_and_args_d["input"]["args"],
                },
                "target": {
                    "train": train_providers_and_args_d["target"]["args"],
                    "test" : test_providers_and_args_d["target"]["args"],
                },
            }

            # print("providers_d['input']", providers_d["input"])
            # print("providers_d['target']", providers_d["target"])
            # print("providers_args_d['input']", providers_args_d["input"])
            # print("providers_args_d['target']", providers_args_d["target"])

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
                test_period=self.test_periods_list,   # one-dim list of TimeRange
                input_provider=providers_d["input"],
                target_provider=providers_d["target"],
                input_provider_args=providers_args_d["input"],
                target_provider_args=providers_args_d["target"],
                save_train=True,
                save_test=True,
            ))
            needs_ca_bundles.append(self._needs_ca_bundle(providers_d["input"], providers_d["target"]))

        return exp_configs, any(needs_ca_bundles)


    def get_exp_data(
        self,
        exp_gen_config: dict,
    ) -> tuple[str, MLBCDatasetGenerator, MLBCExperiment]:
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

        return dataset.experiment_run_name, dataset, MLBCExperiment(config=exp_cfg)


    def _run_exp(
        self,
        mode: MLBCRunMode,
        exp_gen_config: dict,
    ):
        if mode in ("prepare", "train", "test", "train_test", "train_test_on_train"):
            run_name, dataset, exp = self.get_exp_data(exp_gen_config)
        else:
            raise ValueError(f"Invalid run mode {mode}")
        if mode in ("train", "train_test", "train_test_on_train"):
            exp.train()
        if mode in ("test", "train_test", "train_test_on_train"):
            exp.test()
        if mode == "train_test_on_train":
            exp.test_on_train()
        return run_name, dataset, exp


    def run(self):
        failed_runs: list[tuple[int, str, str]] = []
        exp_gen_cfgs, needs_ca_bundle = self._gen_exp_configs()
        for run_idx, exp_cfg in enumerate(exp_gen_cfgs):
            success = False
            run_name = None
            e_last = None
            for attempt in range(self.max_retries):
                runtime = Runtime(
                    dask_workers=self.dask_workers,
                    needs_ca_bundle=needs_ca_bundle,
                )
                dask_runtime = runtime.start()
                try:
                    print(f"RUN, ATTEMPT: ({run_idx}, {attempt})")
                    run_name, dataset, exp = self._run_exp(
                        mode=self.run_mode,
                        exp_gen_config=exp_cfg,
                    )
                    print(f"RUN SUCCESSFULL: {run_name}")
                    success = True
                    break
                except (RuntimeError, OSError) as e:
                    e_last = e
                    if attempt + 1 == self.max_retries: # print traceback on last retry only
                        traceback.print_exc()
                finally:
                    dask_runtime.close()
            if not success:
                failed_run_name = run_name if run_name else run_idx
                if e_last is None:
                    failed_runs.append((failed_run_name, "UnknownError", "No exception captured"))
                else:
                    failed_runs.append((failed_run_name, type(e_last).__name__, repr(e_last)))
                continue

        for failed_run_name, exc_type, err in failed_runs:
            print("FAIL", failed_run_name)
            print(f"{exc_type}: {err}")
