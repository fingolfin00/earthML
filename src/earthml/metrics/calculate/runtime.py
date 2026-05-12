from dataclasses import dataclass, field, replace
from typing import Literal
from pathlib import Path

import xarray as xr

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .dataclasses import (
    CalculateMetricsConfig,
    CalculateMetricsGlobalConfig,
    CalculateMetricsFilters,
    CalculateMetricsSaveConfig,
    CalculateMetricsTimeseriesConfig,
    CalculateMetricsMapConfig,
    CalculateMetricsProfileConfig,
    CalculateMetricsVariableColorConfig,
    CalculateMetricsTableConfig,
    CalculateMetricsMetricVsDeltaConfig,
    CalculateMetricsScoreboardConfig,
    CalculateMetricsModelConfig,
)
from .constants import (
    CALCULATE_METRICS_ITEMS,
    METRIC_SECTIONS,
    METRIC_MAPS_CMAPS,
    METRIC_MAPS_DEFAULT_LIMITS,
    METRIC_MAPS_VAR_OVERRIDE_LIMITS,
    PROFILE_METRICS,
    PROFILE_COMBINED_VARS_METRICS,
    NORMALIZED_METRICS,
    METRIC_VS_DELTA_PAIRS,
    SCOREBOARD_METRICS,
    SCOREBOARD_CMAPS,
    SCOREBOARD_LIMITS,
)
from .utils import (
    _print_user_config_table,
    _load_external_mask_dataset,
    _get_variable_units_from_runs,
    _infer_leadtime_unit_from_configs,
    _filter_variables_by_filters,
    _infer_region_extents_from_configs,
    _merge_filename_contexts,
    _filters_filename_context,
    _normalize_profile_axis_fields,
    _format_axis_display_name,
    _table_output_folder,
    _get_variable_table_color,
    deep_merge,
)

from .timeseries import save_field_timeseries_plots
from .maps import save_field_and_metric_map_plots
from .metric_vs_delta import (
    _metric_vs_deltametric_specs,
    save_metric_vs_deltametric_plot,
)
from .profiles import (
    save_metrics_vs_parameter_plots,
    save_combined_variable_metric_profiles
)
from .tables import (
    save_scalar_metric_tables,
)
from .scoreboard import save_scoreboard_plot
from .save import load_metric_datasets, save_metric_datasets

from .. import get_runs_and_metrics, metrics_to_df_single_region
from ..utils import (
    METRIC_GROUP_DIM,
    _discover_region_values,
    metric_group_specs_from_data,
    slice_time_group_data,
)

from ...misc import Dask
from ...logging import configure_logging, LoggingConfig, get_console
from ...experiments.mlbc import MLBCExperimentMode
from ...experiments.mlbc.load import add_ke_to_runs, load_all_exp_from_folder
from ...experiments.mlbc.utils import apply_mask_to_dataset


def _slice_metrics_for_group(
    metrics: dict[str, dict[str, xr.Dataset]],
    *,
    metric_group_label: str,
) -> dict[str, dict[str, xr.Dataset]]:
    sliced: dict[str, dict[str, xr.Dataset]] = {}
    for metric_type, section_dict in metrics.items():
        sliced[metric_type] = {}
        for section, ds in section_dict.items():
            if not isinstance(ds, xr.Dataset):
                sliced[metric_type][section] = ds
                continue
            if METRIC_GROUP_DIM not in ds.dims:
                sliced[metric_type][section] = ds
                continue
            available_groups = [str(value) for value in ds[METRIC_GROUP_DIM].values.tolist()]
            if metric_group_label not in available_groups:
                sliced[metric_type][section] = xr.Dataset(attrs=ds.attrs)
                continue
            sliced[metric_type][section] = ds.sel({METRIC_GROUP_DIM: metric_group_label}, drop=True)
    return sliced


def _build_output_payloads(
    *,
    payload: dict,
    groupby_period: str | None,
    group_source: xr.Dataset,
) -> list[dict]:
    plot_root = payload["plot_root"]
    base_all_payload = {
        **payload,
        "plot_folder": plot_root / "all",
        "scope_label": "all",
        "metric_group_label": "all",
    }
    if "metrics" in payload:
        base_all_payload["metrics"] = _slice_metrics_for_group(payload["metrics"], metric_group_label="all")
    output_payloads = [base_all_payload]

    if groupby_period in {None, "none"}:
        return output_payloads

    for metric_group_label, group_value in metric_group_specs_from_data(group_source, groupby_period):
        grouped_runs = {
            model_name: slice_time_group_data(
                ds,
                period=groupby_period,
                group_value=group_value,
            )
            for model_name, ds in payload["runs"].items()
        }
        output_payloads.append(
            {
                **payload,
                "runs": grouped_runs,
                "plot_folder": plot_root / "grouped" / groupby_period / metric_group_label,
                "scope_label": f"grouped/{groupby_period}/{metric_group_label}",
                "metric_group_label": metric_group_label,
            }
        )
        if "metrics" in payload:
            output_payloads[-1]["metrics"] = _slice_metrics_for_group(
                payload["metrics"],
                metric_group_label=metric_group_label,
            )

    return output_payloads


def _load_runs_by_region(
    *,
    exp_root: Path,
    exp_mode: MLBCExperimentMode,
    load_models: list[str],
    variables: list[str] | None,
    run_filters: dict[str, object] | None,
    external_mask_data: xr.Dataset | None,
) -> dict[str, dict[str, xr.Dataset]]:
    region_values = _discover_region_values(
        exp_root=exp_root,
        exp_mode=exp_mode,
        load_models=load_models,
        variables=variables,
        run_filters=run_filters,
        show_progress=True,
    )
    if not region_values:
        return {}

    runs_by_region: dict[str, dict[str, xr.Dataset]] = {}
    for region_name in region_values:
        region_filters = dict(run_filters or {})
        region_filters["region"] = [region_name]
        loaded_runs, _ = load_all_exp_from_folder(
            exp_root=exp_root,
            exp_mode=exp_mode,
            load_train_preds=(exp_mode == "train"),
            load_models=load_models,
            variables=variables,
            run_filters=region_filters,
            show_progress=True,
        )
        if not loaded_runs:
            continue

        for model_name in load_models:
            if model_name in loaded_runs:
                loaded_runs[model_name] = add_ke_to_runs(
                    loaded_runs[model_name],
                    dataset_keys=None,
                )

        if external_mask_data is not None:
            loaded_runs = {
                model_name: ds if model_name == "mask" else apply_mask_to_dataset(ds, external_mask_data)
                for model_name, ds in loaded_runs.items()
            }

        runs_by_region[str(region_name)] = loaded_runs

    return runs_by_region


@dataclass
class CalculateMetricsRuntime:
    # Globals
    dask_workers: int | None
    memory_limit: str
    root_path: str | Path
    exp_mode: MLBCExperimentMode = "test"
    metrics: list[str] | str | None = None # None means all available metrics
    items: list = field(default_factory=lambda: list(CALCULATE_METRICS_ITEMS))
    log_level: Literal["info", "debug"] = "info"
    print_user_config: bool = False
    print_console_tables: bool = False
    external_mask_path: str | Path | None = None
    external_mask_variable: str | Path | None = None
    disable_flox: bool = False
    metric_groupby_period: Literal["month", "dayofyear", "season", "none"] | None = None
    variable_colors: dict[str, str] = field(default_factory=dict)
    save_calculated_metrics: bool = False
    reuse_saved_metrics: bool = False
    saved_metrics_subfolder: str = "metrics"
    saved_metric_types: tuple[str, ...] = field(default_factory=lambda: tuple(METRIC_SECTIONS))
    saved_metric_kinds: tuple[str, ...] = ("scalar", "timeseries", "map")
    # Labels
    truth: str = "an"
    reference: str = "fc"
    corrected: str | None = "pr"
    gain: str = "gain"
    diff: str | None = "pr-fc"
    # Filters
    leadtime: list[int] | None = None
    variable: list[str] | None = None
    region: list[str] | None = None
    train_period: list[str] | None = None
    loss: list[str] | None = None
    variant: list[str] | None = None
    # Items
    timeseries: bool = False
    maps: bool = False
    profiles: bool = False
    tables: bool = False
    metric_vs_diff: bool = False
    scoreboard: bool = False
    # Items config
    profiles_x_axis: str = "leadtime"
    profiles_enable_combined_variable: bool = False
    metric_vs_delta_byconfig: dict[Literal["color", "marker", "shade"], str] = field(default_factory=dict)
    scoreboard_mode: Literal["absolute", "relative"] = "absolute"

    def __post_init__(self):
        log_config = LoggingConfig(level=self.log_level)
        self.logger = configure_logging(log_config)
        self.console = get_console()

        valid_metric_groupby_periods = {"month", "dayofyear", "season", "none", None}
        if self.metric_groupby_period not in valid_metric_groupby_periods:
            raise ValueError(
                "metric_groupby_period must be one of {'month', 'dayofyear', 'season', 'none', None}"
            )

        metric_vs_delta_byconfig_default = {
            "color": "variable",
            "marker": "region", # e.g. "region" or combined context dimensions ("loss", "variant")
            "shade": "leadtime", # e.g. "loss", "total_months"
        }
        invalid_keys = set(self.metric_vs_delta_byconfig) - set(metric_vs_delta_byconfig_default)
        if invalid_keys:
            raise ValueError(
                "Use only 'color', 'marker' and 'shade' in metric_vs_delta_byconfig"
            )
        self.metric_vs_delta_byconfig = {
            **metric_vs_delta_byconfig_default,
            **self.metric_vs_delta_byconfig,
        }

        self.filters = CalculateMetricsFilters(
            leadtime=self.leadtime,
            variable=self.variable,
            region=self.region,
            train_period=self.train_period,
            loss=self.loss,
            variant=self.variant,
        )

        self.config = CalculateMetricsConfig(
            global_config=CalculateMetricsGlobalConfig(
                exp_root_folder=Path(self.root_path),
                plot_folder_name="plots",
                exp_mode=self.exp_mode,
                metric_names=self.metrics,
                external_mask_path=self.external_mask_path,
                external_mask_variable=self.external_mask_variable,
                use_saved_mask=True,
                items=self.items,
                print_user_config=self.print_user_config,
                print_console_tables=self.print_console_tables,
            ),
            models=CalculateMetricsModelConfig(
                truth_model=self.truth,
                reference_model=self.reference,
                corrected_model=self.corrected,
                gain_label=self.gain,
                display_names={
                    self.truth: self.truth.upper(),
                    self.reference: self.reference.upper(),
                    self.corrected: "MLFC" if self.reference=="fc" else self.corrected.upper(),
                    self.gain: self.gain.capitalize(),
                } if self.reference is not None else {
                    self.truth: self.truth.upper(),
                    self.reference: self.reference.upper(),
                },
                model_colors={
                    self.truth: "tab:orange",
                    self.reference: "tab:green",
                    self.corrected: "tab:blue",
                    self.diff: "tab:red",
                },
            ),
            variable_colors=CalculateMetricsVariableColorConfig(
                overrides=self.variable_colors,
            ),
            saved_metrics=CalculateMetricsSaveConfig(
                output_subfolder=self.saved_metrics_subfolder,
                metric_types=self.saved_metric_types,
                kinds=self.saved_metric_kinds,
                reuse_existing=self.reuse_saved_metrics,
            ) if (self.save_calculated_metrics or self.reuse_saved_metrics) else None,
            timeseries=CalculateMetricsTimeseriesConfig(
                filters=self.filters,
                combine_leadtimes=False,
                leadtime_style="shade",
                plot_realization_members=True,
            ) if self.timeseries else None,
            maps=CalculateMetricsMapConfig(
                filters=self.filters,
                field_cmap="jet",
                # "mean" averages realizations before plotting. "members" saves one map per realization.
                realization_mode="mean",
                metric_cmaps=METRIC_MAPS_CMAPS,
                metric_cmap_default="viridis",
                metric_limits=deep_merge(
                    METRIC_MAPS_DEFAULT_LIMITS,
                    METRIC_MAPS_VAR_OVERRIDE_LIMITS,
                ),
                lon_tick_step=10.0,
                lat_tick_step=10.0,
            ) if self.maps else None,
            profiles=CalculateMetricsProfileConfig(
                filters=self.filters,
                x_axis=self.profiles_x_axis,
                metrics=PROFILE_METRICS,
                shade_by="loss",
                shade_label="Loss",
                # Optional cross-variable profile family saved under all_variables/profiles.
                enable_combined_variable_profiles=self.profiles_enable_combined_variable,
                # Normalized metrics to compare across variables with variable encoded by color.
                combined_variable_metrics=PROFILE_COMBINED_VARS_METRICS,
            ) if self.profiles else None,
            scalar_tables=CalculateMetricsTableConfig(
                scalar_filters=self.filters,
                normalized_filters=self.filters,
                row_index=("metric",),
                column_index=("model", "variable", "stat"),
                stat_order=(
                    "all_dims_avg",
                    "all_dims_spread",
                    "spatial_avg",
                    "spatial_spread",
                    "ens_avg",
                    "prob",
                ),
                significant_digits=3,
                image_dpi=300,
                background_color="#ffffff",
            ) if self.tables else None,
            metric_vs_delta=CalculateMetricsMetricVsDeltaConfig(
                filters=self.filters,
                # Optional multi-plot override. When provided, one plot is generated for
                # each item while the visual settings below stay shared.
                metric_pairs=METRIC_VS_DELTA_PAIRS,
                color_by=self.metric_vs_delta_byconfig["color"],
                marker_by=self.metric_vs_delta_byconfig["marker"],
                shade_by=self.metric_vs_delta_byconfig["shade"],
                shade_label=self.metric_vs_delta_byconfig["shade"].capitalize(),
                point_size=30,
            ) if self.metric_vs_diff else None,
            scoreboards=CalculateMetricsScoreboardConfig(
                filters=self.filters,
                mode=self.scoreboard_mode,
                metrics=SCOREBOARD_METRICS,
                metric_cmaps=SCOREBOARD_CMAPS,
                metric_vlims=SCOREBOARD_LIMITS,
                # Pick which metadata dimensions span the scoreboard grid.
                row_axis="variable",  # e.g. "train_period", "loss", "variant", "leadtime", "variable"
                col_axis="leadtime",
                annotate=True,
            ) if self.scoreboard else None,
        )

        self.plot_folder = self.config.global_config.plot_root

        
    def start(self):
        self.logger.info(f"Dask workers: {self.dask_workers if self.dask_workers is not None else 'auto'}, memory limit: {self.memory_limit}")
        dask_earthml = Dask(
            n_workers=self.dask_workers,
            memory_limit=self.memory_limit,
        )
        dask_earthml.start()
        client = dask_earthml.client
        base_url = "http://localhost:"
        dashboard_url = client.dashboard_link.replace("http://127.0.0.1:", base_url)
        self.logger.info(f"Dask dashboard: {dashboard_url}")

        self.logger.info(f"Loading experiments from: {self.config.global_config.exp_root_folder}")
        if self.config.global_config.print_user_config:
            _print_user_config_table(self.config)

        load_variables = self.filters.get("variable")
        load_run_filters = {
            key: value
            for key, value in self.filters.items()
            if key != "variable"
        }
        self.logger.info(f"Load selection: variables={load_variables or 'all'}, run_filters={load_run_filters or 'all'}")
        external_mask_data = _load_external_mask_dataset(self.config)

        runs_by_region = _load_runs_by_region(
            exp_root=self.config.global_config.exp_root_folder,
            exp_mode=self.config.global_config.exp_mode,
            load_models=self.config.models.load_models,
            variables=load_variables,
            run_filters=load_run_filters,
            external_mask_data=external_mask_data,
        )

        if not runs_by_region:
            raise ValueError("No runs were loaded for the requested filters.")

        self.plot_folder.mkdir(parents=True, exist_ok=True)

        leadtime_unit = _infer_leadtime_unit_from_configs(self.config.global_config.exp_root_folder)

        map_region_filters = self.filters.get("region") if self.filters else None
        map_region_values = (
            list(map_region_filters)
            if isinstance(map_region_filters, (list, tuple, set))
            else ([map_region_filters] if map_region_filters is not None else [])
        )
        region_extents = _infer_region_extents_from_configs(
            self.config.global_config.exp_root_folder,
            selected_regions=map_region_values or None,
        )

        # Print info
        self.logger.info(f"Inferred leadtime unit: {leadtime_unit or 'unknown'}")
        self.logger.info(f"Inferred map regions: {sorted(region_extents) if region_extents else 'unknown'}")
        self.logger.info(f"Inferred region extents: {region_extents or 'unknown'}")

        region_payloads = []
        inventory: dict[str, set[str]] = {}
        for region_name, runs in runs_by_region.items():
            if self.reference not in runs:
                raise ValueError(
                    f"Reference model {self.reference!r} not found in loaded runs for region {region_name!r}. "
                    f"Available models: {list(runs)}"
                )

            variables = [v for v in runs[self.reference].data_vars if v != "_has_var"]
            variable_units = _get_variable_units_from_runs(runs, variables)
            filtered_variables = _filter_variables_by_filters(variables, self.filters)
            region_filters = replace(self.filters, region=[region_name])
            region_plot_folder = self.plot_folder / region_name
            region_plot_folder.mkdir(parents=True, exist_ok=True)
            region_filename_context = _merge_filename_contexts(
                _filters_filename_context(region_filters),
            )

            self.logger.info(f"Processed vars ({region_name}): {dict(zip(variables, variable_units))}")

            region_payloads.append(
                {
                    "region": region_name,
                    "runs": runs,
                    "variables": variables,
                    "variable_units": variable_units,
                    "filtered_variables": filtered_variables,
                    "filters": region_filters,
                    "plot_root": region_plot_folder,
                    "filename_context": region_filename_context,
                }
            )

        output_payloads = []
        for payload in region_payloads:
            group_source = payload["runs"][self.reference]
            output_payloads.extend(
                _build_output_payloads(
                    payload=payload,
                    groupby_period=self.metric_groupby_period,
                    group_source=group_source,
                )
            )

        metrics_loaded_from_saved = False
        if self.config.saved_metrics is not None and self.config.saved_metrics.reuse_existing:
            loaded_payload_metrics = []
            effective_clim_period = self.metric_groupby_period or "month"
            include_group_dim = self.metric_groupby_period is not None
            for payload in output_payloads:
                metrics = load_metric_datasets(
                    output_root=payload["plot_folder"],
                    config=self.config.saved_metrics,
                    filename_context_suffix=payload["filename_context"],
                    model_names=self.config.models.load_models,
                    metric_names=self.metrics,
                    clim_period=effective_clim_period,
                    include_group_dim=include_group_dim,
                )
                if metrics is None:
                    loaded_payload_metrics = []
                    break
                loaded_payload_metrics.append(metrics)

            if loaded_payload_metrics:
                metrics_loaded_from_saved = True
                for payload, metrics in zip(output_payloads, loaded_payload_metrics):
                    payload["metrics"] = metrics

        if not metrics_loaded_from_saved:
            _, metrics_by_region, _ = get_runs_and_metrics(
                exp_root=self.config.global_config.exp_root_folder,
                exp_mode=self.config.global_config.exp_mode,
                load_models=self.config.models.load_models,
                variables=load_variables,
                run_filters=load_run_filters,
                use_saved_mask=self.config.global_config.use_saved_mask,
                external_mask_data=external_mask_data,
                apply_external_mask_to_runs=True,
                metric_names=self.metrics,
                metric_groupby_period=self.metric_groupby_period,
                show_progress=True,
            )

            for payload in output_payloads:
                region_metrics = metrics_by_region[payload["region"]]
                payload["metrics"] = _slice_metrics_for_group(
                    region_metrics,
                    metric_group_label=payload["metric_group_label"],
                )

        for payload in output_payloads:
            metrics = payload["metrics"]
            for metric_type, section_dict in metrics.items():
                ds_scalar = section_dict.get("scalar")
                if ds_scalar is None or "metric" not in ds_scalar.coords:
                    inventory.setdefault(metric_type, set())
                    continue
                inventory.setdefault(metric_type, set()).update(
                    str(metric) for metric in ds_scalar.coords["metric"].values.tolist()
                )

        self.logger.info(
            "Metric source: %s",
            "saved files" if metrics_loaded_from_saved else "fresh computation",
        )
        self.logger.info("Available scalar metrics:")
        for metric_type, metric_names in inventory.items():
            self.logger.info(f"  {metric_type}: {sorted(metric_names) or 'none'}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            auto_refresh=True,
            refresh_per_second=6,
            transient=False,
        ) as progress:
            if self.config.saved_metrics is not None and not metrics_loaded_from_saved:
                save_task = progress.add_task(
                    "Saving calculated metrics",
                    total=len(output_payloads),
                )
                for payload in output_payloads:
                    saved_metric_paths = save_metric_datasets(
                        metrics=payload["metrics"],
                        output_root=payload["plot_folder"],
                        config=self.config.saved_metrics,
                        filename_context_suffix=payload["filename_context"],
                        model_names=self.config.models.load_models,
                        metric_names=self.metrics,
                        clim_period=self.metric_groupby_period or "month",
                        include_group_dim=self.metric_groupby_period is not None,
                        dataset_attrs={
                            "region": payload["region"],
                            "scope_label": payload["scope_label"],
                            "metric_group_label": payload["metric_group_label"],
                            "metric_groupby_period": self.metric_groupby_period or "none",
                        },
                    )
                    progress.advance(save_task)
                    for saved_metric_path in saved_metric_paths:
                        self.logger.debug(f"Saved metric dataset to: {saved_metric_path}")

            if self.config.timeseries is not None:
                for payload in output_payloads:
                    if payload["metric_group_label"] != "all":
                        continue
                    ts_task = progress.add_task(
                        f"Generating field timeseries ({payload['region']} | {payload['scope_label']})",
                        total=len(payload["filtered_variables"]),
                    )
                    field_timeseries_paths = save_field_timeseries_plots(
                        load_models=self.config.models.load_models,
                        runs=payload["runs"],
                        metrics=payload["metrics"],
                        variables=payload["filtered_variables"],
                        plot_folder=payload["plot_folder"],
                        leadtime_unit=leadtime_unit,
                        filters=payload["filters"],
                        config=self.config,
                        combine_leadtimes=self.config.timeseries.combine_leadtimes,
                        plot_realization_members=self.config.timeseries.plot_realization_members,
                        metric_groupby_period=self.metric_groupby_period,
                        disable_flox=self.disable_flox,
                        progress=progress,
                        task_id=ts_task,
                    )
                    for field_timeseries_path in field_timeseries_paths:
                        self.logger.debug(f"Saved field timeseries plot to: {field_timeseries_path}")

            if self.config.maps is not None:
                for payload in output_payloads:
                    map_task = progress.add_task(
                        f"Generating maps ({payload['region']} | {payload['scope_label']})",
                        total=len(payload["filtered_variables"]),
                    )
                    field_map_paths = save_field_and_metric_map_plots(
                        load_models=self.config.models.load_models,
                        runs=payload["runs"],
                        metrics=payload["metrics"],
                        variables=payload["filtered_variables"],
                        plot_folder=payload["plot_folder"],
                        leadtime_unit=leadtime_unit,
                        filters=payload["filters"],
                        config=self.config,
                        realization_mode=self.config.maps.realization_mode,
                        metric_group_label=payload["metric_group_label"],
                        metric_groupby_period=self.metric_groupby_period,
                        region_extents=region_extents,
                        progress=progress,
                        task_id=map_task,
                    )
                    for field_map_path in field_map_paths:
                        self.logger.debug(f"Saved map plot to: {field_map_path}")


            if self.config.metric_vs_delta is not None:
                metric_vs_deltametric_specs = _metric_vs_deltametric_specs(self.config.metric_vs_delta)
                for payload in output_payloads:
                    diff_task = progress.add_task(
                        f"Generating metric-vs-delta-metric plot ({payload['region']} | {payload['scope_label']})",
                        total=len(metric_vs_deltametric_specs),
                    )
                    for metric_spec in metric_vs_deltametric_specs:
                        df_nodiff = metrics_to_df_single_region(
                            metrics=payload["metrics"],
                            variables=payload["variables"],
                            metric_names=metric_spec["forecast_metric"],
                            kind="scalar",
                            metric_type=metric_spec["forecast_metric_type"],
                            diff="no",
                            models=(self.config.models.reference_model,),
                        )
                        df_delta = metrics_to_df_single_region(
                            metrics=payload["metrics"],
                            variables=payload["variables"],
                            metric_names=metric_spec["delta_metric"],
                            kind="scalar",
                            metric_type=metric_spec["delta_metric_type"],
                            diff="delta",
                            models=self.config.models.comparison_models,
                        )

                        plot_path = save_metric_vs_deltametric_plot(
                            df_nodiff=df_nodiff,
                            df_delta=df_delta,
                            plot_folder=payload["plot_folder"],
                            forecast_metric=metric_spec["forecast_metric"],
                            delta_metric=metric_spec["delta_metric"],
                            leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                            region=payload["region"],
                            filters=payload["filters"],
                            variable_unit=None, # TODO might be wrong
                            metric_vs_delta_byconfig=self.metric_vs_delta_byconfig,
                            point_size=20,
                            config=self.config,
                            filename_context_suffix=payload["filename_context"],
                        )
                        progress.advance(diff_task)
                        self.logger.debug(f"Saved leadtime-vs-global-metrics plot to: {plot_path}")

            if self.config.profiles is not None:
                for payload in output_payloads:
                    profile_x_axis_fields = _normalize_profile_axis_fields(self.config.profiles.x_axis)
                    profile_total = 1 if profile_x_axis_fields == ("variable",) else len(payload["variables"])
                    profile_task = progress.add_task(
                        f"Generating metric profiles ({payload['region']} | {payload['scope_label']} | {_format_axis_display_name(self.config.profiles.x_axis)})",
                        total=profile_total,
                    )
                    profile_plot_paths = save_metrics_vs_parameter_plots(
                        metrics=payload["metrics"],
                        variables=payload["variables"],
                        plot_folder=payload["plot_folder"],
                        leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                        variable_units=payload["variable_units"],
                        x_axis=self.config.profiles.x_axis,
                        filters=payload["filters"],
                        config=self.config,
                        progress=progress,
                        task_id=profile_task,
                    )
                    for profile_plot_path in profile_plot_paths:
                        self.logger.debug(f"Saved metric profile plot to: {profile_plot_path}")

                    if self.config.profiles.combined_variable_metrics is not None and len(payload["variables"]) > 1:
                        combined_profile_task = progress.add_task(
                            f"Generating combined variable profiles ({payload['region']} | {payload['scope_label']} | {_format_axis_display_name(self.config.profiles.x_axis)})",
                            total=1,
                        )
                        combined_profile_paths = save_combined_variable_metric_profiles(
                            metrics=payload["metrics"],
                            variables=payload["variables"],
                            plot_folder=payload["plot_folder"],
                            leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                            x_axis=self.config.profiles.x_axis,
                            filters=payload["filters"],
                            config=self.config,
                            progress=progress,
                            task_id=combined_profile_task,
                        )
                        for combined_profile_path in combined_profile_paths:
                            self.logger.debug(f"Saved combined variable profile plot to: {combined_profile_path}")
                    elif self.config.profiles.combined_variable_metrics is not None:
                        self.logger.info(
                            f"Skipping combined variable profiles for region {payload['region']} because only one variable is selected."
                        )

            if self.config.scalar_tables is not None:
                for payload in output_payloads:
                    metrics = payload["metrics"]
                    variables = payload["variables"]
                    raw_metrics = {
                        metric
                        for metric_group in (
                            metrics.get("all_dims", {}).get("scalar"),
                            metrics.get("spatial_mean", {}).get("scalar"),
                            metrics.get("ensemble_mean", {}).get("scalar"),
                            metrics.get("probabilistic", {}).get("scalar"),
                        )
                        if metric_group is not None and "metric" in metric_group.coords
                        for metric in metric_group.coords["metric"].values.tolist()
                    } - NORMALIZED_METRICS

                    table_task = progress.add_task(
                        f"Generating scalar tables ({payload['region']} | {payload['scope_label']})",
                        total=len(variables) + 1,
                    )
                    for variable in variables:
                        progress.update(table_task, description=f"Generating {payload['region']} {variable} scalar tables")
                        variable_plot_folder = _table_output_folder(plot_root=payload["plot_folder"], variable=variable)
                        base_color = _get_variable_table_color(variable=variable, variables=variables)
                        scalar_table_paths = save_scalar_metric_tables(
                            metrics=metrics,
                            variables=[variable],
                            output_folder=variable_plot_folder,
                            filters=payload["filters"],
                            config=self.config,
                            metrics_keep=raw_metrics,
                            filename_prefix="scalar_metrics",
                            leadtime_unit=leadtime_unit,
                            base_color=base_color,
                        )
                        progress.advance(table_task)
                        for scalar_table_path in scalar_table_paths:
                            self.logger.debug(f"Saved scalar metrics table to: {scalar_table_path}")

                    progress.update(table_task, description=f"Generating {payload['region']} all_variables scalar tables")
                    combined_normalized_table_paths = save_scalar_metric_tables(
                        metrics=metrics,
                        variables=variables,
                        output_folder=_table_output_folder(plot_root=payload["plot_folder"], variable=None),
                        filters=payload["filters"],
                        config=self.config,
                        metrics_keep=NORMALIZED_METRICS,
                        filename_prefix="normalized_metrics",
                        title_prefix="Normalized metrics",
                        row_index=("variable", "metric"),
                        column_index=("model", "stat"),
                        stat_label_overrides={"prob": ""},
                        leadtime_unit=leadtime_unit,
                        base_color=_get_variable_table_color(variable=variables[0], variables=variables),
                    )
                    self.logger.info(f"Scalar table variable ({payload['region']}): all_variables")
                    progress.advance(table_task)
                    for normalized_table_path in combined_normalized_table_paths:
                        self.logger.debug(f"Saved combined normalized metrics table to: {normalized_table_path}")

            if self.config.scoreboards is not None:
                for payload in output_payloads:
                    scoreboard_task = progress.add_task(
                        f"Generating scoreboard ({payload['region']} | {payload['scope_label']})",
                        total=1,
                    )
                    scoreboard_path = save_scoreboard_plot(
                        metrics=payload["metrics"],
                        variables=payload["variables"],
                        plot_folder=payload["plot_folder"],
                        config=self.config,
                        leadtime_unit=leadtime_unit,
                        filters=payload["filters"],
                        filename_context_suffix=payload["filename_context"],
                    )
                    progress.advance(scoreboard_task)
                    self.logger.debug(f"Saved scoreboard plot to: {scoreboard_path}")
