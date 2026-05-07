from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path

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
    METRIC_CMAPS,
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

from .. import get_runs_and_metrics, metrics_to_df

from ...misc import Dask
from ...logging import configure_logging, LoggingConfig, get_console
from ...experiments.mlbc import MLBCExperimentMode


@dataclass
class CalculateMetricsRuntime:
    # Globals
    dask_workers: int | None
    memory_limit: str
    root_path: str | Path
    exp_mode: MLBCExperimentMode = "test"
    metrics: str | None = None # None means all available metrics
    items: list = field(default_factory=lambda: list(CALCULATE_METRICS_ITEMS))
    log_level: Literal["info", "debug"] = "info"
    print_user_config: bool = False
    print_console_tables: bool = False
    external_mask_path: str | Path | None = None
    external_mask_variable: str | Path | None = None
    disable_flox: bool = False
    variable_colors: dict[str, str] = field(default_factory=dict)
    # Labels
    truth: str = "an"
    reference: str = "fc"
    corrected: str = "pr"
    gain: str = "gain"
    diff: str = "pr-fc"
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
                metric_cmaps=METRIC_CMAPS,
                metric_cmap_default="viridis",
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

        self.plot_folder = self.config.global_config.exp_root_folder / self.config.global_config.plot_folder_name

        
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

        runs, metrics, _ = get_runs_and_metrics(
            exp_root=self.config.global_config.exp_root_folder,
            exp_mode=self.config.global_config.exp_mode,
            load_models=self.config.models.load_models,
            variables=load_variables,
            run_filters=load_run_filters,
            use_saved_mask=self.config.global_config.use_saved_mask,
            external_mask_data=external_mask_data,
            apply_external_mask_to_runs=True,
            metric_names=self.metrics,
            show_progress=True,
        )

        if self.reference not in runs:
            raise ValueError(
                f"Reference model {self.reference!r} not found in loaded runs. "
                f"Available models: {list(runs)}"
            )

        variables = [v for v in runs[self.reference].data_vars if v != "_has_var"]
        variable_units = _get_variable_units_from_runs(runs, variables)
        self.logger.info(f"Processed vars: {dict(zip(variables, variable_units))}")

        self.plot_folder.mkdir(parents=True, exist_ok=True)

        leadtime_unit = _infer_leadtime_unit_from_configs(self.config.global_config.exp_root_folder)

        filtered_variables = _filter_variables_by_filters(variables, self.filters)

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

        # Metric_vs_delta and scoreboard contexts
        filename_context = _merge_filename_contexts(
            _filters_filename_context(self.filters),
        )

        # Print info
        self.logger.info(f"Inferred leadtime unit: {leadtime_unit or 'unknown'}")
        self.logger.info(f"Inferred map regions: {sorted(region_extents) if region_extents else 'unknown'}")
        self.logger.info(f"Inferred region extents: {region_extents or 'unknown'}")

        inventory = {}
        for metric_type, section_dict in metrics.items():
            ds_scalar = section_dict.get("scalar")
            if ds_scalar is None or "metric" not in ds_scalar.coords:
                inventory[metric_type] = []
                continue
            inventory[metric_type] = [str(metric) for metric in ds_scalar.coords["metric"].values.tolist()]

        self.logger.info("Available scalar metrics:")
        for metric_type, metric_names in inventory.items():
            self.logger.info(f"  {metric_type}: {metric_names or 'none'}")

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
            if self.config.timeseries is not None:
                ts_task = progress.add_task("Generating field timeseries", total=len(filtered_variables))
                field_timeseries_paths = save_field_timeseries_plots(
                    load_models=self.config.models.load_models,
                    runs=runs,
                    metrics=metrics,
                    variables=filtered_variables,
                    plot_folder=self.plot_folder,
                    leadtime_unit=leadtime_unit,
                    filters=self.filters,
                    config=self.config,
                    combine_leadtimes=self.config.timeseries.combine_leadtimes,
                    plot_realization_members=self.config.timeseries.plot_realization_members,
                    disable_flox=self.disable_flox,
                    progress=progress,
                    task_id=ts_task,
                )
                for field_timeseries_path in field_timeseries_paths:
                    self.logger.debug(f"Saved field timeseries plot to: {field_timeseries_path}")

            if self.config.maps is not None:
                map_task = progress.add_task("Generating maps", total=len(filtered_variables))
                field_map_paths = save_field_and_metric_map_plots(
                    load_models=self.config.models.load_models,
                    runs=runs,
                    metrics=metrics,
                    variables=filtered_variables,
                    plot_folder=self.plot_folder,
                    leadtime_unit=leadtime_unit,
                    filters=self.filters,
                    config=self.config,
                    realization_mode=self.config.maps.realization_mode,
                    region_extents=region_extents,
                    progress=progress,
                    task_id=map_task,
                )
                for field_map_path in field_map_paths:
                    self.logger.debug(f"Saved map plot to: {field_map_path}")


            if self.config.metric_vs_delta is not None:
                metric_vs_deltametric_specs = _metric_vs_deltametric_specs(self.config.metric_vs_delta)
                diff_task = progress.add_task(
                    "Generating metric-vs-delta-metric plot",
                    total=len(metric_vs_deltametric_specs),
                )
                for metric_spec in metric_vs_deltametric_specs:
                    df_nodiff = metrics_to_df(
                        metrics=metrics,
                        variables=variables,
                        metric_names=metric_spec["forecast_metric"],
                        kind="scalar",
                        metric_type=metric_spec["forecast_metric_type"],
                        diff="no",
                        models=(self.config.models.reference_model,),
                    )
                    df_delta = metrics_to_df(
                        metrics=metrics,
                        variables=variables,
                        metric_names=metric_spec["delta_metric"],
                        kind="scalar",
                        metric_type=metric_spec["delta_metric_type"],
                        diff="delta",
                        models=self.config.models.comparison_models,
                    )

                    plot_path = save_metric_vs_deltametric_plot(
                        df_nodiff=df_nodiff,
                        df_delta=df_delta,
                        plot_folder=self.plot_folder,
                        forecast_metric=metric_spec["forecast_metric"],
                        delta_metric=metric_spec["delta_metric"],
                        leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                        region=", ".join(str(r) for r in self.filters.region) if self.filters.region else "",
                        filters=self.filters,
                        variable_unit=None, # TODO might be wrong
                        metric_vs_delta_byconfig=self.metric_vs_delta_byconfig,
                        point_size=20,
                        config=self.config,
                        filename_context_suffix=filename_context,
                    )
                    progress.advance(diff_task)
                    self.logger.debug(f"Saved leadtime-vs-global-metrics plot to: {plot_path}")

            if self.config.profiles is not None:
                profile_x_axis_fields = _normalize_profile_axis_fields(self.config.profiles.x_axis)
                profile_total = 1 if profile_x_axis_fields == ("variable",) else len(variables)
                profile_task = progress.add_task(
                    f"Generating metric profiles ({_format_axis_display_name(self.config.profiles.x_axis)})",
                    total=profile_total,
                )
                profile_plot_paths = save_metrics_vs_parameter_plots(
                    metrics=metrics,
                    variables=variables,
                    plot_folder=self.plot_folder,
                    leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                    variable_units=variable_units,
                    x_axis=self.config.profiles.x_axis,
                    filters=self.filters,
                    config=self.config,
                    progress=progress,
                    task_id=profile_task,
                )
                for profile_plot_path in profile_plot_paths:
                    self.logger.debug(f"Saved metric profile plot to: {profile_plot_path}")

                if self.config.profiles.combined_variable_metrics is not None and len(variables) > 1:
                    combined_profile_task = progress.add_task(
                        f"Generating combined variable profiles ({_format_axis_display_name(self.config.profiles.x_axis)})",
                        total=1,
                    )
                    combined_profile_paths = save_combined_variable_metric_profiles(
                        metrics=metrics,
                        variables=variables,
                        plot_folder=self.plot_folder,
                        leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                        x_axis=self.config.profiles.x_axis,
                        filters=self.filters,
                        config=self.config,
                        progress=progress,
                        task_id=combined_profile_task,
                    )
                    for combined_profile_path in combined_profile_paths:
                        self.logger.debug(f"Saved combined variable profile plot to: {combined_profile_path}")
                elif self.config.profiles.combined_variable_metrics is not None:
                    self.logger.info("Skipping combined variable profiles because only one variable is selected.")

            if self.config.scalar_tables is not None:
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

                table_task = progress.add_task("Generating scalar tables", total=len(variables) + 1)
                for variable in variables:
                    progress.update(table_task, description=f"Generating {variable} scalar tables")
                    variable_plot_folder = _table_output_folder(plot_root=self.plot_folder, variable=variable)
                    base_color = _get_variable_table_color(variable=variable, variables=variables)
                    scalar_table_paths = save_scalar_metric_tables(
                        metrics=metrics,
                        variables=[variable],
                        output_folder=variable_plot_folder,
                        filters=self.filters,
                        config=self.config,
                        metrics_keep=raw_metrics,
                        filename_prefix="scalar_metrics",
                        leadtime_unit=leadtime_unit,
                        base_color=base_color,
                    )
                    progress.advance(table_task)
                    for scalar_table_path in scalar_table_paths:
                        self.logger.debug(f"Saved scalar metrics table to: {scalar_table_path}")

                progress.update(table_task, description="Generating all_variables scalar tables")
                combined_normalized_table_paths = save_scalar_metric_tables(
                    # Combined normalized tables live under all_variables/tables.
                    metrics=metrics,
                    variables=variables,
                    output_folder=_table_output_folder(plot_root=self.plot_folder, variable=None),
                    filters=self.filters,
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
                self.logger.info("Scalar table variable: all_variables")
                progress.advance(table_task)
                for normalized_table_path in combined_normalized_table_paths:
                    self.logger.debug(f"Saved combined normalized metrics table to: {normalized_table_path}")

            if self.config.scoreboards is not None:
                scoreboard_task = progress.add_task("Generating scoreboard", total=1)
                scoreboard_path = save_scoreboard_plot(
                    metrics=metrics,
                    variables=variables,
                    plot_folder=self.plot_folder,
                    config=self.config,
                    leadtime_unit=leadtime_unit,
                    filters=self.filters,
                    filename_context_suffix=filename_context,
                )
                progress.advance(scoreboard_task)
                self.logger.debug(f"Saved scoreboard plot to: {scoreboard_path}")
