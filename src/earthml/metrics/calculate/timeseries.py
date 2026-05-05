from pathlib import Path
from contextlib import nullcontext

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from rich.progress import Progress

from ...logging import get_logger
from ...plots import plot_realization_timeseries

from .dataclasses import CalculateMetricsConfig
from .utils import (
    _apply_xarray_filters,
    _context_selection_entries,
    _plot_context_subtitle,
    _set_title_and_subtitle,
    _realization_dim,
    _format_variable_display_name,
    _ensure_grouped_output_folders,
    _get_variable_output_item_folder,
    _has_multi_realization,
    _context_filename_suffix,
    _get_da_unit,
    _resolved_single_region_label,
    _leadtime_values,
    _leadtime_shade_weight_map,
    _leadtime_linestyle_map,
    _members_arg,
    _format_label_with_unit,
    _format_metric_label,
)
from .constants import (
    VARIABLE_SUBFOLDERS,
    CONTEXT_AXES,
    AXIS_DISPLAY_NAMES,
)


logger = get_logger(__name__)


def _build_climatology_da(da: xr.DataArray) -> xr.DataArray:
    time_dim = da.earthml.guessed_dims.time
    if time_dim is None or time_dim not in da.dims:
        raise ValueError("Cannot compute climatology without a time dimension.")
    return da.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)

def _build_anomaly_da(da: xr.DataArray) -> xr.DataArray:
    time_dim = da.earthml.guessed_dims.time
    if time_dim is None or time_dim not in da.dims:
        raise ValueError("Cannot compute anomaly without a time dimension.")
    clim = _build_climatology_da(da)
    return da.groupby(f"{time_dim}.month") - clim

def _build_residual_da(
    left: xr.DataArray,
    right: xr.DataArray
) -> xr.DataArray:
    left_rdim = _realization_dim(left)
    right_rdim = _realization_dim(right)

    if (
        left_rdim is not None
        and right_rdim is not None
        and left_rdim == right_rdim
        and left.sizes.get(left_rdim, 1) > 1
        and right.sizes.get(right_rdim, 1) == 1
    ):
        right = right.squeeze(right_rdim, drop=True)
        right_rdim = None

    exclude_align_dims = tuple(dim for dim in (left_rdim, right_rdim) if dim is not None)
    left, right = xr.align(left, right, join="inner", exclude=exclude_align_dims)
    return left - right

def _shade_model_color(
    color: str | None,
    *,
    weight: float
) -> str | None:
    if color is None:
        return None
    rgb = np.array(mcolors.to_rgb(color))
    white = np.ones(3, dtype=float)
    shaded = (1.0 - weight) * rgb + weight * white
    return mcolors.to_hex(shaded)


def _combined_leadtime_legend_values(
    plot_data: dict[str, xr.DataArray],
    *,
    truth_model: str | None,
) -> list[object]:
    values: list[object] = []
    for model_name, da in plot_data.items():
        if model_name == truth_model:
            continue
        for leadtime_value in _leadtime_values(da):
            if leadtime_value not in values:
                values.append(leadtime_value)
    return values

def _leadtime_legend_label(
    value: object,
    *,
    leadtime_unit: str
) -> str:
    unit = leadtime_unit.strip()
    return f"{value} {unit}" if unit else str(value)

def _set_combined_leadtime_legend(
    ax: Axes,
    config: CalculateMetricsConfig,
    *,
    plot_data: dict[str, xr.DataArray],
    truth_model: str | None,
    leadtime_unit: str,
    leadtime_style: str,
) -> None:
    model_handles = [
        Line2D(
            [0],
            [0],
            color=config.models.model_colors.get(model_name),
            linewidth=2.0,
            alpha=1.0,
            label=config.models.display_names.get(model_name, str(model_name)),
        )
        for model_name in plot_data
    ]

    leadtime_values = _combined_leadtime_legend_values(plot_data, truth_model=truth_model)
    if leadtime_style == "shade":
        shade_weight_map = _leadtime_shade_weight_map(leadtime_values)
        leadtime_handles = [
            Line2D(
                [0],
                [0],
                color=_shade_model_color("#4c4c4c", weight=shade_weight_map[leadtime_value]),
                linewidth=2.0,
                label=_leadtime_legend_label(leadtime_value, leadtime_unit=leadtime_unit),
            )
            for leadtime_value in leadtime_values
        ]
        legend_title = "Color = model, shade = leadtime"
    elif leadtime_style == "linestyle":
        linestyle_map = _leadtime_linestyle_map(leadtime_values)
        leadtime_handles = [
            Line2D(
                [0],
                [0],
                color="0.25",
                linewidth=2.0,
                linestyle=linestyle_map[leadtime_value],
                label=_leadtime_legend_label(leadtime_value, leadtime_unit=leadtime_unit),
            )
            for leadtime_value in leadtime_values
        ]
        legend_title = "Color = model, linestyle = leadtime"
    else:
        raise ValueError(f"Unsupported leadtime style {leadtime_style!r}. Expected 'shade' or 'linestyle'.")

    handles = [*model_handles, *leadtime_handles]
    if handles:
        ax.legend(handles=handles, fontsize=9) #, title=legend_title)

def _metric_output_group(metric_type: str) -> str:
    return "deterministic" if metric_type == "deterministic" else "ensemble"

def _build_model_difference_da(
    corrected: xr.DataArray,
    reference: xr.DataArray,
) -> xr.DataArray | None:
    join_dims = [dim for dim in corrected.dims if dim in reference.dims]
    if not join_dims:
        return None
    corrected_aligned, reference_aligned = xr.align(corrected, reference, join="inner")
    if corrected_aligned.size == 0 or reference_aligned.size == 0:
        return None
    return corrected_aligned - reference_aligned


def save_field_timeseries_plots(
    load_models: list,
    *,
    runs: dict[str, xr.Dataset],
    metrics: dict[str, dict[str, xr.Dataset]],
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    config: CalculateMetricsConfig,
    combine_leadtimes: bool = False,
    plot_realization_members: bool = False,
    disable_flox: bool = False,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_order = [model_name for model_name in load_models if model_name in runs]
    truth_model = load_models[0] if load_models else None
    leadtime_style = config.timeseries.leadtime_style
    if leadtime_style not in {"shade", "linestyle"}:
        raise ValueError(
            f"Unsupported leadtime style {leadtime_style!r}. Expected 'shade' or 'linestyle'."
        )

    def _count_total_timeseries_plots() -> int:
        total = 0
        for variable in variables:
            model_data: dict[str, xr.DataArray] = {}
            context_reference: xr.DataArray | None = None

            for model_name in model_order:
                ds = runs.get(model_name)
                if ds is None or variable not in ds.data_vars:
                    continue
                da = _apply_xarray_filters(ds[variable], filters)
                time_dim = da.earthml.guessed_dims.time
                if time_dim is None or time_dim not in da.dims:
                    continue
                model_data[model_name] = da
                if context_reference is None:
                    context_reference = da

            if model_data and context_reference is not None:
                context_dims = tuple(
                    dim for dim in CONTEXT_AXES
                    if not (combine_leadtimes and dim == "leadtime")
                )
                for selection, context_values in _context_selection_entries(context_reference, context_dims=context_dims):
                    selected_data = {
                        model_name: da.sel({dim: value for dim, value in selection.items() if dim in da.dims}, drop=True)
                        for model_name, da in model_data.items()
                    }
                    selected_data = {model_name: da for model_name, da in selected_data.items() if da.size > 0}
                    if not selected_data:
                        continue

                    total += 1  # raw

                    try:
                        climatology_data = {
                            model_name: _build_climatology_da(da)
                            for model_name, da in selected_data.items()
                        }
                        if climatology_data:
                            total += 1
                    except Exception:
                        pass

                    try:
                        anomaly_data = {
                            model_name: _build_anomaly_da(da)
                            for model_name, da in selected_data.items()
                        }
                        if anomaly_data:
                            total += 1
                    except Exception:
                        pass

                    if truth_model is not None and truth_model in selected_data:
                        truth_da = selected_data[truth_model]
                        residual_plot_data: dict[str, xr.DataArray] = {}
                        for model_name in config.models.comparison_models:
                            if model_name not in selected_data:
                                continue
                            residual_plot_data[model_name] = _build_residual_da(selected_data[model_name], truth_da)
                        if residual_plot_data:
                            total += 1

            for metric_type, section_dict in metrics.items():
                ds_ts = section_dict.get("timeseries", xr.Dataset())
                if not isinstance(ds_ts, xr.Dataset) or variable not in ds_ts.data_vars:
                    continue

                da_var = _apply_xarray_filters(ds_ts[variable], filters)
                if "metric" not in da_var.dims:
                    continue

                for metric_name in [str(value) for value in da_var["metric"].values.tolist()]:
                    da_metric = da_var.sel(metric=metric_name, drop=True)
                    context_dims = tuple(
                        dim for dim in CONTEXT_AXES
                        if not (combine_leadtimes and dim == "leadtime")
                    )
                    for selection, context_values in _context_selection_entries(da_metric, context_dims=context_dims):
                        da_sel = da_metric.sel(
                            {dim: value for dim, value in selection.items() if dim in da_metric.dims},
                            drop=True,
                        )
                        if da_sel.size == 0 or "model" not in da_sel.dims:
                            continue

                        per_model_ts: dict[str, xr.DataArray] = {}
                        for model_name in [str(value) for value in da_sel["model"].values.tolist()]:
                            da_model = da_sel.sel(model=model_name, drop=True)
                            if da_model.size == 0:
                                continue
                            per_model_ts[model_name] = da_model

                        sample_da = next(iter(per_model_ts.values()), None)
                        time_dim = sample_da.earthml.guessed_dims.time if sample_da is not None else None
                        if per_model_ts and time_dim is not None and time_dim in sample_da.dims:
                            total += 1

        return total

    total_plots = _count_total_timeseries_plots()
    if progress is not None and task_id is not None:
        progress.update(task_id, total=total_plots, completed=0, description="Generating field timeseries")

    options_context = (
        xr.set_options(use_flox=False, use_numbagg=False)
        if disable_flox
        else nullcontext()
    )
    with options_context:
        for variable in variables:
            # print(f"Field timeseries variable: {variable}")
            _ensure_grouped_output_folders(
                plot_root=plot_folder,
                variable=variable,
                subfolder="timeseries",
            )
            timeseries_folder = _get_variable_output_item_folder(
                plot_root=plot_folder,
                variable=variable,
                subfolder="timeseries",
                output_group="field",
                item_name="fields",
            )
            model_data: dict[str, xr.DataArray] = {}
            context_reference: xr.DataArray | None = None

            for model_name in model_order:
                ds = runs.get(model_name)
                if ds is None or variable not in ds.data_vars:
                    continue
                da = _apply_xarray_filters(ds[variable], filters)
                time_dim = da.earthml.guessed_dims.time
                if time_dim is None or time_dim not in da.dims:
                    continue
                model_data[model_name] = da
                if context_reference is None:
                    context_reference = da

            if not model_data or context_reference is None:
                continue

            context_dims = tuple(
                dim for dim in CONTEXT_AXES
                if not (combine_leadtimes and dim == "leadtime")
            )
            for selection, context_values in _context_selection_entries(context_reference, context_dims=context_dims):
                selected_data = {
                    model_name: da.sel({dim: value for dim, value in selection.items() if dim in da.dims}, drop=True)
                    for model_name, da in model_data.items()
                }
                selected_data = {model_name: da for model_name, da in selected_data.items() if da.size > 0}
                if not selected_data:
                    continue

                reference_model = config.models.reference_model
                corrected_model = config.models.corrected_model
                fc_has_spread = _has_multi_realization(selected_data.get(reference_model))
                pr_has_spread = _has_multi_realization(selected_data.get(corrected_model))
                enable_spread = fc_has_spread or pr_has_spread
                show_legend = enable_spread

                base_title = _format_variable_display_name(variable)
                filename_context = _context_filename_suffix(context_values)
                base_unit = _get_da_unit(context_reference, variable)

                plot_jobs: list[tuple[str, str, dict[str, xr.DataArray]]] = [
                    ("raw", f"{base_title} spatial mean timeseries", selected_data),
                ]

                try:
                    climatology_data = {
                        model_name: _build_climatology_da(da)
                        for model_name, da in selected_data.items()
                    }
                    plot_jobs.append(("climatology", f"{base_title} climatology", climatology_data))
                except Exception:
                    climatology_data = {}

                try:
                    anomaly_data = {
                        model_name: _build_anomaly_da(da)
                        for model_name, da in selected_data.items()
                    }
                    plot_jobs.append(("anomaly", f"{base_title} anomaly", anomaly_data))
                except Exception:
                    anomaly_data = {}

                if truth_model is not None and truth_model in selected_data:
                    truth_da = selected_data[truth_model]
                    residual_plot_data: dict[str, xr.DataArray] = {}
                    for model_name in config.models.comparison_models:
                        if model_name not in selected_data:
                            continue
                        residual_plot_data[model_name] = _build_residual_da(selected_data[model_name], truth_da)
                    if residual_plot_data:
                        plot_jobs.append(
                            (
                                f"residual_minus_{truth_model}",
                                f"{base_title} residuals (model - {config.models.display_names.get(truth_model, truth_model)})",
                                residual_plot_data,
                            )
                        )

                for plot_kind, title, plot_data in plot_jobs:
                    if not plot_data:
                        continue
                    sample_da = next(iter(plot_data.values()))
                    plot_region = _resolved_single_region_label(sample_da, filters=filters)
                    x_dim = "month" if plot_kind == "climatology" else (sample_da.earthml.guessed_dims.time or "time")
                    fig, ax = plt.subplots(figsize=(12, 5.5))
                    plotted = False

                    for model_name, da in plot_data.items():
                        leadtime_values = _leadtime_values(da) if combine_leadtimes else []
                        if combine_leadtimes and "leadtime" in da.dims and leadtime_values:
                            shade_weight_map = _leadtime_shade_weight_map(leadtime_values)
                            linestyle_map = _leadtime_linestyle_map(leadtime_values)
                            truth_drawn = False
                            for leadtime_value in leadtime_values:
                                da_lt = da.sel(leadtime=leadtime_value, drop=True)
                                if model_name == truth_model:
                                    if truth_drawn:
                                        continue
                                    truth_drawn = True
                                    label = config.models.display_names.get(model_name, str(model_name))
                                    mean_label = label
                                    mean_alpha = 1.0
                                    line_color = config.models.model_colors.get(model_name)
                                    mean_ls = "-"
                                    members_ls = "--"
                                else:
                                    label = config.models.display_names.get(model_name, str(model_name))
                                    mean_label = label if leadtime_value == leadtime_values[-1] else None
                                    mean_alpha = 1.0
                                    if leadtime_style == "shade":
                                        line_color = _shade_model_color(
                                            config.models.model_colors.get(model_name),
                                            weight=shade_weight_map[leadtime_value],
                                        )
                                        mean_ls = "-"
                                        members_ls = "--"
                                    else:
                                        line_color = config.models.model_colors.get(model_name)
                                        mean_ls = linestyle_map[leadtime_value]
                                        members_ls = mean_ls

                                plot_realization_timeseries(
                                    da_lt,
                                    members=_members_arg(da_lt, enable_spread=enable_spread),
                                    ax=ax,
                                    x_dim=x_dim,
                                    ens_dim=_realization_dim(da_lt) or "realization",
                                    x_label="Month" if x_dim == "month" else "Time",
                                    y_label=_format_label_with_unit(base_title, base_unit),
                                    label=label,
                                    mean_label=mean_label,
                                    color=line_color,
                                    plot_members=plot_realization_members,
                                    members_ls=members_ls,
                                    members_alpha=0.08,
                                    mean_ls=mean_ls,
                                    mean_alpha=mean_alpha,
                                    spread_alpha=0.12,
                                    eager_compute=disable_flox,
                                )
                                plotted = True
                        else:
                            plot_realization_timeseries(
                                da,
                                members=_members_arg(da, enable_spread=enable_spread),
                                ax=ax,
                                x_dim=x_dim,
                                ens_dim=_realization_dim(da) or "realization",
                                x_label="Month" if x_dim == "month" else "Time",
                                y_label=_format_label_with_unit(base_title, base_unit),
                                label=config.models.display_names.get(model_name, str(model_name)),
                                mean_label=config.models.display_names.get(model_name, str(model_name)),
                                color=config.models.model_colors.get(model_name),
                                plot_members=plot_realization_members,
                                eager_compute=disable_flox,
                            )
                            plotted = True

                    if not plotted:
                        plt.close(fig)
                        continue

                    _set_title_and_subtitle(
                        fig,
                        ax,
                        title=title,
                        subtitle=_plot_context_subtitle(
                            context_values,
                            leadtime_unit=leadtime_unit,
                            filters=filters,
                        ),
                    )
                    if plot_kind.startswith("residual_"):
                        ax.axhline(0.0, color="0.3", lw=1.0, ls=":")
                    if combine_leadtimes:
                        _set_combined_leadtime_legend(
                            ax,
                            config,
                            plot_data=plot_data,
                            truth_model=truth_model,
                            leadtime_unit=leadtime_unit,
                            leadtime_style=leadtime_style,
                        )
                    elif show_legend:
                        ax.legend(fontsize=9)
                    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

                    filename = f"{plot_kind}_timeseries_{filename_context}"
                    if combine_leadtimes and "leadtime" not in context_values:
                        filename = f"{filename}_combined_leadtime"
                    plot_path = timeseries_folder / f"{filename}.png"
                    if progress is not None and task_id is not None:
                        progress.update(
                            task_id,
                            description=f"Timeseries | {variable} | field | {plot_kind}",
                        )
                    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    saved_paths.append(plot_path)
                    if progress is not None and task_id is not None:
                        progress.advance(task_id)
                    logger.debug(f"Saved {plot_kind} timeseries for region={plot_region or 'unknown'} to: {plot_path}")

            for metric_type, section_dict in metrics.items():
                ds_ts = section_dict.get("timeseries", xr.Dataset())
                if not isinstance(ds_ts, xr.Dataset) or variable not in ds_ts.data_vars:
                    continue

                da_var = _apply_xarray_filters(ds_ts[variable], filters)
                if "metric" not in da_var.dims:
                    continue

                metric_values = [str(value) for value in da_var["metric"].values.tolist()]
                for metric_name in metric_values:
                    metric_folder = _get_variable_output_item_folder(
                        plot_root=plot_folder,
                        variable=variable,
                        subfolder="timeseries",
                        output_group=_metric_output_group(metric_type),
                        item_name=metric_name,
                    )
                    da_metric = da_var.sel(metric=metric_name, drop=True)
                    context_dims = tuple(
                        dim for dim in CONTEXT_AXES
                        if not (combine_leadtimes and dim == "leadtime")
                    )
                    for selection, context_values in _context_selection_entries(da_metric, context_dims=context_dims):
                        da_sel = da_metric.sel(
                            {dim: value for dim, value in selection.items() if dim in da_metric.dims},
                            drop=True,
                        )
                        if da_sel.size == 0 or "model" not in da_sel.dims:
                            continue

                        per_model_ts: dict[str, xr.DataArray] = {}
                        for model_name in [str(value) for value in da_sel["model"].values.tolist()]:
                            da_model = da_sel.sel(model=model_name, drop=True)
                            if da_model.size == 0:
                                continue
                            per_model_ts[model_name] = da_model

                        if not per_model_ts:
                            continue

                        sample_da = next(iter(per_model_ts.values()))
                        plot_region = _resolved_single_region_label(sample_da, filters=filters)
                        time_dim = sample_da.earthml.guessed_dims.time
                        if time_dim is None or time_dim not in sample_da.dims:
                            continue

                        base_unit = _get_da_unit(sample_da) or _get_da_unit(
                            runs.get(config.models.truth_model, xr.Dataset()),
                            variable,
                        )
                        metric_label = _format_metric_label(metric_name, base_unit=base_unit)
                        filename_context = _context_filename_suffix(context_values)
                        fig, ax = plt.subplots(figsize=(12, 5.5))
                        diff_fig, diff_ax = plt.subplots(figsize=(12, 5.5))
                        plotted = False
                        diff_plotted = False
                        show_legend = any(_has_multi_realization(da) for da in per_model_ts.values())

                        for model_name, da in per_model_ts.items():
                            leadtime_values = _leadtime_values(da) if combine_leadtimes else []
                            if combine_leadtimes and "leadtime" in da.dims and leadtime_values:
                                shade_weight_map = _leadtime_shade_weight_map(leadtime_values)
                                linestyle_map = _leadtime_linestyle_map(leadtime_values)
                                truth_drawn = False
                                for leadtime_value in leadtime_values:
                                    da_lt = da.sel(leadtime=leadtime_value, drop=True)
                                    if model_name == truth_model:
                                        if truth_drawn:
                                            continue
                                        truth_drawn = True
                                        mean_label = config.models.display_names.get(model_name, str(model_name))
                                        mean_alpha = 1.0
                                        line_color = config.models.model_colors.get(model_name)
                                        mean_ls = "-"
                                        members_ls = "--"
                                    else:
                                        mean_label = (
                                            config.models.display_names.get(model_name, str(model_name))
                                            if leadtime_value == leadtime_values[-1]
                                            else None
                                        )
                                        mean_alpha = 1.0
                                        if leadtime_style == "shade":
                                            line_color = _shade_model_color(
                                                config.models.model_colors.get(model_name),
                                                weight=shade_weight_map[leadtime_value],
                                            )
                                            mean_ls = "-"
                                            members_ls = "--"
                                        else:
                                            line_color = config.models.model_colors.get(model_name)
                                            mean_ls = linestyle_map[leadtime_value]
                                            members_ls = mean_ls

                                    plot_realization_timeseries(
                                        da_lt,
                                        members=_members_arg(da_lt, enable_spread=show_legend),
                                        ax=ax,
                                        x_dim=time_dim,
                                        ens_dim=_realization_dim(da_lt) or "realization",
                                        x_label="Time",
                                        y_label=metric_label,
                                        label=config.models.display_names.get(model_name, str(model_name)),
                                        mean_label=mean_label,
                                        color=line_color,
                                        plot_members=plot_realization_members,
                                        members_ls=members_ls,
                                        members_alpha=0.08,
                                        mean_ls=mean_ls,
                                        mean_alpha=mean_alpha,
                                        spread_alpha=0.12,
                                        eager_compute=disable_flox,
                                    )
                                    plotted = True
                            else:
                                plot_realization_timeseries(
                                    da,
                                    members=_members_arg(da, enable_spread=show_legend),
                                    ax=ax,
                                    x_dim=time_dim,
                                    ens_dim=_realization_dim(da) or "realization",
                                    x_label="Time",
                                    y_label=metric_label,
                                    label=config.models.display_names.get(model_name, str(model_name)),
                                    mean_label=config.models.display_names.get(model_name, str(model_name)),
                                    color=config.models.model_colors.get(model_name),
                                    plot_members=plot_realization_members,
                                    eager_compute=disable_flox,
                                )
                                plotted = True

                        if not plotted:
                            plt.close(fig)
                            plt.close(diff_fig)
                            continue

                        diff_model_data: dict[str, xr.DataArray] = {}
                        corrected_model = config.models.corrected_model
                        reference_model = config.models.reference_model
                        corrected_da = per_model_ts.get(corrected_model)
                        reference_da = per_model_ts.get(reference_model)
                        if corrected_da is not None and reference_da is not None:
                            diff_da = _build_model_difference_da(corrected_da, reference_da)
                            if diff_da is not None and diff_da.size > 0:
                                diff_model_data[config.models.difference_model] = diff_da
                                diff_show_legend = _has_multi_realization(diff_da)
                                diff_leadtime_values = _leadtime_values(diff_da) if combine_leadtimes else []
                                if combine_leadtimes and "leadtime" in diff_da.dims and diff_leadtime_values:
                                    shade_weight_map = _leadtime_shade_weight_map(diff_leadtime_values)
                                    linestyle_map = _leadtime_linestyle_map(diff_leadtime_values)
                                    for leadtime_value in diff_leadtime_values:
                                        da_lt = diff_da.sel(leadtime=leadtime_value, drop=True)
                                        if leadtime_style == "shade":
                                            line_color = _shade_model_color(
                                                config.models.model_colors.get(config.models.difference_model, config.models.model_colors.get(corrected_model)),
                                                weight=shade_weight_map[leadtime_value],
                                            )
                                            mean_ls = "-"
                                            members_ls = "--"
                                        else:
                                            line_color = config.models.model_colors.get(config.models.difference_model, config.models.model_colors.get(corrected_model))
                                            mean_ls = linestyle_map[leadtime_value]
                                            members_ls = mean_ls

                                        plot_realization_timeseries(
                                            da_lt,
                                            members=_members_arg(da_lt, enable_spread=diff_show_legend),
                                            ax=diff_ax,
                                            x_dim=time_dim,
                                            ens_dim=_realization_dim(da_lt) or "realization",
                                            x_label="Time",
                                            y_label=f"Delta {metric_label}",
                                            label=config.models.display_names.get(config.models.difference_model, config.models.difference_model),
                                            mean_label=(
                                                config.models.display_names.get(config.models.difference_model, config.models.difference_model)
                                                if leadtime_value == diff_leadtime_values[-1]
                                                else None
                                            ),
                                            color=line_color,
                                            plot_members=plot_realization_members,
                                            members_ls=members_ls,
                                            members_alpha=0.08,
                                            mean_ls=mean_ls,
                                            mean_alpha=1.0,
                                            spread_alpha=0.12,
                                            eager_compute=disable_flox,
                                        )
                                        diff_plotted = True
                                else:
                                    plot_realization_timeseries(
                                        diff_da,
                                        members=_members_arg(diff_da, enable_spread=diff_show_legend),
                                        ax=diff_ax,
                                        x_dim=time_dim,
                                        ens_dim=_realization_dim(diff_da) or "realization",
                                        x_label="Time",
                                        y_label=f"Delta {metric_label}",
                                        label=config.models.display_names.get(config.models.difference_model, config.models.difference_model),
                                        mean_label=config.models.display_names.get(config.models.difference_model, config.models.difference_model),
                                        color=config.models.model_colors.get(config.models.difference_model, config.models.model_colors.get(corrected_model)),
                                        plot_members=plot_realization_members,
                                        eager_compute=disable_flox,
                                    )
                                    diff_plotted = True

                        _set_title_and_subtitle(
                            fig,
                            ax,
                            title=f"{metric_label} timeseries ({_format_variable_display_name(variable)})",
                            subtitle=_plot_context_subtitle(
                                context_values,
                                leadtime_unit=leadtime_unit,
                                filters=filters,
                            ),
                        )
                        if combine_leadtimes:
                            _set_combined_leadtime_legend(
                                ax,
                                config,
                                plot_data=per_model_ts,
                                truth_model=truth_model,
                                leadtime_unit=leadtime_unit,
                                leadtime_style=leadtime_style,
                            )
                        elif show_legend:
                            ax.legend(fontsize=9)
                        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

                        filename = f"{metric_type}_{metric_name}_timeseries_{filename_context}"
                        if combine_leadtimes and "leadtime" not in context_values:
                            filename = f"{filename}_combined_leadtime"
                        plot_path = metric_folder / f"{filename}.png"
                        if progress is not None and task_id is not None:
                            progress.update(
                                task_id,
                                description=f"Timeseries | {variable} | {metric_type} | {metric_name}",
                            )
                        fig.savefig(plot_path, bbox_inches="tight", dpi=150)
                        plt.close(fig)
                        saved_paths.append(plot_path)
                        if progress is not None and task_id is not None:
                            progress.advance(task_id)
                        logger.debug(f"Saved metric timeseries for metric={metric_name} type={metric_type} region={plot_region or 'unknown'} to: {plot_path}")

                        if diff_plotted:
                            _set_title_and_subtitle(
                                diff_fig,
                                diff_ax,
                                title=f"Delta {metric_label} timeseries ({_format_variable_display_name(variable)})",
                                subtitle=_plot_context_subtitle(
                                    context_values,
                                    leadtime_unit=leadtime_unit,
                                    filters=filters,
                                ),
                            )
                            diff_ax.axhline(0.0, color="0.3", lw=1.0, ls=":")
                            if combine_leadtimes:
                                _set_combined_leadtime_legend(
                                    diff_ax,
                                    config,
                                    plot_data=diff_model_data,
                                    truth_model=None,
                                    leadtime_unit=leadtime_unit,
                                    leadtime_style=leadtime_style,
                                )
                            elif _has_multi_realization(next(iter(diff_model_data.values()))):
                                diff_ax.legend(fontsize=9)
                            diff_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

                            diff_path = metric_folder / f"{filename}_diff.png"
                            diff_fig.savefig(diff_path, bbox_inches="tight", dpi=150)
                            saved_paths.append(diff_path)
                            logger.debug(
                                f"Saved diff metric timeseries for metric={metric_name} "
                                f"type={metric_type} region={plot_region or 'unknown'} to: {diff_path}"
                            )
                        plt.close(diff_fig)

    return saved_paths
