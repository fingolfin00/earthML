from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from rich.progress import Progress

from ...logging import get_logger
from .. import metrics_to_df_single_region

from .dataclasses import CalculateMetricsConfig
from .utils import (
    _normalize_profile_axis_fields,
    _apply_df_filters_except_axis,
    _format_axis_display_name,
    _ordered_unique_variables,
    _safe_relative_change,
    _format_variable_display_name,
    _format_metric_label,
    _plot_context_subtitle,
    _set_title_and_subtitle,
    _sanitize_filename_fragment,
    _merge_filename_contexts,
    _filters_filename_context,
    _get_variable_colors,
    _ensure_grouped_output_folders,
    _get_variable_output_item_folder,
)
from .constants import (
    VARIABLE_SUBFOLDERS,
    CONTEXT_AXES,
    AXIS_DISPLAY_NAMES,
)

logger = get_logger(__name__)


def _profile_axis_column_name(
    x_axis: str | Sequence[str]
) -> str:
    x_axis_fields = _normalize_profile_axis_fields(x_axis)
    if len(x_axis_fields) == 1:
        return x_axis_fields[0]
    return f"__profile_x_axis_{'_'.join(x_axis_fields)}__"

def _ensure_profile_axis_column(
    df: pd.DataFrame,
    *,
    x_axis: str | Sequence[str]
) -> pd.DataFrame:
    x_axis_fields = _normalize_profile_axis_fields(x_axis)
    if len(x_axis_fields) == 1:
        return df
    if any(field not in df.columns for field in x_axis_fields):
        return df
    out = df.copy()
    out[_profile_axis_column_name(x_axis_fields)] = list(
        out.loc[:, list(x_axis_fields)].itertuples(index=False, name=None)
    )
    return out

def _profile_output_folder(
    *,
    plot_root: Path,
    variable: str | None = None,
) -> Path:
    folder_variable = variable if variable is not None else "all_variables"
    return get_variable_subfolder(plot_root=plot_root, variable=folder_variable, subfolder="profiles")

def _profile_output_group(metric_type: str) -> str:
    if metric_type == "all_dims_with_ensemble_overlay":
        return "all_dims"
    return metric_type

def _expand_profile_metric_modes(
    metric_modes: dict[str, str | tuple[str, ...] | list[str]],
) -> list[tuple[str, str]]:
    expanded: list[tuple[str, str]] = []
    for metric_name, metric_mode in metric_modes.items():
        modes = [metric_mode] if isinstance(metric_mode, str) else list(metric_mode)
        expanded.extend((metric_name, str(mode)) for mode in modes)
    return expanded

def _profile_item_name(metric_name: str, metric_type: str) -> str:
    if metric_type == "all_dims_with_ensemble_overlay":
        return f"{metric_name}_with_ensemble_overlay"
    return metric_name

def _profile_metric_output_folder(
    *,
    plot_root: Path,
    variable: str | None,
    metric_name: str,
    metric_type: str,
) -> Path:
    folder_variable = variable if variable is not None else "all_variables"
    _ensure_grouped_output_folders(
        plot_root=plot_root,
        variable=folder_variable,
        subfolder="profiles",
    )
    return _get_variable_output_item_folder(
        plot_root=plot_root,
        variable=folder_variable,
        subfolder="profiles",
        output_group=_profile_output_group(metric_type),
        item_name=_profile_item_name(metric_name, metric_type),
    )


def get_variable_plot_folder(
    *,
    plot_root: Path,
    variable: str
) -> Path:
    variable_folder = plot_root / variable
    variable_folder.mkdir(parents=True, exist_ok=True)
    return variable_folder

def get_variable_subfolder(
    *,
    plot_root: Path,
    variable: str,
    subfolder: str
) -> Path:
    if subfolder not in VARIABLE_SUBFOLDERS:
        raise ValueError(f"Unsupported variable subfolder {subfolder!r}. Expected one of {VARIABLE_SUBFOLDERS}.")
    output_folder = get_variable_plot_folder(plot_root=plot_root, variable=variable) / subfolder
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder

def _profile_template_df(
    metric_type: str,
    *,
    df_det: pd.DataFrame,
    df_spatial: pd.DataFrame,
    df_ens: pd.DataFrame,
    df_prob: pd.DataFrame,
) -> pd.DataFrame:
    if metric_type == "probabilistic":
        return df_prob
    if metric_type == "all_dims":
        return df_det
    if metric_type == "spatial_mean":
        return df_spatial
    if metric_type == "ensemble_mean":
        return df_ens
    if metric_type == "all_dims_with_ensemble_overlay":
        return df_ens
    raise ValueError(f"Unsupported metric profile plot mode {metric_type!r}")

def _profile_deterministic_mode_df(
    metric_type: str,
    *,
    df_det: pd.DataFrame,
    df_spatial: pd.DataFrame,
    df_ens: pd.DataFrame,
) -> pd.DataFrame:
    if metric_type == "all_dims":
        return df_det
    if metric_type == "spatial_mean":
        return df_spatial
    if metric_type == "ensemble_mean":
        return df_ens
    raise ValueError(f"Unsupported deterministic metric profile plot mode {metric_type!r}")

def _profile_context_columns(
    df: pd.DataFrame,
    *,
    x_axis: str | Sequence[str],
    shade_by: str | None = None,
) -> list[str]:
    x_axis_fields = set(_normalize_profile_axis_fields(x_axis))
    return [
        col for col in CONTEXT_AXES
        if (
            col in df.columns
            and col not in x_axis_fields
            and col != shade_by
            and df[col].nunique(dropna=False) > 1
        )
    ]


def _matches_profile_axis_filters(
    value: object,
    *,
    x_axis_fields: Sequence[str],
    filters: dict | None,
) -> bool:
    if not filters:
        return True

    if len(x_axis_fields) == 1:
        components = (value,)
    else:
        if not isinstance(value, tuple) or len(value) != len(x_axis_fields):
            return False
        components = value

    for field, component in zip(x_axis_fields, components):
        allowed = filters.get(field)
        if allowed is None:
            continue
        allowed_values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        normalized_allowed = {str(item) for item in allowed_values if item is not None}
        if str(component) not in normalized_allowed:
            return False
    return True

def _profile_axis_value_present(value: object) -> bool:
    if isinstance(value, tuple):
        return all(not pd.isna(component) for component in value)
    return pd.notna(value)

def _profile_axis_sort_key(
    value: object,
    *,
    x_axis_fields: tuple[str, ...],
    variables: list[str],
) -> tuple[object, ...]:
    variable_order = {variable: idx for idx, variable in enumerate(_ordered_unique_variables(variables))}
    components = value if isinstance(value, tuple) else (value,)
    sort_key: list[object] = []
    for field, component in zip(x_axis_fields, components):
        if pd.isna(component):
            sort_key.append((2, ""))
            continue
        if field == "leadtime":
            sort_key.append((0, component))
        elif field == "variant":
            sort_key.append((0, str(component) != "", str(component)))
        elif field == "variable":
            sort_key.append((0, variable_order.get(str(component), len(variable_order)), str(component)))
        else:
            sort_key.append((0, str(component)))
    return tuple(sort_key)

def _ordered_profile_axis_values(
    df: pd.DataFrame,
    *,
    x_axis: str | Sequence[str],
    variables: list[str],
    filters: dict | None = None,
) -> list[object]:
    x_axis_fields = _normalize_profile_axis_fields(x_axis)
    x_axis_column = _profile_axis_column_name(x_axis_fields)
    if x_axis_column not in df.columns:
        return []
    if "value" in df.columns:
        df = df[df["value"].notna()]
        if df.empty:
            return []

    values = [value for value in pd.unique(df[x_axis_column]) if _profile_axis_value_present(value)]
    values = [
        value
        for value in values
        if _matches_profile_axis_filters(value, x_axis_fields=x_axis_fields, filters=filters)
    ]
    if len(x_axis_fields) == 1 and x_axis_fields[0] == "leadtime":
        return sorted(values)
    if len(x_axis_fields) == 1 and x_axis_fields[0] == "variant":
        return sorted(values, key=lambda value: (str(value) != "", str(value)))
    if len(x_axis_fields) == 1 and x_axis_fields[0] == "variable":
        ordered_variables = _ordered_unique_variables(variables)
        return [value for value in ordered_variables if value in values]
    if len(x_axis_fields) > 1:
        return sorted(values, key=lambda value: _profile_axis_sort_key(value, x_axis_fields=x_axis_fields, variables=variables))
    return values

def _profile_axis_source_df(
    df: pd.DataFrame,
    config: CalculateMetricsConfig,
) -> pd.DataFrame:
    if df.empty or "model" not in df.columns:
        return df

    for preferred_model in (
        config.models.corrected_model,
        config.models.reference_model,
        config.models.truth_model,
    ):
        df_model = df[df["model"] == preferred_model]
        if not df_model.empty:
            return df_model
    return df

def _profile_axis_index(x_values: list[object], *, x_axis: str | Sequence[str]) -> pd.Index:
    # Force tuple-valued composite axes to stay a plain object Index instead of
    # being auto-promoted by pandas into a MultiIndex.
    values = np.empty(len(x_values), dtype=object)
    values[:] = list(x_values)
    return pd.Index(values, name=x_axis)

def _format_duplicate_profile_key(
    *,
    key_names: tuple[str, ...],
    key_values: tuple[object, ...],
) -> str:
    parts: list[str] = []
    for name, value in zip(key_names, key_values):
        if pd.isna(value):
            value_repr = "NaN"
        else:
            value_repr = str(value)
        parts.append(f"{name}={value_repr}")
    return ", ".join(parts)

def _build_strict_profile_series(
    df: pd.DataFrame,
    *,
    x_axis: str | Sequence[str],
    x_values: list[object],
) -> pd.Series:
    if df.empty:
        return pd.Series(index=_profile_axis_index(x_values, x_axis=x_axis), dtype=float)

    duplicate_counts = (
        df.groupby([x_axis], dropna=False)
        .size()
        .reset_index(name="count")
    )
    duplicate_counts = duplicate_counts[duplicate_counts["count"] > 1]
    if not duplicate_counts.empty:
        duplicate_examples = [
            _format_duplicate_profile_key(
                key_names=(x_axis,),
                key_values=(record[x_axis],),
            )
            for record in duplicate_counts.head(5).to_dict("records")
        ]
        raise ValueError(
            "Metric profile generation found duplicate x-axis values after filtering/splitting. "
            "This would require forbidden aggregation in calculate_metrics.py. "
            f"Duplicate keys (showing up to 5): {duplicate_examples}"
        )

    series = df.set_index(x_axis)["value"]
    return series.reindex(x_values)

def _build_strict_member_matrix(
    df: pd.DataFrame,
    *,
    x_axis: str | Sequence[str],
    x_values: list[object],
) -> pd.DataFrame | None:
    if df.empty or "realization" not in df.columns:
        return None

    key_cols = [x_axis, "realization"]
    duplicate_counts = (
        df.groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )
    duplicate_counts = duplicate_counts[duplicate_counts["count"] > 1]
    if not duplicate_counts.empty:
        duplicate_examples = [
            _format_duplicate_profile_key(
                key_names=(x_axis, "realization"),
                key_values=(record[x_axis], record["realization"]),
            )
            for record in duplicate_counts.head(5).to_dict("records")
        ]
        raise ValueError(
            "Metric profile generation found duplicate x-axis/member pairs after filtering/splitting. "
            "This would require forbidden aggregation in calculate_metrics.py. "
            f"Duplicate keys (showing up to 5): {duplicate_examples}"
        )

    members = pd.pivot(df, index=x_axis, columns="realization", values="value").reindex(x_values)
    return members if not members.empty else None

def _build_deterministic_profile_series(
    df: pd.DataFrame,
    *,
    x_axis: str | Sequence[str],
    x_values: list[object],
) -> tuple[pd.Series, pd.DataFrame | None]:
    empty_series = pd.Series(index=_profile_axis_index(x_values, x_axis=x_axis), dtype=float)
    if df.empty:
        return empty_series, None

    if "realization" in df.columns:
        members = _build_strict_member_matrix(df, x_axis=x_axis, x_values=x_values)
        if members is not None and pd.notna(members.to_numpy()).any():
            mean_values = np.nanmean(members.to_numpy(), axis=1)
            return (
                pd.Series(mean_values, index=_profile_axis_index(x_values, x_axis=x_axis), dtype=float),
                members,
            )

    return _build_strict_profile_series(df, x_axis=x_axis, x_values=x_values), None

def _format_profile_axis_tick(value: object, *, x_axis: str | Sequence[str]) -> str:
    x_axis_fields = _normalize_profile_axis_fields(x_axis)
    if len(x_axis_fields) == 1:
        return _format_profile_axis_component(x_axis_fields[0], value)
    components = value if isinstance(value, tuple) else (value,)
    return ", ".join(
        f"{AXIS_DISPLAY_NAMES.get(field, field.replace('_', ' ').title())}={_format_profile_axis_component(field, component)}"
        for field, component in zip(x_axis_fields, components)
    )

def _format_profile_axis_component(field: str, value: object) -> str:
    if field == "variable":
        return _format_variable_display_name(str(value))
    if field == "variant":
        return str(value) if str(value) != "" else "default"
    return str(value)

def _profile_axis_filename_fragment(x_axis: str | Sequence[str]) -> str:
    return _sanitize_filename_fragment("_".join(_normalize_profile_axis_fields(x_axis)))

def _model_linestyle(
    model_name: str,
    config: CalculateMetricsConfig,
    *,
    probabilistic: bool = False,
) -> str:
    if model_name == config.models.truth_model:
        return "-"
    if model_name == config.models.reference_model:
        return ":" if probabilistic else "--"
    if model_name == config.models.corrected_model:
        return "-." if probabilistic else "-"
    return "-"


def _set_combined_variable_profile_legend(
    ax: Axes,
    *,
    variables: list[str],
    variable_colors: dict[str, str],
    model_values: list[object],
    metric_type: str,
    shade_values: list[object],
    config: CalculateMetricsConfig,
) -> None:
    handles: list[Line2D] = []

    def _section(title: str) -> None:
        handles.append(Line2D([], [], linestyle="None", label=title))

    visible_variables = [variable for variable in variables if variable in variable_colors]
    if visible_variables:
        _section("Variable")
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=variable_colors[variable],
                    marker="o",
                    linestyle="-",
                    linewidth=2.0,
                    markersize=5,
                    label=_format_variable_display_name(str(variable)),
                )
                for variable in visible_variables
            ]
        )

    visible_models = [str(model_name) for model_name in model_values if pd.notna(model_name)]
    if visible_models:
        _section("Series")
        handles.extend(
            [
                    Line2D(
                        [0],
                        [0],
                        color="0.25",
                        linestyle=_model_linestyle(model_name, config, probabilistic=(metric_type == "probabilistic")),
                        linewidth=2.0,
                        marker="o",
                        markersize=5,
                    label=config.models.display_names.get(model_name, model_name),
                )
                for model_name in visible_models
            ]
        )
        if metric_type == "all_dims_with_ensemble_overlay":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.25",
                    linestyle=":",
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    label="Ensemble mean",
                )
            )

    if len(shade_values) > 1:
        _section(config.profiles.shade_label)
        shade_count = max(1, len(shade_values))
        for idx, shade_value in enumerate(shade_values):
            alpha = 0.35 + 0.55 * (idx + 1) / shade_count
            label = f"{shade_value}M" if config.profiles.shade_by == "total_months" else str(shade_value)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=(0.15, 0.15, 0.15, alpha),
                    linestyle="-",
                    linewidth=3.0,
                    label=label,
                )
            )

    if handles:
        legend = ax.legend(handles=handles, fontsize=8, loc="best", frameon=True, ncol=1)
        for text in legend.get_texts():
            label = text.get_text()
            if label in {"Variable", "Series", config.profiles.shade_label}:
                text.set_fontweight("bold")

def _set_transformed_combined_variable_profile_legend(
    ax: Axes,
    *,
    variables: list[str],
    variable_colors: dict[str, str],
    metric_type: str,
    shade_values: list[object],
    config: CalculateMetricsConfig,
) -> None:
    handles: list[Line2D] = []

    def _section(title: str) -> None:
        handles.append(Line2D([], [], linestyle="None", label=title))

    visible_variables = [variable for variable in variables if variable in variable_colors]
    if visible_variables:
        _section("Variable")
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color=variable_colors[variable],
                    marker="o",
                    linestyle="-",
                    linewidth=2.0,
                    markersize=5,
                    label=_format_variable_display_name(str(variable)),
                )
                for variable in visible_variables
            ]
        )

    _section("Series")
    handles.append(
        Line2D(
            [0],
            [0],
            color=config.models.model_colors.get(config.models.difference_model, "0.25"),
            linestyle="-",
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=config.models.display_names.get(config.models.difference_model, config.models.difference_model),
        )
    )
    if metric_type == "all_dims_with_ensemble_overlay":
        handles.append(
            Line2D(
                [0],
                [0],
                color=config.models.model_colors.get(config.models.difference_model, "0.25"),
                linestyle=":",
                linewidth=2.0,
                marker="o",
                markersize=5,
                label="Ensemble mean",
            )
        )

    if len(shade_values) > 1:
        _section(config.profiles.shade_label)
        shade_count = max(1, len(shade_values))
        for idx, shade_value in enumerate(shade_values):
            alpha = 0.35 + 0.55 * (idx + 1) / shade_count
            label = f"{shade_value}M" if config.profiles.shade_by == "total_months" else str(shade_value)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=(0.15, 0.15, 0.15, alpha),
                    linestyle="-",
                    linewidth=3.0,
                    label=label,
                )
            )

    if handles:
        legend = ax.legend(handles=handles, fontsize=8, loc="best", frameon=True, ncol=1)
        for text in legend.get_texts():
            label = text.get_text()
            if label in {"Variable", "Series", config.profiles.shade_label}:
                text.set_fontweight("bold")


def save_metrics_vs_parameter_plots(
    *,
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    variable_units: dict[str, str | None] | None,
    x_axis: str,
    filters: dict | None,
    config: CalculateMetricsConfig,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    supported_axes = {"leadtime", "train_period", "loss", "variant", "variable", "region"}
    x_axis_fields = _normalize_profile_axis_fields(x_axis)
    unsupported_axes = [field for field in x_axis_fields if field not in supported_axes]
    if unsupported_axes:
        raise ValueError(
            f"Unsupported metric profile axis {x_axis!r}. "
            f"Unsupported fields={unsupported_axes!r}. Expected fields from {sorted(supported_axes)}."
        )
    if len(x_axis_fields) > 1 and "variable" in x_axis_fields:
        raise ValueError("Combined metric profile x_axis does not support 'variable' as one of the combined fields.")
    x_axis_column = _profile_axis_column_name(x_axis_fields)

    expanded_metric_modes = _expand_profile_metric_modes(config.profiles.metrics)
    metric_names = list(dict.fromkeys(metric_name for metric_name, _ in expanded_metric_modes))
    df_det = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="all_dims",
        diff="no",
    )
    df_spatial = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="spatial_mean",
        diff="no",
    )
    df_ens = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="ensemble_mean",
        diff="no",
    )
    df_prob = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="probabilistic",
        diff="no",
    )

    df_det = df_det[df_det["metric"].isin(metric_names)]
    df_spatial = df_spatial[df_spatial["metric"].isin(metric_names)]
    df_ens = df_ens[df_ens["metric"].isin(metric_names)]
    df_prob = df_prob[df_prob["metric"].isin(metric_names)]

    df_det = _ensure_profile_axis_column(df_det, x_axis=x_axis_fields)
    df_spatial = _ensure_profile_axis_column(df_spatial, x_axis=x_axis_fields)
    df_ens = _ensure_profile_axis_column(df_ens, x_axis=x_axis_fields)
    df_prob = _ensure_profile_axis_column(df_prob, x_axis=x_axis_fields)

    df_det = _apply_df_filters_except_axis(df_det, filters=filters, x_axis=x_axis)
    df_spatial = _apply_df_filters_except_axis(df_spatial, filters=filters, x_axis=x_axis)
    df_ens = _apply_df_filters_except_axis(df_ens, filters=filters, x_axis=x_axis)
    df_prob = _apply_df_filters_except_axis(df_prob, filters=filters, x_axis=x_axis)

    panel_variables = [None] if x_axis_fields == ("variable",) else variables

    for variable in panel_variables:
        profile_variable = variable if variable is not None else "all_variables"
        if progress is not None and task_id is not None:
            progress.update(
                task_id,
                description=f"Generating {profile_variable} metric profiles ({_format_axis_display_name(x_axis)})",
            )
        logger.info(f"Metric profile variable: {variable if variable is not None else 'all_variables'}")
        variable_mask = slice(None) if variable is None else variable

        for metric_name, metric_type in expanded_metric_modes:
            output_folder = _profile_metric_output_folder(
                plot_root=plot_folder,
                variable=variable,
                metric_name=metric_name,
                metric_type=metric_type,
            )
            df_plot_ens = df_ens[df_ens["metric"] == metric_name]
            df_plot_det = df_det[df_det["metric"] == metric_name]
            df_plot_spatial = df_spatial[df_spatial["metric"] == metric_name]
            df_plot_prob = df_prob[df_prob["metric"] == metric_name]

            if variable is not None:
                df_plot_ens = df_plot_ens[df_plot_ens["variable"] == variable_mask]
                df_plot_det = df_plot_det[df_plot_det["variable"] == variable_mask]
                df_plot_spatial = df_plot_spatial[df_plot_spatial["variable"] == variable_mask]
                df_plot_prob = df_plot_prob[df_plot_prob["variable"] == variable_mask]

            df_template = _profile_template_df(
                metric_type,
                df_det=df_plot_det,
                df_spatial=df_plot_spatial,
                df_ens=df_plot_ens,
                df_prob=df_plot_prob,
            )
            if df_template.empty or x_axis_column not in df_template.columns:
                continue

            context_columns = _profile_context_columns(
                df_template,
                x_axis=x_axis,
                shade_by=None if config.profiles.shade_by in x_axis_fields else config.profiles.shade_by,
            )
            grouped_items = (
                list(df_template.groupby(context_columns, dropna=False, sort=True))
                if context_columns
                else [((), df_template)]
            )

            for group_key, _ in grouped_items:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)

                context_filters = {
                    col: value
                    for col, value in zip(context_columns, group_key)
                    if not pd.isna(value)
                }
                context_values = {
                    col: value for col, value in context_filters.items()
                    if str(value) != ""
                }

                df_context_ens = df_plot_ens.copy()
                df_context_det = df_plot_det.copy()
                df_context_spatial = df_plot_spatial.copy()
                df_context_prob = df_plot_prob.copy()
                for col, value in context_filters.items():
                    df_context_ens = df_context_ens[df_context_ens[col] == value]
                    df_context_det = df_context_det[df_context_det[col] == value]
                    df_context_spatial = df_context_spatial[df_context_spatial[col] == value]
                    df_context_prob = df_context_prob[df_context_prob[col] == value]

                df_context_template = _profile_template_df(
                    metric_type,
                    df_det=df_context_det,
                    df_spatial=df_context_spatial,
                    df_ens=df_context_ens,
                    df_prob=df_context_prob,
                )
                x_values = _ordered_profile_axis_values(
                    _profile_axis_source_df(df_context_template, config),
                    x_axis=x_axis,
                    variables=variables,
                    filters=filters,
                )
                if not x_values:
                    continue

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
                diff_fig, diff_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
                rel_fig, rel_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
                x_positions = np.arange(len(x_values))

                if config.profiles.shade_by not in x_axis_fields and config.profiles.shade_by in df_context_template.columns:
                    shade_values = [value for value in pd.unique(df_context_template[config.profiles.shade_by]) if pd.notna(value)]
                else:
                    shade_values = [None]

                model_values = [value for value in pd.unique(df_context_template["model"]) if pd.notna(value)]
                plotted = False
                diff_plotted = False
                rel_plotted = False
                probabilistic_series: dict[tuple[str, object], pd.Series] = {}
                deterministic_series: dict[tuple[str, object], pd.Series] = {}
                deterministic_mean_series: dict[tuple[str, object], pd.Series] = {}
                deterministic_member_matrices: dict[tuple[str, object], pd.DataFrame] = {}
                deterministic_ensemble_series: dict[tuple[str, object], pd.Series] = {}

                for model_name in model_values:
                    base_color = config.models.model_colors.get(str(model_name), None)
                    model_label = config.models.display_names.get(str(model_name), str(model_name))

                    for j, shade_value in enumerate(shade_values):
                        alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))
                        shade_suffix = f", {config.profiles.shade_label}={shade_value}" if shade_value is not None else ""

                        if metric_type == "probabilistic":
                            df_model = df_context_prob[df_context_prob["model"] == model_name]
                            if shade_value is not None:
                                df_model = df_model[df_model[config.profiles.shade_by] == shade_value]
                            y_series = _build_strict_profile_series(
                                df_model,
                                x_axis=x_axis_column,
                                x_values=x_values,
                            )
                            if not pd.notna(y_series).any():
                                continue
                            plotted = True
                            probabilistic_series[(str(model_name), shade_value)] = y_series
                            ax.plot(
                                x_positions,
                                y_series.values,
                                marker="o",
                                linewidth=2.0,
                                alpha=alpha,
                                color=base_color,
                                label=f"{model_label}{shade_suffix}",
                            )
                        elif metric_type in {"all_dims", "spatial_mean", "ensemble_mean"}:
                            df_model_det = _profile_deterministic_mode_df(
                                metric_type,
                                df_det=df_context_det,
                                df_spatial=df_context_spatial,
                                df_ens=df_context_ens,
                            )
                            df_model_det = df_model_det[df_model_det["model"] == model_name]
                            if shade_value is not None:
                                df_model_det = df_model_det[df_model_det[config.profiles.shade_by] == shade_value]

                            y_det, y_members = _build_deterministic_profile_series(
                                df_model_det,
                                x_axis=x_axis_column,
                                x_values=x_values,
                            )
                            if not pd.notna(y_det).any():
                                continue

                            plotted = True
                            deterministic_series[(str(model_name), shade_value)] = y_det
                            if y_members is not None:
                                y_member_values = y_members.to_numpy()
                                ax.fill_between(
                                    x_positions,
                                    np.nanmin(y_member_values, axis=1),
                                    np.nanmax(y_member_values, axis=1),
                                    color=base_color,
                                    alpha=0.16 * alpha,
                                    linewidth=0.0,
                                )
                                for member_idx in range(y_member_values.shape[1]):
                                    ax.plot(
                                        x_positions,
                                        y_member_values[:, member_idx],
                                        color=base_color,
                                        alpha=0.06 * alpha,
                                        linewidth=0.8,
                                        linestyle="--",
                                        marker="o",
                                        markersize=2.5,
                                    )

                            ax.plot(
                                x_positions,
                                y_det.values,
                                marker="o",
                                linewidth=2.0,
                                alpha=alpha,
                                color=base_color,
                                label=f"{model_label} {metric_name}{shade_suffix}",
                            )
                        elif metric_type == "all_dims_with_ensemble_overlay":
                            df_model_ens = df_context_ens[df_context_ens["model"] == model_name]
                            df_model_det = df_context_det[df_context_det["model"] == model_name]
                            if shade_value is not None:
                                df_model_ens = df_model_ens[df_model_ens[config.profiles.shade_by] == shade_value]
                                df_model_det = df_model_det[df_model_det[config.profiles.shade_by] == shade_value]

                            y_ens = _build_strict_profile_series(
                                df_model_ens,
                                x_axis=x_axis_column,
                                x_values=x_values,
                            )
                            y_members = _build_strict_member_matrix(
                                df_model_det,
                                x_axis=x_axis_column,
                                x_values=x_values,
                            )
                            if y_members is None and not pd.notna(y_ens).any():
                                continue

                            if y_members is not None:
                                y_member_values = y_members.to_numpy()
                                if pd.notna(y_member_values).any():
                                    y_mean = np.nanmean(y_member_values, axis=1)
                                    plotted = True
                                    deterministic_mean_series[(str(model_name), shade_value)] = pd.Series(
                                        y_mean,
                                        index=_profile_axis_index(x_values, x_axis=x_axis_column),
                                        dtype=float,
                                    )
                                    deterministic_member_matrices[(str(model_name), shade_value)] = y_members
                                    y_min = np.nanmin(y_member_values, axis=1)
                                    y_max = np.nanmax(y_member_values, axis=1)
                                    ax.fill_between(
                                        x_positions,
                                        y_min,
                                        y_max,
                                        color=base_color,
                                        alpha=0.16 * alpha,
                                        linewidth=0.0,
                                    )
                                    for member_idx in range(y_member_values.shape[1]):
                                        ax.plot(
                                            x_positions,
                                            y_member_values[:, member_idx],
                                            color=base_color,
                                            alpha=0.06 * alpha,
                                            linewidth=0.8,
                                            linestyle="--",
                                            marker="o",
                                            markersize=2.5,
                                        )

                                    ax.plot(
                                        x_positions,
                                        y_mean,
                                        color=base_color,
                                        linewidth=2.0,
                                        alpha=alpha,
                                        marker="o",
                                        markersize=5,
                                        label=f"{model_label} {metric_name}{shade_suffix}",
                                    )

                            if pd.notna(y_ens).any():
                                plotted = True
                                deterministic_ensemble_series[(str(model_name), shade_value)] = y_ens
                                ax.plot(
                                    x_positions,
                                    y_ens.values,
                                    color=base_color,
                                    linewidth=2.0,
                                    alpha=alpha,
                                    linestyle=":",
                                    marker="o",
                                    markersize=5,
                                    label=f"{model_label} {metric_name} ensemble{shade_suffix}",
                                )
                        else:
                            raise ValueError(f"Unsupported metric profile plot mode {metric_type!r}")

                if not plotted:
                    plt.close(fig)
                    plt.close(diff_fig)
                    plt.close(rel_fig)
                    continue

                difference_color = config.models.model_colors.get(
                    config.models.difference_model,
                    config.models.model_colors.get(config.models.corrected_model, "tab:blue"),
                )
                for j, shade_value in enumerate(shade_values):
                    alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))
                    shade_suffix = f", {config.profiles.shade_label}={shade_value}" if shade_value is not None else ""

                    if metric_type == "probabilistic":
                        ref_series = probabilistic_series.get((config.models.reference_model, shade_value))
                        corr_series = probabilistic_series.get((config.models.corrected_model, shade_value))
                        if ref_series is not None and corr_series is not None:
                            diff_series = corr_series - ref_series
                            if pd.notna(diff_series).any():
                                diff_plotted = True
                                diff_ax.plot(
                                    x_positions,
                                    diff_series.values,
                                    marker="o",
                                    linewidth=2.0,
                                    alpha=alpha,
                                    color=difference_color,
                                    label=f"{config.models.difference_model}{shade_suffix}",
                                )

                                rel_series = _safe_relative_change(diff_series, ref_series)
                                if pd.notna(rel_series).any():
                                    rel_plotted = True
                                    rel_ax.plot(
                                        x_positions,
                                        rel_series.values,
                                        marker="o",
                                        linewidth=2.0,
                                        alpha=alpha,
                                        color=difference_color,
                                        label=f"{config.models.difference_model}{shade_suffix}",
                                    )
                    elif metric_type in {"all_dims", "spatial_mean", "ensemble_mean"}:
                        ref_series = deterministic_series.get((config.models.reference_model, shade_value))
                        corr_series = deterministic_series.get((config.models.corrected_model, shade_value))
                        if ref_series is not None and corr_series is not None:
                            diff_series = corr_series - ref_series
                            if pd.notna(diff_series).any():
                                diff_plotted = True
                                diff_ax.plot(
                                    x_positions,
                                    diff_series.values,
                                    marker="o",
                                    linewidth=2.0,
                                    alpha=alpha,
                                    color=difference_color,
                                    label=f"{config.models.difference_model}{shade_suffix}",
                                )

                                rel_series = _safe_relative_change(diff_series, ref_series)
                                if pd.notna(rel_series).any():
                                    rel_plotted = True
                                    rel_ax.plot(
                                        x_positions,
                                        rel_series.values,
                                        marker="o",
                                        linewidth=2.0,
                                        alpha=alpha,
                                        color=difference_color,
                                        label=f"{config.models.difference_model}{shade_suffix}",
                                    )
                    elif metric_type == "all_dims_with_ensemble_overlay":
                        ref_mean = deterministic_mean_series.get((config.models.reference_model, shade_value))
                        corr_mean = deterministic_mean_series.get((config.models.corrected_model, shade_value))
                        ref_members = deterministic_member_matrices.get((config.models.reference_model, shade_value))
                        corr_members = deterministic_member_matrices.get((config.models.corrected_model, shade_value))
                        if ref_mean is not None and corr_mean is not None:
                            diff_mean = corr_mean - ref_mean
                            if pd.notna(diff_mean).any():
                                diff_plotted = True
                                if ref_members is not None and corr_members is not None:
                                    common_members = [
                                        col for col in corr_members.columns
                                        if col in ref_members.columns
                                    ]
                                    if common_members:
                                        diff_members = corr_members[common_members] - ref_members[common_members]
                                        diff_values = diff_members.to_numpy()
                                        if pd.notna(diff_values).any():
                                            diff_ax.fill_between(
                                                x_positions,
                                                np.nanmin(diff_values, axis=1),
                                                np.nanmax(diff_values, axis=1),
                                                color=difference_color,
                                                alpha=0.16 * alpha,
                                                linewidth=0.0,
                                            )
                                diff_ax.plot(
                                    x_positions,
                                    diff_mean.values,
                                    color=difference_color,
                                    linewidth=2.0,
                                    alpha=alpha,
                                    marker="o",
                                    markersize=5,
                                    label=f"{config.models.difference_model}{shade_suffix}",
                                )

                                rel_mean = _safe_relative_change(diff_mean, ref_mean)
                                if pd.notna(rel_mean).any():
                                    rel_plotted = True
                                    if ref_members is not None and corr_members is not None:
                                        common_members = [
                                            col for col in corr_members.columns
                                            if col in ref_members.columns
                                        ]
                                        if common_members:
                                            diff_members = corr_members[common_members] - ref_members[common_members]
                                            rel_members = _safe_relative_change(diff_members, ref_members[common_members])
                                            rel_values = rel_members.to_numpy()
                                            if pd.notna(rel_values).any():
                                                rel_ax.fill_between(
                                                    x_positions,
                                                    np.nanmin(rel_values, axis=1),
                                                    np.nanmax(rel_values, axis=1),
                                                    color=difference_color,
                                                    alpha=0.16 * alpha,
                                                    linewidth=0.0,
                                                )
                                    rel_ax.plot(
                                        x_positions,
                                        rel_mean.values,
                                        color=difference_color,
                                        linewidth=2.0,
                                        alpha=alpha,
                                        marker="o",
                                        markersize=5,
                                        label=f"{config.models.difference_model}{shade_suffix}",
                                    )

                        ref_ens = deterministic_ensemble_series.get((config.models.reference_model, shade_value))
                        corr_ens = deterministic_ensemble_series.get((config.models.corrected_model, shade_value))
                        if ref_ens is not None and corr_ens is not None:
                            diff_ens = corr_ens - ref_ens
                            if pd.notna(diff_ens).any():
                                diff_plotted = True
                                diff_ax.plot(
                                    x_positions,
                                    diff_ens.values,
                                    color=difference_color,
                                    linewidth=2.0,
                                    alpha=alpha,
                                    linestyle=":",
                                    marker="o",
                                    markersize=5,
                                    label=f"{config.models.difference_model} ensemble{shade_suffix}",
                                )

                                rel_ens = _safe_relative_change(diff_ens, ref_ens)
                                if pd.notna(rel_ens).any():
                                    rel_plotted = True
                                    rel_ax.plot(
                                        x_positions,
                                        rel_ens.values,
                                        color=difference_color,
                                        linewidth=2.0,
                                        alpha=alpha,
                                        linestyle=":",
                                        marker="o",
                                        markersize=5,
                                        label=f"{config.models.difference_model} ensemble{shade_suffix}",
                                    )
                    else:
                        raise ValueError(f"Unsupported metric profile plot mode {metric_type!r}")

                axis_label = _format_axis_display_name(x_axis, leadtime_unit=leadtime_unit.strip() if leadtime_unit else None)
                title_subject = (
                    "all variables"
                    if x_axis_fields == ("variable",)
                    else _format_variable_display_name(str(variable))
                )
                metric_label = _format_metric_label(
                    metric_name,
                    base_unit=None if variable is None else (variable_units or {}).get(str(variable)),
                )
                context_subtitle = _plot_context_subtitle(
                    context_values,
                    leadtime_unit=leadtime_unit,
                    filters=filters,
                )

                for current_fig, current_ax, current_title, current_ylabel, show_legend in [
                    (
                        fig,
                        ax,
                        f"{metric_label} vs {_format_axis_display_name(x_axis).lower()} ({title_subject})",
                        metric_label,
                        True,
                    ),
                    (
                        diff_fig,
                        diff_ax,
                        f"Delta {metric_label} vs {_format_axis_display_name(x_axis).lower()} ({title_subject})",
                        f"Delta {metric_label}",
                        diff_plotted,
                    ),
                    (
                        rel_fig,
                        rel_ax,
                        f"Relative change in {metric_label} vs {_format_axis_display_name(x_axis).lower()} ({title_subject})",
                        f"Relative change in {metric_label}",
                        rel_plotted,
                    ),
                ]:
                    _set_title_and_subtitle(
                        current_fig,
                        current_ax,
                        title=current_title,
                        subtitle=context_subtitle,
                    )
                    current_ax.set_xlabel(axis_label)
                    current_ax.set_ylabel(current_ylabel)
                    current_ax.set_xticks(x_positions)
                    current_ax.set_xticklabels(
                        [_format_profile_axis_tick(value, x_axis=x_axis) for value in x_values],
                        rotation=30 if any(field in {"train_period", "loss", "variant", "variable", "region"} for field in x_axis_fields) else 0,
                        ha="right" if any(field in {"train_period", "loss", "variant", "variable", "region"} for field in x_axis_fields) else "center",
                    )
                    current_ax.grid(True, linestyle="--", alpha=0.35)

                    ymin, ymax = current_ax.get_ylim()
                    if ymin <= 0 <= ymax:
                        current_ax.axhline(0.0, color="0.3", lw=1.0, ls=":")

                    if show_legend:
                        current_ax.legend(fontsize=9)
                    current_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

                file_axis = _profile_axis_filename_fragment(x_axis)
                filename = f"{metric_name}_vs_{file_axis}"
                filename_context_suffix = _merge_filename_contexts(
                    _filters_filename_context(filters),
                    context_values,
                )
                if filename_context_suffix:
                    filename = f"{filename}_{filename_context_suffix}"
                plot_path = output_folder / f"{filename}.png"
                fig.savefig(plot_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
                saved_paths.append(plot_path)

                if diff_plotted:
                    diff_path = output_folder / f"{filename}_diff.png"
                    diff_fig.savefig(diff_path, bbox_inches="tight", dpi=150)
                    saved_paths.append(diff_path)
                plt.close(diff_fig)

                if rel_plotted:
                    rel_path = output_folder / f"{filename}_relative_change.png"
                    rel_fig.savefig(rel_path, bbox_inches="tight", dpi=150)
                    saved_paths.append(rel_path)
                plt.close(rel_fig)

        if progress is not None and task_id is not None:
            progress.advance(task_id)

    return saved_paths


def save_combined_variable_metric_profiles(
    *,
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    x_axis: str | Sequence[str],
    filters: dict | None,
    config: CalculateMetricsConfig,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    x_axis_fields = _normalize_profile_axis_fields(x_axis)
    if "variable" in x_axis_fields:
        raise ValueError("Combined variable metric profiles do not support x_axis containing 'variable'.")
    if len(variables) <= 1:
        return []

    saved_paths: list[Path] = []
    variable_colors = _get_variable_colors(variables=variables)
    metric_modes = config.profiles.combined_variable_metrics
    expanded_metric_modes = _expand_profile_metric_modes(metric_modes)
    metric_names = list(dict.fromkeys(metric_name for metric_name, _ in expanded_metric_modes))
    x_axis_column = _profile_axis_column_name(x_axis_fields)

    df_det = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="all_dims",
        diff="no",
    )
    df_spatial = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="spatial_mean",
        diff="no",
    )
    df_ens = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="ensemble_mean",
        diff="no",
    )
    df_prob = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="probabilistic",
        diff="no",
    )

    df_det = df_det[df_det["metric"].isin(metric_names)]
    df_spatial = df_spatial[df_spatial["metric"].isin(metric_names)]
    df_ens = df_ens[df_ens["metric"].isin(metric_names)]
    df_prob = df_prob[df_prob["metric"].isin(metric_names)]

    df_det = _ensure_profile_axis_column(df_det, x_axis=x_axis_fields)
    df_spatial = _ensure_profile_axis_column(df_spatial, x_axis=x_axis_fields)
    df_ens = _ensure_profile_axis_column(df_ens, x_axis=x_axis_fields)
    df_prob = _ensure_profile_axis_column(df_prob, x_axis=x_axis_fields)

    df_det = _apply_df_filters_except_axis(df_det, filters=filters, x_axis=x_axis)
    df_spatial = _apply_df_filters_except_axis(df_spatial, filters=filters, x_axis=x_axis)
    df_ens = _apply_df_filters_except_axis(df_ens, filters=filters, x_axis=x_axis)
    df_prob = _apply_df_filters_except_axis(df_prob, filters=filters, x_axis=x_axis)

    for metric_name, metric_type in expanded_metric_modes:
        if progress is not None and task_id is not None:
            progress.update(task_id, description=f"Generating all_variables combined profile: {metric_name}")
        output_folder = _profile_metric_output_folder(
            plot_root=plot_folder,
            variable=None,
            metric_name=metric_name,
            metric_type=metric_type,
        )

        df_plot_ens = df_ens[df_ens["metric"] == metric_name]
        df_plot_det = df_det[df_det["metric"] == metric_name]
        df_plot_spatial = df_spatial[df_spatial["metric"] == metric_name]
        df_plot_prob = df_prob[df_prob["metric"] == metric_name]

        df_template = _profile_template_df(
            metric_type,
            df_det=df_plot_det,
            df_spatial=df_plot_spatial,
            df_ens=df_plot_ens,
            df_prob=df_plot_prob,
        )
        if df_template.empty or x_axis_column not in df_template.columns:
            continue

        context_columns = _profile_context_columns(
            df_template,
            x_axis=x_axis,
            shade_by=None if config.profiles.shade_by in x_axis_fields else config.profiles.shade_by,
        )
        grouped_items = (
            list(df_template.groupby(context_columns, dropna=False, sort=True))
            if context_columns
            else [((), df_template)]
        )

        for group_key, _ in grouped_items:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            context_filters = {
                col: value
                for col, value in zip(context_columns, group_key)
                if not pd.isna(value)
            }
            context_values = {
                col: value for col, value in context_filters.items()
                if str(value) != ""
            }

            df_context_ens = df_plot_ens.copy()
            df_context_det = df_plot_det.copy()
            df_context_spatial = df_plot_spatial.copy()
            df_context_prob = df_plot_prob.copy()
            for col, value in context_filters.items():
                df_context_ens = df_context_ens[df_context_ens[col] == value]
                df_context_det = df_context_det[df_context_det[col] == value]
                df_context_spatial = df_context_spatial[df_context_spatial[col] == value]
                df_context_prob = df_context_prob[df_context_prob[col] == value]

            df_context_template = _profile_template_df(
                metric_type,
                df_det=df_context_det,
                df_spatial=df_context_spatial,
                df_ens=df_context_ens,
                df_prob=df_context_prob,
            )
            x_values = _ordered_profile_axis_values(
                _profile_axis_source_df(df_context_template, config),
                x_axis=x_axis,
                variables=variables,
                filters=filters,
            )
            if not x_values:
                continue

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
            diff_fig, diff_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
            rel_fig, rel_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
            x_positions = np.arange(len(x_values))

            if config.profiles.shade_by not in x_axis_fields and config.profiles.shade_by in df_context_template.columns:
                shade_values = [value for value in pd.unique(df_context_template[config.profiles.shade_by]) if pd.notna(value)]
            else:
                shade_values = [None]

            plotted = False
            diff_plotted = False
            rel_plotted = False
            plotted_variables: list[str] = []
            plotted_models: list[object] = []
            plotted_shade_values: list[object] = []
            transformed_variables: list[str] = []
            transformed_shade_values: list[object] = []
            probabilistic_series: dict[tuple[str, str, object], pd.Series] = {}
            deterministic_series: dict[tuple[str, str, object], pd.Series] = {}
            deterministic_mean_series: dict[tuple[str, str, object], pd.Series] = {}
            deterministic_member_matrices: dict[tuple[str, str, object], pd.DataFrame] = {}
            deterministic_ensemble_series: dict[tuple[str, str, object], pd.Series] = {}

            for variable in variables:
                variable_color = variable_colors.get(variable)
                df_var_ens = df_context_ens[df_context_ens["variable"] == variable]
                df_var_det = df_context_det[df_context_det["variable"] == variable]
                df_var_spatial = df_context_spatial[df_context_spatial["variable"] == variable]
                df_var_prob = df_context_prob[df_context_prob["variable"] == variable]

                if df_var_ens.empty and df_var_det.empty and df_var_spatial.empty and df_var_prob.empty:
                    continue

                model_values = [
                    value for value in pd.unique(
                        _profile_template_df(
                            metric_type,
                            df_det=df_var_det,
                            df_spatial=df_var_spatial,
                            df_ens=df_var_ens,
                            df_prob=df_var_prob,
                        )["model"]
                    ) if pd.notna(value)
                ]

                for model_name in model_values:
                    for j, shade_value in enumerate(shade_values):
                        alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))

                        if metric_type == "probabilistic":
                            df_model = df_var_prob[df_var_prob["model"] == model_name]
                            if shade_value is not None:
                                df_model = df_model[df_model[config.profiles.shade_by] == shade_value]
                            y_series = _build_strict_profile_series(
                                df_model,
                                x_axis=x_axis_column,
                                x_values=x_values,
                            )
                            if not pd.notna(y_series).any():
                                continue
                            plotted = True
                            probabilistic_series[(str(variable), str(model_name), shade_value)] = y_series
                            if variable not in plotted_variables:
                                plotted_variables.append(variable)
                            if model_name not in plotted_models:
                                plotted_models.append(model_name)
                            if shade_value is not None and shade_value not in plotted_shade_values:
                                plotted_shade_values.append(shade_value)
                            ax.plot(
                                x_positions,
                                y_series.values,
                                marker="o",
                                linewidth=2.0,
                                alpha=alpha,
                                color=variable_color,
                                linestyle=_model_linestyle(str(model_name), config, probabilistic=True),
                                label="_nolegend_",
                            )
                        elif metric_type in {"all_dims", "spatial_mean", "ensemble_mean"}:
                            df_model_det = _profile_deterministic_mode_df(
                                metric_type,
                                df_det=df_var_det,
                                df_spatial=df_var_spatial,
                                df_ens=df_var_ens,
                            )
                            df_model_det = df_model_det[df_model_det["model"] == model_name]
                            if shade_value is not None:
                                df_model_det = df_model_det[df_model_det[config.profiles.shade_by] == shade_value]

                            y_det, y_members = _build_deterministic_profile_series(
                                df_model_det,
                                x_axis=x_axis_column,
                                x_values=x_values,
                            )
                            if not pd.notna(y_det).any():
                                continue

                            plotted = True
                            deterministic_series[(str(variable), str(model_name), shade_value)] = y_det
                            if variable not in plotted_variables:
                                plotted_variables.append(variable)
                            if model_name not in plotted_models:
                                plotted_models.append(model_name)
                            if shade_value is not None and shade_value not in plotted_shade_values:
                                plotted_shade_values.append(shade_value)
                            if y_members is not None:
                                y_member_values = y_members.to_numpy()
                                ax.fill_between(
                                    x_positions,
                                    np.nanmin(y_member_values, axis=1),
                                    np.nanmax(y_member_values, axis=1),
                                    color=variable_color,
                                    alpha=0.16 * alpha,
                                    linewidth=0.0,
                                )
                                for member_idx in range(y_member_values.shape[1]):
                                    ax.plot(
                                        x_positions,
                                        y_member_values[:, member_idx],
                                        color=variable_color,
                                        alpha=0.06 * alpha,
                                        linewidth=0.8,
                                        linestyle="--",
                                        marker="o",
                                        markersize=2.5,
                                        label="_nolegend_",
                                    )
                            ax.plot(
                                x_positions,
                                y_det.values,
                                color=variable_color,
                                linewidth=2.0,
                                alpha=alpha,
                                linestyle=_model_linestyle(str(model_name), config),
                                marker="o",
                                markersize=5,
                                label="_nolegend_",
                            )
                        elif metric_type == "all_dims_with_ensemble_overlay":
                            df_model_ens = df_var_ens[df_var_ens["model"] == model_name]
                            df_model_det = df_var_det[df_var_det["model"] == model_name]
                            if shade_value is not None:
                                df_model_ens = df_model_ens[df_model_ens[config.profiles.shade_by] == shade_value]
                                df_model_det = df_model_det[df_model_det[config.profiles.shade_by] == shade_value]

                            y_ens = _build_strict_profile_series(
                                df_model_ens,
                                x_axis=x_axis_column,
                                x_values=x_values,
                            )
                            y_members = _build_strict_member_matrix(
                                df_model_det,
                                x_axis=x_axis_column,
                                x_values=x_values,
                            )
                            if y_members is None and not pd.notna(y_ens).any():
                                continue

                            if y_members is not None:
                                y_member_values = y_members.to_numpy()
                                if pd.notna(y_member_values).any():
                                    y_mean = np.nanmean(y_member_values, axis=1)
                                    plotted = True
                                    deterministic_mean_series[(str(variable), str(model_name), shade_value)] = pd.Series(
                                        y_mean,
                                        index=_profile_axis_index(x_values, x_axis=x_axis_column),
                                        dtype=float,
                                    )
                                    deterministic_member_matrices[(str(variable), str(model_name), shade_value)] = y_members
                                    if variable not in plotted_variables:
                                        plotted_variables.append(variable)
                                    if model_name not in plotted_models:
                                        plotted_models.append(model_name)
                                    if shade_value is not None and shade_value not in plotted_shade_values:
                                        plotted_shade_values.append(shade_value)
                                    ax.fill_between(
                                        x_positions,
                                        np.nanmin(y_member_values, axis=1),
                                        np.nanmax(y_member_values, axis=1),
                                        color=variable_color,
                                        alpha=0.16 * alpha,
                                        linewidth=0.0,
                                    )
                                    for member_idx in range(y_member_values.shape[1]):
                                        ax.plot(
                                            x_positions,
                                            y_member_values[:, member_idx],
                                            color=variable_color,
                                            alpha=0.06 * alpha,
                                            linewidth=0.8,
                                            linestyle="--",
                                            marker="o",
                                            markersize=2.5,
                                            label="_nolegend_",
                                        )
                                    ax.plot(
                                        x_positions,
                                        y_mean,
                                        color=variable_color,
                                        linewidth=2.0,
                                        alpha=alpha,
                                        linestyle=_model_linestyle(str(model_name), config),
                                        marker="o",
                                        markersize=5,
                                        label="_nolegend_",
                                    )

                            if pd.notna(y_ens).any():
                                plotted = True
                                deterministic_ensemble_series[(str(variable), str(model_name), shade_value)] = y_ens
                                if variable not in plotted_variables:
                                    plotted_variables.append(variable)
                                if model_name not in plotted_models:
                                    plotted_models.append(model_name)
                                if shade_value is not None and shade_value not in plotted_shade_values:
                                    plotted_shade_values.append(shade_value)
                                ax.plot(
                                    x_positions,
                                    y_ens.values,
                                    color=variable_color,
                                    linewidth=2.0,
                                    alpha=alpha,
                                    linestyle=":",
                                    marker="o",
                                    markersize=5,
                                    label="_nolegend_",
                                )
                        else:
                            raise ValueError(f"Unsupported combined variable profile mode {metric_type!r}")

            if not plotted:
                plt.close(fig)
                plt.close(diff_fig)
                plt.close(rel_fig)
                continue

            for variable in variables:
                variable_color = variable_colors.get(variable)
                if variable_color is None:
                    continue

                for j, shade_value in enumerate(shade_values):
                    alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))

                    if metric_type == "probabilistic":
                        ref_series = probabilistic_series.get((str(variable), config.models.reference_model, shade_value))
                        corr_series = probabilistic_series.get((str(variable), config.models.corrected_model, shade_value))
                        if ref_series is not None and corr_series is not None:
                            diff_series = corr_series - ref_series
                            if pd.notna(diff_series).any():
                                diff_plotted = True
                                if variable not in transformed_variables:
                                    transformed_variables.append(variable)
                                if shade_value is not None and shade_value not in transformed_shade_values:
                                    transformed_shade_values.append(shade_value)
                                diff_ax.plot(
                                    x_positions,
                                    diff_series.values,
                                    marker="o",
                                    linewidth=2.0,
                                    alpha=alpha,
                                    color=variable_color,
                                    linestyle="-",
                                    label="_nolegend_",
                                )

                                rel_series = _safe_relative_change(diff_series, ref_series)
                                if pd.notna(rel_series).any():
                                    rel_plotted = True
                                    rel_ax.plot(
                                        x_positions,
                                        rel_series.values,
                                        marker="o",
                                        linewidth=2.0,
                                        alpha=alpha,
                                        color=variable_color,
                                        linestyle="-",
                                        label="_nolegend_",
                                    )
                    elif metric_type in {"all_dims", "spatial_mean", "ensemble_mean"}:
                        ref_series = deterministic_series.get((str(variable), config.models.reference_model, shade_value))
                        corr_series = deterministic_series.get((str(variable), config.models.corrected_model, shade_value))
                        if ref_series is not None and corr_series is not None:
                            diff_series = corr_series - ref_series
                            if pd.notna(diff_series).any():
                                diff_plotted = True
                                if variable not in transformed_variables:
                                    transformed_variables.append(variable)
                                if shade_value is not None and shade_value not in transformed_shade_values:
                                    transformed_shade_values.append(shade_value)
                                diff_ax.plot(
                                    x_positions,
                                    diff_series.values,
                                    marker="o",
                                    linewidth=2.0,
                                    alpha=alpha,
                                    color=variable_color,
                                    linestyle="-",
                                    label="_nolegend_",
                                )

                                rel_series = _safe_relative_change(diff_series, ref_series)
                                if pd.notna(rel_series).any():
                                    rel_plotted = True
                                    rel_ax.plot(
                                        x_positions,
                                        rel_series.values,
                                        marker="o",
                                        linewidth=2.0,
                                        alpha=alpha,
                                        color=variable_color,
                                        linestyle="-",
                                        label="_nolegend_",
                                    )
                    elif metric_type == "all_dims_with_ensemble_overlay":
                        ref_mean = deterministic_mean_series.get((str(variable), config.models.reference_model, shade_value))
                        corr_mean = deterministic_mean_series.get((str(variable), config.models.corrected_model, shade_value))
                        ref_members = deterministic_member_matrices.get((str(variable), config.models.reference_model, shade_value))
                        corr_members = deterministic_member_matrices.get((str(variable), config.models.corrected_model, shade_value))
                        if ref_mean is not None and corr_mean is not None:
                            diff_mean = corr_mean - ref_mean
                            if pd.notna(diff_mean).any():
                                diff_plotted = True
                                if variable not in transformed_variables:
                                    transformed_variables.append(variable)
                                if shade_value is not None and shade_value not in transformed_shade_values:
                                    transformed_shade_values.append(shade_value)
                                if ref_members is not None and corr_members is not None:
                                    common_members = [
                                        col for col in corr_members.columns
                                        if col in ref_members.columns
                                    ]
                                    if common_members:
                                        diff_members = corr_members[common_members] - ref_members[common_members]
                                        diff_values = diff_members.to_numpy()
                                        if pd.notna(diff_values).any():
                                            diff_ax.fill_between(
                                                x_positions,
                                                np.nanmin(diff_values, axis=1),
                                                np.nanmax(diff_values, axis=1),
                                                color=variable_color,
                                                alpha=0.16 * alpha,
                                                linewidth=0.0,
                                            )
                                diff_ax.plot(
                                    x_positions,
                                    diff_mean.values,
                                    color=variable_color,
                                    linewidth=2.0,
                                    alpha=alpha,
                                    linestyle="-",
                                    marker="o",
                                    markersize=5,
                                    label="_nolegend_",
                                )

                                rel_mean = _safe_relative_change(diff_mean, ref_mean)
                                if pd.notna(rel_mean).any():
                                    rel_plotted = True
                                    if ref_members is not None and corr_members is not None:
                                        common_members = [
                                            col for col in corr_members.columns
                                            if col in ref_members.columns
                                        ]
                                        if common_members:
                                            diff_members = corr_members[common_members] - ref_members[common_members]
                                            rel_members = _safe_relative_change(diff_members, ref_members[common_members])
                                            rel_values = rel_members.to_numpy()
                                            if pd.notna(rel_values).any():
                                                rel_ax.fill_between(
                                                    x_positions,
                                                    np.nanmin(rel_values, axis=1),
                                                    np.nanmax(rel_values, axis=1),
                                                    color=variable_color,
                                                    alpha=0.16 * alpha,
                                                    linewidth=0.0,
                                                )
                                    rel_ax.plot(
                                        x_positions,
                                        rel_mean.values,
                                        color=variable_color,
                                        linewidth=2.0,
                                        alpha=alpha,
                                        linestyle="-",
                                        marker="o",
                                        markersize=5,
                                        label="_nolegend_",
                                    )

                        ref_ens = deterministic_ensemble_series.get((str(variable), config.models.reference_model, shade_value))
                        corr_ens = deterministic_ensemble_series.get((str(variable), config.models.corrected_model, shade_value))
                        if ref_ens is not None and corr_ens is not None:
                            diff_ens = corr_ens - ref_ens
                            if pd.notna(diff_ens).any():
                                diff_plotted = True
                                if variable not in transformed_variables:
                                    transformed_variables.append(variable)
                                if shade_value is not None and shade_value not in transformed_shade_values:
                                    transformed_shade_values.append(shade_value)
                                diff_ax.plot(
                                    x_positions,
                                    diff_ens.values,
                                    color=variable_color,
                                    linewidth=2.0,
                                    alpha=alpha,
                                    linestyle=":",
                                    marker="o",
                                    markersize=5,
                                    label="_nolegend_",
                                )

                                rel_ens = _safe_relative_change(diff_ens, ref_ens)
                                if pd.notna(rel_ens).any():
                                    rel_plotted = True
                                    rel_ax.plot(
                                        x_positions,
                                        rel_ens.values,
                                        color=variable_color,
                                        linewidth=2.0,
                                        alpha=alpha,
                                        linestyle=":",
                                        marker="o",
                                        markersize=5,
                                        label="_nolegend_",
                                    )
                    else:
                        raise ValueError(f"Unsupported combined variable profile mode {metric_type!r}")

            axis_label = _format_axis_display_name(x_axis, leadtime_unit=leadtime_unit.strip() if leadtime_unit else None)
            metric_label = _format_metric_label(metric_name, base_unit=None)
            context_subtitle = _plot_context_subtitle(
                context_values,
                leadtime_unit=leadtime_unit,
                filters=filters,
            )

            for current_fig, current_ax, current_title, current_ylabel in [
                (
                    fig,
                    ax,
                    f"{metric_label} vs {_format_axis_display_name(x_axis).lower()} (all variables)",
                    metric_label,
                ),
                (
                    diff_fig,
                    diff_ax,
                    f"Delta {metric_label} vs {_format_axis_display_name(x_axis).lower()} (all variables)",
                    f"Delta {metric_label}",
                ),
                (
                    rel_fig,
                    rel_ax,
                    f"Relative change in {metric_label} vs {_format_axis_display_name(x_axis).lower()} (all variables)",
                    f"Relative change in {metric_label}",
                ),
            ]:
                _set_title_and_subtitle(
                    current_fig,
                    current_ax,
                    title=current_title,
                    subtitle=context_subtitle,
                )
                current_ax.set_xlabel(axis_label)
                current_ax.set_ylabel(current_ylabel)
                current_ax.set_xticks(x_positions)
                current_ax.set_xticklabels(
                    [_format_profile_axis_tick(value, x_axis=x_axis) for value in x_values],
                    rotation=30 if any(field in {"train_period", "loss", "variant", "variable", "region"} for field in x_axis_fields) else 0,
                    ha="right" if any(field in {"train_period", "loss", "variant", "variable", "region"} for field in x_axis_fields) else "center",
                )
                current_ax.grid(True, linestyle="--", alpha=0.35)

                ymin, ymax = current_ax.get_ylim()
                if ymin <= 0 <= ymax:
                    current_ax.axhline(0.0, color="0.3", lw=1.0, ls=":")

            _set_combined_variable_profile_legend(
                ax,
                variables=plotted_variables,
                variable_colors=variable_colors,
                model_values=plotted_models,
                metric_type=metric_type,
                shade_values=plotted_shade_values,
                config=config,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

            if diff_plotted:
                _set_transformed_combined_variable_profile_legend(
                    diff_ax,
                    variables=transformed_variables,
                    variable_colors=variable_colors,
                    metric_type=metric_type,
                    shade_values=transformed_shade_values,
                    config=config,
                )
                diff_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

            if rel_plotted:
                _set_transformed_combined_variable_profile_legend(
                    rel_ax,
                    variables=transformed_variables,
                    variable_colors=variable_colors,
                    metric_type=metric_type,
                    shade_values=transformed_shade_values,
                    config=config,
                )
                rel_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

            filename = f"all_variables_{metric_name}_vs_{_profile_axis_filename_fragment(x_axis)}"
            filename_context_suffix = _merge_filename_contexts(
                _filters_filename_context(filters),
                context_values,
            )
            if filename_context_suffix:
                filename = f"{filename}_{filename_context_suffix}"
            plot_path = output_folder / f"{filename}.png"
            fig.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved_paths.append(plot_path)

            if diff_plotted:
                diff_path = output_folder / f"{filename}_diff.png"
                diff_fig.savefig(diff_path, bbox_inches="tight", dpi=150)
                saved_paths.append(diff_path)
            plt.close(diff_fig)

            if rel_plotted:
                rel_path = output_folder / f"{filename}_relative_change.png"
                rel_fig.savefig(rel_path, bbox_inches="tight", dpi=150)
                saved_paths.append(rel_path)
            plt.close(rel_fig)

    if progress is not None and task_id is not None:
        progress.advance(task_id)

    return saved_paths
