from pathlib import Path

import numpy as np
import pandas as pd

from .dataclasses import CalculateMetricsConfig
from .constants import METRIC_DISPLAY_NAMES
from .utils import (
    _format_axis_display_name,
    _format_variable_display_name,
    _format_variable_short_name,
)

from .. import metrics_to_df
from ...plots import plot_scoreboard


def _build_relative_abs_improvement_df(
    *,
    metrics,
    variables: list[str],
    metric_name: str,
    metric_type: str,
    models: tuple[str, str],
) -> pd.DataFrame:
    raw_df = metrics_to_df(
        metrics=metrics,
        variables=variables,
        metric_names=metric_name,
        kind="scalar",
        metric_type=metric_type,
        diff="no",
        models=models,
    )
    if raw_df.empty:
        return raw_df

    model_a, model_b = models
    id_cols = [col for col in raw_df.columns if col not in {"model", "value"}]
    pivot = raw_df.pivot_table(index=id_cols, columns="model", values="value", aggfunc="first")

    if model_a not in pivot.columns or model_b not in pivot.columns:
        return pd.DataFrame(columns=raw_df.columns)

    baseline = pivot[model_a].abs()
    corrected = pivot[model_b].abs()
    baseline_safe = baseline.where(~np.isclose(baseline, 0.0), np.nan)
    relative_improvement = (baseline - corrected) / baseline_safe

    result = relative_improvement.rename("value").reset_index()
    result["model"] = f"{model_b}-{model_a}"

    ordered_cols = [col for col in raw_df.columns if col in result.columns]
    extra_cols = [col for col in result.columns if col not in ordered_cols]
    return result[ordered_cols + extra_cols]


def save_scoreboard_plot(
    *,
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    config: CalculateMetricsConfig,
    leadtime_unit: str,
    filters: dict,
    filename_context_suffix: str | None = None,
) -> Path:
    scoreboard_frames: list[pd.DataFrame] = []
    diff_mode = "delta" if config.scoreboards.mode == "relative" else "no"
    outer_x_values = (
        [config.models.difference_model]
        if config.scoreboards.mode == "relative"
        else list(config.models.comparison_models)
    )

    for metric_name, metric_type in config.scoreboards.metrics.items():
        if config.scoreboards.mode == "relative" and metric_name in {"bias", "nbias"}:
            scoreboard_frames.append(
                _build_relative_abs_improvement_df(
                    metrics=metrics,
                    variables=variables,
                    metric_name=metric_name,
                    metric_type=metric_type,
                    models=config.models.comparison_models,
                )
            )
        else:
            scoreboard_frames.append(
                metrics_to_df(
                    metrics=metrics,
                    variables=variables,
                    metric_names=metric_name,
                    kind="scalar",
                    metric_type=metric_type,
                    diff=diff_mode,
                    models=config.models.comparison_models if diff_mode == "delta" else None,
                )
            )

    df_scoreboard = pd.concat(scoreboard_frames, ignore_index=True)

    scoreboard_plot_filters = {}
    for col, val in filters.items():
        if val is None:
            continue
        if col in {config.scoreboards.row_axis, config.scoreboards.col_axis, "metric", "model"}:
            continue
        scoreboard_plot_filters[col] = val

    filename = f"scoreboard_{config.scoreboards.mode}_{config.scoreboards.row_axis}_vs_{config.scoreboards.col_axis}"
    if filename_context_suffix:
        filename = f"{filename}_{filename_context_suffix}"
    plot_path = plot_folder / f"{filename}.png"
    plot_scoreboard(
        df=df_scoreboard,
        outer_y={"index": "metric", "values": list(config.scoreboards.metrics.keys())},
        outer_x={"index": "model", "values": outer_x_values},
        inner_y={
            "index": config.scoreboards.row_axis,
            "label": _format_axis_display_name(config.scoreboards.row_axis, leadtime_unit=leadtime_unit),
        },
        inner_x={
            "index": config.scoreboards.col_axis,
            "label": _format_axis_display_name(config.scoreboards.col_axis, leadtime_unit=leadtime_unit),
        },
        filters=scoreboard_plot_filters,
        agg="mean",
        metric_cmaps=config.scoreboards.metric_cmaps,
        metric_vlims=config.scoreboards.metric_vlims,
        metric_names=METRIC_DISPLAY_NAMES,
        display_name_maps={
            "variable": _format_variable_display_name,
            "model": config.models.display_names,
            "metric": METRIC_DISPLAY_NAMES,
        },
        inner_y_display_name_map=(
            _format_variable_short_name
            if config.scoreboards.row_axis == "variable"
            else None
        ),
        annotate=config.scoreboards.annotate,
        annotate_fmt="{:.2f}",
        annotate_color="auto",
        save_path=plot_path,
        title=f"Scoreboard ({config.scoreboards.mode})",
    )
    return plot_path
