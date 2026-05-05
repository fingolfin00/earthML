from typing import Literal
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .dataclasses import CalculateMetricsConfig, CalculateMetricsMetricVsDeltaConfig
from .utils import (
    _ordered_unique_variables,
    _format_metric_label,
    _get_variable_colors,
)

from ...plots import plot_metric_vs_diff


def _metric_vs_deltametric_specs(
    knobs: CalculateMetricsMetricVsDeltaConfig
) -> list[dict[str, str]]:
    metric_pairs = knobs.get("metric_pairs")
    if metric_pairs is None:
        return [
            {
                "forecast_metric": str(knobs["forecast_metric"]),
                "forecast_metric_type": str(knobs["forecast_metric_type"]),
                "delta_metric": str(knobs["delta_metric"]),
                "delta_metric_type": str(knobs["delta_metric_type"]),
            }
        ]

    if not isinstance(metric_pairs, (list, tuple)) or not metric_pairs:
        raise ValueError("CalculateMetricsMetricVsDeltaConfig.metric_pairs must be a non-empty list when provided.")

    specs: list[dict[str, str]] = []
    required_keys = (
        "forecast_metric",
        "forecast_metric_type",
        "delta_metric",
        "delta_metric_type",
    )
    for idx, pair in enumerate(metric_pairs):
        if not isinstance(pair, dict):
            raise ValueError(
                "Each CalculateMetricsMetricVsDeltaConfig.metric_pairs item must be a dict "
                f"with keys {required_keys!r}. Problem at index {idx}: {pair!r}"
            )
        missing = [key for key in required_keys if key not in pair or pair[key] is None or str(pair[key]) == ""]
        if missing:
            raise ValueError(
                "Each CalculateMetricsMetricVsDeltaConfig.metric_pairs item must define "
                f"{required_keys!r}. Missing {missing!r} at index {idx}."
            )
        specs.append({key: str(pair[key]) for key in required_keys})
    return specs


def save_metric_vs_deltametric_plot(
    *,
    df_nodiff,
    df_delta,
    plot_folder: Path,
    forecast_metric: str,
    delta_metric: str,
    leadtime_unit: str,
    region: str,
    filters: dict | None,
    variable_unit: str | None,
    metric_vs_delta_byconfig: dict[Literal["color", "marker", "shade"], str],
    point_size: int, 
    config: CalculateMetricsConfig,
    filename_context_suffix: str | None = None,
) -> Path:
    plot_folder.mkdir(parents=True, exist_ok=True)
    plot_variables = _ordered_unique_variables(
        [
            *df_nodiff.get("variable", pd.Series(dtype=object)).dropna().astype(str).tolist(),
            *df_delta.get("variable", pd.Series(dtype=object)).dropna().astype(str).tolist(),
        ]
    )
    forecast_metric_label = _format_metric_label(forecast_metric, base_unit=variable_unit)
    delta_metric_label = _format_metric_label(delta_metric, base_unit=variable_unit)
    region_suffix = f" - {region}" if region else ""

    fig, ax, merged = plot_metric_vs_diff(
        df_nodiff,
        df_delta,
        forecast_metric=forecast_metric,
        diff_metric=delta_metric,
        x_metric_name=f"delta_{delta_metric}",
        y_metric_name=forecast_metric,
        color_by=metric_vs_delta_byconfig["color"],
        marker_by=metric_vs_delta_byconfig["marker"],
        shade_by=metric_vs_delta_byconfig["shade"],
        shade_label=config.metric_vs_delta.shade_by,
        fit_lines=False,
        fit_ci=False,
        leadtime_unit=leadtime_unit,
        point_size=point_size,
        variable_colors=_get_variable_colors(
            variables=plot_variables,
            base_colors=config.variable_colors.base_colors,
            overrides=config.variable_colors.overrides,
        ),
        filters=filters,
        title=f"Forecast {forecast_metric_label} vs Δ {delta_metric_label} ({config.models.difference_model}){region_suffix}",
        xlabel=f"Δ {delta_metric_label} ({config.models.difference_model})",
        ylabel=f"Forecast {forecast_metric_label}",
        legend_loc="lower left",
    )

    if merged.empty:
        available_y = sorted(df_nodiff["metric"].dropna().astype(str).unique())
        available_x = sorted(df_delta["metric"].dropna().astype(str).unique())
        raise ValueError(
            "Plot dataframe is empty after merging. "
            f"Requested forecast_metric={forecast_metric!r}, delta_metric={delta_metric!r}. "
            f"Available no-diff metrics={available_y}. "
            f"Available delta metrics={available_x}."
        )

    filename = f"{forecast_metric}_vs_deltametric_{delta_metric}"
    if filename_context_suffix:
        filename = f"{filename}_{filename_context_suffix}"
    plot_path = plot_folder / f"{filename}.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return plot_path
