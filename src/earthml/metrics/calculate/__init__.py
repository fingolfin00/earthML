from typing import TYPE_CHECKING, Any

from .dataclasses import CalculateMetricsConfig

__all__ = [
    "CalculateMetricsConfig",
    "CalculateMetricsRuntime",
    "save_field_and_metric_map_plots",
    "save_metric_vs_deltametric_plot",
    "save_combined_variable_metric_profiles",
    "save_metrics_vs_parameter_plots",
    "save_scoreboard_plot",
    "save_scalar_metric_tables",
    "save_field_timeseries_plots",
]


if TYPE_CHECKING:
    from .maps import save_field_and_metric_map_plots
    from .metric_vs_delta import save_metric_vs_deltametric_plot
    from .profiles import (
        save_combined_variable_metric_profiles,
        save_metrics_vs_parameter_plots,
    )
    from .runtime import CalculateMetricsRuntime
    from .scoreboard import save_scoreboard_plot
    from .tables import save_scalar_metric_tables
    from .timeseries import save_field_timeseries_plots


def __getattr__(name: str) -> Any:
    if name == "CalculateMetricsRuntime":
        from .runtime import CalculateMetricsRuntime

        return CalculateMetricsRuntime
    if name == "save_field_and_metric_map_plots":
        from .maps import save_field_and_metric_map_plots

        return save_field_and_metric_map_plots
    if name == "save_metric_vs_deltametric_plot":
        from .metric_vs_delta import save_metric_vs_deltametric_plot

        return save_metric_vs_deltametric_plot
    if name == "save_combined_variable_metric_profiles":
        from .profiles import save_combined_variable_metric_profiles

        return save_combined_variable_metric_profiles
    if name == "save_metrics_vs_parameter_plots":
        from .profiles import save_metrics_vs_parameter_plots

        return save_metrics_vs_parameter_plots
    if name == "save_scoreboard_plot":
        from .scoreboard import save_scoreboard_plot

        return save_scoreboard_plot
    if name == "save_scalar_metric_tables":
        from .tables import save_scalar_metric_tables

        return save_scalar_metric_tables
    if name == "save_field_timeseries_plots":
        from .timeseries import save_field_timeseries_plots

        return save_field_timeseries_plots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
