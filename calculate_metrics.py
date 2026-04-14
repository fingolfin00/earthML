from typing import Sequence
import logging
import multiprocessing
import os
import socket
from contextlib import nullcontext
from itertools import product
from pathlib import Path
import warnings
import joblib

# Silence Intel OpenMP deprecation/info chatter emitted by downstream libraries.
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import numpy as np
import pandas as pd
import xarray as xr
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.style import Style
from rich.table import Table as RichTable
from rich.text import Text

from earthml import Dask, get_runs_and_metrics
from earthml.experiments.mlbc.load import get_leadtime_value_and_unit
from earthml.metrics.utils import metrics_to_df
from earthml.misc import Table
from earthml.plots import (
    plot_metric_vs_diff,
    plot_realization_timeseries,
    plot_scoreboard,
    plot_temporal_mean_map,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Dask.*")
warnings.filterwarnings("ignore", message=".*distributed.scheduler.*")
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
    module=r"dask\.array\.numpy_compat",
)
logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
logging.getLogger("distributed.worker").setLevel(logging.ERROR)
logging.getLogger("distributed.nanny").setLevel(logging.ERROR)

DEBUG = False
CONSOLE = Console()


#
# User knobs
# Edit these blocks first. They are grouped by output so users can quickly
# switch products on/off or retarget a plot without digging through the code.
#

GLOBAL_KNOBS = {
    # Main experiment folder to read from.
    "exp_root_folder": Path("/Users/jacopodallaglio/ML/experiments_earthML_seasonal_atmo/"),
    # "exp_root_folder": Path("/Users/jacopodallaglio/ML/experiments_earthML_seasonal_ocean/"),
    # All artifacts land in exp_root_folder / plot_folder_name.
    "plot_folder_name": "plots",
    # Dataset split and model loading.
    "type_data": "test",
    # Keep None to load every scalar metric available.
    "metric_names": None,
    # Global output switches.
    "enable_field_timeseries": True,
    "enable_maps": True,
    "enable_scalar_tables": True,
    "enable_metric_vs_deltametric_plot": True,
    "enable_metric_profile_plots": True,
    "enable_scoreboard": True,
    # Print the user config summary table at startup.
    "print_user_config": False,
    # Toggle rich table output in the terminal while still saving table files/images.
    "print_console_tables": True,
}

BASE_FILTERS = {
    # Default analysis slice shared across timeseries, maps, tables and profiles unless overridden below.
    # For scoreboard and metric-vs-delta-metric plots, use dedicated filters below
    "leadtime": [1, 2, 3, 4, 5, 6],
    "variable": ["t2m"], # None, ["t2m"]
    "region": ["ConUS", "Europe"], # ConUS, Europe
    # "leadtime": [6],
    # "variable": ["sst"],
    # "region": ["CentralPacific"], # CentralPacific, NorthAtlantic
    "train_period": None,
    "loss": None,
}

FIELD_TIMESERIES_KNOBS = {
    # Per-output overrides on top of BASE_FILTERS.
    "filters": {
        "variable": None,
        "leadtime": None,
        "region": None,
        "train_period": None,
        "loss": None,
    },
    # When True, plot all available leadtimes in the same figure.
    "combine_leadtimes": True,
    # Visual encoding for leadtime when combine_leadtimes is enabled.
    # "shade" uses lighter/darker shades of the model color.
    # "linestyle" uses different line styles while keeping the model color fixed.
    "leadtime_style": "linestyle",
    # When True, draw individual realization members in field-timeseries plots.
    "plot_realization_members": True,
    # Disable flox/numbagg in field-timeseries plotting. Useful on some clusters
    # where lazy reductions on float16-backed arrays fail.
    "disable_flox_for_field_timeseries": False,
}

MAP_PLOT_KNOBS = {
    # Per-output overrides on top of BASE_FILTERS.
    "filters": {
        "variable": None,
        "leadtime": None,
        "region": None,
        "train_period": None,
        "loss": None,
    },
    # Colormap for raw field temporal-mean maps.
    "field_cmap": "jet",
    # "mean" averages realizations before plotting. "members" saves one map per realization.
    "realization_mode": "mean",
    # Per-metric colormap overrides for metric maps.
    "metric_cmaps": {
        "bias": "RdBu_r",
        "nbias": "RdBu_r",
        "r2": "RdBu",
        "rmse_skill_clim": "RdBu",
        "spread_error_ratio": "RdBu",
    },
    # Fallback colormap when a metric is not listed above.
    "metric_cmap_default": "jet",
    # Gridline label spacing in degrees for map plots.
    "lon_tick_step": 10.0,
    "lat_tick_step": 10.0,
}

METRIC_PROFILE_PLOT_KNOBS = {
    # Per-output overrides on top of BASE_FILTERS.
    "filters": {
        "variable": ["t2m", "msl", "d2m", "u10", "v10", "msl"],
        "leadtime": None,
        "region": None,
        "train_period": None,
        "loss": None,
    },
    # Pick the x-axis used for metric profile plots.
    "x_axis": "leadtime",  # "leadtime", "train_period", "loss", "variable"
    # Each metric maps to the plotting mode expected by save_metrics_vs_parameter_plots.
    "metrics": {
        "r2": "deterministic_with_ensemble_overlay",
        "rmse": "deterministic_with_ensemble_overlay",
        # "nrmse": "deterministic_with_ensemble_overlay",
        "rmse_skill_clim": "deterministic_with_ensemble_overlay",
        "mae": "deterministic_with_ensemble_overlay",
        # "nmae": "deterministic_with_ensemble_overlay",
        "bias": "deterministic_with_ensemble_overlay",
        "clim_acc": "deterministic_with_ensemble_overlay",
        # "nbias": "deterministic_with_ensemble_overlay",
        "crps": "probabilistic",
        "spread": "probabilistic",
        "spread_error_ratio": "probabilistic",
    },
    "shade_by": "loss",  # e.g. "loss", "total_months"
    "shade_label": "Loss",
    # Optional cross-variable profile family saved under all_variables/profiles.
    "enable_combined_variable_profiles": True,
    # Normalized metrics to compare across variables with variable encoded by color.
    "combined_variable_metrics": {
        "nbias": "deterministic_with_ensemble_overlay",
        "nrmse": "deterministic_with_ensemble_overlay",
        "nmae": "deterministic_with_ensemble_overlay",
        "r2": "deterministic_with_ensemble_overlay",
        "clim_acc": "deterministic_with_ensemble_overlay",
    },
}

VARIABLE_COLOR_KNOBS = {
    # Single source of truth for variable colors across tables and profile plots.
    "base_colors": (
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#1f78b4",
    ),
    # Optional fixed per-variable overrides.
    "overrides": {},
}

TABLE_KNOBS = {
    # Per-variable raw scalar tables overrides on top of BASE_FILTERS.
    "scalar_filters": {
        "train_period": None,
        "loss": None,
        "leadtime": None,
        "variable": None,
        "region": None,
    },
    # Combined normalized tables overrides on top of BASE_FILTERS.
    "normalized_filters": {
        "train_period": None,
        "loss": None,
        "leadtime": None,
        "variable": None,
        "region": None,
    },
    "row_index": ("metric",),
    "column_index": ("model", "variable", "stat"),
    "stat_order": ("avg", "spread", "ens", "prob"),
    "significant_digits": 3,
    "image_dpi": 300,
    "background_color": "#ffffff",
}


METRIC_VS_DELTAMETRIC_PLOT_KNOBS = {
    "filters": {
        "variable": None,
        "leadtime": [1, 2, 3, 4, 5, 6],
        "region": ["ConUS", "Europe"], # ConUS, Europe
        # "region": ["CentralPacific"], # CentralPacific, NorthAtlantic
        "train_period": None,
        "loss": None,
    },
    # x-axis: how much the ML correction changes the chosen metric.
    "delta_metric": "nrmse",
    "delta_metric_type": "ensemble",
    # y-axis: the forecast quality to compare against.
    "forecast_metric": "r2",
    "forecast_metric_type": "ensemble",
    # Visual encodings for the scatter plot.
    "color_by": "variable",
    "marker_by": "region",
    "shade_by": "leadtime",  # e.g. "loss", "total_months"
    "shade_label": "Leadtime",
    "point_size": 30,
}

SCOREBOARD_KNOBS = {
    # These act only when the dimension is not already on an axis.
    "filters": {
        "variable": None,
        "loss": "mseloss",
        "train_period": None,
        "leadtime": [1, 2, 3, 4, 5, 6],
        "region": ["ConUS"], # ConUS, Europe
        # "region": ["CentralPacific"], # CentralPacific, NorthAtlantic
    },
    "mode": "relative",  # "absolute" or "relative"
    "metrics": {
        "r2": "ensemble",
        "nrmse": "ensemble",
        "ncrmse": "ensemble",
        "nmae": "ensemble",
        "nbias": "ensemble",
        # "crps": "probabilistic",
        "rmse_skill_clim": "ensemble",
        "clim_acc": "ensemble",
        "spatial_acc": "ensemble",
        "spread_error_ratio": "probabilistic",
    },
    "metric_cmaps": {
        "r2": "RdBu",
        "nrmse": "RdBu_r",
        "ncrmse": "RdBu_r",
        "nmae": "RdBu_r",
        "nbias": "RdBu_r",
        "rmse_skill_clim": "RdBu",
        "clim_acc": "RdBu",
        "spatial_acc": "RdBu",
        "spread_error_ratio": "RdBu",
    },
    "metric_vlims": {
        # "r2": (-0.4, 0.4),
        # "nrmse": (-0.2, 0.2),
        # "crps": (-0.2, 0.2),
    },
    # Pick which metadata dimensions span the scoreboard grid.
    "row_axis": "variable",  # e.g. "train_period", "loss", "leadtime", "variable"
    "col_axis": "leadtime",
    "annotate": True,
}


MODEL_KNOBS = {
    # Raw model ids as they appear in the loaded runs/metrics.
    "truth_model": "an",
    "reference_model": "fc",
    "corrected_model": "pr",
    # Label used for improvement columns in scalar tables.
    "gain_model": "gain",
    # User-facing names for legends, titles, tables and scoreboards.
    "display_names": {
        "fc": "FC",
        "pr": "MLFC",
        "an": "AN",
        "gain": "Gain",
    },
    # Consistent model colors across plots.
    "model_colors": {
        "fc": "tab:green",
        "pr": "tab:blue",
        "pr-fc": "tab:red",
        "an": "tab:orange",
    },
}

# Derived internal aliases
PLOT_FOLDER = GLOBAL_KNOBS["exp_root_folder"] / GLOBAL_KNOBS["plot_folder_name"]
LOAD_MODELS = (
    MODEL_KNOBS["truth_model"],
    MODEL_KNOBS["reference_model"],
    MODEL_KNOBS["corrected_model"],
)
COMPARISON_MODELS = (
    MODEL_KNOBS["reference_model"],
    MODEL_KNOBS["corrected_model"],
)
PLOT_MODELS = (
    MODEL_KNOBS["reference_model"],
    MODEL_KNOBS["corrected_model"],
)
DIFFERENCE_MODEL = f"{MODEL_KNOBS['corrected_model']}-{MODEL_KNOBS['reference_model']}"
TABLE_MODEL_ORDER = (
    MODEL_KNOBS["reference_model"],
    MODEL_KNOBS["corrected_model"],
    MODEL_KNOBS["gain_model"],
)
MODEL_COLORS = MODEL_KNOBS["model_colors"]
MODEL_DISPLAY_NAMES = {
    **MODEL_KNOBS["display_names"],
    DIFFERENCE_MODEL: (
        f"{MODEL_KNOBS['display_names'][MODEL_KNOBS['corrected_model']]} - "
        f"{MODEL_KNOBS['display_names'][MODEL_KNOBS['reference_model']]}"
    ),
}
NORMALIZED_METRICS = {"nrmse", "ncrmse", "nmae", "nbias", "r2"}
VARIABLE_SUBFOLDERS = ("timeseries", "maps", "profiles", "tables")
OUTPUT_GROUP_SUBFOLDERS = ("field", "deterministic", "ensemble") # ensemble containes also probabilistic metrics
RUN_CONTEXT_DIMS = ("leadtime", "train_period", "loss", "region")
TABLE_CONTEXT_FIELDS = ("train_period", "loss", "leadtime", "region")


METRIC_DISPLAY_NAMES = {
    "r2": "R2",
    "rmse": "RMSE",
    "rmse_skill_clim": "RMSE vs clim Skill",
    "crmse": "CRMSE",
    "mae": "MAE",
    "bias": "Bias",
    "error_std": "Error Std",
    "variance_ratio": "Variance Ratio",
    "nrmse": "nRMSE",
    "ncrmse": "nCRMSE",
    "nmae": "nMAE",
    "nbias": "nBias",
    "crps": "CRPS",
    "spread": "Spread",
    "spread_error_ratio": "Spread/Error",
}

VARIABLE_DISPLAY_NAMES = {
    "t2m": "2m temperature",
    "d2m": "2m dewpoint temperature",
    "msl": "MSL pressure",
    "mslp": "MSL pressure",
    "u10": "10m zonal wind",
    "v10": "10m meridional wind",
    "tp": "Precipitation",
    "tcc": "Cloud cover",
    "sst": "Sea surface temperature",
    "ssh": "Sea surface height",
    "sss": "Sea surface salinity",
}

AXIS_DISPLAY_NAMES = {
    "variable": "",
    "leadtime": "Lead time",
    "loss": "Loss",
    "train_period": "Train period",
    "region": "Region",
    "model": "Model",
    "metric": "Metric",
}

def debug_print(*args, **kwargs) -> None:
    if DEBUG:
        CONSOLE.print(*args, **kwargs)


def _resolved_single_region_label(
    data: xr.Dataset | xr.DataArray,
    *,
    filters: dict | None = None,
) -> str:
    if "region" in data.coords or "region" in data.dims:
        values = np.asarray(data["region"].values).reshape(-1).tolist()
        region_values = sorted({str(value) for value in values if value is not None and str(value) != ""})
        if len(region_values) > 1:
            raise ValueError(f"Expected a single region after filtering, got {region_values}")
        if len(region_values) == 1:
            return region_values[0]
    return _single_filter_value(filters, "region")


def build_scalar_metric_df(
    *,
    metrics,
    variables: list[str],
    metric_name: str | None,
    metric_type: str,
    diff: str,
    models: tuple[str, str] | None = None,
) -> pd.DataFrame:
    metric_names = (metric_name,) if metric_name is not None else GLOBAL_KNOBS["metric_names"]
    return metrics_to_df(
        metrics=metrics,
        variables=variables,
        metric_names=metric_names,
        kind="scalar",
        metric_type=metric_type,
        diff=diff,
        models=models,
    )


def _build_relative_abs_improvement_df(
    *,
    metrics,
    variables: list[str],
    metric_name: str,
    metric_type: str,
    models: tuple[str, str],
) -> pd.DataFrame:
    raw_df = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=metric_name,
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


def get_variable_plot_folder(*, plot_root: Path, variable: str) -> Path:
    variable_folder = plot_root / variable
    variable_folder.mkdir(parents=True, exist_ok=True)
    return variable_folder


def get_variable_subfolder(*, plot_root: Path, variable: str, subfolder: str) -> Path:
    if subfolder not in VARIABLE_SUBFOLDERS:
        raise ValueError(f"Unsupported variable subfolder {subfolder!r}. Expected one of {VARIABLE_SUBFOLDERS}.")
    output_folder = get_variable_plot_folder(plot_root=plot_root, variable=variable) / subfolder
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def get_variable_output_group_folder(
    *,
    plot_root: Path,
    variable: str,
    subfolder: str,
    output_group: str,
) -> Path:
    if output_group not in OUTPUT_GROUP_SUBFOLDERS:
        raise ValueError(
            f"Unsupported output group {output_group!r}. Expected one of {OUTPUT_GROUP_SUBFOLDERS}."
        )
    output_folder = get_variable_subfolder(
        plot_root=plot_root,
        variable=variable,
        subfolder=subfolder,
    ) / output_group
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def get_variable_output_item_folder(
    *,
    plot_root: Path,
    variable: str,
    subfolder: str,
    output_group: str,
    item_name: str,
) -> Path:
    output_folder = get_variable_output_group_folder(
        plot_root=plot_root,
        variable=variable,
        subfolder=subfolder,
        output_group=output_group,
    ) / item_name
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


def _ensure_grouped_output_folders(*, plot_root: Path, variable: str, subfolder: str) -> dict[str, Path]:
    return {
        output_group: get_variable_output_group_folder(
            plot_root=plot_root,
            variable=variable,
            subfolder=subfolder,
            output_group=output_group,
        )
        for output_group in OUTPUT_GROUP_SUBFOLDERS
    }


def _ordered_unique_variables(variables: list[str]) -> list[str]:
    unique_variables: list[str] = []
    for variable in variables:
        variable_str = str(variable)
        if variable_str not in unique_variables:
            unique_variables.append(variable_str)
    return unique_variables


def _filter_variables_by_filters(
    variables: list[str],
    filters: dict | None,
) -> list[str]:
    ordered_variables = _ordered_unique_variables(variables)
    if not filters or filters.get("variable") is None:
        return ordered_variables

    allowed = filters["variable"]
    allowed_values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
    allowed_set = {str(value) for value in allowed_values if value is not None and str(value) != ""}
    return [variable for variable in ordered_variables if str(variable) in allowed_set]


def _merge_selection_values(existing: object | None, new_value: object | None) -> object | None:
    if new_value is None:
        return None
    new_values = new_value if isinstance(new_value, (list, tuple, set)) else [new_value]
    normalized_new = [value for value in new_values if value is not None and str(value) != ""]
    if not normalized_new:
        return existing
    if existing is None:
        return list(dict.fromkeys(normalized_new))

    existing_values = existing if isinstance(existing, (list, tuple, set)) else [existing]
    merged = [value for value in existing_values if value is not None and str(value) != ""]
    for value in normalized_new:
        if str(value) not in {str(item) for item in merged}:
            merged.append(value)
    return merged


def _build_load_selection(*filter_sets: dict | None) -> dict[str, object]:
    selection: dict[str, object] = {}
    tracked_keys = ("variable", "leadtime", "train_period", "loss", "region")

    for key in tracked_keys:
        merged_value: object | None = []
        saw_filters = False

        for filters in filter_sets:
            if not filters:
                continue
            saw_filters = True
            merged_value = _merge_selection_values(merged_value, filters.get(key))
            if merged_value is None:
                break

        if saw_filters and merged_value is not None:
            values = merged_value if isinstance(merged_value, (list, tuple, set)) else [merged_value]
            cleaned = [value for value in values if value is not None and str(value) != ""]
            if cleaned:
                selection[key] = cleaned

    return selection


def get_variable_colors(
    *,
    variables: list[str],
    base_colors: Sequence[str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    ordered_variables = _ordered_unique_variables(variables)
    palette = tuple(base_colors or VARIABLE_COLOR_KNOBS["base_colors"])
    if not palette:
        return {}

    color_map = {
        variable: palette[idx % len(palette)]
        for idx, variable in enumerate(ordered_variables)
    }
    merged_overrides = VARIABLE_COLOR_KNOBS["overrides"].copy()
    if overrides:
        merged_overrides.update(overrides)
    if merged_overrides:
        for variable, color in merged_overrides.items():
            if variable in color_map:
                color_map[variable] = color
    return color_map


def get_variable_table_color(*, variable: str, variables: list[str]) -> str:
    variable_colors = get_variable_colors(variables=variables)
    if not variable_colors:
        return VARIABLE_COLOR_KNOBS["base_colors"][0]
    return variable_colors.get(str(variable), VARIABLE_COLOR_KNOBS["base_colors"][0])


def _table_variables(df: pd.DataFrame) -> list[str]:
    variables: list[str] = []

    if isinstance(df.index, pd.MultiIndex) and "variable" in df.index.names:
        for value in df.index.get_level_values("variable"):
            variable = str(value)
            if variable not in variables:
                variables.append(variable)

    if isinstance(df.columns, pd.MultiIndex) and "variable" in df.columns.names:
        variable_idx = list(df.columns.names).index("variable")
        for col in df.columns:
            variable = str(col[variable_idx])
            if variable not in variables:
                variables.append(variable)

    return variables


def _variable_from_row(df: pd.DataFrame, row_pos: int) -> str | None:
    if not isinstance(df.index, pd.MultiIndex) or "variable" not in df.index.names:
        return None
    variable_idx = list(df.index.names).index("variable")
    idx_value = df.index[row_pos]
    return str(idx_value[variable_idx])


def _variable_from_column(df: pd.DataFrame, col_pos: int) -> str | None:
    if not isinstance(df.columns, pd.MultiIndex) or "variable" not in df.columns.names:
        return None
    variable_idx = list(df.columns.names).index("variable")
    col_value = df.columns[col_pos]
    return str(col_value[variable_idx])


def _apply_df_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if df.empty or not filters:
        return df

    out = df.copy()
    for col, allowed in filters.items():
        if allowed is None or col not in out.columns:
            continue
        values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        out = out[out[col].isin(values)]
    return out


def _apply_df_filters_except_axis(
    df: pd.DataFrame,
    *,
    filters: dict | None,
    x_axis: str,
) -> pd.DataFrame:
    if not filters:
        return df
    effective_filters = {
        col: allowed
        for col, allowed in filters.items()
        if col != x_axis
    }
    return _apply_df_filters(df, effective_filters)


def _apply_xarray_filters(data: xr.Dataset | xr.DataArray, filters: dict | None) -> xr.Dataset | xr.DataArray:
    if filters is None:
        return data

    out = data
    for dim, allowed in filters.items():
        if allowed is None or (dim not in out.dims and dim not in out.coords):
            continue
        values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        out = out.sel({dim: list(values)})
    return out


def _single_filter_value(filters: dict | None, key: str) -> str:
    if not filters or key not in filters or filters[key] is None:
        return ""
    allowed = filters[key]
    values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
    normalized = [str(value) for value in values if value is not None and str(value) != ""]
    if len(normalized) == 1:
        return normalized[0]
    return ""


def _scalar_coord_value(data: xr.Dataset | xr.DataArray, dim: str) -> object | None:
    if dim not in data.coords and dim not in data.dims:
        return None
    values = np.asarray(data[dim].values)
    if values.size != 1:
        return None
    value = values.reshape(-1)[0]
    return value.item() if isinstance(value, np.generic) else value


def _reduce_extra_map_dims(da: xr.DataArray) -> xr.DataArray:
    lat_dim = da.earthml.guessed_dims.latitude
    lon_dim = da.earthml.guessed_dims.longitude
    time_dim = da.earthml.guessed_dims.time
    realization_dim = da.earthml.guessed_dims.realization
    keep_dims = {dim for dim in (lat_dim, lon_dim, time_dim, realization_dim) if dim is not None}
    reduce_dims = [dim for dim in da.dims if dim not in keep_dims]
    if reduce_dims:
        da = da.mean(dim=reduce_dims, skipna=True)
    return da


def _materialize_map_array(da: xr.DataArray) -> xr.DataArray:
    return da.load()


def _collapse_realization_for_map(da: xr.DataArray) -> xr.DataArray:
    realization_dim = _realization_dim(da)
    if realization_dim is not None and realization_dim in da.dims:
        da = da.mean(dim=realization_dim, skipna=True)
    return da


def _map_realization_slices(
    da: xr.DataArray,
    *,
    realization_mode: str,
) -> list[tuple[object | None, xr.DataArray]]:
    realization_dim = _realization_dim(da)
    if realization_mode == "members" and realization_dim is not None and da.sizes.get(realization_dim, 1) > 1:
        return [
            (
                value.item() if isinstance(value, np.generic) else value,
                _materialize_map_array(da.sel({realization_dim: value}, drop=True)),
            )
            for value in da[realization_dim].values.tolist()
        ]
    return [(None, _materialize_map_array(_collapse_realization_for_map(da)))]


def _realization_label(realization_value: object | None) -> str:
    return "" if realization_value is None else f" | realization={realization_value}"


def _realization_filename_fragment(realization_value: object | None) -> str:
    if realization_value is None:
        return ""
    return f"_realization_{_sanitize_filename_fragment(str(realization_value))}"


def _realization_dim(da: xr.DataArray) -> str | None:
    realization_dim = da.earthml.guessed_dims.realization
    if realization_dim is None or realization_dim not in da.dims:
        return None
    return realization_dim


def _has_multi_realization(da: xr.DataArray | None) -> bool:
    if da is None:
        return False
    realization_dim = _realization_dim(da)
    return realization_dim is not None and int(da.sizes.get(realization_dim, 1)) > 1


def _members_arg(da: xr.DataArray, *, enable_spread: bool) -> xr.DataArray | None:
    return da if enable_spread and _has_multi_realization(da) else None


def _leadtime_values(da: xr.DataArray) -> list[object]:
    if "leadtime" not in da.dims:
        return []
    return list(da["leadtime"].values.tolist())


def _leadtime_shade_weight_map(values: list[object]) -> dict[object, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {values[0]: 0.0}
    return {
        value: 0.55 * (1.0 - idx / (len(values) - 1))
        for idx, value in enumerate(values)
    }


def _leadtime_linestyle_map(values: list[object]) -> dict[object, str]:
    linestyles = [
        "-",
        "--",
        ":",
        "-.",
        (0, (5, 1)),
        (0, (3, 1, 1, 1)),
        (0, (1, 1)),
    ]
    if not values:
        return {}
    return {
        value: linestyles[idx % len(linestyles)]
        for idx, value in enumerate(values)
    }


def _shade_model_color(color: str | None, *, weight: float) -> str | None:
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


def _leadtime_legend_label(value: object, *, leadtime_unit: str) -> str:
    unit = leadtime_unit.strip()
    return f"{value} {unit}" if unit else str(value)


def _set_combined_leadtime_legend(
    ax: Axes,
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
            color=MODEL_COLORS.get(model_name),
            linewidth=2.0,
            alpha=1.0,
            label=MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
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


def _set_combined_variable_profile_legend(
    ax: Axes,
    *,
    variables: list[str],
    variable_colors: dict[str, str],
    model_values: list[object],
    metric_type: str,
    shade_values: list[object],
    shade_by: str,
    shade_label: str,
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
                    label=format_variable_display_name(str(variable)),
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
                    linestyle=_model_linestyle(model_name, probabilistic=(metric_type == "probabilistic")),
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    label=MODEL_DISPLAY_NAMES.get(model_name, model_name),
                )
                for model_name in visible_models
            ]
        )
        if metric_type == "deterministic_with_ensemble_overlay":
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
        _section(shade_label)
        shade_count = max(1, len(shade_values))
        for idx, shade_value in enumerate(shade_values):
            alpha = 0.35 + 0.55 * (idx + 1) / shade_count
            label = f"{shade_value}M" if shade_by == "total_months" else str(shade_value)
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
            if label in {"Variable", "Series", shade_label}:
                text.set_fontweight("bold")


def _set_transformed_combined_variable_profile_legend(
    ax: Axes,
    *,
    variables: list[str],
    variable_colors: dict[str, str],
    metric_type: str,
    shade_values: list[object],
    shade_by: str,
    shade_label: str,
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
                    label=format_variable_display_name(str(variable)),
                )
                for variable in visible_variables
            ]
        )

    _section("Series")
    handles.append(
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS.get(DIFFERENCE_MODEL, "0.25"),
            linestyle="-",
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=MODEL_DISPLAY_NAMES.get(DIFFERENCE_MODEL, DIFFERENCE_MODEL),
        )
    )
    if metric_type == "deterministic_with_ensemble_overlay":
        handles.append(
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS.get(DIFFERENCE_MODEL, "0.25"),
                linestyle=":",
                linewidth=2.0,
                marker="o",
                markersize=5,
                label="Ensemble mean",
            )
        )

    if len(shade_values) > 1:
        _section(shade_label)
        shade_count = max(1, len(shade_values))
        for idx, shade_value in enumerate(shade_values):
            alpha = 0.35 + 0.55 * (idx + 1) / shade_count
            label = f"{shade_value}M" if shade_by == "total_months" else str(shade_value)
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
            if label in {"Variable", "Series", shade_label}:
                text.set_fontweight("bold")


def _metric_output_group(metric_type: str) -> str:
    return "deterministic" if metric_type == "deterministic" else "ensemble"


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


def _build_residual_da(left: xr.DataArray, right: xr.DataArray) -> xr.DataArray:
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


def _map_title(*parts: str | None) -> str:
    return " | ".join(part for part in parts if part)


def _map_cbar_label(*, variable: str, unit: str | None) -> str:
    return _format_label_with_unit(format_variable_display_name(variable), unit)


def _shared_map_limits(*arrays: xr.DataArray) -> tuple[float | None, float | None]:
    mins: list[float] = []
    maxs: list[float] = []
    for array in arrays:
        values = np.asarray(array.values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        mins.append(float(finite.min()))
        maxs.append(float(finite.max()))
    if not mins or not maxs:
        return None, None
    vmin = min(mins)
    vmax = max(maxs)
    if vmin == vmax:
        eps = abs(vmin) * 0.05 or 1e-6
        return vmin - eps, vmax + eps
    return vmin, vmax


def _symmetric_map_limits(array: xr.DataArray) -> tuple[float | None, float | None]:
    values = np.asarray(array.values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    bound = float(np.max(np.abs(finite)))
    if bound == 0.0:
        bound = 1e-6
    return -bound, bound


def _context_selection_entries(
    data: xr.Dataset | xr.DataArray,
    *,
    context_dims: tuple[str, ...] = RUN_CONTEXT_DIMS,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    dims_present = [dim for dim in context_dims if dim in data.dims]
    if not dims_present:
        return [({}, {})]

    varying_dims = [dim for dim in dims_present if data.sizes.get(dim, 1) > 1]
    if not varying_dims:
        labels = {
            dim: _scalar_coord_value(data, dim)
            for dim in dims_present
            if _scalar_coord_value(data, dim) is not None
        }
        return [({}, labels)]

    entries: list[tuple[dict[str, object], dict[str, object]]] = []
    for values in product(*[data[dim].values.tolist() for dim in varying_dims]):
        selection = {dim: value for dim, value in zip(varying_dims, values)}
        labels = {}
        for dim in dims_present:
            if dim in selection:
                labels[dim] = selection[dim]
            else:
                scalar_value = _scalar_coord_value(data, dim)
                if scalar_value is not None:
                    labels[dim] = scalar_value
        entries.append((selection, labels))
    return entries


def _format_scalar_value(value: object, significant_digits: int) -> str:
    if value is None or pd.isna(value):
        return "-"
    value_float = float(value)
    if np.isinf(value_float):
        return "inf" if value_float > 0 else "-inf"
    if significant_digits < 1:
        raise ValueError("significant_digits must be at least 1")
    if value_float == 0:
        return "0" if significant_digits == 1 else f"0.{('0' * (significant_digits - 1))}"

    abs_value = abs(value_float)
    exponent = int(np.floor(np.log10(abs_value)))
    if exponent >= significant_digits or exponent < -3:
        return f"{value_float:.{significant_digits - 1}e}"

    decimals = significant_digits - exponent - 1
    return f"{value_float:.{max(decimals, 0)}f}"


def _flatten_label(parts: tuple[object, ...] | object) -> str:
    if not isinstance(parts, tuple):
        parts = (parts,)
    cleaned = [str(part) for part in parts if pd.notna(part) and str(part) != ""]
    return " | ".join(cleaned) if cleaned else "-"


def _sanitize_filename_fragment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "all"


def _ordered_profile_axis_values(
    df: pd.DataFrame,
    *,
    x_axis: str,
    variables: list[str],
) -> list[object]:
    if x_axis not in df.columns:
        return []

    values = [value for value in pd.unique(df[x_axis]) if pd.notna(value)]
    if x_axis == "leadtime":
        return sorted(values)
    if x_axis == "variable":
        ordered_variables = _ordered_unique_variables(variables)
        return [value for value in ordered_variables if value in values]
    return values


def _format_profile_axis_tick(value: object, *, x_axis: str) -> str:
    if x_axis == "variable":
        return format_variable_display_name(str(value))
    return str(value)


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
    x_axis: str,
    x_values: list[object],
) -> pd.Series:
    if df.empty:
        return pd.Series(index=pd.Index(x_values, name=x_axis), dtype=float)

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
    x_axis: str,
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


def _safe_relative_change(
    values: pd.Series | pd.DataFrame,
    baseline: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    baseline_safe = baseline.where(~np.isclose(baseline, 0.0), np.nan)
    result = values / baseline_safe
    return result.where(np.isfinite(result), np.nan)


def _profile_output_folder(
    *,
    plot_root: Path,
    variable: str | None = None,
) -> Path:
    folder_variable = variable if variable is not None else "all_variables"
    return get_variable_subfolder(plot_root=plot_root, variable=folder_variable, subfolder="profiles")


def _table_output_folder(
    *,
    plot_root: Path,
    variable: str | None = None,
) -> Path:
    folder_variable = variable if variable is not None else "all_variables"
    return get_variable_subfolder(plot_root=plot_root, variable=folder_variable, subfolder="tables")


def _prettify_stat_label(stat: str) -> str:
    pretty = {
        "avg": "avg",
        "spread": "spread",
        "ens": "ens",
        "gain": "gain",
        "prob": "prob",
    }
    return pretty.get(stat, stat.replace("_", " "))


def _mix_colors(color_a: str, color_b: str, weight: float) -> str:
    rgb_a = np.array(mcolors.to_rgb(color_a))
    rgb_b = np.array(mcolors.to_rgb(color_b))
    mixed = (1 - weight) * rgb_a + weight * rgb_b
    return mcolors.to_hex(mixed)


def format_metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " "))

def format_variable_display_name(variable: str) -> str:
    return VARIABLE_DISPLAY_NAMES.get(variable, str(variable))


def format_variable_short_name(variable: str) -> str:
    return str(variable)

def _format_label_with_unit(label: str, unit: str | None) -> str:
    return f"{label} [{unit}]" if unit else label


def _get_da_unit(data: xr.Dataset | xr.DataArray, variable: str | None = None) -> str | None:
    try:
        if isinstance(data, xr.Dataset):
            if variable is not None and variable in data.data_vars:
                da = data[variable]
            elif len(data.data_vars) == 1:
                da = data[next(iter(data.data_vars))]
            else:
                return None
        else:
            da = data
        unit = da.attrs.get("units") or da.attrs.get("unit")
        return str(unit) if unit else None
    except Exception:
        return None


def _get_variable_units_from_runs(runs: dict[str, xr.Dataset], variables: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for variable in variables:
        unit = None
        for ds in runs.values():
            if variable in ds.data_vars:
                unit = _get_da_unit(ds, variable)
                if unit:
                    break
        out[str(variable)] = unit
    return out


def _metric_unit(metric_name: str, base_unit: str | None) -> str | None:
    if metric_name in {"rmse", "crmse", "mae", "bias", "error_std", "crps", "spread"}:
        return base_unit
    return None


def format_metric_label(metric_name: str, *, base_unit: str | None = None) -> str:
    return _format_label_with_unit(format_metric_display_name(metric_name), _metric_unit(metric_name, base_unit))


def _resolve_metric_map_cmap(metric_name: str) -> str:
    return MAP_PLOT_KNOBS["metric_cmaps"].get(metric_name, MAP_PLOT_KNOBS["metric_cmap_default"])


def format_axis_display_name(axis_name: str, *, leadtime_unit: str | None = None) -> str:
    if axis_name == "leadtime" and leadtime_unit:
        return f"Lead time [{leadtime_unit}]"
    return AXIS_DISPLAY_NAMES.get(axis_name, axis_name.replace("_", " ").title())


def _format_context_label(*, field: str, value: object, leadtime_unit: str | None = None) -> str:
    field_name = format_axis_display_name(field, leadtime_unit=leadtime_unit)
    if field == "leadtime" and leadtime_unit:
        return f"{field_name}={value} {leadtime_unit}"
    return f"{field_name}={value}"


def _context_title_suffix(context_values: dict[str, object], *, leadtime_unit: str | None = None) -> str:
    if not context_values:
        return ""
    return " | ".join(
        _format_context_label(field=field, value=value, leadtime_unit=leadtime_unit)
        for field, value in context_values.items()
    )


def _plot_context_subtitle(
    context_values: dict[str, object] | None = None,
    *,
    leadtime_unit: str | None = None,
    filters: dict | None = None,
) -> str:
    subtitle_values: dict[str, object] = {}
    selected_region = _single_filter_value(filters, "region")
    if selected_region:
        subtitle_values["region"] = selected_region
    if context_values:
        subtitle_values.update(context_values)
    return _context_title_suffix(subtitle_values, leadtime_unit=leadtime_unit)


def _set_title_and_subtitle(
    fig: Figure,
    ax: Axes,
    *,
    title: str,
    subtitle: str | None = None,
) -> None:
    fig.suptitle(title)
    ax.set_title(subtitle or "", fontsize=10, color="0.35", pad=8)


def _context_filename_suffix(context_values: dict[str, object]) -> str:
    if not context_values:
        return "all"
    return "__".join(
        f"{field}_{_sanitize_filename_fragment(str(value))}"
        for field, value in context_values.items()
    )


def _filters_filename_context(filters: dict | None) -> dict[str, object]:
    if not filters:
        return {}

    context: dict[str, object] = {}
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = [item for item in value if item is not None and str(item) != ""]
            if not values:
                continue
            context[key] = values[0] if len(values) == 1 else "-".join(map(str, values))
            continue
        context[key] = value
    return context


def _merge_filters(base_filters: dict | None, override_filters: dict | None) -> dict[str, object]:
    merged: dict[str, object] = {}
    for filters in (base_filters, override_filters):
        if not filters:
            continue
        for key, value in filters.items():
            if value is None:
                continue
            merged[key] = value
    return merged


def _merge_filename_contexts(*contexts: dict[str, object]) -> str:
    merged: dict[str, object] = {}
    for context in contexts:
        for key, value in context.items():
            if value is None or str(value) == "":
                continue
            merged[key] = value
    return "" if not merged else _context_filename_suffix(merged)


def _ordered_context_filename_suffix(
    context_values: dict[str, object],
    *,
    priority_fields: tuple[str, ...] = ("train_period", "loss", "leadtime"),
) -> str:
    if not context_values:
        return "all"

    ordered_items: list[tuple[str, object]] = [
        (field, context_values[field]) for field in priority_fields if field in context_values
    ]
    ordered_items.extend(
        (field, value)
        for field, value in context_values.items()
        if field not in priority_fields
    )
    return _context_filename_suffix(dict(ordered_items))


def _profile_context_columns(
    df: pd.DataFrame,
    *,
    x_axis: str,
    shade_by: str | None = None,
) -> list[str]:
    return [
        col for col in RUN_CONTEXT_DIMS
        if (
            col in df.columns
            and col != x_axis
            and col != shade_by
            and df[col].nunique(dropna=False) > 1
        )
    ]


def _model_linestyle(model_name: str, *, probabilistic: bool = False) -> str:
    if model_name == MODEL_KNOBS["truth_model"]:
        return "-"
    if model_name == MODEL_KNOBS["reference_model"]:
        return ":" if probabilistic else "--"
    if model_name == MODEL_KNOBS["corrected_model"]:
        return "-." if probabilistic else "-"
    return "-"


def print_user_config_table() -> None:
    if not GLOBAL_KNOBS["print_user_config"]:
        return

    config_sections = [
        ("debug", {"value": DEBUG}),
        ("models", MODEL_KNOBS),
        ("global", GLOBAL_KNOBS),
        ("base_filters", BASE_FILTERS),
        ("field_timeseries", FIELD_TIMESERIES_KNOBS),
        ("map_plot", MAP_PLOT_KNOBS),
        ("metric_vs_deltametric_plot", METRIC_VS_DELTAMETRIC_PLOT_KNOBS),
        ("metric_profile_plot", METRIC_PROFILE_PLOT_KNOBS),
        ("tables", TABLE_KNOBS),
        ("scoreboard", SCOREBOARD_KNOBS),
    ]

    chunk_size = 2
    for chunk_idx in range(0, len(config_sections), chunk_size):
        chunk = dict(config_sections[chunk_idx:chunk_idx + chunk_size])
        title_suffix = f" ({chunk_idx // chunk_size + 1})"
        CONSOLE.print(Table(chunk, title=f"Calculate Metrics Config{title_suffix}").table)


def build_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
        auto_refresh=True,
        refresh_per_second=6,
        transient=False,
    )


def _metric_name_from_index(df: pd.DataFrame, row_pos: int) -> str | None:
    idx_value = df.index[row_pos]
    return _metric_name_from_index_value(idx_value, index_names=list(df.index.names))


def _metric_name_from_index_value(idx_value: object, *, index_names: list[str | None]) -> str | None:
    if "metric" in index_names:
        metric_idx = index_names.index("metric")
        if isinstance(idx_value, tuple):
            return str(idx_value[metric_idx])
        return str(idx_value)
    if isinstance(idx_value, tuple):
        return None
    return str(idx_value)


def _metric_higher_is_better(metric: str) -> bool | None:
    mapping = {
        "corr": True,
        "clim_acc": True,
        "spatial_acc": True,
        "r2": True,
        "rmse_skill_clim": True,
        "rmse": False,
        "crmse": False,
        "mae": False,
        "bias": False,
        "error_std": False,
        "nrmse": False,
        "ncrmse": False,
        "nmae": False,
        "nbias": False,
        "crps": False,
        "spread": None,
        "variance_ratio": None,
        "spread_error_ratio": None,
    }
    return mapping.get(metric)


def _metric_gain_value(metric_name: str | None, reference_value: float, corrected_value: float) -> float:
    if metric_name is None or pd.isna(reference_value) or pd.isna(corrected_value):
        return np.nan
    if metric_name in {"bias", "nbias"}:
        return abs(reference_value) - abs(corrected_value)
    if metric_name == "variance_ratio":
        return abs(reference_value - 1.0) - abs(corrected_value - 1.0)
    if metric_name == "spread_error_ratio":
        return abs(reference_value - 1.0) - abs(corrected_value - 1.0)

    direction = _metric_higher_is_better(metric_name)
    if direction is True:
        return corrected_value - reference_value
    if direction is False:
        return reference_value - corrected_value
    return np.nan


def _rich_gain_cell(
    *,
    raw_df: pd.DataFrame,
    display_value: object,
    row_pos: int,
    col_pos: int,
    significant_digits: int,
    base_color: str,
) -> str | Text:
    text = _format_scalar_value(display_value, significant_digits)
    if not isinstance(raw_df.columns, pd.MultiIndex):
        return text

    raw_col_key = raw_df.columns[col_pos]
    col_name_to_value = {
        str(name): raw_col_key[i]
        for i, name in enumerate(raw_df.columns.names)
    }
    stat_name = str(col_name_to_value.get("stat", ""))
    model_name = str(col_name_to_value.get("model", ""))
    if model_name != MODEL_KNOBS["gain_model"] or stat_name not in {"avg", "ens", "prob"}:
        return text

    raw_value = raw_df.iloc[row_pos, col_pos]
    if pd.isna(raw_value):
        style = Style(color="bright_black", bold=True)
    else:
        if raw_value > 0:
            style = Style(color="green", bold=True)
        elif raw_value < 0:
            style = Style(color="red", bold=True)
        else:
            style = Style(color="bright_black", bold=True)

    return Text(text, style=style)


def _format_table_context_value(*, field: str, value: object, leadtime_unit: str | None = None) -> str:
    if field == "leadtime" and leadtime_unit:
        return f"{value} {leadtime_unit}"
    return str(value)


def _table_context_subtitle(
    df: pd.DataFrame,
    *,
    fields: tuple[str, ...] = TABLE_CONTEXT_FIELDS,
    leadtime_unit: str | None = None,
) -> str:
    parts: list[str] = []
    for field in fields:
        if field not in df.columns:
            continue
        values = [value for value in df[field].dropna().unique().tolist() if str(value) != ""]
        if not values:
            continue
        if len(values) == 1:
            parts.append(
                f"{field}: {_format_table_context_value(field=field, value=values[0], leadtime_unit=leadtime_unit)}"
            )
        else:
            joined = ", ".join(
                _format_table_context_value(field=field, value=value, leadtime_unit=leadtime_unit)
                for value in values
            )
            parts.append(f"{field}: {joined}")
    return " | ".join(parts)


def _print_rich_dataframe_table(
    df: pd.DataFrame,
    *,
    title: str,
    subtitle: str | None = None,
    significant_digits: int,
    raw_df: pd.DataFrame | None = None,
    base_color: str | None = None,
) -> None:
    if not GLOBAL_KNOBS["print_console_tables"]:
        return

    rich_table = RichTable(
        title=title,
        show_lines=False,
        footer_style="bold white",
    )
    index_labels = list(df.index.names) if isinstance(df.index, pd.MultiIndex) else [df.index.name or "row"]
    index_labels = [label if label is not None else "row" for label in index_labels]

    for label in index_labels:
        rich_table.add_column(str(label), style="magenta")
    for col in df.columns:
        rich_table.add_column(_flatten_label(col), justify="right", style="cyan")

    for row_pos, (idx, row) in enumerate(df.iterrows()):
        idx_parts = idx if isinstance(idx, tuple) else (idx,)
        row_cells = [str(part) for part in idx_parts]
        for col_pos, val in enumerate(row.tolist()):
            if raw_df is not None and base_color is not None:
                row_cells.append(
                    _rich_gain_cell(
                        raw_df=raw_df,
                        display_value=val,
                        row_pos=row_pos,
                        col_pos=col_pos,
                        significant_digits=significant_digits,
                        base_color=base_color,
                    )
                )
            else:
                row_cells.append(_format_scalar_value(val, significant_digits))
        rich_table.add_row(*row_cells)

    CONSOLE.print(rich_table)
    if subtitle:
        CONSOLE.print(Text(subtitle, style="italic grey70"))


def _format_table_for_display(
    df: pd.DataFrame,
    *,
    stat_label_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    df_display = df.copy()
    if isinstance(df_display.columns, pd.MultiIndex):
        df_display.columns = pd.MultiIndex.from_tuples(
            [
                tuple(
                    (
                        stat_label_overrides.get(str(part), _prettify_stat_label(str(part)))
                        if stat_label_overrides is not None
                        else _prettify_stat_label(str(part))
                    ) if i == len(col) - 1 else part
                    for i, part in enumerate(col)
                )
                for col in df_display.columns
            ],
            names=df_display.columns.names,
        )
    return df_display


def _filter_metric_rows(df: pd.DataFrame, *, metrics_keep: set[str] | None = None) -> pd.DataFrame:
    if df.empty or metrics_keep is None:
        return df
    if "metric" not in df.index.names:
        return df
    metric_values = df.index.get_level_values("metric").astype(str)
    return df.loc[metric_values.isin(metrics_keep)]


def _drop_empty_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep_mask = ~df.isna().all(axis=0)
    return df.loc[:, keep_mask]


def _add_model_gain_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex) or "stat" not in df.columns.names or "model" not in df.columns.names:
        return df

    col_names = list(df.columns.names)
    stat_idx = col_names.index("stat")
    model_idx = col_names.index("model")
    prefix_indices = [i for i in range(len(col_names)) if i not in {stat_idx, model_idx}]
    prefix_stat_pairs = sorted({(tuple(col[i] for i in prefix_indices), col[stat_idx]) for col in df.columns})

    out = df.copy()
    gain_stats = {"avg", "ens", "prob"}
    for prefix, stat in prefix_stat_pairs:
        if stat not in gain_stats:
            continue
        reference_key = list(prefix)
        reference_key.insert(model_idx, MODEL_KNOBS["reference_model"])
        reference_key.insert(stat_idx, stat)
        corrected_key = list(prefix)
        corrected_key.insert(model_idx, MODEL_KNOBS["corrected_model"])
        corrected_key.insert(stat_idx, stat)
        gain_key = list(prefix)
        gain_key.insert(model_idx, MODEL_KNOBS["gain_model"])
        gain_key.insert(stat_idx, stat)

        reference_key = tuple(reference_key)
        corrected_key = tuple(corrected_key)
        gain_key = tuple(gain_key)

        if reference_key in out.columns and corrected_key in out.columns:
            reference_series = out[reference_key]
            corrected_series = out[corrected_key]
            gain_values = [
                _metric_gain_value(
                    _metric_name_from_index_value(out.index[row_pos], index_names=list(out.index.names)),
                    reference_series.iloc[row_pos],
                    corrected_series.iloc[row_pos],
                )
                for row_pos in range(len(out.index))
            ]
            out[gain_key] = pd.Series(gain_values, index=out.index, dtype=float)

    return out


def _order_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex) or "stat" not in df.columns.names:
        return df

    col_names = list(df.columns.names)
    stat_idx = col_names.index("stat")
    model_idx = col_names.index("model") if "model" in col_names else None

    def _sort_key(col: tuple[object, ...]) -> tuple:
        prefix = tuple("" if col[i] is None else str(col[i]) for i in range(len(col)) if i not in {stat_idx, model_idx})
        model = str(col[model_idx]) if model_idx is not None else ""
        model_order = (
            TABLE_MODEL_ORDER.index(model)
            if model in TABLE_MODEL_ORDER
            else len(TABLE_MODEL_ORDER)
        )
        stat = str(col[stat_idx])
        stat_order = (
            TABLE_KNOBS["stat_order"].index(stat)
            if stat in TABLE_KNOBS["stat_order"]
            else len(TABLE_KNOBS["stat_order"])
        )
        return (*prefix, model_order, stat_order, model, stat)

    return df.loc[:, sorted(df.columns.tolist(), key=_sort_key)]


def save_scalar_metric_table_image(
    *,
    df: pd.DataFrame,
    title: str,
    subtitle: str | None,
    image_path: Path,
    significant_digits: int,
    dpi: int = TABLE_KNOBS["image_dpi"],
    stat_label_overrides: dict[str, str] | None = None,
    base_color: str,
) -> Path:
    df_display = _format_table_for_display(df, stat_label_overrides=stat_label_overrides)
    n_rows, n_cols = df_display.shape
    table_variables = _table_variables(df)
    background_color = TABLE_KNOBS["background_color"]

    fig_width = max(14.0, 2.2 + 1.75 * (n_cols + df_display.index.nlevels))
    fig_height = max(2.8, 1.4 + 0.42 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(background_color)
    ax.axis("off")

    col_labels = []
    for col in df_display.columns:
        parts = col if isinstance(col, tuple) else (col,)
        col_labels.append("\n".join(map(str, parts)))

    row_labels = [
        "\n".join(map(str, idx if isinstance(idx, tuple) else (idx,)))
        for idx in df_display.index
    ]
    cell_text = [
        [_format_scalar_value(value, significant_digits) for value in row]
        for row in df_display.to_numpy()
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.01, 0.10, 0.98, 0.81],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    header_color = base_color
    header_text = "#fdfaf3"
    edge_color = _mix_colors(base_color, background_color, 0.58)
    text_color = "#1b1b1b"
    gain_good = "#d7efdc"
    gain_bad = "#f6d7d2"
    gain_neutral = _mix_colors(base_color, background_color, 0.76)

    def _resolve_variable_color(variable: str | None) -> str:
        if variable is None or not table_variables:
            return base_color
        return get_variable_table_color(variable=variable, variables=table_variables)

    for (row, col), cell in table.get_celld().items():
        cell_edge_color = edge_color
        cell.set_linewidth(0.8)
        if row == 0:
            column_color = _resolve_variable_color(_variable_from_column(df, col)) if col >= 0 else header_color
            cell.set_height(cell.get_height() * 1.45)
            cell.PAD = 0.08
            cell.set_facecolor(column_color)
            cell.set_text_props(color=header_text, weight="bold")
        elif col == -1:
            row_color = _resolve_variable_color(_variable_from_row(df, row - 1))
            row_label_color = _mix_colors(row_color, background_color, 0.50)
            cell_edge_color = _mix_colors(row_color, background_color, 0.32)
            cell.set_facecolor(row_label_color)
            cell.set_text_props(color=text_color, weight="bold")
        else:
            row_color = _resolve_variable_color(_variable_from_row(df, row - 1))
            stripe_a = _mix_colors(row_color, background_color, 0.78)
            stripe_b = _mix_colors(row_color, background_color, 0.68)
            cell_edge_color = _mix_colors(row_color, background_color, 0.42)
            base_face = stripe_a if row % 2 else stripe_b
            cell.set_facecolor(base_face)
            cell.set_text_props(color=text_color)

            if isinstance(df.columns, pd.MultiIndex):
                raw_col_key = df.columns[col]
                col_name_to_value = {
                    str(name): raw_col_key[i]
                    for i, name in enumerate(df.columns.names)
                }
                stat_name = str(col_name_to_value.get("stat", ""))
                model_name = str(col_name_to_value.get("model", ""))
                if model_name == MODEL_KNOBS["gain_model"] and stat_name in {"avg", "ens", "prob"}:
                    raw_value = df.iloc[row - 1, col]
                    if pd.isna(raw_value):
                        cell.set_facecolor(gain_neutral)
                    else:
                        if raw_value > 0:
                            cell.set_facecolor(gain_good)
                        elif raw_value < 0:
                            cell.set_facecolor(gain_bad)
                        else:
                            cell.set_facecolor(gain_neutral)
                    cell.set_text_props(color=text_color, weight="bold")
        cell.set_edgecolor(cell_edge_color)

    ax.text(
        0.01,
        0.97,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=base_color,
    )
    if subtitle:
        ax.text(
            0.01,
            0.035,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            color="#5b5b5b",
        )

    fig.savefig(image_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return image_path


def print_available_scalar_metrics(*, metrics: dict) -> None:
    inventory = {}
    for metric_type, section_dict in metrics.items():
        ds_scalar = section_dict.get("scalar")
        if ds_scalar is None or "metric" not in ds_scalar.coords:
            inventory[metric_type] = []
            continue
        inventory[metric_type] = [str(metric) for metric in ds_scalar.coords["metric"].values.tolist()]

    CONSOLE.print("Available scalar metrics:")
    for metric_type, metric_names in inventory.items():
        CONSOLE.print(f"  {metric_type}: {metric_names or 'none'}")


def _build_deterministic_summary_frame(
    *,
    metrics: dict,
    variables: list[str],
) -> pd.DataFrame:
    df_det = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="deterministic",
        diff="no",
    )
    if df_det.empty:
        return pd.DataFrame()

    group_cols = [col for col in df_det.columns if col not in {"value", "realization"}]

    if "realization" in df_det.columns:
        df_mean = df_det.groupby(group_cols, dropna=False, as_index=False)["value"].mean()
        df_mean["stat"] = "avg"

        df_spread = df_det.groupby(group_cols, dropna=False, as_index=False)["value"].std(ddof=0)
        df_spread["value"] = df_spread["value"].fillna(0.0)
        df_spread["stat"] = "spread"

        return pd.concat([df_mean, df_spread], ignore_index=True)

    df_det = df_det.copy()
    df_det["stat"] = "avg"
    df_spread = df_det.copy()
    df_spread["value"] = 0.0
    df_spread["stat"] = "spread"
    return pd.concat([df_det, df_spread], ignore_index=True)


def _build_metric_stat_frame(
    *,
    metrics: dict,
    variables: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    df_det = _build_deterministic_summary_frame(metrics=metrics, variables=variables)
    if not df_det.empty:
        frames.append(df_det)

    df_ens = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="ensemble",
        diff="no",
    )
    if not df_ens.empty:
        df_ens = df_ens.copy()
        df_ens["stat"] = "ens"
        frames.append(df_ens)

    df_prob = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="probabilistic",
        diff="no",
    )
    if not df_prob.empty:
        df_prob = df_prob.copy()
        df_prob["stat"] = "prob"
        frames.append(df_prob)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _format_duplicate_table_key(
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


def _build_strict_scalar_table_pivot(
    df: pd.DataFrame,
    *,
    row_index: tuple[str, ...],
    column_index: tuple[str, ...],
) -> pd.DataFrame:
    key_cols = list(row_index) + list(column_index)
    duplicate_counts = (
        df.groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )
    duplicate_counts = duplicate_counts[duplicate_counts["count"] > 1]
    if not duplicate_counts.empty:
        key_names = tuple(key_cols)
        duplicate_examples = [
            _format_duplicate_table_key(
                key_names=key_names,
                key_values=tuple(record[col] for col in key_cols),
            )
            for record in duplicate_counts.head(5).to_dict("records")
        ]
        raise ValueError(
            "Scalar table generation found duplicate cells after realization aggregation. "
            "This would require forbidden extra aggregation in calculate_metrics.py. "
            f"Duplicate keys (showing up to 5): {duplicate_examples}"
        )

    pivot = pd.pivot(
        df,
        index=list(row_index),
        columns=list(column_index),
        values="value",
    )
    return pivot.sort_index(axis=0).sort_index(axis=1)


def build_scalar_metric_tables(
    *,
    metrics: dict,
    variables: list[str],
    filters: dict | None = None,
    row_index: tuple[str, ...] = TABLE_KNOBS["row_index"],
    column_index: tuple[str, ...] = TABLE_KNOBS["column_index"],
    leadtime_unit: str | None = None,
) -> list[tuple[str, str | None, pd.DataFrame, str]]:
    df_all = _build_metric_stat_frame(metrics=metrics, variables=variables)
    df_all = _apply_df_filters(df_all, filters)
    if df_all.empty:
        return []

    single_variable = len(variables) == 1
    effective_column_index = tuple(
        col for col in column_index
        if not (single_variable and col == "variable")
    )

    context_cols = [
        col for col in TABLE_CONTEXT_FIELDS
        if (
            col in df_all.columns
            and col not in row_index
            and col not in effective_column_index
            and df_all[col].nunique(dropna=False) > 1
        )
    ]

    if context_cols:
        grouped_items = list(df_all.groupby(context_cols, dropna=False, sort=True))
    else:
        grouped_items = [((), df_all)]

    tables: list[tuple[str, str | None, pd.DataFrame, str]] = []
    for group_key, df_group in grouped_items:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        pivot = _build_strict_scalar_table_pivot(
            df_group,
            row_index=row_index,
            column_index=effective_column_index,
        )
        pivot = _add_model_gain_columns(pivot)
        pivot = _order_table_columns(pivot)

        context_values: dict[str, object] = {}
        for col, value in zip(context_cols, group_key):
            if pd.isna(value):
                continue
            context_values[col] = value

        for col in TABLE_CONTEXT_FIELDS:
            if col in context_values or col not in df_group.columns:
                continue
            values = [value for value in df_group[col].dropna().unique().tolist() if str(value) != ""]
            if len(values) == 1:
                context_values[col] = values[0]

        title = "Scalar metrics"
        if single_variable:
            title = f"{title} ({format_variable_display_name(str(variables[0]))})"

        tables.append(
            (
                title,
                _table_context_subtitle(df_group, leadtime_unit=leadtime_unit),
                pivot,
                _ordered_context_filename_suffix(context_values),
            )
        )

    return tables


def save_scalar_metric_tables(
    *,
    metrics: dict,
    variables: list[str],
    output_folder: Path,
    filters: dict | None = None,
    significant_digits: int = TABLE_KNOBS["significant_digits"],
    metrics_keep: set[str] | None = None,
    filename_prefix: str = "scalar_metrics",
    title_prefix: str | None = None,
    row_index: tuple[str, ...] = TABLE_KNOBS["row_index"],
    column_index: tuple[str, ...] = TABLE_KNOBS["column_index"],
    stat_label_overrides: dict[str, str] | None = None,
    leadtime_unit: str | None = None,
    base_color: str,
) -> list[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for title, subtitle, table_df, filename_stem in build_scalar_metric_tables(
        metrics=metrics,
        variables=variables,
        filters=filters,
        row_index=row_index,
        column_index=column_index,
        leadtime_unit=leadtime_unit,
    ):
        if table_df.empty:
            continue

        table_df = _filter_metric_rows(table_df, metrics_keep=metrics_keep)
        if table_df.empty:
            continue
        table_df = _drop_empty_table_columns(table_df)
        if table_df.empty:
            continue

        if title_prefix:
            title = f"{title_prefix} | {title}"

        table_df_display = _format_table_for_display(
            table_df,
            stat_label_overrides=stat_label_overrides,
        )
        _print_rich_dataframe_table(
            table_df_display,
            title=title,
            subtitle=subtitle,
            significant_digits=significant_digits,
            raw_df=table_df,
            base_color=base_color,
        )

        filename_parts = [filename_prefix, filename_stem]
        base_filename = "_".join(filename_parts)
        image_path = output_folder / f"{base_filename}.png"
        save_scalar_metric_table_image(
            df=table_df,
            title=title,
            subtitle=subtitle,
            image_path=image_path,
            significant_digits=significant_digits,
            stat_label_overrides=stat_label_overrides,
            base_color=base_color,
        )
        saved_paths.append(image_path)

    return saved_paths


def infer_leadtime_unit_from_configs(exp_root: Path) -> str:
    units: set[str] = set()

    for cfg_path in exp_root.rglob("experiment.cfg"):
        experiment = joblib.load(cfg_path)
        config = experiment["config"]

        train_datasets = config.train_dataset
        train_datasets = train_datasets if isinstance(train_datasets, (list, tuple)) else [train_datasets]

        train_input = next((td for td in train_datasets if td.role == "input"), None)
        if train_input is None:
            continue

        source_configs = train_input.source_configs
        source_config = source_configs[0] if isinstance(source_configs, (list, tuple)) else source_configs
        leadtime_rd = source_config.leadtime

        try:
            _, leadtime_unit = get_leadtime_value_and_unit(leadtime_rd)
        except ValueError as exc:
            raise ValueError(f"Unsupported leadtime in experiment config {cfg_path}: {leadtime_rd}") from exc

        units.add(leadtime_unit)

    if not units:
        return ""
    if len(units) > 1:
        raise ValueError(f"Multiple leadtime units found in {exp_root}: {sorted(units)}")
    return next(iter(units))


def infer_region_from_configs(exp_root: Path, *, selected_region: str = "") -> str:
    regions: set[str] = set()

    for cfg_path in exp_root.rglob("experiment.cfg"):
        experiment = joblib.load(cfg_path)
        config = experiment["config"]

        train_datasets = config.train_dataset
        train_datasets = train_datasets if isinstance(train_datasets, (list, tuple)) else [train_datasets]

        train_input = next((td for td in train_datasets if td.role == "input"), None)
        if train_input is None:
            continue

        datasources = train_input.datasource
        datasources = datasources if isinstance(datasources, (list, tuple)) else [datasources]
        if not datasources:
            continue

        region = getattr(datasources[0].data_selection.region, "name", None)
        if region:
            regions.add(str(region))

    if not regions:
        return ""
    if selected_region:
        if selected_region not in regions:
            raise ValueError(
                f"Selected region {selected_region!r} was not found in {exp_root}. "
                f"Available regions: {sorted(regions)}"
            )
        return selected_region
    if len(regions) > 1:
        raise ValueError(f"Multiple regions found in {exp_root}: {sorted(regions)}")
    return next(iter(regions))


def infer_region_extent_from_configs(
    exp_root: Path,
    *,
    selected_region: str = "",
) -> tuple[float, float, float, float] | None:
    extents = infer_region_extents_from_configs(
        exp_root,
        selected_regions=[selected_region] if selected_region else None,
    )
    if not extents:
        return None
    if selected_region:
        return extents[selected_region]
    if len(extents) > 1:
        raise ValueError(f"Multiple regions found in {exp_root}: {sorted(extents)}")
    return next(iter(extents.values()))


def infer_region_extents_from_configs(
    exp_root: Path,
    *,
    selected_regions: Sequence[str] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    extents: dict[str, tuple[float, float, float, float]] = {}

    for cfg_path in exp_root.rglob("experiment.cfg"):
        experiment = joblib.load(cfg_path)
        config = experiment["config"]

        train_datasets = config.train_dataset
        train_datasets = train_datasets if isinstance(train_datasets, (list, tuple)) else [train_datasets]

        train_input = next((td for td in train_datasets if td.role == "input"), None)
        if train_input is None:
            continue

        datasources = train_input.datasource
        datasources = datasources if isinstance(datasources, (list, tuple)) else [datasources]
        if not datasources:
            continue

        region_obj = getattr(datasources[0].data_selection, "region", None)
        region_name = getattr(region_obj, "name", None)
        lon = getattr(region_obj, "lon", None)
        lat = getattr(region_obj, "lat", None)
        if not region_name or lon is None or lat is None:
            continue

        lon0, lon1 = map(float, lon)
        lat0, lat1 = map(float, lat)
        extents[str(region_name)] = (lon0, lon1, min(lat0, lat1), max(lat0, lat1))

    if selected_regions is None:
        return extents

    normalized_selected = [str(region) for region in selected_regions if region is not None and str(region) != ""]
    if not normalized_selected:
        return extents

    missing = [region for region in normalized_selected if region not in extents]
    if missing:
        raise ValueError(
            f"Selected region(s) {missing!r} were not found in {exp_root}. "
            f"Available regions: {sorted(extents)}"
        )

    return {region: extents[region] for region in normalized_selected}


def _region_extent_for_plot(
    *,
    data: xr.DataArray,
    context_values: dict[str, object] | None,
    filters: dict | None,
    region_extents: dict[str, tuple[float, float, float, float]] | None,
) -> tuple[float, float, float, float] | None:
    if not region_extents:
        return None

    region_name = ""
    if context_values and context_values.get("region") is not None:
        region_name = str(context_values["region"])
    if not region_name:
        region_name = _resolved_single_region_label(data, filters=filters)
    if region_name and region_name in region_extents:
        return region_extents[region_name]
    if len(region_extents) == 1:
        return next(iter(region_extents.values()))
    return None


def save_scoreboard_plot(
    *,
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict,
    filename_context_suffix: str | None = None,
) -> Path:
    scoreboard_frames: list[pd.DataFrame] = []
    diff_mode = "delta" if SCOREBOARD_KNOBS["mode"] == "relative" else "no"
    outer_x_values = (
        [DIFFERENCE_MODEL]
        if SCOREBOARD_KNOBS["mode"] == "relative"
        else list(PLOT_MODELS)
    )

    for metric_name, metric_type in SCOREBOARD_KNOBS["metrics"].items():
        if SCOREBOARD_KNOBS["mode"] == "relative" and metric_name in {"bias", "nbias"}:
            scoreboard_frames.append(
                _build_relative_abs_improvement_df(
                    metrics=metrics,
                    variables=variables,
                    metric_name=metric_name,
                    metric_type=metric_type,
                    models=COMPARISON_MODELS,
                )
            )
        else:
            scoreboard_frames.append(
                build_scalar_metric_df(
                    metrics=metrics,
                    variables=variables,
                    metric_name=metric_name,
                    metric_type=metric_type,
                    diff=diff_mode,
                    models=COMPARISON_MODELS if diff_mode == "delta" else None,
                )
            )

    df_scoreboard = pd.concat(scoreboard_frames, ignore_index=True)

    scoreboard_plot_filters = {}
    for col, val in filters.items():
        if val is None:
            continue
        if col in {SCOREBOARD_KNOBS["row_axis"], SCOREBOARD_KNOBS["col_axis"], "metric", "model"}:
            continue
        scoreboard_plot_filters[col] = val

    filename = f"scoreboard_{SCOREBOARD_KNOBS['mode']}_{SCOREBOARD_KNOBS['row_axis']}_vs_{SCOREBOARD_KNOBS['col_axis']}"
    if filename_context_suffix:
        filename = f"{filename}_{filename_context_suffix}"
    plot_path = plot_folder / f"{filename}.png"
    plot_scoreboard(
        df=df_scoreboard,
        outer_y={"index": "metric", "values": list(SCOREBOARD_KNOBS["metrics"].keys())},
        outer_x={"index": "model", "values": outer_x_values},
        inner_y={
            "index": SCOREBOARD_KNOBS["row_axis"],
            "label": format_axis_display_name(SCOREBOARD_KNOBS["row_axis"], leadtime_unit=leadtime_unit),
        },
        inner_x={
            "index": SCOREBOARD_KNOBS["col_axis"],
            "label": format_axis_display_name(SCOREBOARD_KNOBS["col_axis"], leadtime_unit=leadtime_unit),
        },
        filters=scoreboard_plot_filters,
        agg="mean",
        metric_cmaps=SCOREBOARD_KNOBS["metric_cmaps"],
        metric_vlims=SCOREBOARD_KNOBS["metric_vlims"],
        metric_names=METRIC_DISPLAY_NAMES,
        display_name_maps={
            "variable": format_variable_display_name,
            "model": MODEL_DISPLAY_NAMES,
            "metric": METRIC_DISPLAY_NAMES,
        },
        inner_y_display_name_map=(
            format_variable_short_name
            if SCOREBOARD_KNOBS["row_axis"] == "variable"
            else None
        ),
        annotate=SCOREBOARD_KNOBS["annotate"],
        annotate_fmt="{:.2f}",
        annotate_color="auto",
        save_path=plot_path,
        title=f"Scoreboard ({SCOREBOARD_KNOBS['mode']})",
    )
    return plot_path


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
    shade_by: str,
    shade_label: str,
    filename_context_suffix: str | None = None,
) -> Path:
    plot_folder.mkdir(parents=True, exist_ok=True)
    plot_variables = _ordered_unique_variables(
        [
            *df_nodiff.get("variable", pd.Series(dtype=object)).dropna().astype(str).tolist(),
            *df_delta.get("variable", pd.Series(dtype=object)).dropna().astype(str).tolist(),
        ]
    )
    forecast_metric_label = format_metric_label(forecast_metric, base_unit=variable_unit)
    delta_metric_label = format_metric_label(delta_metric, base_unit=variable_unit)
    region_suffix = f" - {region}" if region else ""

    fig, ax, merged = plot_metric_vs_diff(
        df_nodiff,
        df_delta,
        forecast_metric=forecast_metric,
        diff_metric=delta_metric,
        x_metric_name=f"delta_{delta_metric}",
        y_metric_name=forecast_metric,
        color_by=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["color_by"],
        marker_by=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["marker_by"],
        shade_by=shade_by,
        shade_label=shade_label,
        fit_lines=False,
        fit_ci=False,
        leadtime_unit=leadtime_unit,
        point_size=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["point_size"],
        variable_colors=get_variable_colors(variables=plot_variables),
        filters=filters,
        title=f"Forecast {forecast_metric_label} vs Δ {delta_metric_label} (ML-corrected - forecast){region_suffix}",
        xlabel=f"Δ {delta_metric_label} (ML-corrected - forecast)",
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


def save_field_timeseries_plots(
    *,
    runs: dict[str, xr.Dataset],
    metrics: dict[str, dict[str, xr.Dataset]],
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    combine_leadtimes: bool = False,
    plot_realization_members: bool = False,
    disable_flox: bool = False,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_order = [model_name for model_name in LOAD_MODELS if model_name in runs]
    truth_model = LOAD_MODELS[0] if LOAD_MODELS else None
    leadtime_style = FIELD_TIMESERIES_KNOBS["leadtime_style"]
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
                    dim for dim in RUN_CONTEXT_DIMS
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
                        for model_name in PLOT_MODELS:
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
                        dim for dim in RUN_CONTEXT_DIMS
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
            timeseries_folder = get_variable_output_item_folder(
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
                dim for dim in RUN_CONTEXT_DIMS
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

                reference_model = MODEL_KNOBS["reference_model"]
                corrected_model = MODEL_KNOBS["corrected_model"]
                fc_has_spread = _has_multi_realization(selected_data.get(reference_model))
                pr_has_spread = _has_multi_realization(selected_data.get(corrected_model))
                enable_spread = fc_has_spread or pr_has_spread
                show_legend = enable_spread

                base_title = format_variable_display_name(variable)
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
                    for model_name in PLOT_MODELS:
                        if model_name not in selected_data:
                            continue
                        residual_plot_data[model_name] = _build_residual_da(selected_data[model_name], truth_da)
                    if residual_plot_data:
                        plot_jobs.append(
                            (
                                f"residual_minus_{truth_model}",
                                f"{base_title} residuals (model - {MODEL_DISPLAY_NAMES.get(truth_model, truth_model)})",
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
                                    label = MODEL_DISPLAY_NAMES.get(model_name, str(model_name))
                                    mean_label = label
                                    mean_alpha = 1.0
                                    line_color = MODEL_COLORS.get(model_name)
                                    mean_ls = "-"
                                    members_ls = "--"
                                else:
                                    label = MODEL_DISPLAY_NAMES.get(model_name, str(model_name))
                                    mean_label = label if leadtime_value == leadtime_values[-1] else None
                                    mean_alpha = 1.0
                                    if leadtime_style == "shade":
                                        line_color = _shade_model_color(
                                            MODEL_COLORS.get(model_name),
                                            weight=shade_weight_map[leadtime_value],
                                        )
                                        mean_ls = "-"
                                        members_ls = "--"
                                    else:
                                        line_color = MODEL_COLORS.get(model_name)
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
                                label=MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
                                mean_label=MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
                                color=MODEL_COLORS.get(model_name),
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
                    debug_print(f"Saved {plot_kind} timeseries for region={plot_region or 'unknown'} to:", plot_path)

            for metric_type, section_dict in metrics.items():
                ds_ts = section_dict.get("timeseries", xr.Dataset())
                if not isinstance(ds_ts, xr.Dataset) or variable not in ds_ts.data_vars:
                    continue

                da_var = _apply_xarray_filters(ds_ts[variable], filters)
                if "metric" not in da_var.dims:
                    continue

                metric_values = [str(value) for value in da_var["metric"].values.tolist()]
                for metric_name in metric_values:
                    metric_folder = get_variable_output_item_folder(
                        plot_root=plot_folder,
                        variable=variable,
                        subfolder="timeseries",
                        output_group=_metric_output_group(metric_type),
                        item_name=metric_name,
                    )
                    da_metric = da_var.sel(metric=metric_name, drop=True)
                    context_dims = tuple(
                        dim for dim in RUN_CONTEXT_DIMS
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
                            runs.get(MODEL_KNOBS["truth_model"], xr.Dataset()),
                            variable,
                        )
                        metric_label = format_metric_label(metric_name, base_unit=base_unit)
                        filename_context = _context_filename_suffix(context_values)
                        fig, ax = plt.subplots(figsize=(12, 5.5))
                        plotted = False
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
                                        mean_label = MODEL_DISPLAY_NAMES.get(model_name, str(model_name))
                                        mean_alpha = 1.0
                                        line_color = MODEL_COLORS.get(model_name)
                                        mean_ls = "-"
                                        members_ls = "--"
                                    else:
                                        mean_label = (
                                            MODEL_DISPLAY_NAMES.get(model_name, str(model_name))
                                            if leadtime_value == leadtime_values[-1]
                                            else None
                                        )
                                        mean_alpha = 1.0
                                        if leadtime_style == "shade":
                                            line_color = _shade_model_color(
                                                MODEL_COLORS.get(model_name),
                                                weight=shade_weight_map[leadtime_value],
                                            )
                                            mean_ls = "-"
                                            members_ls = "--"
                                        else:
                                            line_color = MODEL_COLORS.get(model_name)
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
                                        label=MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
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
                                    label=MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
                                    mean_label=MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
                                    color=MODEL_COLORS.get(model_name),
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
                            title=f"{metric_label} timeseries ({format_variable_display_name(variable)})",
                            subtitle=_plot_context_subtitle(
                                context_values,
                                leadtime_unit=leadtime_unit,
                                filters=filters,
                            ),
                        )
                        if combine_leadtimes:
                            _set_combined_leadtime_legend(
                                ax,
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
                        debug_print(
                            f"Saved metric timeseries for metric={metric_name} type={metric_type} region={plot_region or 'unknown'} to:",
                            plot_path,
                        )

    return saved_paths


def save_field_and_metric_map_plots(
    *,
    runs: dict[str, xr.Dataset],
    metrics: dict[str, dict[str, xr.Dataset]],
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    region_extents: dict[str, tuple[float, float, float, float]] | None = None,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_order = [model_name for model_name in LOAD_MODELS if model_name in runs]
    realization_mode = MAP_PLOT_KNOBS["realization_mode"]
    if realization_mode not in {"mean", "members"}:
        raise ValueError(
            f"Unsupported map realization mode {realization_mode!r}. Expected 'mean' or 'members'."
        )

    def _count_total_map_plots() -> int:
        total = 0
        for variable in variables:
            for model_name in model_order:
                ds = runs.get(model_name)
                if ds is None or variable not in ds.data_vars:
                    continue

                da = _apply_xarray_filters(ds[variable], filters)
                for selection, context_values in _context_selection_entries(da):
                    da_sel = da.sel({dim: value for dim, value in selection.items() if dim in da.dims}, drop=True)
                    if da_sel.size == 0:
                        continue
                    da_sel = _reduce_extra_map_dims(da_sel)
                    total += sum(1 for _ in _map_realization_slices(da_sel, realization_mode=realization_mode))

            for metric_type, section_dict in metrics.items():
                ds_map = section_dict.get("map", xr.Dataset())
                if not isinstance(ds_map, xr.Dataset) or variable not in ds_map.data_vars:
                    continue

                da_var = _apply_xarray_filters(ds_map[variable], filters)
                if "metric" not in da_var.dims:
                    continue

                metric_values = [str(value) for value in da_var["metric"].values.tolist()]
                model_values = (
                    [str(value) for value in da_var["model"].values.tolist()]
                    if "model" in da_var.dims
                    else [None]
                )

                for metric_name in metric_values:
                    da_metric = da_var.sel(metric=metric_name, drop=True)
                    for selection, context_values in _context_selection_entries(da_metric):
                        per_model_maps: dict[str, dict[object | None, xr.DataArray]] = {}

                        for model_name in model_values:
                            da_model = da_metric.sel(model=model_name, drop=True) if model_name is not None else da_metric
                            if da_model.size == 0:
                                continue
                            da_sel = da_model.sel(
                                {dim: value for dim, value in selection.items() if dim in da_model.dims},
                                drop=True,
                            )
                            if da_sel.size == 0:
                                continue
                            da_sel = _reduce_extra_map_dims(da_sel)
                            per_model_maps[str(model_name)] = dict(
                                _map_realization_slices(
                                    da_sel,
                                    realization_mode=realization_mode,
                                )
                            )

                        if not per_model_maps:
                            continue

                        total += sum(len(realization_maps) for realization_maps in per_model_maps.values())

                        reference_model = MODEL_KNOBS["reference_model"]
                        corrected_model = MODEL_KNOBS["corrected_model"]
                        if reference_model in per_model_maps and corrected_model in per_model_maps:
                            reference_realizations = per_model_maps[reference_model]
                            corrected_realizations = per_model_maps[corrected_model]
                            total += sum(
                                1
                                for realization_value in corrected_realizations
                                if realization_value in reference_realizations
                            )

        return total

    total_plots = _count_total_map_plots()
    if progress is not None and task_id is not None:
        progress.update(task_id, total=total_plots, completed=0, description="Generating maps")

    for variable in variables:
        CONSOLE.print(f"Map variable: {variable}")
        _ensure_grouped_output_folders(
            plot_root=plot_folder,
            variable=variable,
            subfolder="maps",
        )
        field_maps_folder = get_variable_output_item_folder(
            plot_root=plot_folder,
            variable=variable,
            subfolder="maps",
            output_group="field",
            item_name="fields",
        )

        for model_name in model_order:
            ds = runs.get(model_name)
            if ds is None or variable not in ds.data_vars:
                continue

            da = _apply_xarray_filters(ds[variable], filters)
            for selection, context_values in _context_selection_entries(da):
                da_sel = da.sel({dim: value for dim, value in selection.items() if dim in da.dims}, drop=True)
                if da_sel.size == 0:
                    continue
                da_sel = _reduce_extra_map_dims(da_sel)
                base_unit = _get_da_unit(da_sel)
                for realization_value, da_plot in _map_realization_slices(
                    da_sel,
                    realization_mode=realization_mode,
                ):
                    plot_region = _resolved_single_region_label(da_plot, filters=filters)
                    extent = _region_extent_for_plot(
                        data=da_plot,
                        context_values=context_values,
                        filters=filters,
                        region_extents=region_extents,
                    )
                    save_path = field_maps_folder / (
                        f"field_{model_name}_{_context_filename_suffix(context_values)}"
                        f"{_realization_filename_fragment(realization_value)}_temporal_mean_map.png"
                    )
                    if progress is not None and task_id is not None:
                        progress.update(
                            task_id,
                            description=f"Maps | {variable} | field | {model_name}",
                        )
                    plot_temporal_mean_map(
                        da_plot,
                        save_path=save_path,
                        title=_map_title(
                            format_variable_display_name(variable),
                            MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
                            f"Temporal mean{_realization_label(realization_value)}",
                            _context_title_suffix(context_values, leadtime_unit=leadtime_unit),
                        ),
                        extent=extent,
                        cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                        cmap=MAP_PLOT_KNOBS["field_cmap"],
                        lon_tick_step=MAP_PLOT_KNOBS["lon_tick_step"],
                        lat_tick_step=MAP_PLOT_KNOBS["lat_tick_step"],
                    )
                    saved_paths.append(save_path)
                    if progress is not None and task_id is not None:
                        progress.advance(task_id)
                    debug_print(
                        f"Saved field map for model={model_name} region={plot_region or 'unknown'} to:",
                        save_path,
                    )

        for metric_type, section_dict in metrics.items():
            ds_map = section_dict.get("map", xr.Dataset())
            if not isinstance(ds_map, xr.Dataset) or variable not in ds_map.data_vars:
                continue

            da_var = _apply_xarray_filters(ds_map[variable], filters)
            if "metric" not in da_var.dims:
                continue

            metric_values = [str(value) for value in da_var["metric"].values.tolist()]
            model_values = (
                [str(value) for value in da_var["model"].values.tolist()]
                if "model" in da_var.dims
                else [None]
            )

            for metric_name in metric_values:
                da_metric = da_var.sel(metric=metric_name, drop=True)
                context_source = da_metric
                for selection, context_values in _context_selection_entries(context_source):
                    per_model_maps: dict[str, dict[object | None, xr.DataArray]] = {}
                    base_unit: str | None = None

                    for model_name in model_values:
                        da_model = da_metric.sel(model=model_name, drop=True) if model_name is not None else da_metric
                        if da_model.size == 0:
                            continue
                        da_sel = da_model.sel(
                            {dim: value for dim, value in selection.items() if dim in da_model.dims},
                            drop=True,
                        )
                        if da_sel.size == 0:
                            continue
                        da_sel = _reduce_extra_map_dims(da_sel)
                        per_model_maps[str(model_name)] = dict(
                            _map_realization_slices(
                                da_sel,
                                realization_mode=realization_mode,
                            )
                        )
                        if base_unit is None:
                            base_unit = _get_da_unit(da_sel) or _get_da_unit(
                                runs.get(MODEL_KNOBS["truth_model"], xr.Dataset()),
                                variable,
                            )

                    if not per_model_maps:
                        continue

                    context_suffix = _context_title_suffix(context_values, leadtime_unit=leadtime_unit)
                    sample_da = next(iter(next(iter(per_model_maps.values())).values()))
                    extent = _region_extent_for_plot(
                        data=sample_da,
                        context_values=context_values,
                        filters=filters,
                        region_extents=region_extents,
                    )
                    metric_label = format_metric_label(metric_name, base_unit=base_unit)
                    shared_vmin, shared_vmax = _shared_map_limits(
                        *(da for per_model in per_model_maps.values() for da in per_model.values())
                    )

                    for model_name, realization_maps in per_model_maps.items():
                        metric_folder = get_variable_output_item_folder(
                            plot_root=plot_folder,
                            variable=variable,
                            subfolder="maps",
                            output_group=_metric_output_group(metric_type),
                            item_name=metric_name,
                        )
                        for realization_value, da_sel in realization_maps.items():
                            map_region = _resolved_single_region_label(da_sel, filters=filters)
                            filename_parts = [metric_type, metric_name, model_name, _context_filename_suffix(context_values)]
                            save_path = metric_folder / (
                                f"{'_'.join(filename_parts)}{_realization_filename_fragment(realization_value)}_map.png"
                            )

                            if progress is not None and task_id is not None:
                                progress.update(
                                    task_id,
                                    description=f"Maps | {variable} | {metric_type} | {metric_name}",
                                )
                            plot_temporal_mean_map(
                                da_sel,
                                save_path=save_path,
                                title=_map_title(
                                    metric_label,
                                    MODEL_DISPLAY_NAMES.get(model_name, model_name),
                                    f"{context_suffix}{_realization_label(realization_value)}" if context_suffix else _realization_label(realization_value).lstrip(" |"),
                                ),
                                extent=extent,
                                cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                                cmap=_resolve_metric_map_cmap(metric_name),
                                vmin=shared_vmin,
                                vmax=shared_vmax,
                                lon_tick_step=MAP_PLOT_KNOBS["lon_tick_step"],
                                lat_tick_step=MAP_PLOT_KNOBS["lat_tick_step"],
                            )
                            saved_paths.append(save_path)
                            if progress is not None and task_id is not None:
                                progress.advance(task_id)
                            debug_print(
                                f"Saved metric map for metric={metric_name} model={model_name} region={map_region or 'unknown'} to:",
                                save_path,
                            )

                    reference_model = MODEL_KNOBS["reference_model"]
                    corrected_model = MODEL_KNOBS["corrected_model"]
                    difference_model = DIFFERENCE_MODEL
                    if reference_model in per_model_maps and corrected_model in per_model_maps:
                        metric_folder = get_variable_output_item_folder(
                            plot_root=plot_folder,
                            variable=variable,
                            subfolder="maps",
                            output_group=_metric_output_group(metric_type),
                            item_name=metric_name,
                        )
                        reference_realizations = per_model_maps[reference_model]
                        corrected_realizations = per_model_maps[corrected_model]
                        common_realizations = [
                            realization_value
                            for realization_value in corrected_realizations
                            if realization_value in reference_realizations
                        ]
                        diff_maps = [
                            (
                                realization_value,
                                corrected_realizations[realization_value] - reference_realizations[realization_value],
                            )
                            for realization_value in common_realizations
                        ]
                        if diff_maps:
                            diff_vmin, diff_vmax = _symmetric_map_limits(
                                xr.concat([diff_da for _, diff_da in diff_maps], dim="_diff_map")
                            )
                            for realization_value, diff_da in diff_maps:
                                diff_region = _resolved_single_region_label(diff_da, filters=filters)
                                diff_save_path = metric_folder / (
                                    f"{metric_type}_{metric_name}_{difference_model}_{_context_filename_suffix(context_values)}"
                                    f"{_realization_filename_fragment(realization_value)}_map.png"
                                )
                                if progress is not None and task_id is not None:
                                    progress.update(
                                        task_id,
                                        description=f"Maps | {variable} | diff | {metric_name}",
                                    )
                                plot_temporal_mean_map(
                                    diff_da,
                                    save_path=diff_save_path,
                                    title=_map_title(
                                        metric_label,
                                        MODEL_DISPLAY_NAMES[difference_model],
                                        f"{context_suffix}{_realization_label(realization_value)}" if context_suffix else _realization_label(realization_value).lstrip(" |"),
                                    ),
                                    extent=extent,
                                    cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                                    cmap="RdBu_r",
                                    vmin=diff_vmin,
                                    vmax=diff_vmax,
                                    lon_tick_step=MAP_PLOT_KNOBS["lon_tick_step"],
                                    lat_tick_step=MAP_PLOT_KNOBS["lat_tick_step"],
                                )
                                saved_paths.append(diff_save_path)
                                if progress is not None and task_id is not None:
                                    progress.advance(task_id)
                                debug_print(
                                    f"Saved diff map for metric={metric_name} model={difference_model} region={diff_region or 'unknown'} to:",
                                    diff_save_path,
                                )

    return saved_paths


def save_metrics_vs_parameter_plots(
    *,
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    variable_units: dict[str, str | None] | None,
    x_axis: str,
    filters: dict | None,
    shade_by: str,
    shade_label: str,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_colors = MODEL_COLORS
    supported_axes = {"leadtime", "train_period", "loss", "variable", "region"}
    if x_axis not in supported_axes:
        raise ValueError(f"Unsupported metric profile axis {x_axis!r}. Expected one of {sorted(supported_axes)}.")

    metric_names = list(METRIC_PROFILE_PLOT_KNOBS["metrics"])
    df_det = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="deterministic",
        diff="no",
    )
    df_ens = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="ensemble",
        diff="no",
    )
    df_prob = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="probabilistic",
        diff="no",
    )

    df_det = df_det[df_det["metric"].isin(metric_names)]
    df_ens = df_ens[df_ens["metric"].isin(metric_names)]
    df_prob = df_prob[df_prob["metric"].isin(metric_names)]

    df_det = _apply_df_filters_except_axis(df_det, filters=filters, x_axis=x_axis)
    df_ens = _apply_df_filters_except_axis(df_ens, filters=filters, x_axis=x_axis)
    df_prob = _apply_df_filters_except_axis(df_prob, filters=filters, x_axis=x_axis)

    panel_variables = [None] if x_axis == "variable" else variables

    for variable in panel_variables:
        profile_variable = variable if variable is not None else "all_variables"
        if progress is not None and task_id is not None:
            progress.update(
                task_id,
                description=f"Generating {profile_variable} metric profiles ({x_axis})",
            )
        CONSOLE.print(f"Metric profile variable: {variable if variable is not None else 'all_variables'}")
        output_folder = _profile_output_folder(plot_root=plot_folder, variable=variable)
        variable_mask = slice(None) if variable is None else variable

        for metric_name, metric_type in METRIC_PROFILE_PLOT_KNOBS["metrics"].items():
            df_plot_ens = df_ens[df_ens["metric"] == metric_name]
            df_plot_det = df_det[df_det["metric"] == metric_name]
            df_plot_prob = df_prob[df_prob["metric"] == metric_name]

            if variable is not None:
                df_plot_ens = df_plot_ens[df_plot_ens["variable"] == variable_mask]
                df_plot_det = df_plot_det[df_plot_det["variable"] == variable_mask]
                df_plot_prob = df_plot_prob[df_plot_prob["variable"] == variable_mask]

            df_template = df_plot_prob if metric_type == "probabilistic" else df_plot_ens
            if df_template.empty or x_axis not in df_template.columns:
                continue

            context_columns = _profile_context_columns(
                df_template,
                x_axis=x_axis,
                shade_by=None if shade_by == x_axis else shade_by,
            )
            grouped_items = (
                list(df_template.groupby(context_columns, dropna=False, sort=True))
                if context_columns
                else [((), df_template)]
            )

            for group_key, _ in grouped_items:
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)

                context_values = {
                    col: value
                    for col, value in zip(context_columns, group_key)
                    if not pd.isna(value)
                }

                df_context_ens = df_plot_ens.copy()
                df_context_det = df_plot_det.copy()
                df_context_prob = df_plot_prob.copy()
                for col, value in context_values.items():
                    df_context_ens = df_context_ens[df_context_ens[col] == value]
                    df_context_det = df_context_det[df_context_det[col] == value]
                    df_context_prob = df_context_prob[df_context_prob[col] == value]

                df_context_template = df_context_prob if metric_type == "probabilistic" else df_context_ens
                x_values = _ordered_profile_axis_values(df_context_template, x_axis=x_axis, variables=variables)
                if not x_values:
                    continue

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
                diff_fig, diff_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
                rel_fig, rel_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
                x_positions = np.arange(len(x_values))

                if shade_by != x_axis and shade_by in df_context_template.columns:
                    shade_values = [value for value in pd.unique(df_context_template[shade_by]) if pd.notna(value)]
                else:
                    shade_values = [None]

                model_values = [value for value in pd.unique(df_context_template["model"]) if pd.notna(value)]
                plotted = False
                diff_plotted = False
                rel_plotted = False
                probabilistic_series: dict[tuple[str, object], pd.Series] = {}
                deterministic_mean_series: dict[tuple[str, object], pd.Series] = {}
                deterministic_member_matrices: dict[tuple[str, object], pd.DataFrame] = {}
                deterministic_ensemble_series: dict[tuple[str, object], pd.Series] = {}

                for model_name in model_values:
                    base_color = model_colors.get(str(model_name), None)
                    model_label = MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name))

                    for j, shade_value in enumerate(shade_values):
                        alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))
                        shade_suffix = f", {shade_label}={shade_value}" if shade_value is not None else ""

                        if metric_type == "probabilistic":
                            df_model = df_context_prob[df_context_prob["model"] == model_name]
                            if shade_value is not None:
                                df_model = df_model[df_model[shade_by] == shade_value]
                            y_series = _build_strict_profile_series(
                                df_model,
                                x_axis=x_axis,
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
                        elif metric_type == "deterministic_with_ensemble_overlay":
                            df_model_ens = df_context_ens[df_context_ens["model"] == model_name]
                            df_model_det = df_context_det[df_context_det["model"] == model_name]
                            if shade_value is not None:
                                df_model_ens = df_model_ens[df_model_ens[shade_by] == shade_value]
                                df_model_det = df_model_det[df_model_det[shade_by] == shade_value]

                            y_ens = _build_strict_profile_series(
                                df_model_ens,
                                x_axis=x_axis,
                                x_values=x_values,
                            )
                            y_members = _build_strict_member_matrix(
                                df_model_det,
                                x_axis=x_axis,
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
                                        index=pd.Index(x_values, name=x_axis),
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

                difference_color = model_colors.get(
                    DIFFERENCE_MODEL,
                    model_colors.get(MODEL_KNOBS["corrected_model"], "tab:blue"),
                )
                for j, shade_value in enumerate(shade_values):
                    alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))
                    shade_suffix = f", {shade_label}={shade_value}" if shade_value is not None else ""

                    if metric_type == "probabilistic":
                        ref_series = probabilistic_series.get((MODEL_KNOBS["reference_model"], shade_value))
                        corr_series = probabilistic_series.get((MODEL_KNOBS["corrected_model"], shade_value))
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
                                    label=f"{MODEL_DISPLAY_NAMES[DIFFERENCE_MODEL]}{shade_suffix}",
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
                                        label=f"{MODEL_DISPLAY_NAMES[DIFFERENCE_MODEL]}{shade_suffix}",
                                    )
                    elif metric_type == "deterministic_with_ensemble_overlay":
                        ref_mean = deterministic_mean_series.get((MODEL_KNOBS["reference_model"], shade_value))
                        corr_mean = deterministic_mean_series.get((MODEL_KNOBS["corrected_model"], shade_value))
                        ref_members = deterministic_member_matrices.get((MODEL_KNOBS["reference_model"], shade_value))
                        corr_members = deterministic_member_matrices.get((MODEL_KNOBS["corrected_model"], shade_value))
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
                                    label=f"{MODEL_DISPLAY_NAMES[DIFFERENCE_MODEL]}{shade_suffix}",
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
                                        label=f"{MODEL_DISPLAY_NAMES[DIFFERENCE_MODEL]}{shade_suffix}",
                                    )

                        ref_ens = deterministic_ensemble_series.get((MODEL_KNOBS["reference_model"], shade_value))
                        corr_ens = deterministic_ensemble_series.get((MODEL_KNOBS["corrected_model"], shade_value))
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
                                    label=f"{MODEL_DISPLAY_NAMES[DIFFERENCE_MODEL]} ensemble{shade_suffix}",
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
                                        label=f"{MODEL_DISPLAY_NAMES[DIFFERENCE_MODEL]} ensemble{shade_suffix}",
                                    )
                    else:
                        raise ValueError(f"Unsupported metric profile plot mode {metric_type!r}")

                axis_label = format_axis_display_name(x_axis, leadtime_unit=leadtime_unit.strip() if leadtime_unit else None)
                title_subject = (
                    "all variables"
                    if x_axis == "variable"
                    else format_variable_display_name(str(variable))
                )
                metric_label = format_metric_label(
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
                        f"{metric_label} vs {format_axis_display_name(x_axis).lower()} ({title_subject})",
                        metric_label,
                        True,
                    ),
                    (
                        diff_fig,
                        diff_ax,
                        f"Delta {metric_label} vs {format_axis_display_name(x_axis).lower()} ({title_subject})",
                        f"Delta {metric_label}",
                        diff_plotted,
                    ),
                    (
                        rel_fig,
                        rel_ax,
                        f"Relative change in {metric_label} vs {format_axis_display_name(x_axis).lower()} ({title_subject})",
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
                        rotation=30 if x_axis in {"train_period", "loss", "variable"} else 0,
                        ha="right" if x_axis in {"train_period", "loss", "variable"} else "center",
                    )
                    current_ax.grid(True, linestyle="--", alpha=0.35)

                    ymin, ymax = current_ax.get_ylim()
                    if ymin <= 0 <= ymax:
                        current_ax.axhline(0.0, color="0.3", lw=1.0, ls=":")

                    if show_legend:
                        current_ax.legend(fontsize=9)
                    current_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

                file_axis = _sanitize_filename_fragment(x_axis)
                filename = f"{metric_name}_vs_{file_axis}"
                context_suffix = _context_filename_suffix(context_values)
                if context_suffix != "all":
                    filename = f"{filename}_{context_suffix}"
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
    x_axis: str,
    filters: dict | None,
    shade_by: str,
    shade_label: str,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    if x_axis == "variable":
        raise ValueError("Combined variable metric profiles do not support x_axis='variable'.")

    saved_paths: list[Path] = []
    variable_colors = get_variable_colors(variables=variables)
    metric_modes = METRIC_PROFILE_PLOT_KNOBS["combined_variable_metrics"]
    metric_names = list(metric_modes)

    df_det = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="deterministic",
        diff="no",
    )
    df_ens = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="ensemble",
        diff="no",
    )
    df_prob = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="probabilistic",
        diff="no",
    )

    df_det = df_det[df_det["metric"].isin(metric_names)]
    df_ens = df_ens[df_ens["metric"].isin(metric_names)]
    df_prob = df_prob[df_prob["metric"].isin(metric_names)]

    df_det = _apply_df_filters_except_axis(df_det, filters=filters, x_axis=x_axis)
    df_ens = _apply_df_filters_except_axis(df_ens, filters=filters, x_axis=x_axis)
    df_prob = _apply_df_filters_except_axis(df_prob, filters=filters, x_axis=x_axis)

    output_folder = _profile_output_folder(plot_root=plot_folder, variable=None)

    for metric_name, metric_type in metric_modes.items():
        if progress is not None and task_id is not None:
            progress.update(task_id, description=f"Generating all_variables combined profile: {metric_name}")

        df_plot_ens = df_ens[df_ens["metric"] == metric_name]
        df_plot_det = df_det[df_det["metric"] == metric_name]
        df_plot_prob = df_prob[df_prob["metric"] == metric_name]

        df_template = df_plot_prob if metric_type == "probabilistic" else df_plot_ens
        if df_template.empty or x_axis not in df_template.columns:
            continue

        context_columns = _profile_context_columns(
            df_template,
            x_axis=x_axis,
            shade_by=None if shade_by == x_axis else shade_by,
        )
        grouped_items = (
            list(df_template.groupby(context_columns, dropna=False, sort=True))
            if context_columns
            else [((), df_template)]
        )

        for group_key, _ in grouped_items:
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            context_values = {
                col: value
                for col, value in zip(context_columns, group_key)
                if not pd.isna(value)
            }

            df_context_ens = df_plot_ens.copy()
            df_context_det = df_plot_det.copy()
            df_context_prob = df_plot_prob.copy()
            for col, value in context_values.items():
                df_context_ens = df_context_ens[df_context_ens[col] == value]
                df_context_det = df_context_det[df_context_det[col] == value]
                df_context_prob = df_context_prob[df_context_prob[col] == value]

            df_context_template = df_context_prob if metric_type == "probabilistic" else df_context_ens
            x_values = _ordered_profile_axis_values(df_context_template, x_axis=x_axis, variables=variables)
            if not x_values:
                continue

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
            diff_fig, diff_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
            rel_fig, rel_ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
            x_positions = np.arange(len(x_values))

            if shade_by != x_axis and shade_by in df_context_template.columns:
                shade_values = [value for value in pd.unique(df_context_template[shade_by]) if pd.notna(value)]
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
            deterministic_mean_series: dict[tuple[str, str, object], pd.Series] = {}
            deterministic_member_matrices: dict[tuple[str, str, object], pd.DataFrame] = {}
            deterministic_ensemble_series: dict[tuple[str, str, object], pd.Series] = {}

            for variable in variables:
                variable_color = variable_colors.get(variable)
                df_var_ens = df_context_ens[df_context_ens["variable"] == variable]
                df_var_det = df_context_det[df_context_det["variable"] == variable]
                df_var_prob = df_context_prob[df_context_prob["variable"] == variable]

                if df_var_ens.empty and df_var_det.empty and df_var_prob.empty:
                    continue

                model_values = [
                    value for value in pd.unique(
                        (df_var_prob if metric_type == "probabilistic" else df_var_ens)["model"]
                    ) if pd.notna(value)
                ]

                for model_name in model_values:
                    for j, shade_value in enumerate(shade_values):
                        alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))

                        if metric_type == "probabilistic":
                            df_model = df_var_prob[df_var_prob["model"] == model_name]
                            if shade_value is not None:
                                df_model = df_model[df_model[shade_by] == shade_value]
                            y_series = _build_strict_profile_series(
                                df_model,
                                x_axis=x_axis,
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
                                linestyle=_model_linestyle(str(model_name), probabilistic=True),
                                label="_nolegend_",
                            )
                        elif metric_type == "deterministic_with_ensemble_overlay":
                            df_model_ens = df_var_ens[df_var_ens["model"] == model_name]
                            df_model_det = df_var_det[df_var_det["model"] == model_name]
                            if shade_value is not None:
                                df_model_ens = df_model_ens[df_model_ens[shade_by] == shade_value]
                                df_model_det = df_model_det[df_model_det[shade_by] == shade_value]

                            y_ens = _build_strict_profile_series(
                                df_model_ens,
                                x_axis=x_axis,
                                x_values=x_values,
                            )
                            y_members = _build_strict_member_matrix(
                                df_model_det,
                                x_axis=x_axis,
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
                                        index=pd.Index(x_values, name=x_axis),
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
                                        linestyle=_model_linestyle(str(model_name)),
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
                        ref_series = probabilistic_series.get((str(variable), MODEL_KNOBS["reference_model"], shade_value))
                        corr_series = probabilistic_series.get((str(variable), MODEL_KNOBS["corrected_model"], shade_value))
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
                    elif metric_type == "deterministic_with_ensemble_overlay":
                        ref_mean = deterministic_mean_series.get((str(variable), MODEL_KNOBS["reference_model"], shade_value))
                        corr_mean = deterministic_mean_series.get((str(variable), MODEL_KNOBS["corrected_model"], shade_value))
                        ref_members = deterministic_member_matrices.get((str(variable), MODEL_KNOBS["reference_model"], shade_value))
                        corr_members = deterministic_member_matrices.get((str(variable), MODEL_KNOBS["corrected_model"], shade_value))
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

                        ref_ens = deterministic_ensemble_series.get((str(variable), MODEL_KNOBS["reference_model"], shade_value))
                        corr_ens = deterministic_ensemble_series.get((str(variable), MODEL_KNOBS["corrected_model"], shade_value))
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

            axis_label = format_axis_display_name(x_axis, leadtime_unit=leadtime_unit.strip() if leadtime_unit else None)
            metric_label = format_metric_label(metric_name, base_unit=None)
            context_subtitle = _plot_context_subtitle(
                context_values,
                leadtime_unit=leadtime_unit,
                filters=filters,
            )

            for current_fig, current_ax, current_title, current_ylabel in [
                (
                    fig,
                    ax,
                    f"{metric_label} vs {format_axis_display_name(x_axis).lower()} (all variables)",
                    metric_label,
                ),
                (
                    diff_fig,
                    diff_ax,
                    f"Delta {metric_label} vs {format_axis_display_name(x_axis).lower()} (all variables)",
                    f"Delta {metric_label}",
                ),
                (
                    rel_fig,
                    rel_ax,
                    f"Relative change in {metric_label} vs {format_axis_display_name(x_axis).lower()} (all variables)",
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
                    rotation=30 if x_axis in {"train_period", "loss", "variable"} else 0,
                    ha="right" if x_axis in {"train_period", "loss", "variable"} else "center",
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
                shade_by=shade_by,
                shade_label=shade_label,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

            if diff_plotted:
                _set_transformed_combined_variable_profile_legend(
                    diff_ax,
                    variables=transformed_variables,
                    variable_colors=variable_colors,
                    metric_type=metric_type,
                    shade_values=transformed_shade_values,
                    shade_by=shade_by,
                    shade_label=shade_label,
                )
                diff_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

            if rel_plotted:
                _set_transformed_combined_variable_profile_legend(
                    rel_ax,
                    variables=transformed_variables,
                    variable_colors=variable_colors,
                    metric_type=metric_type,
                    shade_values=transformed_shade_values,
                    shade_by=shade_by,
                    shade_label=shade_label,
                )
                rel_fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

            filename = f"all_variables_{metric_name}_vs_{_sanitize_filename_fragment(x_axis)}"
            context_suffix = _context_filename_suffix(context_values)
            if context_suffix != "all":
                filename = f"{filename}_{context_suffix}"
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


def _guess_dask_workers_for_host() -> tuple[int | None, str]:
    hostname = socket.gethostname().lower()
    cpu_count = os.cpu_count() or multiprocessing.cpu_count() or 1

    # Keep a few known machine classes on stable defaults for interactive use.
    host_overrides = (
        (("mac-studio",), 5, "Mac Studio default"),
        (("macbook", "mbp", "air"), min(4, cpu_count), "laptop default"),
        (("juno", "login", "node", "hpc", "cluster"), min(max(8, cpu_count // 2), 16), "shared cluster default"),
    )
    for patterns, workers, reason in host_overrides:
        if any(pattern in hostname for pattern in patterns):
            return max(1, min(workers, cpu_count)), f"{reason} for host {hostname}"

    # Unknown hosts fall back to the generic heuristic inside earthml.misc.Dask.
    return None, f"using Dask auto sizing for host {hostname}"


def main() -> None:
    dask_workers, dask_worker_reason = _guess_dask_workers_for_host()
    CONSOLE.print(f"Dask worker guess: {dask_workers if dask_workers is not None else 'auto'} ({dask_worker_reason})")
    dask_earthml = Dask(n_workers=dask_workers, memory_limit="100GiB")
    dask_earthml.start()
    client = dask_earthml.client
    base_url = "http://localhost:"
    CONSOLE.print("Dask dashboard:", client.dashboard_link.replace("http://127.0.0.1:", base_url))

    CONSOLE.print(f"Loading experiments from: {GLOBAL_KNOBS['exp_root_folder']}")
    print_user_config_table()

    field_timeseries_filters = _merge_filters(BASE_FILTERS, FIELD_TIMESERIES_KNOBS["filters"])
    map_filters = _merge_filters(BASE_FILTERS, MAP_PLOT_KNOBS["filters"])
    metric_vs_deltametric_filters = METRIC_VS_DELTAMETRIC_PLOT_KNOBS["filters"]
    metric_profile_filters = _merge_filters(BASE_FILTERS, METRIC_PROFILE_PLOT_KNOBS["filters"])
    scalar_table_filters = _merge_filters(BASE_FILTERS, TABLE_KNOBS["scalar_filters"])
    normalized_table_filters = _merge_filters(BASE_FILTERS, TABLE_KNOBS["normalized_filters"])
    scoreboard_filters = SCOREBOARD_KNOBS["filters"]

    enabled_filter_sets: list[dict | None] = []
    if GLOBAL_KNOBS["enable_field_timeseries"]:
        enabled_filter_sets.append(field_timeseries_filters)
    if GLOBAL_KNOBS["enable_maps"]:
        enabled_filter_sets.append(map_filters)
    if GLOBAL_KNOBS["enable_metric_vs_deltametric_plot"]:
        enabled_filter_sets.append(metric_vs_deltametric_filters)
    if GLOBAL_KNOBS["enable_metric_profile_plots"]:
        enabled_filter_sets.append(metric_profile_filters)
    if GLOBAL_KNOBS["enable_scalar_tables"]:
        enabled_filter_sets.extend((scalar_table_filters, normalized_table_filters))
    if GLOBAL_KNOBS["enable_scoreboard"]:
        enabled_filter_sets.append(scoreboard_filters)

    load_selection = _build_load_selection(*enabled_filter_sets)
    load_variables = load_selection.get("variable")
    load_run_filters = {
        key: value
        for key, value in load_selection.items()
        if key != "variable"
    }
    CONSOLE.print(f"Load selection: variables={load_variables or 'all'}, run_filters={load_run_filters or 'all'}")

    runs, metrics, _ = get_runs_and_metrics(
        exp_root=GLOBAL_KNOBS["exp_root_folder"],
        type_data=GLOBAL_KNOBS["type_data"],
        load_models=LOAD_MODELS,
        variables=load_variables,
        run_filters=load_run_filters,
        metric_names=GLOBAL_KNOBS["metric_names"],
        show_progress=True,
    )

    reference_model = MODEL_KNOBS["reference_model"]
    variables = [v for v in runs[reference_model].data_vars if v != "_has_var"]
    variable_units = _get_variable_units_from_runs(runs, variables)
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
    CONSOLE.print(f"Processed vars: {variables}")
    leadtime_unit = infer_leadtime_unit_from_configs(GLOBAL_KNOBS["exp_root_folder"])
    field_timeseries_variables = _filter_variables_by_filters(variables, field_timeseries_filters)
    map_variables = _filter_variables_by_filters(variables, map_filters)
    metric_profile_variables = _filter_variables_by_filters(variables, metric_profile_filters)
    scalar_table_variables = _filter_variables_by_filters(variables, scalar_table_filters)
    normalized_table_variables = _filter_variables_by_filters(variables, normalized_table_filters)
    map_region_filters = map_filters.get("region") if map_filters else None
    map_region_values = (
        list(map_region_filters)
        if isinstance(map_region_filters, (list, tuple, set))
        else ([map_region_filters] if map_region_filters is not None else [])
    )
    region_extents = infer_region_extents_from_configs(
        GLOBAL_KNOBS["exp_root_folder"],
        selected_regions=map_region_values or None,
    )
    diff_region = _single_filter_value(metric_vs_deltametric_filters, "region")
    metric_vs_deltametric_filename_context = _merge_filename_contexts(
        _filters_filename_context(metric_vs_deltametric_filters),
    )
    scoreboard_filename_context = _merge_filename_contexts(
        _filters_filename_context(scoreboard_filters),
    )
    CONSOLE.print(f"Inferred leadtime unit: {leadtime_unit or 'unknown'}")
    CONSOLE.print(f"Inferred map regions: {sorted(region_extents) if region_extents else 'unknown'}")
    CONSOLE.print(f"Inferred region extents: {region_extents or 'unknown'}")
    print_available_scalar_metrics(metrics=metrics)

    with build_progress() as progress:
        if GLOBAL_KNOBS["enable_field_timeseries"]:
            ts_task = progress.add_task("Generating field timeseries", total=len(field_timeseries_variables))
            field_timeseries_paths = save_field_timeseries_plots(
                runs=runs,
                metrics=metrics,
                variables=field_timeseries_variables,
                plot_folder=PLOT_FOLDER,
                leadtime_unit=leadtime_unit,
                filters=field_timeseries_filters,
                combine_leadtimes=FIELD_TIMESERIES_KNOBS["combine_leadtimes"],
                plot_realization_members=FIELD_TIMESERIES_KNOBS["plot_realization_members"],
                disable_flox=FIELD_TIMESERIES_KNOBS["disable_flox_for_field_timeseries"],
                progress=progress,
                task_id=ts_task,
            )
            for field_timeseries_path in field_timeseries_paths:
                debug_print("Saved field timeseries plot to:", field_timeseries_path)
        else:
            CONSOLE.print("Skipping field timeseries: GLOBAL_KNOBS['enable_field_timeseries'] is False")

        if GLOBAL_KNOBS["enable_maps"]:
            map_task = progress.add_task("Generating maps", total=len(map_variables))
            field_map_paths = save_field_and_metric_map_plots(
                runs=runs,
                metrics=metrics,
                variables=map_variables,
                plot_folder=PLOT_FOLDER,
                leadtime_unit=leadtime_unit,
                filters=map_filters,
                region_extents=region_extents,
                progress=progress,
                task_id=map_task,
            )
            for field_map_path in field_map_paths:
                debug_print("Saved map plot to:", field_map_path)
        else:
            CONSOLE.print("Skipping maps: GLOBAL_KNOBS['enable_maps'] is False")

        if GLOBAL_KNOBS["enable_metric_vs_deltametric_plot"]:
            diff_task = progress.add_task("Generating metric-vs-delta-metric plot", total=1)
            df_nodiff = build_scalar_metric_df(
                metrics=metrics,
                variables=variables,
                metric_name=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["forecast_metric"],
                metric_type=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["forecast_metric_type"],
                diff="no",
            )
            df_delta = build_scalar_metric_df(
                metrics=metrics,
                variables=variables,
                metric_name=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["delta_metric"],
                metric_type=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["delta_metric_type"],
                diff="delta",
                models=COMPARISON_MODELS,
            )

            plot_path = save_metric_vs_deltametric_plot(
                df_nodiff=df_nodiff,
                df_delta=df_delta,
                plot_folder=PLOT_FOLDER,
                forecast_metric=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["forecast_metric"],
                delta_metric=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["delta_metric"],
                leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                region=diff_region,
                filters=metric_vs_deltametric_filters,
                variable_unit=variable_units.get(_single_filter_value(metric_vs_deltametric_filters, "variable")),
                shade_by=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["shade_by"],
                shade_label=METRIC_VS_DELTAMETRIC_PLOT_KNOBS["shade_label"],
                filename_context_suffix=metric_vs_deltametric_filename_context,
            )
            progress.advance(diff_task)
            debug_print("Saved leadtime-vs-global-metrics plot to:", plot_path)
        else:
            CONSOLE.print(
                "Skipping metric-vs-delta-metric plot: "
                "GLOBAL_KNOBS['enable_metric_vs_deltametric_plot'] is False"
            )

        if GLOBAL_KNOBS["enable_metric_profile_plots"]:
            profile_total = 1 if METRIC_PROFILE_PLOT_KNOBS["x_axis"] == "variable" else len(metric_profile_variables)
            profile_task = progress.add_task(
                f"Generating metric profiles ({METRIC_PROFILE_PLOT_KNOBS['x_axis']})",
                total=profile_total,
            )
            profile_plot_paths = save_metrics_vs_parameter_plots(
                metrics=metrics,
                variables=metric_profile_variables,
                plot_folder=PLOT_FOLDER,
                leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                variable_units=variable_units,
                x_axis=METRIC_PROFILE_PLOT_KNOBS["x_axis"],
                filters=metric_profile_filters,
                shade_by=METRIC_PROFILE_PLOT_KNOBS["shade_by"],
                shade_label=METRIC_PROFILE_PLOT_KNOBS["shade_label"],
                progress=progress,
                task_id=profile_task,
            )
            for profile_plot_path in profile_plot_paths:
                debug_print("Saved metric profile plot to:", profile_plot_path)
        else:
            CONSOLE.print("Skipping metric profile plots: GLOBAL_KNOBS['enable_metric_profile_plots'] is False")

        if METRIC_PROFILE_PLOT_KNOBS["enable_combined_variable_profiles"]:
            combined_profile_task = progress.add_task(
                f"Generating combined variable profiles ({METRIC_PROFILE_PLOT_KNOBS['x_axis']})",
                total=1,
            )
            combined_profile_paths = save_combined_variable_metric_profiles(
                metrics=metrics,
                variables=metric_profile_variables,
                plot_folder=PLOT_FOLDER,
                leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                x_axis=METRIC_PROFILE_PLOT_KNOBS["x_axis"],
                filters=metric_profile_filters,
                shade_by=METRIC_PROFILE_PLOT_KNOBS["shade_by"],
                shade_label=METRIC_PROFILE_PLOT_KNOBS["shade_label"],
                progress=progress,
                task_id=combined_profile_task,
            )
            for combined_profile_path in combined_profile_paths:
                debug_print("Saved combined variable profile plot to:", combined_profile_path)

        if GLOBAL_KNOBS["enable_scalar_tables"]:
            raw_metrics = {
                metric
                for metric_group in (
                    metrics.get("deterministic", {}).get("scalar"),
                    metrics.get("ensemble", {}).get("scalar"),
                    metrics.get("probabilistic", {}).get("scalar"),
                )
                if metric_group is not None and "metric" in metric_group.coords
                for metric in metric_group.coords["metric"].values.tolist()
            } - NORMALIZED_METRICS

            table_task = progress.add_task("Generating scalar tables", total=len(scalar_table_variables) + 1)
            for variable in scalar_table_variables:
                progress.update(table_task, description=f"Generating {variable} scalar tables")
                CONSOLE.print(f"Scalar table variable: {variable}")
                variable_plot_folder = _table_output_folder(plot_root=PLOT_FOLDER, variable=variable)
                base_color = get_variable_table_color(variable=variable, variables=variables)
                scalar_table_paths = save_scalar_metric_tables(
                    metrics=metrics,
                    variables=[variable],
                    output_folder=variable_plot_folder,
                    filters=scalar_table_filters,
                    metrics_keep=raw_metrics,
                    filename_prefix="scalar_metrics",
                    leadtime_unit=leadtime_unit,
                    base_color=base_color,
                )
                progress.advance(table_task)
                for scalar_table_path in scalar_table_paths:
                    debug_print("Saved scalar metrics table to:", scalar_table_path)

            progress.update(table_task, description="Generating all_variables scalar tables")
            combined_normalized_table_paths = save_scalar_metric_tables(
                # Combined normalized tables live under all_variables/tables.
                metrics=metrics,
                variables=normalized_table_variables,
                output_folder=_table_output_folder(plot_root=PLOT_FOLDER, variable=None),
                filters=normalized_table_filters,
                metrics_keep=NORMALIZED_METRICS,
                filename_prefix="normalized_metrics",
                title_prefix="Normalized metrics",
                row_index=("variable", "metric"),
                column_index=("model", "stat"),
                stat_label_overrides={"prob": ""},
                leadtime_unit=leadtime_unit,
                base_color=get_variable_table_color(variable=variables[0], variables=variables),
            )
            CONSOLE.print("Scalar table variable: all_variables")
            progress.advance(table_task)
            for normalized_table_path in combined_normalized_table_paths:
                debug_print("Saved combined normalized metrics table to:", normalized_table_path)
        else:
            CONSOLE.print("Skipping scalar tables: GLOBAL_KNOBS['enable_scalar_tables'] is False")

        if GLOBAL_KNOBS["enable_scoreboard"]:
            scoreboard_task = progress.add_task("Generating scoreboard", total=1)
            scoreboard_path = save_scoreboard_plot(
                metrics=metrics,
                variables=variables,
                plot_folder=PLOT_FOLDER,
                leadtime_unit=leadtime_unit,
                filters=scoreboard_filters,
                filename_context_suffix=scoreboard_filename_context,
            )
            progress.advance(scoreboard_task)
            debug_print("Saved scoreboard plot to:", scoreboard_path)
        else:
            CONSOLE.print("Skipping scoreboard: GLOBAL_KNOBS['enable_scoreboard'] is False")


if __name__ == "__main__":
    main()
