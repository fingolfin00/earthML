import logging
import os
from contextlib import nullcontext
from itertools import product
from pathlib import Path
import warnings
import joblib

# Silence Intel OpenMP deprecation/info chatter emitted by downstream libraries.
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import xarray as xr
from rich import print
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
    # All artifacts land in exp_root_folder / plot_folder_name.
    "plot_folder_name": "plots",
    # Dataset split and model loading.
    "type_data": "test",
    "load_models": ("an", "fc", "pr"),
    # Keep None to load every scalar metric available.
    "metric_names": None,
    # Model pair used for delta comparisons like pr - fc.
    "models_diff": ("fc", "pr"),
    # Global output switches.
    "enable_scalar_tables": True,
    "enable_diff_plot": False,
    "enable_metric_profile_plots": True,
    "enable_scoreboard": True,
    # Toggle rich table output in the terminal while still saving table files/images.
    "print_console_tables": False,
}

COMMON_PLOT_KNOBS = {
    # Applied to maps, timeseries, diff plot and the metric-profile plots.
    "filters": {
        "leadtime": [1],
        "variable": None,
        "region": ["ConUS"], # ConUS, Europe
    },
    # Used to color/shade runs consistently across plots.
    "shade_by": "loss",  # e.g. "loss", "total_months"
    "shade_label": "Loss",
    # Disable flox/numbagg in field-timeseries plotting. Useful on some clusters
    # where lazy reductions on float16-backed arrays fail.
    "disable_flox_for_field_timeseries": False,
}

MAP_PLOT_KNOBS = {
    # Colormap for raw field temporal-mean maps.
    "field_cmap": "jet",
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

DIFF_PLOT_KNOBS = {
    # x-axis: how much the ML correction changes the chosen metric.
    "diff_metric": "nrmse",
    "diff_metric_type": "ensemble",
    # y-axis: the forecast quality you want to compare against.
    "forecast_metric": "corr",
    "forecast_metric_type": "ensemble",
    "cmap": "magma",
}

METRIC_PROFILE_PLOT_KNOBS = {
    # Pick the x-axis used for metric profile plots.
    "x_axis": "variable",  # "leadtime", "train_period", "loss", "variable"
    # Each metric maps to the plotting mode expected by save_metrics_vs_parameter_plots.
    "metrics": {
        "r2": "deterministic_with_ensemble_overlay",
        # "rmse": "deterministic_with_ensemble_overlay",
        "nrmse": "deterministic_with_ensemble_overlay",
        # "rmse_skill_clim": "deterministic_with_ensemble_overlay",
        "nmae": "deterministic_with_ensemble_overlay",
        "nbias": "deterministic_with_ensemble_overlay",
        "crps": "probabilistic",
        "spread": "probabilistic",
        "spread_error_ratio": "probabilistic",
    },
    "model_colors": {
        "fc": "tab:green",
        "pr": "tab:blue",
        "an": "tab:orange",
    },
}

TABLE_KNOBS = {
    # Filter rows before building summary tables.
    "filters": {
        "train_period": None,
        "loss": "mseloss",
        "leadtime": 1,
    },
    "row_index": ("metric",),
    "column_index": ("model", "variable", "stat"),
    # Tables split when these dimensions have more than one surviving value.
    "group_by": ("train_period", "loss", "leadtime"),
    "model_order": ("fc", "pr", "pr-fc"),
    "stat_order": ("avg", "spread", "ens", "prob"),
    "significant_digits": 3,
    "image_dpi": 300,
    "base_colors": ("#3c8be4", "#c9be2c", "#bb41c6"),
}

SCOREBOARD_KNOBS = {
    "mode": "relative",  # "absolute" or "relative"
    "metrics": {
        "r2": "ensemble",
        "nrmse": "ensemble",
        "crps": "probabilistic",
        "rmse_skill_clim": "ensemble",
        "spread_error_ratio": "probabilistic",
    },
    "metric_cmaps": {
        "r2": "RdBu",
        "nrmse": "RdBu_r",
        "nmae": "RdBu_r",
        "nbias": "RdBu_r",
        "rmse_skill_clim": "RdBu",
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
    # These act only when the dimension is not already on an axis.
    "filters": {
        "variable": None,  # e.g. ["t2m"]
        "loss": "mseloss",
        "train_period": None,
        "leadtime": None,
    },
    "annotate": True,
}


# Derived internal aliases. The grouped knobs above are the intended edit points.
EXP_ROOT_FOLDER = GLOBAL_KNOBS["exp_root_folder"]
PLOT_FOLDER = EXP_ROOT_FOLDER / GLOBAL_KNOBS["plot_folder_name"]

TYPE_DATA = GLOBAL_KNOBS["type_data"]
LOAD_MODELS = GLOBAL_KNOBS["load_models"]
METRIC_NAMES = GLOBAL_KNOBS["metric_names"]
MODELS_DIFF = GLOBAL_KNOBS["models_diff"]

FORECAST_METRIC = DIFF_PLOT_KNOBS["forecast_metric"]
FORECAST_METRIC_TYPE = DIFF_PLOT_KNOBS["forecast_metric_type"]
DIFF_METRIC = DIFF_PLOT_KNOBS["diff_metric"]
DIFF_METRIC_TYPE = DIFF_PLOT_KNOBS["diff_metric_type"]
DIFF_PLOT_CMAP = DIFF_PLOT_KNOBS["cmap"]

METRIC_PROFILE_AXIS = METRIC_PROFILE_PLOT_KNOBS["x_axis"]
METRIC_PROFILE_METRICS = METRIC_PROFILE_PLOT_KNOBS["metrics"]
MODEL_COLORS = METRIC_PROFILE_PLOT_KNOBS["model_colors"]
PLOT_FILTERS = COMMON_PLOT_KNOBS["filters"]
SHADE_BY = COMMON_PLOT_KNOBS["shade_by"]
SHADE_LABEL = COMMON_PLOT_KNOBS["shade_label"]
DISABLE_FLOX_FOR_FIELD_TIMESERIES = COMMON_PLOT_KNOBS["disable_flox_for_field_timeseries"]
FIELD_MAP_CMAP = MAP_PLOT_KNOBS["field_cmap"]
METRIC_MAP_CMAPS = MAP_PLOT_KNOBS["metric_cmaps"]
METRIC_MAP_CMAP_DEFAULT = MAP_PLOT_KNOBS["metric_cmap_default"]
MAP_LON_TICK_STEP = MAP_PLOT_KNOBS["lon_tick_step"]
MAP_LAT_TICK_STEP = MAP_PLOT_KNOBS["lat_tick_step"]

SCOREBOARD_MODE = SCOREBOARD_KNOBS["mode"]
SCOREBOARD_METRICS = SCOREBOARD_KNOBS["metrics"]
SCOREBOARD_METRIC_CMAPS = SCOREBOARD_KNOBS["metric_cmaps"]
SCOREBOARD_METRIC_VLIMS = SCOREBOARD_KNOBS["metric_vlims"]
SCOREBOARD_ROW_AXIS = SCOREBOARD_KNOBS["row_axis"]
SCOREBOARD_COL_AXIS = SCOREBOARD_KNOBS["col_axis"]
SCOREBOARD_FILTERS = SCOREBOARD_KNOBS["filters"]
SCOREBOARD_ANNOTATE = SCOREBOARD_KNOBS["annotate"]

SCALAR_TABLE_IMAGE_DPI = TABLE_KNOBS["image_dpi"]
SCALAR_TABLE_BASE_COLORS = TABLE_KNOBS["base_colors"]
SCALAR_TABLE_MODEL_ORDER = TABLE_KNOBS["model_order"]
SCALAR_TABLE_STAT_ORDER = TABLE_KNOBS["stat_order"]
SCALAR_TABLE_FILTERS = TABLE_KNOBS["filters"]
SCALAR_TABLE_ROW_INDEX = TABLE_KNOBS["row_index"]
SCALAR_TABLE_COLUMN_INDEX = TABLE_KNOBS["column_index"]
SCALAR_TABLE_GROUP_BY = TABLE_KNOBS["group_by"]
SCALAR_TABLE_SIGNIFICANT_DIGITS = TABLE_KNOBS["significant_digits"]
NORMALIZED_METRICS = {"nrmse", "ncrmse", "nmae", "nbias", "r2"}
VARIABLE_SUBFOLDERS = ("timeseries", "maps", "leadtime")
RUN_CONTEXT_DIMS = ("leadtime", "train_period", "loss", "region")


METRIC_DISPLAY_NAMES = {
    "r2": "R2",
    "rmse": "RMSE",
    "rmse_skill_clim": "RMSE Skill",
    "crmse": "CRMSE",
    "mae": "MAE",
    "bias": "Bias",
    "nrmse": "nRMSE",
    "ncrmse": "nCRMSE",
    "nmae": "nMAE",
    "nbias": "nBias",
    "crps": "CRPS",
    "spread": "Spread",
    "spread_error_ratio": "Spread/Error",
}

VARIABLE_DISPLAY_NAMES = {
    "t2m": "2m temp",
    "d2m": "2m dewpoint",
    "msl": "MSL pressure",
    "mslp": "MSL pressure",
    "u10": "10m u-wind",
    "v10": "10m v-wind",
    "tp": "Precipitation",
    "tcc": "Cloud cover",
    "sst": "SST",
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

MODEL_DISPLAY_NAMES = {
    "fc": "FC",
    "pr": "PR",
    "an": "AN",
    "pr-fc": "PR - FC",
}


def debug_print(*args, **kwargs) -> None:
    if DEBUG:
        print(*args, **kwargs)


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
    metric_names = (metric_name,) if metric_name is not None else METRIC_NAMES
    return metrics_to_df(
        metrics=metrics,
        variables=variables,
        metric_names=metric_names,
        kind="scalar",
        metric_type=metric_type,
        diff=diff,
        models=models,
    )


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


def _ordered_unique_variables(variables: list[str]) -> list[str]:
    unique_variables: list[str] = []
    for variable in variables:
        variable_str = str(variable)
        if variable_str not in unique_variables:
            unique_variables.append(variable_str)
    return unique_variables


def get_variable_colors(*, variables: list[str]) -> dict[str, str]:
    ordered_variables = _ordered_unique_variables(variables)
    return {
        variable: SCALAR_TABLE_BASE_COLORS[idx % len(SCALAR_TABLE_BASE_COLORS)]
        for idx, variable in enumerate(ordered_variables)
    }


def get_variable_table_color(*, variable: str, variables: list[str]) -> str:
    variable_colors = get_variable_colors(variables=variables)
    if not variable_colors:
        return SCALAR_TABLE_BASE_COLORS[0]
    return variable_colors.get(str(variable), SCALAR_TABLE_BASE_COLORS[0])


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


def _mean_series_by_axis(
    df: pd.DataFrame,
    *,
    x_axis: str,
    x_values: list[object],
) -> pd.Series:
    if df.empty:
        return pd.Series(index=pd.Index(x_values, name=x_axis), dtype=float)
    series = df.groupby(x_axis, dropna=False)["value"].mean()
    return series.reindex(x_values)


def _member_matrix_by_axis(
    df: pd.DataFrame,
    *,
    x_axis: str,
    x_values: list[object],
) -> pd.DataFrame | None:
    if df.empty or "realization" not in df.columns:
        return None

    members = (
        df.groupby([x_axis, "realization"], dropna=False)["value"]
        .mean()
        .unstack("realization")
        .reindex(x_values)
    )
    return members if not members.empty else None


def _profile_output_folder(
    *,
    plot_root: Path,
    x_axis: str,
    variable: str | None = None,
) -> Path:
    if x_axis == "leadtime" and variable is not None:
        return get_variable_subfolder(plot_root=plot_root, variable=variable, subfolder="leadtime")
    if variable is not None:
        output_folder = get_variable_plot_folder(plot_root=plot_root, variable=variable) / "profiles"
    else:
        output_folder = plot_root / f"{x_axis}_profiles"
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


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
    if metric_name in {"rmse", "crmse", "mae", "bias", "crps", "spread"}:
        return base_unit
    return None


def format_metric_label(metric_name: str, *, base_unit: str | None = None) -> str:
    return _format_label_with_unit(format_metric_display_name(metric_name), _metric_unit(metric_name, base_unit))


def _resolve_metric_map_cmap(metric_name: str) -> str:
    return METRIC_MAP_CMAPS.get(metric_name, METRIC_MAP_CMAP_DEFAULT)


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


def print_user_config_table() -> None:
    if not GLOBAL_KNOBS["print_console_tables"]:
        return

    config_sections = [
        ("debug", {"value": DEBUG}),
        ("global", GLOBAL_KNOBS),
        ("common_plot", COMMON_PLOT_KNOBS),
        ("map_plot", MAP_PLOT_KNOBS),
        ("diff_plot", DIFF_PLOT_KNOBS),
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
    if isinstance(df.index, pd.MultiIndex):
        index_names = list(df.index.names)
        if "metric" in index_names:
            metric_idx = index_names.index("metric")
            return str(idx_value[metric_idx])
    return str(idx_value[0] if isinstance(idx_value, tuple) else idx_value)


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
        "nrmse": False,
        "ncrmse": False,
        "nmae": False,
        "nbias": False,
        "crps": False,
        "spread": None,
        "spread_error_ratio": None,
    }
    return mapping.get(metric)


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
    if model_name != "pr-fc" or stat_name not in {"avg", "ens", "prob"}:
        return text

    metric_name = _metric_name_from_index(raw_df, row_pos)
    raw_value = raw_df.iloc[row_pos, col_pos]
    direction = _metric_higher_is_better(metric_name)
    if pd.isna(raw_value) or direction is None:
        style = Style(color="bright_black", bold=True)
    else:
        improved = raw_value > 0 if direction else raw_value < 0
        degraded = raw_value < 0 if direction else raw_value > 0
        if improved:
            style = Style(color="green", bold=True)
        elif degraded:
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
    fields: tuple[str, ...] = ("train_period", "loss", "leadtime"),
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

    rich_table = RichTable(title=title, show_lines=False)
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

    print(rich_table)
    if subtitle:
        print(Text(subtitle, style="italic bright_black"))


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
        fc_key = list(prefix)
        fc_key.insert(model_idx, "fc")
        fc_key.insert(stat_idx, stat)
        pr_key = list(prefix)
        pr_key.insert(model_idx, "pr")
        pr_key.insert(stat_idx, stat)
        gain_key = list(prefix)
        gain_key.insert(model_idx, "pr-fc")
        gain_key.insert(stat_idx, stat)

        fc_key = tuple(fc_key)
        pr_key = tuple(pr_key)
        gain_key = tuple(gain_key)

        if fc_key in out.columns and pr_key in out.columns:
            out[gain_key] = out[pr_key] - out[fc_key]

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
            SCALAR_TABLE_MODEL_ORDER.index(model)
            if model in SCALAR_TABLE_MODEL_ORDER
            else len(SCALAR_TABLE_MODEL_ORDER)
        )
        stat = str(col[stat_idx])
        stat_order = (
            SCALAR_TABLE_STAT_ORDER.index(stat)
            if stat in SCALAR_TABLE_STAT_ORDER
            else len(SCALAR_TABLE_STAT_ORDER)
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
    dpi: int = SCALAR_TABLE_IMAGE_DPI,
    stat_label_overrides: dict[str, str] | None = None,
    base_color: str,
) -> Path:
    df_display = _format_table_for_display(df, stat_label_overrides=stat_label_overrides)
    n_rows, n_cols = df_display.shape
    table_variables = _table_variables(df)

    fig_width = max(14.0, 2.2 + 1.75 * (n_cols + df_display.index.nlevels))
    fig_height = max(2.8, 1.4 + 0.42 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#f6f1e8")
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
    edge_color = _mix_colors(base_color, "#ffffff", 0.58)
    text_color = "#1b1b1b"
    gain_good = "#d7efdc"
    gain_bad = "#f6d7d2"
    gain_neutral = _mix_colors(base_color, "#ffffff", 0.76)

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
            row_label_color = _mix_colors(row_color, "#ffffff", 0.50)
            cell_edge_color = _mix_colors(row_color, "#ffffff", 0.32)
            cell.set_facecolor(row_label_color)
            cell.set_text_props(color=text_color, weight="bold")
        else:
            row_color = _resolve_variable_color(_variable_from_row(df, row - 1))
            stripe_a = _mix_colors(row_color, "#ffffff", 0.78)
            stripe_b = _mix_colors(row_color, "#ffffff", 0.68)
            cell_edge_color = _mix_colors(row_color, "#ffffff", 0.42)
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
                if model_name == "pr-fc" and stat_name in {"avg", "ens", "prob"}:
                    metric_name = _metric_name_from_index(df, row - 1)
                    raw_value = df.iloc[row - 1, col]
                    direction = _metric_higher_is_better(metric_name)
                    if pd.isna(raw_value) or direction is None:
                        cell.set_facecolor(gain_neutral)
                    else:
                        improved = raw_value > 0 if direction else raw_value < 0
                        degraded = raw_value < 0 if direction else raw_value > 0
                        if improved:
                            cell.set_facecolor(gain_good)
                        elif degraded:
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

    print("Available scalar metrics:")
    for metric_type, metric_names in inventory.items():
        print(f"  {metric_type}: {metric_names or 'none'}")


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


def build_scalar_metric_tables(
    *,
    metrics: dict,
    variables: list[str],
    filters: dict | None = None,
    row_index: tuple[str, ...] = SCALAR_TABLE_ROW_INDEX,
    column_index: tuple[str, ...] = SCALAR_TABLE_COLUMN_INDEX,
    group_by: tuple[str, ...] = SCALAR_TABLE_GROUP_BY,
    leadtime_unit: str | None = None,
) -> list[tuple[str, str | None, pd.DataFrame, str]]:
    df_all = _build_metric_stat_frame(metrics=metrics, variables=variables)
    df_all = _apply_df_filters(df_all, filters)
    if df_all.empty:
        return []

    group_cols = [
        col for col in group_by
        if col in df_all.columns and col not in row_index and col not in column_index and df_all[col].nunique(dropna=False) > 1
    ]

    if group_cols:
        grouped_items = list(df_all.groupby(group_cols, dropna=False, sort=True))
    else:
        grouped_items = [((), df_all)]

    tables: list[tuple[str, str | None, pd.DataFrame, str]] = []
    for group_key, df_group in grouped_items:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        pivot = pd.pivot_table(
            df_group,
            index=list(row_index),
            columns=list(column_index),
            values="value",
            aggfunc="mean",
            dropna=False,
        ).sort_index(axis=0).sort_index(axis=1)
        pivot = _add_model_gain_columns(pivot)
        pivot = _order_table_columns(pivot)

        context_values: dict[str, object] = {}
        for col, value in zip(group_cols, group_key):
            if pd.isna(value):
                continue
            context_values[col] = value

        for col in ("train_period", "loss", "leadtime"):
            if col in context_values or col not in df_group.columns:
                continue
            values = [value for value in df_group[col].dropna().unique().tolist() if str(value) != ""]
            if len(values) == 1:
                context_values[col] = values[0]

        filename_parts = [
            f"{col}_{_sanitize_filename_fragment(str(context_values[col]))}"
            for col in ("train_period", "loss", "leadtime")
            if col in context_values
        ]

        for col, value in context_values.items():
            if col in {"train_period", "loss", "leadtime"}:
                continue
            filename_parts.append(f"{col}_{_sanitize_filename_fragment(str(value))}")

        tables.append(
            (
                "Scalar metrics",
                _table_context_subtitle(df_group, leadtime_unit=leadtime_unit),
                pivot,
                "__".join(filename_parts) if filename_parts else "all",
            )
        )

    return tables


def save_scalar_metric_tables(
    *,
    metrics: dict,
    variables: list[str],
    output_folder: Path,
    filters: dict | None = None,
    significant_digits: int = SCALAR_TABLE_SIGNIFICANT_DIGITS,
    metrics_keep: set[str] | None = None,
    filename_prefix: str = "scalar_metrics",
    title_prefix: str | None = None,
    row_index: tuple[str, ...] = SCALAR_TABLE_ROW_INDEX,
    column_index: tuple[str, ...] = SCALAR_TABLE_COLUMN_INDEX,
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

        parquet_path = output_folder / f"{filename_prefix}_{filename_stem}.parquet"
        image_path = output_folder / f"{filename_prefix}_{filename_stem}.png"
        table_df_display.to_parquet(parquet_path)
        save_scalar_metric_table_image(
            df=table_df,
            title=title,
            subtitle=subtitle,
            image_path=image_path,
            significant_digits=significant_digits,
            stat_label_overrides=stat_label_overrides,
            base_color=base_color,
        )
        saved_paths.extend([parquet_path, image_path])

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

    if not extents:
        return None
    if selected_region:
        if selected_region not in extents:
            raise ValueError(
                f"Selected region {selected_region!r} was not found in {exp_root}. "
                f"Available regions: {sorted(extents)}"
            )
        return extents[selected_region]
    if len(extents) > 1:
        raise ValueError(f"Multiple regions found in {exp_root}: {sorted(extents)}")
    return next(iter(extents.values()))


def save_scoreboard_plot(
    *,
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
) -> Path:
    scoreboard_frames: list[pd.DataFrame] = []
    diff_mode = "delta" if SCOREBOARD_MODE == "relative" else "no"
    outer_x_values = ["pr-fc"] if SCOREBOARD_MODE == "relative" else ["fc", "pr"]

    for metric_name, metric_type in SCOREBOARD_METRICS.items():
        scoreboard_frames.append(
            build_scalar_metric_df(
                metrics=metrics,
                variables=variables,
                metric_name=metric_name,
                metric_type=metric_type,
                diff=diff_mode,
                models=MODELS_DIFF if diff_mode == "delta" else None,
            )
        )

    df_scoreboard = pd.concat(scoreboard_frames, ignore_index=True)

    filters = {}
    for col, val in SCOREBOARD_FILTERS.items():
        if val is None:
            continue
        if col in {SCOREBOARD_ROW_AXIS, SCOREBOARD_COL_AXIS, "metric", "model"}:
            continue
        filters[col] = val

    plot_path = plot_folder / f"scoreboard_{SCOREBOARD_MODE}_{SCOREBOARD_ROW_AXIS}_vs_{SCOREBOARD_COL_AXIS}.png"
    plot_scoreboard(
        df=df_scoreboard,
        outer_y={"index": "metric", "values": list(SCOREBOARD_METRICS.keys())},
        outer_x={"index": "model", "values": outer_x_values},
        inner_y={
            "index": SCOREBOARD_ROW_AXIS,
            "label": format_axis_display_name(SCOREBOARD_ROW_AXIS, leadtime_unit=leadtime_unit),
        },
        inner_x={
            "index": SCOREBOARD_COL_AXIS,
            "label": format_axis_display_name(SCOREBOARD_COL_AXIS, leadtime_unit=leadtime_unit),
        },
        filters=filters,
        agg="mean",
        metric_cmaps=SCOREBOARD_METRIC_CMAPS,
        metric_vlims=SCOREBOARD_METRIC_VLIMS,
        metric_names=METRIC_DISPLAY_NAMES,
        display_name_maps={
            "variable": format_variable_display_name,
            "model": MODEL_DISPLAY_NAMES,
            "metric": METRIC_DISPLAY_NAMES,
        },
        annotate=SCOREBOARD_ANNOTATE,
        annotate_fmt="{:.2f}",
        annotate_color="auto",
        save_path=plot_path,
        title=f"Scoreboard ({SCOREBOARD_MODE})",
    )
    return plot_path


def save_metric_vs_diff_plot(
    *,
    df_nodiff,
    df_delta,
    plot_folder: Path,
    forecast_metric: str,
    diff_metric: str,
    leadtime_unit: str,
    region: str,
    filters: dict | None,
    variable_unit: str | None,
    shade_by: str,
    shade_label: str,
) -> Path:
    plot_folder.mkdir(parents=True, exist_ok=True)
    plot_variables = _ordered_unique_variables(
        [
            *df_nodiff.get("variable", pd.Series(dtype=object)).dropna().astype(str).tolist(),
            *df_delta.get("variable", pd.Series(dtype=object)).dropna().astype(str).tolist(),
        ]
    )
    forecast_metric_label = format_metric_label(forecast_metric, base_unit=variable_unit)
    diff_metric_label = format_metric_label(diff_metric, base_unit=variable_unit)
    region_suffix = f" - {region}" if region else ""

    fig, ax, merged = plot_metric_vs_diff(
        df_nodiff,
        df_delta,
        forecast_metric=forecast_metric,
        diff_metric=diff_metric,
        x_metric_name=f"delta_{diff_metric}",
        y_metric_name=forecast_metric,
        shade_by=shade_by,
        shade_label=shade_label,
        fit_lines=False,
        fit_ci=False,
        leadtime_unit=leadtime_unit,
        cmap_name=DIFF_PLOT_CMAP,
        variable_colors=get_variable_colors(variables=plot_variables),
        filters=filters,
        title=f"Forecast {forecast_metric_label} vs Δ {diff_metric_label} (ML-corrected - forecast){region_suffix}",
        xlabel=f"Δ {diff_metric_label} (ML-corrected - forecast)",
        ylabel=f"Forecast {forecast_metric_label}",
        legend_loc="lower left",
    )

    if merged.empty:
        available_y = sorted(df_nodiff["metric"].dropna().astype(str).unique())
        available_x = sorted(df_delta["metric"].dropna().astype(str).unique())
        raise ValueError(
            "Plot dataframe is empty after merging. "
            f"Requested forecast_metric={forecast_metric!r}, diff_metric={diff_metric!r}. "
            f"Available no-diff metrics={available_y}. "
            f"Available delta metrics={available_x}."
        )

    plot_path = plot_folder / f"{forecast_metric}_vs_delta_{diff_metric}.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return plot_path


def save_field_timeseries_plots(
    *,
    runs: dict[str, xr.Dataset],
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    disable_flox: bool = False,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_order = [model_name for model_name in LOAD_MODELS if model_name in runs]
    truth_model = LOAD_MODELS[0] if LOAD_MODELS else None

    options_context = (
        xr.set_options(use_flox=False, use_numbagg=False)
        if disable_flox
        else nullcontext()
    )
    with options_context:
        for variable in variables:
            timeseries_folder = get_variable_subfolder(
                plot_root=plot_folder,
                variable=variable,
                subfolder="timeseries",
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

            for selection, context_values in _context_selection_entries(context_reference):
                selected_data = {
                    model_name: da.sel({dim: value for dim, value in selection.items() if dim in da.dims}, drop=True)
                    for model_name, da in model_data.items()
                }
                selected_data = {model_name: da for model_name, da in selected_data.items() if da.size > 0}
                if not selected_data:
                    continue

                fc_has_spread = _has_multi_realization(selected_data.get("fc"))
                pr_has_spread = _has_multi_realization(selected_data.get("pr"))
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
                    for model_name in ("fc", "pr"):
                        if model_name not in selected_data:
                            continue
                        residual_da = _build_residual_da(selected_data[model_name], truth_da)
                        plot_jobs.append(
                            (
                                f"residual_{model_name}_minus_{truth_model}",
                                f"{base_title} residual ({MODEL_DISPLAY_NAMES.get(model_name, model_name)} - {MODEL_DISPLAY_NAMES.get(truth_model, truth_model)})",
                                {model_name: residual_da},
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
                            plot_members=False,
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
                    if show_legend:
                        ax.legend(fontsize=9)
                    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

                    plot_path = timeseries_folder / f"{plot_kind}_timeseries_{filename_context}.png"
                    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    saved_paths.append(plot_path)
                    debug_print(f"Saved {plot_kind} timeseries for region={plot_region or 'unknown'} to:", plot_path)

            if progress is not None and task_id is not None:
                progress.advance(task_id)

    return saved_paths


def save_field_and_metric_map_plots(
    *,
    runs: dict[str, xr.Dataset],
    metrics: dict[str, dict[str, xr.Dataset]],
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    region_extent: tuple[float, float, float, float] | None = None,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_order = [model_name for model_name in LOAD_MODELS if model_name in runs]

    for variable in variables:
        maps_folder = get_variable_subfolder(
            plot_root=plot_folder,
            variable=variable,
            subfolder="maps",
        )

        for model_name in model_order:
            ds = runs.get(model_name)
            if ds is None or variable not in ds.data_vars:
                continue

            da = _apply_xarray_filters(ds[variable], filters)
            dataset_region = _resolved_single_region_label(da, filters=filters)
            for selection, context_values in _context_selection_entries(da):
                da_sel = da.sel({dim: value for dim, value in selection.items() if dim in da.dims}, drop=True)
                if da_sel.size == 0:
                    continue
                da_sel = _reduce_extra_map_dims(da_sel)
                base_unit = _get_da_unit(da_sel)
                save_path = maps_folder / (
                    f"field_{model_name}_{_context_filename_suffix(context_values)}_temporal_mean_map.png"
                )
                plot_temporal_mean_map(
                    da_sel,
                    save_path=save_path,
                    title=_map_title(
                        format_variable_display_name(variable),
                        MODEL_DISPLAY_NAMES.get(model_name, str(model_name)),
                        "Temporal mean",
                        _context_title_suffix(context_values, leadtime_unit=leadtime_unit),
                    ),
                    extent=region_extent,
                    cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                    cmap=FIELD_MAP_CMAP,
                    lon_tick_step=MAP_LON_TICK_STEP,
                    lat_tick_step=MAP_LAT_TICK_STEP,
                )
                saved_paths.append(save_path)
                debug_print(f"Saved field map for model={model_name} region={dataset_region or 'unknown'} to:", save_path)

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
                    per_model_maps: dict[str, xr.DataArray] = {}
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
                        per_model_maps[str(model_name)] = da_sel
                        if base_unit is None:
                            base_unit = _get_da_unit(da_sel) or _get_da_unit(runs.get("an", xr.Dataset()), variable)

                    if not per_model_maps:
                        continue

                    shared_vmin, shared_vmax = _shared_map_limits(*per_model_maps.values())
                    context_suffix = _context_title_suffix(context_values, leadtime_unit=leadtime_unit)
                    metric_label = format_metric_label(metric_name, base_unit=base_unit)

                    for model_name, da_sel in per_model_maps.items():
                        map_region = _resolved_single_region_label(da_sel, filters=filters)
                        filename_parts = [metric_type, metric_name, model_name, _context_filename_suffix(context_values)]
                        save_path = maps_folder / f"{'_'.join(filename_parts)}_map.png"

                        plot_temporal_mean_map(
                            da_sel,
                            save_path=save_path,
                            title=_map_title(metric_label, MODEL_DISPLAY_NAMES.get(model_name, model_name), context_suffix),
                            extent=region_extent,
                            cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                            cmap=_resolve_metric_map_cmap(metric_name),
                            vmin=shared_vmin,
                            vmax=shared_vmax,
                            lon_tick_step=MAP_LON_TICK_STEP,
                            lat_tick_step=MAP_LAT_TICK_STEP,
                        )
                        saved_paths.append(save_path)
                        debug_print(
                            f"Saved metric map for metric={metric_name} model={model_name} region={map_region or 'unknown'} to:",
                            save_path,
                        )

                    if "fc" in per_model_maps and "pr" in per_model_maps:
                        diff_da = per_model_maps["pr"] - per_model_maps["fc"]
                        diff_region = _resolved_single_region_label(diff_da, filters=filters)
                        diff_vmin, diff_vmax = _symmetric_map_limits(diff_da)
                        diff_save_path = maps_folder / (
                            f"{metric_type}_{metric_name}_pr-fc_{_context_filename_suffix(context_values)}_map.png"
                        )
                        plot_temporal_mean_map(
                            diff_da,
                            save_path=diff_save_path,
                            title=_map_title(metric_label, MODEL_DISPLAY_NAMES["pr-fc"], context_suffix),
                            extent=region_extent,
                            cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                            cmap="RdBu_r",
                            vmin=diff_vmin,
                            vmax=diff_vmax,
                            lon_tick_step=MAP_LON_TICK_STEP,
                            lat_tick_step=MAP_LAT_TICK_STEP,
                        )
                        saved_paths.append(diff_save_path)
                        debug_print(
                            f"Saved diff map for metric={metric_name} model=pr-fc region={diff_region or 'unknown'} to:",
                            diff_save_path,
                        )

        if progress is not None and task_id is not None:
            progress.advance(task_id)

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

    metric_names = list(METRIC_PROFILE_METRICS)
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
        output_folder = _profile_output_folder(plot_root=plot_folder, x_axis=x_axis, variable=variable)
        variable_mask = slice(None) if variable is None else variable

        for metric_name, metric_type in METRIC_PROFILE_METRICS.items():
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

            x_values = _ordered_profile_axis_values(df_template, x_axis=x_axis, variables=variables)
            if not x_values:
                continue

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5.5), squeeze=True)
            x_positions = np.arange(len(x_values))

            if shade_by != x_axis and shade_by in df_template.columns:
                shade_values = [value for value in pd.unique(df_template[shade_by]) if pd.notna(value)]
            else:
                shade_values = [None]

            model_values = [value for value in pd.unique(df_template["model"]) if pd.notna(value)]
            plotted = False

            for model_name in model_values:
                base_color = model_colors.get(str(model_name), None)

                for j, shade_value in enumerate(shade_values):
                    alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))
                    shade_suffix = f", {shade_label}={shade_value}" if shade_value is not None else ""

                    if metric_type == "probabilistic":
                        df_model = df_plot_prob[df_plot_prob["model"] == model_name]
                        if shade_value is not None:
                            df_model = df_model[df_model[shade_by] == shade_value]
                        y_series = _mean_series_by_axis(df_model, x_axis=x_axis, x_values=x_values)
                        if not pd.notna(y_series).any():
                            continue
                        plotted = True
                        ax.plot(
                            x_positions,
                            y_series.values,
                            marker="o",
                            linewidth=2.0,
                            alpha=alpha,
                            color=base_color,
                            label=f"{model_name}{shade_suffix}",
                        )
                    elif metric_type == "deterministic_with_ensemble_overlay":
                        df_model_ens = df_plot_ens[df_plot_ens["model"] == model_name]
                        df_model_det = df_plot_det[df_plot_det["model"] == model_name]
                        if shade_value is not None:
                            df_model_ens = df_model_ens[df_model_ens[shade_by] == shade_value]
                            df_model_det = df_model_det[df_model_det[shade_by] == shade_value]

                        y_ens = _mean_series_by_axis(df_model_ens, x_axis=x_axis, x_values=x_values)
                        y_members = _member_matrix_by_axis(df_model_det, x_axis=x_axis, x_values=x_values)
                        if y_members is None and not pd.notna(y_ens).any():
                            continue

                        if y_members is not None:
                            y_member_values = y_members.to_numpy()
                            if pd.notna(y_member_values).any():
                                y_mean = np.nanmean(y_member_values, axis=1)
                                y_min = np.nanmin(y_member_values, axis=1)
                                y_max = np.nanmax(y_member_values, axis=1)
                                plotted = True
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

                                ax.fill_between(
                                    x_positions,
                                    y_min,
                                    y_max,
                                    color=base_color,
                                    alpha=0.12 * alpha,
                                )
                                ax.plot(
                                    x_positions,
                                    y_mean,
                                    color=base_color,
                                    linewidth=2.0,
                                    alpha=alpha,
                                    marker="o",
                                    markersize=5,
                                    label=f"{model_name} {metric_name}{shade_suffix}",
                                )

                        if pd.notna(y_ens).any():
                            plotted = True
                            ax.plot(
                                x_positions,
                                y_ens.values,
                                color=base_color,
                                linewidth=2.0,
                                alpha=alpha,
                                linestyle=":",
                                marker="o",
                                markersize=5,
                                label=f"{model_name} {metric_name} ensemble{shade_suffix}",
                            )
                    else:
                        raise ValueError(f"Unsupported metric profile plot mode {metric_type!r}")

            if not plotted:
                plt.close(fig)
                continue

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
            _set_title_and_subtitle(
                fig,
                ax,
                title=f"{metric_label} vs {format_axis_display_name(x_axis).lower()} ({title_subject})",
                subtitle=_plot_context_subtitle(leadtime_unit=leadtime_unit, filters=filters),
            )
            ax.set_xlabel(axis_label)
            ax.set_ylabel(metric_label)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [_format_profile_axis_tick(value, x_axis=x_axis) for value in x_values],
                rotation=30 if x_axis in {"train_period", "loss", "variable"} else 0,
                ha="right" if x_axis in {"train_period", "loss", "variable"} else "center",
            )
            ax.grid(True, linestyle="--", alpha=0.35)

            ymin, ymax = ax.get_ylim()
            if ymin <= 0 <= ymax:
                ax.axhline(0.0, color="0.3", lw=1.0, ls=":")

            ax.legend(fontsize=9)
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

            file_axis = _sanitize_filename_fragment(x_axis)
            if variable is None:
                plot_path = output_folder / f"{metric_name}_vs_{file_axis}.png"
            else:
                plot_path = output_folder / f"{metric_name}_vs_{file_axis}.png"
            fig.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved_paths.append(plot_path)

        if progress is not None and task_id is not None:
            progress.advance(task_id)

    return saved_paths


def main() -> None:
    dask_earthml = Dask(n_workers=5, memory_limit="100GiB")
    dask_earthml.start()
    client = dask_earthml.client
    base_url = "http://localhost:"
    print("Dask dashboard:", client.dashboard_link.replace("http://127.0.0.1:", base_url))

    print(f"Loading experiments from: {EXP_ROOT_FOLDER}")
    print_user_config_table()

    runs, metrics, _ = get_runs_and_metrics(
        exp_root=EXP_ROOT_FOLDER,
        type_data=TYPE_DATA,
        load_models=LOAD_MODELS,
        metric_names=METRIC_NAMES,
    )

    vars_from_fc = [v for v in runs["fc"].data_vars if v != "_has_var"]
    variable_units = _get_variable_units_from_runs(runs, vars_from_fc)
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Processed vars: {vars_from_fc}")
    leadtime_unit = infer_leadtime_unit_from_configs(EXP_ROOT_FOLDER)
    selected_region = _single_filter_value(PLOT_FILTERS, "region")
    region = infer_region_from_configs(EXP_ROOT_FOLDER, selected_region=selected_region)
    region_extent = infer_region_extent_from_configs(EXP_ROOT_FOLDER, selected_region=selected_region)
    print(f"Inferred leadtime unit: {leadtime_unit or 'unknown'}")
    print(f"Inferred region: {region or 'unknown'}")
    print(f"Inferred region extent: {region_extent or 'unknown'}")
    print_available_scalar_metrics(metrics=metrics)

    with build_progress() as progress:
        ts_task = progress.add_task("Generating field timeseries", total=len(vars_from_fc))
        field_timeseries_paths = save_field_timeseries_plots(
            runs=runs,
            variables=vars_from_fc,
            plot_folder=PLOT_FOLDER,
            leadtime_unit=leadtime_unit,
            filters=PLOT_FILTERS,
            disable_flox=DISABLE_FLOX_FOR_FIELD_TIMESERIES,
            progress=progress,
            task_id=ts_task,
        )
        for field_timeseries_path in field_timeseries_paths:
            debug_print("Saved field timeseries plot to:", field_timeseries_path)

        map_task = progress.add_task("Generating maps", total=len(vars_from_fc))
        field_map_paths = save_field_and_metric_map_plots(
            runs=runs,
            metrics=metrics,
            variables=vars_from_fc,
            plot_folder=PLOT_FOLDER,
            leadtime_unit=leadtime_unit,
            filters=PLOT_FILTERS,
            region_extent=region_extent,
            progress=progress,
            task_id=map_task,
        )
        for field_map_path in field_map_paths:
            debug_print("Saved map plot to:", field_map_path)

        if GLOBAL_KNOBS["enable_diff_plot"]:
            diff_task = progress.add_task("Generating metric-vs-delta plot", total=1)
            df_nodiff = build_scalar_metric_df(
                metrics=metrics,
                variables=vars_from_fc,
                metric_name=FORECAST_METRIC,
                metric_type=FORECAST_METRIC_TYPE,
                diff="no",
            )
            df_delta = build_scalar_metric_df(
                metrics=metrics,
                variables=vars_from_fc,
                metric_name=DIFF_METRIC,
                metric_type=DIFF_METRIC_TYPE,
                diff="delta",
                models=MODELS_DIFF,
            )

            plot_path = save_metric_vs_diff_plot(
                df_nodiff=df_nodiff,
                df_delta=df_delta,
                plot_folder=PLOT_FOLDER,
                forecast_metric=FORECAST_METRIC,
                diff_metric=DIFF_METRIC,
                leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                region=region,
                filters=PLOT_FILTERS,
                variable_unit=variable_units.get(_single_filter_value(PLOT_FILTERS, "variable")),
                shade_by=SHADE_BY,
                shade_label=SHADE_LABEL,
            )
            progress.advance(diff_task)
            debug_print("Saved leadtime-vs-global-metrics plot to:", plot_path)
        else:
            print("Skipping diff plot: GLOBAL_KNOBS['enable_diff_plot'] is False")

        if GLOBAL_KNOBS["enable_metric_profile_plots"]:
            profile_total = 1 if METRIC_PROFILE_AXIS == "variable" else len(vars_from_fc)
            profile_task = progress.add_task(
                f"Generating metric profiles ({METRIC_PROFILE_AXIS})",
                total=profile_total,
            )
            profile_plot_paths = save_metrics_vs_parameter_plots(
                metrics=metrics,
                variables=vars_from_fc,
                plot_folder=PLOT_FOLDER,
                leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
                variable_units=variable_units,
                x_axis=METRIC_PROFILE_AXIS,
                filters=PLOT_FILTERS,
                shade_by=SHADE_BY,
                shade_label=SHADE_LABEL,
                progress=progress,
                task_id=profile_task,
            )
            for profile_plot_path in profile_plot_paths:
                debug_print("Saved metric profile plot to:", profile_plot_path)
        else:
            print("Skipping metric profile plots: GLOBAL_KNOBS['enable_metric_profile_plots'] is False")

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

            table_task = progress.add_task("Generating scalar tables", total=len(vars_from_fc) + 1)
            for variable in vars_from_fc:
                variable_plot_folder = get_variable_plot_folder(plot_root=PLOT_FOLDER, variable=variable)
                base_color = get_variable_table_color(variable=variable, variables=vars_from_fc)
                scalar_table_paths = save_scalar_metric_tables(
                    metrics=metrics,
                    variables=[variable],
                    output_folder=variable_plot_folder,
                    filters=SCALAR_TABLE_FILTERS,
                    metrics_keep=raw_metrics,
                    filename_prefix="scalar_metrics",
                    leadtime_unit=leadtime_unit,
                    base_color=base_color,
                )
                progress.advance(table_task)
                for scalar_table_path in scalar_table_paths:
                    debug_print("Saved scalar metrics table to:", scalar_table_path)

            combined_normalized_table_paths = save_scalar_metric_tables(
                metrics=metrics,
                variables=vars_from_fc,
                output_folder=PLOT_FOLDER,
                filters=SCALAR_TABLE_FILTERS,
                metrics_keep=NORMALIZED_METRICS,
                filename_prefix="normalized_metrics",
                title_prefix="Normalized metrics",
                row_index=("variable", "metric"),
                column_index=("model", "stat"),
                stat_label_overrides={"prob": ""},
                leadtime_unit=leadtime_unit,
                base_color=get_variable_table_color(variable=vars_from_fc[0], variables=vars_from_fc),
            )
            progress.advance(table_task)
            for normalized_table_path in combined_normalized_table_paths:
                debug_print("Saved combined normalized metrics table to:", normalized_table_path)
        else:
            print("Skipping scalar tables: GLOBAL_KNOBS['enable_scalar_tables'] is False")

        if GLOBAL_KNOBS["enable_scoreboard"]:
            scoreboard_task = progress.add_task("Generating scoreboard", total=1)
            scoreboard_path = save_scoreboard_plot(
                metrics=metrics,
                variables=vars_from_fc,
                plot_folder=PLOT_FOLDER,
                leadtime_unit=leadtime_unit,
            )
            progress.advance(scoreboard_task)
            debug_print("Saved scoreboard plot to:", scoreboard_path)
        else:
            print("Skipping scoreboard: GLOBAL_KNOBS['enable_scoreboard'] is False")


if __name__ == "__main__":
    main()
