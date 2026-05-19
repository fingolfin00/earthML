from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from rich.progress import Progress

from ...logging import get_logger
from ...plots import plot_temporal_mean_map
from ...experiments.mlbc.load import harmonize_leadtime_int
from ...experiments.mlbc.utils import (
    project_mask_to_reference_grid,
    select_mask_for_indexers,
)

from .dataclasses import CalculateMetricsConfig
from .utils import (
    _apply_xarray_filters,
    _context_filename_suffix,
    _context_selection_entries,
    _context_title_suffix,
    _realization_dim,
    _get_da_unit,
    _format_variable_display_name,
    _ensure_grouped_output_folders,
    _get_variable_output_item_folder,
    _resolved_single_region_label,
    _format_metric_label,
)
from .constants import CONTEXT_AXES
from ..metrics.deterministic import DeterministicMetrics
from ..metrics.correlation import CorrelationMetrics
from ..metrics.probabilistic import ProbabilisticMetrics
from ..utils import METRIC_GROUP_DIM, _compute_metric_bundle


logger = get_logger(__name__)


MapLimitTuple = tuple[float | None, float | None]
LEADTIME_WINDOW_CONTEXT_KEY = "leadtime_window"
LEADTIME_AVG_WINDOW_CONTEXT_KEY = "leadtime_avg_window"


def _realization_filename_fragment(realization_value: object | None) -> str:
    if realization_value is None:
        return ""
    return f"_realization_{realization_value}"

def _realization_label(realization_value: object | None) -> str:
    if realization_value is None:
        return ""
    return f" | realization={realization_value}"

def _map_cbar_label(*, variable: str, unit: str | None) -> str:
    label = _format_variable_display_name(variable)
    if unit:
        return f"{label} [{unit}]"
    return label

def _metric_output_group(metric_type: str) -> str:
    return metric_type

def _shared_map_limits(*arrays: xr.DataArray) -> tuple[float | None, float | None]:
    finite_values: list[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array.values, dtype=float).ravel()
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return None, None
    combined = np.concatenate(finite_values)
    return float(combined.min()), float(combined.max())

def _symmetric_map_limits(array: xr.DataArray) -> tuple[float | None, float | None]:
    values = np.asarray(array.values, dtype=float).ravel()
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    bound = float(np.max(np.abs(finite)))
    return -bound, bound

def _resolve_metric_map_cmap(
    metric_name: str,
    *,
    config: CalculateMetricsConfig,
) -> str:
    return config.maps.metric_cmaps.get(metric_name, config.maps.metric_cmap_default)


def _is_map_limit_tuple(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2


def _metric_group_container_keys(metric_groupby_period: str | None) -> list[str]:
    if metric_groupby_period in {None, "none"}:
        return []

    period_key = str(metric_groupby_period)
    plural_map = {
        "month": "months",
        "season": "seasons",
    }
    plural_key = plural_map.get(period_key, f"{period_key}s")
    keys = [plural_key, period_key]
    return list(dict.fromkeys(keys))


def _resolve_map_limit_scope(
    scope_entry: object,
    *,
    metric_group_label: str | None,
    metric_groupby_period: str | None,
) -> MapLimitTuple | None:
    if _is_map_limit_tuple(scope_entry):
        return scope_entry

    if not isinstance(scope_entry, dict):
        return None

    if metric_group_label not in {None, "", "all"} and metric_groupby_period not in {None, "none"}:
        for container_key in _metric_group_container_keys(metric_groupby_period):
            grouped_entry = scope_entry.get(container_key)
            if not isinstance(grouped_entry, dict):
                continue
            for group_key in (metric_group_label, "*", "default", "all"):
                vlimits = grouped_entry.get(group_key)
                if _is_map_limit_tuple(vlimits):
                    return vlimits

    for default_key in ("all", "default", "*"):
        vlimits = scope_entry.get(default_key)
        if _is_map_limit_tuple(vlimits):
            return vlimits

    return None

def _configured_metric_map_limits(
    *,
    config: CalculateMetricsConfig,
    metric_name: str,
    variable: str,
    map_region: str | None,
    metric_group_label: str | None = None,
    metric_groupby_period: str | None = None,
    is_difference: bool = False,
) -> MapLimitTuple | None:
    metric_limits = config.maps.metric_limits.get(metric_name)
    if not isinstance(metric_limits, dict):
        return None

    candidate_variable_keys: list[str | None] = [variable, "*", "default"]
    candidate_region_keys: list[str | None] = []
    if map_region is not None:
        candidate_region_keys.append(map_region)
    candidate_region_keys.extend(["*", "default"])

    for variable_key in candidate_variable_keys:
        if variable_key is None:
            continue
        variable_limits = metric_limits.get(variable_key)
        if not isinstance(variable_limits, dict):
            continue
        for region_key in candidate_region_keys:
            if region_key is None:
                continue
            region_limits = variable_limits.get(region_key)
            if is_difference:
                if not isinstance(region_limits, dict):
                    continue
                vlimits = _resolve_map_limit_scope(
                    region_limits.get("diff"),
                    metric_group_label=metric_group_label,
                    metric_groupby_period=metric_groupby_period,
                )
            else:
                vlimits = _resolve_map_limit_scope(
                    region_limits,
                    metric_group_label=metric_group_label,
                    metric_groupby_period=metric_groupby_period,
                )
            if vlimits is not None:
                return vlimits

    return None

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
                da.sel({realization_dim: value}, drop=True).load(),
            )
            for value in da[realization_dim].values.tolist()
        ]
    return [(None, _collapse_realization_for_map(da).load())]

def _map_realization_count(
    da: xr.DataArray,
    *,
    realization_mode: str,
) -> int:
    realization_dim = _realization_dim(da)
    if realization_mode == "members" and realization_dim is not None:
        return int(da.sizes.get(realization_dim, 0))
    return 1

def _common_map_realization_count(
    reference_da: xr.DataArray,
    corrected_da: xr.DataArray,
    *,
    realization_mode: str,
) -> int:
    if realization_mode != "members":
        return 1

    reference_dim = _realization_dim(reference_da)
    corrected_dim = _realization_dim(corrected_da)
    if reference_dim is None or corrected_dim is None:
        return 1

    reference_values = set(reference_da[reference_dim].values.tolist())
    corrected_values = set(corrected_da[corrected_dim].values.tolist())
    return len(reference_values & corrected_values)

def _map_title(*parts: str | None) -> str:
    return " | ".join(part for part in parts if part)


def _grouped_map_title_context(
    context_values: dict[str, object],
    *,
    metric_group_label: str,
    metric_groupby_period: str | None,
) -> dict[str, object]:
    if metric_group_label == "all" or metric_groupby_period in {None, "none"}:
        return context_values
    return {
        key: value
        for key, value in context_values.items()
        if key not in {"leadtime", LEADTIME_WINDOW_CONTEXT_KEY, LEADTIME_AVG_WINDOW_CONTEXT_KEY}
    }


def _valid_time_mask_for_reference_dates(da: xr.DataArray) -> np.ndarray | None:
    time_dim = da.earthml.guessed_dims.time
    if time_dim is None or time_dim not in da.dims:
        return None

    valid = da.notnull()
    reduce_dims = [dim for dim in valid.dims if dim != time_dim]
    if reduce_dims:
        valid = valid.any(dim=reduce_dims)

    values = np.asarray(valid.values).reshape(-1)
    if values.size != da.sizes.get(time_dim, 0):
        return None
    return values.astype(bool, copy=False)


def _shift_reference_times(
    valid_times: pd.DatetimeIndex,
    *,
    leadtime_value: int,
    leadtime_unit: str,
) -> pd.DatetimeIndex:
    if leadtime_unit == "months":
        return valid_times - pd.DateOffset(months=leadtime_value)
    if leadtime_unit == "days":
        return valid_times - pd.to_timedelta(leadtime_value, unit="D")
    if leadtime_unit == "hours":
        return valid_times - pd.to_timedelta(leadtime_value, unit="h")
    return valid_times


def _reference_forecast_start_times(
    da: xr.DataArray,
    *,
    leadtime_value: object | None,
    leadtime_unit: str,
) -> pd.DatetimeIndex:
    time_dim = da.earthml.guessed_dims.time
    if time_dim is None or time_dim not in da.dims:
        return pd.DatetimeIndex([])

    if leadtime_value is None:
        return pd.DatetimeIndex([])

    try:
        leadtime_int = int(leadtime_value)
    except (TypeError, ValueError):
        return pd.DatetimeIndex([])

    time_mask = _valid_time_mask_for_reference_dates(da)

    if "reftime" in da.coords and time_dim in da["reftime"].dims:
        reference_values = pd.to_datetime(np.asarray(da["reftime"].values).reshape(-1), errors="coerce")
    else:
        valid_times = pd.to_datetime(np.asarray(da[time_dim].values).reshape(-1), errors="coerce")
        reference_values = _shift_reference_times(
            pd.DatetimeIndex(valid_times),
            leadtime_value=leadtime_int,
            leadtime_unit=leadtime_unit,
        )

    reference_index = pd.DatetimeIndex(reference_values)
    if time_mask is not None and len(time_mask) == len(reference_index):
        reference_index = reference_index[time_mask]
    return reference_index.dropna()


def _format_reference_forecast_start_label(
    da: xr.DataArray,
    *,
    leadtime_value: object | None,
    leadtime_unit: str,
    metric_group_label: str,
    metric_groupby_period: str | None,
) -> str | None:
    if metric_group_label == "all" or metric_groupby_period in {None, "none"}:
        return None

    reference_times = _reference_forecast_start_times(
        da,
        leadtime_value=leadtime_value,
        leadtime_unit=leadtime_unit,
    )
    if reference_times.empty:
        return None

    if metric_groupby_period in {"month", "season", "dayofyear"}:
        labels = pd.Index(reference_times.strftime("%b %d")).unique().tolist()
        if len(labels) == 1:
            return labels[0]
        if len(labels) <= 4:
            return ", ".join(labels)
        return f"{labels[0]} to {labels[-1]}"

    unique_times = pd.DatetimeIndex(reference_times.unique()).sort_values()
    if len(unique_times) == 1:
        return unique_times[0].strftime("%Y-%m-%d")
    return f"{unique_times[0]:%Y-%m-%d} to {unique_times[-1]:%Y-%m-%d}"


def _grouped_map_subtitle(
    da: xr.DataArray,
    *,
    context_values: dict[str, object],
    leadtime_unit: str,
    metric_group_label: str,
    metric_groupby_period: str | None,
) -> str | None:
    if metric_group_label == "all" or metric_groupby_period in {None, "none"}:
        return None

    leadtime_value = context_values.get("leadtime")
    if leadtime_value is None or str(leadtime_value) == "":
        return None

    parts: list[str] = []
    reference_label = _format_reference_forecast_start_label(
        da,
        leadtime_value=leadtime_value,
        leadtime_unit=leadtime_unit,
        metric_group_label=metric_group_label,
        metric_groupby_period=metric_groupby_period,
    )
    if reference_label:
        parts.append(f"Reference forecast start date={reference_label}")

    leadtime_label = _context_title_suffix({"leadtime": leadtime_value}, leadtime_unit=leadtime_unit)
    if leadtime_label:
        parts.append(leadtime_label)

    return " | ".join(parts) if parts else None


def _map_context_dims(*, group_leadtimes: bool) -> tuple[str, ...]:
    if not group_leadtimes:
        return CONTEXT_AXES
    return tuple(dim for dim in CONTEXT_AXES if dim != "leadtime")


def _metric_context_dims_for_data(
    data: xr.Dataset | xr.DataArray,
    *,
    base_context_dims: tuple[str, ...],
) -> tuple[str, ...]:
    dims = list(base_context_dims)
    for dim in data.dims:
        if dim == METRIC_GROUP_DIM or dim.endswith("_group"):
            if dim not in dims:
                dims.append(dim)
    return tuple(dims)


def _metric_context_selection_entries(
    data: xr.Dataset | xr.DataArray,
    *,
    base_context_dims: tuple[str, ...],
) -> list[tuple[dict[str, object], dict[str, object]]]:
    return _context_selection_entries(
        data,
        context_dims=_metric_context_dims_for_data(
            data,
            base_context_dims=base_context_dims,
        ),
    )


def _leadtime_window_label(window_values: Sequence[object]) -> str:
    if not window_values:
        return "all"
    if len(window_values) == 1:
        return str(window_values[0])
    return f"{window_values[0]}-{window_values[-1]}"


def _leadtime_window_entries(
    da: xr.DataArray,
    *,
    enabled: bool,
    window_size: int,
    window_stride: int,
) -> list[tuple[str | None, tuple[object, ...] | None]]:
    leadtime_dim = da.earthml.guessed_dims.leadtime
    if not enabled or leadtime_dim is None or leadtime_dim not in da.dims:
        return [(None, None)]

    leadtime_values = da[leadtime_dim].values.tolist()
    if not leadtime_values:
        return []

    if len(leadtime_values) <= window_size:
        window = tuple(leadtime_values)
        return [(_leadtime_window_label(window), window)]

    windows: list[tuple[str, tuple[object, ...]]] = []
    for start in range(0, len(leadtime_values) - window_size + 1, window_stride):
        window = tuple(leadtime_values[start:start + window_size])
        windows.append((_leadtime_window_label(window), window))
    return windows


def _window_context_values(
    context_values: dict[str, object],
    *,
    leadtime_window_label: str | None,
    window_context_key: str = LEADTIME_WINDOW_CONTEXT_KEY,
) -> dict[str, object]:
    if leadtime_window_label is None:
        return dict(context_values)

    out = {
        key: value
        for key, value in context_values.items()
        if key != "leadtime"
    }
    out[window_context_key] = leadtime_window_label
    return out


def _window_context_title(
    context_values: dict[str, object],
    *,
    leadtime_unit: str,
) -> str | None:
    base_context = {
        key: value
        for key, value in context_values.items()
        if key not in {LEADTIME_WINDOW_CONTEXT_KEY, LEADTIME_AVG_WINDOW_CONTEXT_KEY}
    }
    parts: list[str] = []
    base_title = _context_title_suffix(base_context, leadtime_unit=leadtime_unit)
    if base_title:
        parts.append(base_title)
    for key, label_prefix in (
        (LEADTIME_WINDOW_CONTEXT_KEY, "Lead time window"),
        (LEADTIME_AVG_WINDOW_CONTEXT_KEY, "Lead time average window"),
    ):
        leadtime_window_label = context_values.get(key)
        if leadtime_window_label is None:
            continue
        if leadtime_unit:
            parts.append(f"{label_prefix} [{leadtime_unit}]={leadtime_window_label}")
        else:
            parts.append(f"{label_prefix}={leadtime_window_label}")
    return " | ".join(parts) if parts else None


def _window_context_subtitle(
    da: xr.DataArray,
    *,
    context_values: dict[str, object],
    leadtime_unit: str,
    metric_group_label: str,
    metric_groupby_period: str | None,
) -> str | None:
    parts: list[str] = []
    base_subtitle = _grouped_map_subtitle(
        da,
        context_values=context_values,
        leadtime_unit=leadtime_unit,
        metric_group_label=metric_group_label,
        metric_groupby_period=metric_groupby_period,
    )
    if base_subtitle:
        parts.append(base_subtitle)

    for key, label_prefix in (
        (LEADTIME_WINDOW_CONTEXT_KEY, "Lead time window"),
        (LEADTIME_AVG_WINDOW_CONTEXT_KEY, "Lead time average window"),
    ):
        leadtime_window_label = context_values.get(key)
        if leadtime_window_label is None:
            continue
        if leadtime_unit:
            parts.append(f"{label_prefix} [{leadtime_unit}]={leadtime_window_label}")
        else:
            parts.append(f"{label_prefix}={leadtime_window_label}")

    return " | ".join(parts) if parts else None


def _select_leadtime_window(
    data: xr.Dataset | xr.DataArray,
    *,
    selection: dict[str, object],
    leadtime_window_values: tuple[object, ...] | None,
) -> xr.Dataset | xr.DataArray:
    out = data.sel({dim: value for dim, value in selection.items() if dim in data.dims}, drop=True)
    leadtime_dim = out.earthml.guessed_dims.leadtime
    if leadtime_window_values is not None and leadtime_dim is not None and leadtime_dim in out.dims:
        selector = xr.DataArray(
            np.isin(out[leadtime_dim].values, list(leadtime_window_values)),
            dims=out[leadtime_dim].dims,
            coords=out[leadtime_dim].coords,
        )
        out = out.where(selector, drop=True)
    return out


def _reduce_field_leadtime_window(
    da: xr.DataArray,
    *,
    leadtime_window_values: tuple[object, ...] | None,
) -> xr.DataArray:
    out = da
    leadtime_dim = out.earthml.guessed_dims.leadtime
    if leadtime_window_values is not None and leadtime_dim is not None and leadtime_dim in out.dims:
        out = out.mean(dim=leadtime_dim, skipna=True)
    return out


def _average_metric_leadtime_window(
    da: xr.DataArray,
    *,
    leadtime_window_values: tuple[object, ...] | None,
) -> xr.DataArray:
    out = da
    leadtime_dim = out.earthml.guessed_dims.leadtime
    if leadtime_window_values is not None and leadtime_dim is not None and leadtime_dim in out.dims:
        selector = xr.DataArray(
            np.isin(out[leadtime_dim].values, list(leadtime_window_values)),
            dims=out[leadtime_dim].dims,
            coords=out[leadtime_dim].coords,
        )
        out = out.where(selector, drop=True)
        out = out.mean(dim=leadtime_dim, skipna=True)
    return out


def _available_metric_map_inventory(
    metrics: dict[str, dict[str, xr.Dataset]],
) -> dict[str, list[str]]:
    inventory: dict[str, list[str]] = {}
    for metric_type, section_dict in metrics.items():
        ds_map = section_dict.get("map", xr.Dataset())
        if not isinstance(ds_map, xr.Dataset) or "metric" not in ds_map.coords:
            continue
        inventory[metric_type] = [str(value) for value in ds_map["metric"].values.tolist()]
    return inventory


def _select_all_leadtime_group(
    data: xr.Dataset | xr.DataArray,
) -> tuple[xr.Dataset | xr.DataArray, bool]:
    leadtime_group_dim = "leadtime_group"
    if leadtime_group_dim not in data.dims:
        return data, False

    available = [str(value) for value in data[leadtime_group_dim].values.tolist()]
    if "all" not in available:
        return data, False

    return data.sel({leadtime_group_dim: ["all"]}), True


def _grouped_metric_cache_key(
    *,
    selection: dict[str, object],
    leadtime_window_values: tuple[object, ...] | None,
) -> tuple[tuple[tuple[str, object], ...], tuple[object, ...] | None]:
    return (
        tuple(sorted(selection.items())),
        leadtime_window_values,
    )


def _compute_grouped_window_metric_bundle(
    *,
    runs: dict[str, xr.Dataset],
    selection: dict[str, object],
    leadtime_window_values: tuple[object, ...] | None,
    metric_inventory: dict[str, list[str]],
    config: CalculateMetricsConfig,
    clim_period: str,
) -> dict[str, dict[str, xr.Dataset]]:
    truth_model = config.models.truth_model
    if truth_model not in runs:
        return {}

    truth_data = _select_leadtime_window(
        runs[truth_model],
        selection=selection,
        leadtime_window_values=leadtime_window_values,
    )
    if not truth_data.data_vars or not bool(truth_data.to_array().notnull().any()):
        return {}

    model_names_this_run: list[str] = []
    model_data: list[xr.Dataset] = []
    for model_name in config.models.comparison_models:
        ds = runs.get(model_name)
        if ds is None:
            continue
        ds = _select_leadtime_window(
            ds,
            selection=selection,
            leadtime_window_values=leadtime_window_values,
        )
        if not ds.data_vars or not bool(ds.to_array().notnull().any()):
            continue
        model_names_this_run.append(model_name)
        model_data.append(ds)

    if not model_data:
        return {}

    mask_indexers = dict(selection)
    leadtime_dim = truth_data.earthml.guessed_dims.leadtime
    if leadtime_window_values is not None and leadtime_dim is not None:
        mask_indexers[leadtime_dim] = list(leadtime_window_values)
    mask_data = project_mask_to_reference_grid(
        select_mask_for_indexers(runs.get("mask"), mask_indexers),
        truth_data,
    )

    to_harmonize = [truth_data, *model_data]
    if mask_data is not None and "leadtime" in mask_data.coords:
        to_harmonize.append(mask_data)
        has_mask = True
    else:
        has_mask = False

    try:
        harmonized = harmonize_leadtime_int(*to_harmonize, coord="leadtime", method="int_source")
        truth_data = harmonized[0]
        model_data = harmonized[1:1 + len(model_data)]
        if has_mask:
            mask_data = harmonized[-1]
    except Exception:
        pass

    deterministic = DeterministicMetrics(
        truth_data=truth_data,
        model_data=model_data,
        truth_name=truth_model,
        model_names=model_names_this_run,
        clim_data=None,
        mask_data=mask_data,
    )
    correlation = CorrelationMetrics(
        truth_data=truth_data,
        model_data=model_data,
        truth_name=truth_model,
        model_names=model_names_this_run,
        clim_data=None,
        mask_data=mask_data,
    )
    try:
        probabilistic = ProbabilisticMetrics(
            truth_data=truth_data,
            model_data=model_data,
            truth_name=truth_model,
            model_names=model_names_this_run,
            clim_data=None,
            mask_data=mask_data,
        )
    except ValueError:
        probabilistic = None

    metric_names = sorted({metric_name for names in metric_inventory.values() for metric_name in names})
    leadtime_reduce_dim = truth_data.earthml.guessed_dims.leadtime
    metric_mean_extra_dims = None
    if leadtime_reduce_dim is not None and leadtime_reduce_dim in truth_data.dims:
        metric_mean_extra_dims = {"map": (leadtime_reduce_dim,)}

    return _compute_metric_bundle(
        deterministic=deterministic,
        correlation=correlation,
        probabilistic=probabilistic,
        norm="std",
        clim_period=clim_period,
        metric_names=metric_names,
        metric_sections=("map",),
        metric_types=tuple(metric_inventory),
        metric_mean_extra_dims=metric_mean_extra_dims,
    )

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


def save_field_and_metric_map_plots(
    load_models: list,
    *,
    runs: dict[str, xr.Dataset],
    metrics: dict[str, dict[str, xr.Dataset]],
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    config: CalculateMetricsConfig,
    realization_mode: Literal["mean", "members"] = "mean",
    clim_period: str = "month",
    metric_group_label: str = "all",
    metric_groupby_period: str | None = None,
    region_extents: dict[str, tuple[float, float, float, float]] | None = None,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_order = [model_name for model_name in load_models if model_name in runs]
    group_leadtimes = bool(getattr(config.maps, "group_leadtimes", False))
    average_metric_leadtimes = bool(getattr(config.maps, "average_leadtimes", False))
    metric_realization_mode = realization_mode
    metric_window_leadtimes = group_leadtimes or average_metric_leadtimes
    window_size = int(getattr(config.maps, "leadtime_window_size", 3))
    window_stride = int(getattr(config.maps, "leadtime_window_stride", 1))
    field_context_dims = _map_context_dims(group_leadtimes=group_leadtimes)
    metric_context_dims = _map_context_dims(group_leadtimes=metric_window_leadtimes)
    metric_window_context_key = (
        LEADTIME_AVG_WINDOW_CONTEXT_KEY
        if average_metric_leadtimes
        else LEADTIME_WINDOW_CONTEXT_KEY
    )
    metric_inventory = _available_metric_map_inventory(metrics)
    grouped_metric_cache: dict[
        tuple[tuple[tuple[str, object], ...], tuple[object, ...] | None],
        dict[str, dict[str, xr.Dataset]],
    ] = {}
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
                for selection, context_values in _context_selection_entries(da, context_dims=field_context_dims):
                    da_context = da.sel({dim: value for dim, value in selection.items() if dim in da.dims}, drop=True)
                    if da_context.size == 0:
                        continue
                    for _, leadtime_window_values in _leadtime_window_entries(
                        da_context,
                        enabled=group_leadtimes,
                        window_size=window_size,
                        window_stride=window_stride,
                    ):
                        da_sel = _select_leadtime_window(
                            da_context,
                            selection={},
                            leadtime_window_values=leadtime_window_values,
                        )
                        if da_sel.size == 0:
                            continue
                        da_sel = _reduce_field_leadtime_window(
                            da_sel,
                            leadtime_window_values=leadtime_window_values,
                        )
                        da_sel = _reduce_extra_map_dims(da_sel)
                        total += _map_realization_count(da_sel, realization_mode=realization_mode)

            if group_leadtimes:
                context_model_name = (
                    config.models.reference_model
                    if config.models.reference_model in runs
                    else config.models.truth_model
                )
                context_ds = runs.get(context_model_name)
                if context_ds is None or variable not in context_ds.data_vars:
                    continue
                da_context_source = _apply_xarray_filters(context_ds[variable], filters)
                for selection, context_values in _context_selection_entries(
                    da_context_source,
                    context_dims=metric_context_dims,
                ):
                    da_context = da_context_source.sel(
                        {dim: value for dim, value in selection.items() if dim in da_context_source.dims},
                        drop=True,
                    )
                    if da_context.size == 0:
                        continue
                    for _, leadtime_window_values in _leadtime_window_entries(
                        da_context,
                        enabled=True,
                        window_size=window_size,
                        window_stride=window_stride,
                    ):
                        cache_key = _grouped_metric_cache_key(
                            selection=selection,
                            leadtime_window_values=leadtime_window_values,
                        )
                        if cache_key not in grouped_metric_cache:
                            grouped_metric_cache[cache_key] = _compute_grouped_window_metric_bundle(
                                runs=runs,
                                selection=selection,
                                leadtime_window_values=leadtime_window_values,
                                metric_inventory=metric_inventory,
                                config=config,
                                clim_period=clim_period,
                            )
                        grouped_metrics = grouped_metric_cache[cache_key]
                        for metric_type, section_dict in grouped_metrics.items():
                            ds_map = section_dict.get("map", xr.Dataset())
                            if not isinstance(ds_map, xr.Dataset) or variable not in ds_map.data_vars:
                                continue
                            da_var = ds_map[variable]
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
                                per_model_maps: dict[str, xr.DataArray] = {}
                                for model_name in model_values:
                                    da_model = da_metric.sel(model=model_name, drop=True) if model_name is not None else da_metric
                                    if da_model.size == 0:
                                        continue
                                    da_model = _reduce_extra_map_dims(da_model)
                                    per_model_maps[str(model_name)] = da_model
                                if not per_model_maps:
                                    continue
                                total += sum(
                                    _map_realization_count(da_model, realization_mode=metric_realization_mode)
                                    for da_model in per_model_maps.values()
                                )
                                reference_model = config.models.reference_model
                                corrected_model = config.models.corrected_model
                                if corrected_model is not None and reference_model in per_model_maps and corrected_model in per_model_maps:
                                    total += _common_map_realization_count(
                                        per_model_maps[reference_model],
                                        per_model_maps[corrected_model],
                                        realization_mode=metric_realization_mode,
                                    )
                continue

            if average_metric_leadtimes:
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
                        da_metric, using_all_leadtime_group = _select_all_leadtime_group(da_metric)
                        for selection, context_values in _metric_context_selection_entries(
                            da_metric,
                            base_context_dims=metric_context_dims,
                        ):
                            da_context = da_metric.sel(
                                {dim: value for dim, value in selection.items() if dim in da_metric.dims},
                                drop=True,
                            )
                            if da_context.size == 0:
                                continue
                            for _, leadtime_window_values in _leadtime_window_entries(
                                da_context,
                                enabled=not using_all_leadtime_group,
                                window_size=window_size,
                                window_stride=window_stride,
                            ):
                                per_model_maps: dict[str, xr.DataArray] = {}
                                for model_name in model_values:
                                    da_model = da_context.sel(model=model_name, drop=True) if model_name is not None else da_context
                                    if da_model.size == 0:
                                        continue
                                    da_model = _average_metric_leadtime_window(
                                        da_model,
                                        leadtime_window_values=leadtime_window_values,
                                    )
                                    da_model = _reduce_extra_map_dims(da_model)
                                    per_model_maps[str(model_name)] = da_model
                                if not per_model_maps:
                                    continue
                                total += sum(
                                    _map_realization_count(da_model, realization_mode=metric_realization_mode)
                                    for da_model in per_model_maps.values()
                                )
                                reference_model = config.models.reference_model
                                corrected_model = config.models.corrected_model
                                if corrected_model is not None and reference_model in per_model_maps and corrected_model in per_model_maps:
                                    total += _common_map_realization_count(
                                        per_model_maps[reference_model],
                                        per_model_maps[corrected_model],
                                        realization_mode=metric_realization_mode,
                                    )
                continue

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
                    for selection, context_values in _metric_context_selection_entries(
                        da_metric,
                        base_context_dims=CONTEXT_AXES,
                    ):
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
                            per_model_maps[str(model_name)] = da_sel

                        if not per_model_maps:
                            continue

                        total += sum(
                            _map_realization_count(da_model, realization_mode=metric_realization_mode)
                            for da_model in per_model_maps.values()
                        )

                        reference_model = config.models.reference_model
                        corrected_model = config.models.corrected_model
                        if reference_model in per_model_maps and corrected_model in per_model_maps:
                            total += _common_map_realization_count(
                                per_model_maps[reference_model],
                                per_model_maps[corrected_model],
                                realization_mode=metric_realization_mode,
                            )

        return total

    total_plots = _count_total_map_plots()
    if progress is not None and task_id is not None:
        progress.update(task_id, total=total_plots, completed=0, description="Generating maps")

    for variable in variables:
        logger.info(f"Map variable: {variable}")
        _ensure_grouped_output_folders(
            plot_root=plot_folder,
            variable=variable,
            subfolder="maps",
        )
        field_maps_folder = _get_variable_output_item_folder(
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
            for selection, context_values in _context_selection_entries(da, context_dims=field_context_dims):
                da_context = da.sel({dim: value for dim, value in selection.items() if dim in da.dims}, drop=True)
                if da_context.size == 0:
                    continue
                for leadtime_window_label, leadtime_window_values in _leadtime_window_entries(
                    da_context,
                    enabled=group_leadtimes,
                    window_size=window_size,
                    window_stride=window_stride,
                ):
                    da_sel = _select_leadtime_window(
                        da_context,
                        selection={},
                        leadtime_window_values=leadtime_window_values,
                    )
                    if da_sel.size == 0:
                        continue
                    da_sel = _reduce_field_leadtime_window(
                        da_sel,
                        leadtime_window_values=leadtime_window_values,
                    )
                    da_sel = _reduce_extra_map_dims(da_sel)
                    base_unit = _get_da_unit(da_sel)
                    window_context_values = _window_context_values(
                        context_values,
                        leadtime_window_label=leadtime_window_label,
                    )
                    context_title_values = _grouped_map_title_context(
                        window_context_values,
                        metric_group_label=metric_group_label,
                        metric_groupby_period=metric_groupby_period,
                    )
                    for realization_value, da_plot in _map_realization_slices(
                        da_sel,
                        realization_mode=realization_mode,
                    ):
                        plot_region = _resolved_single_region_label(da_plot, filters=filters)
                        extent = _region_extent_for_plot(
                            data=da_plot,
                            context_values=window_context_values,
                            filters=filters,
                            region_extents=region_extents,
                        )
                        save_path = field_maps_folder / (
                            f"field_{model_name}_{_context_filename_suffix(window_context_values)}"
                            f"{_realization_filename_fragment(realization_value)}_temporal_mean_map.png"
                        )
                        if progress is not None and task_id is not None:
                            progress.update(
                                task_id,
                                description=f"Maps | {plot_region or 'unknown'} | {variable} | field | {model_name}",
                            )
                        plot_temporal_mean_map(
                            da_plot,
                            save_path=save_path,
                            title=_map_title(
                                _format_variable_display_name(variable),
                                config.models.display_names.get(model_name, str(model_name)),
                                f"Temporal mean{_realization_label(realization_value)}",
                                _window_context_title(context_title_values, leadtime_unit=leadtime_unit),
                            ),
                            subtitle=_window_context_subtitle(
                                da_plot,
                                context_values=window_context_values,
                                leadtime_unit=leadtime_unit,
                                metric_group_label=metric_group_label,
                                metric_groupby_period=metric_groupby_period,
                            ),
                            extent=extent,
                            cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                            cmap=config.maps.field_cmap,
                            lon_tick_step=config.maps.lon_tick_step,
                            lat_tick_step=config.maps.lat_tick_step,
                        )
                        saved_paths.append(save_path)
                        if progress is not None and task_id is not None:
                            progress.advance(task_id)
                        logger.debug(
                            f"Saved field map for model={model_name} region={plot_region or 'unknown'} to: {save_path}"
                        )

        if group_leadtimes:
            context_model_name = (
                config.models.reference_model
                if config.models.reference_model in runs
                else config.models.truth_model
            )
            context_ds = runs.get(context_model_name)
            if context_ds is None or variable not in context_ds.data_vars:
                continue

            da_context_source = _apply_xarray_filters(context_ds[variable], filters)
            for selection, context_values in _context_selection_entries(
                da_context_source,
                context_dims=metric_context_dims,
            ):
                da_context = da_context_source.sel(
                    {dim: value for dim, value in selection.items() if dim in da_context_source.dims},
                    drop=True,
                )
                if da_context.size == 0:
                    continue

                for leadtime_window_label, leadtime_window_values in _leadtime_window_entries(
                    da_context,
                    enabled=True,
                    window_size=window_size,
                    window_stride=window_stride,
                ):
                    window_context_values = _window_context_values(
                        context_values,
                        leadtime_window_label=leadtime_window_label,
                        window_context_key=metric_window_context_key,
                    )
                    cache_key = _grouped_metric_cache_key(
                        selection=selection,
                        leadtime_window_values=leadtime_window_values,
                    )
                    if cache_key not in grouped_metric_cache:
                        grouped_metric_cache[cache_key] = _compute_grouped_window_metric_bundle(
                            runs=runs,
                            selection=selection,
                            leadtime_window_values=leadtime_window_values,
                            metric_inventory=metric_inventory,
                            config=config,
                            clim_period=clim_period,
                        )
                    grouped_metrics = grouped_metric_cache[cache_key]

                    for metric_type, section_dict in grouped_metrics.items():
                        ds_map = section_dict.get("map", xr.Dataset())
                        if not isinstance(ds_map, xr.Dataset) or variable not in ds_map.data_vars:
                            continue

                        da_var = ds_map[variable]
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
                            per_model_maps: dict[str, dict[object | None, xr.DataArray]] = {}
                            base_unit: str | None = None

                            for model_name in model_values:
                                da_model = da_metric.sel(model=model_name, drop=True) if model_name is not None else da_metric
                                if da_model.size == 0:
                                    continue
                                da_model = _reduce_extra_map_dims(da_model)
                                per_model_maps[str(model_name)] = dict(
                                    _map_realization_slices(
                                        da_model,
                                        realization_mode=metric_realization_mode,
                                    )
                                )
                                if base_unit is None:
                                    base_unit = _get_da_unit(da_model) or _get_da_unit(
                                        runs.get(config.models.truth_model, xr.Dataset()),
                                        variable,
                                    )

                            if not per_model_maps:
                                continue

                            context_title_values = _grouped_map_title_context(
                                window_context_values,
                                metric_group_label=metric_group_label,
                                metric_groupby_period=metric_groupby_period,
                            )
                            context_title = _window_context_title(
                                context_title_values,
                                leadtime_unit=leadtime_unit,
                            )
                            sample_da = next(iter(next(iter(per_model_maps.values())).values()))
                            extent = _region_extent_for_plot(
                                data=sample_da,
                                context_values=window_context_values,
                                filters=filters,
                                region_extents=region_extents,
                            )
                            metric_label = _format_metric_label(metric_name, base_unit=base_unit)
                            shared_vmin, shared_vmax = _shared_map_limits(
                                *(da for per_model in per_model_maps.values() for da in per_model.values())
                            )

                            metric_folder = _get_variable_output_item_folder(
                                plot_root=plot_folder,
                                variable=variable,
                                subfolder="maps",
                                output_group=_metric_output_group(metric_type),
                                item_name=metric_name,
                            )

                            for model_name, realization_maps in per_model_maps.items():
                                for realization_value, da_sel in realization_maps.items():
                                    map_region = _resolved_single_region_label(da_sel, filters=filters)
                                    filename_parts = [
                                        metric_type,
                                        metric_name,
                                        model_name,
                                        _context_filename_suffix(window_context_values),
                                    ]
                                    save_path = metric_folder / (
                                        f"{'_'.join(filename_parts)}"
                                        f"{_realization_filename_fragment(realization_value)}_map.png"
                                    )

                                    if progress is not None and task_id is not None:
                                        progress.update(
                                            task_id,
                                            description=f"Maps | {map_region or 'unknown'} | {variable} | {metric_type} | {metric_name}",
                                        )

                                    vlimits = _configured_metric_map_limits(
                                        config=config,
                                        metric_name=metric_name,
                                        variable=variable,
                                        map_region=map_region,
                                        metric_group_label=metric_group_label,
                                        metric_groupby_period=metric_groupby_period,
                                    )
                                    if vlimits is not None:
                                        vmin, vmax = vlimits[0], vlimits[1]
                                    else:
                                        vmin, vmax = shared_vmin, shared_vmax

                                    plot_temporal_mean_map(
                                        da_sel,
                                        save_path=save_path,
                                        title=_map_title(
                                            metric_label,
                                            config.models.display_names.get(model_name, model_name),
                                            f"{context_title}{_realization_label(realization_value)}"
                                            if context_title else _realization_label(realization_value).lstrip(" |"),
                                        ),
                                        subtitle=_window_context_subtitle(
                                            da_sel,
                                            context_values=window_context_values,
                                            leadtime_unit=leadtime_unit,
                                            metric_group_label=metric_group_label,
                                            metric_groupby_period=metric_groupby_period,
                                        ),
                                        extent=extent,
                                        cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                                        cmap=_resolve_metric_map_cmap(metric_name, config=config),
                                        vmin=vmin,
                                        vmax=vmax,
                                        lon_tick_step=config.maps.lon_tick_step,
                                        lat_tick_step=config.maps.lat_tick_step,
                                    )
                                    saved_paths.append(save_path)
                                    if progress is not None and task_id is not None:
                                        progress.advance(task_id)
                                    logger.debug(
                                        f"Saved grouped metric map for metric={metric_name} "
                                        f"model={model_name} region={map_region or 'unknown'} to: {save_path}"
                                    )

                            reference_model = config.models.reference_model
                            corrected_model = config.models.corrected_model
                            difference_model = config.models.difference_model
                            if corrected_model is not None and reference_model in per_model_maps and corrected_model in per_model_maps:
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
                                    fallback_diff_vlimits = _symmetric_map_limits(
                                        xr.concat([diff_da for _, diff_da in diff_maps], dim="_diff_map")
                                    )
                                    for realization_value, diff_da in diff_maps:
                                        diff_region = _resolved_single_region_label(diff_da, filters=filters)
                                        diff_vlimits = _configured_metric_map_limits(
                                            config=config,
                                            metric_name=metric_name,
                                            variable=variable,
                                            map_region=diff_region,
                                            metric_group_label=metric_group_label,
                                            metric_groupby_period=metric_groupby_period,
                                            is_difference=True,
                                        )
                                        if diff_vlimits is not None:
                                            diff_vmin, diff_vmax = diff_vlimits
                                        else:
                                            diff_vmin, diff_vmax = fallback_diff_vlimits
                                        diff_save_path = metric_folder / (
                                            f"{metric_type}_{metric_name}_{difference_model}_{_context_filename_suffix(window_context_values)}"
                                            f"{_realization_filename_fragment(realization_value)}_map.png"
                                        )
                                        if progress is not None and task_id is not None:
                                            progress.update(
                                                task_id,
                                                description=f"Maps | {diff_region or 'unknown'} | {variable} | diff | {metric_name}",
                                            )
                                        plot_temporal_mean_map(
                                            diff_da,
                                            save_path=diff_save_path,
                                            title=_map_title(
                                                metric_label,
                                                config.models.display_names.get(difference_model, difference_model),
                                                f"{context_title}{_realization_label(realization_value)}"
                                                if context_title else _realization_label(realization_value).lstrip(" |"),
                                            ),
                                            subtitle=_window_context_subtitle(
                                                diff_da,
                                                context_values=window_context_values,
                                                leadtime_unit=leadtime_unit,
                                                metric_group_label=metric_group_label,
                                                metric_groupby_period=metric_groupby_period,
                                            ),
                                            extent=extent,
                                            cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                                            cmap="RdBu_r",
                                            vmin=diff_vmin,
                                            vmax=diff_vmax,
                                            lon_tick_step=config.maps.lon_tick_step,
                                            lat_tick_step=config.maps.lat_tick_step,
                                        )
                                        saved_paths.append(diff_save_path)
                                        if progress is not None and task_id is not None:
                                            progress.advance(task_id)
                                        logger.debug(
                                            f"Saved grouped diff map for metric={metric_name} "
                                            f"model={difference_model} region={diff_region or 'unknown'} to: {diff_save_path}"
                                        )
            continue

        if average_metric_leadtimes:
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
                    da_metric, using_all_leadtime_group = _select_all_leadtime_group(da_metric)
                    for selection, context_values in _metric_context_selection_entries(
                        da_metric,
                        base_context_dims=metric_context_dims,
                    ):
                        da_context = da_metric.sel(
                            {dim: value for dim, value in selection.items() if dim in da_metric.dims},
                            drop=True,
                        )
                        if da_context.size == 0:
                            continue

                        for leadtime_window_label, leadtime_window_values in _leadtime_window_entries(
                            da_context,
                            enabled=not using_all_leadtime_group,
                            window_size=window_size,
                            window_stride=window_stride,
                        ):
                            window_context_values = _window_context_values(
                                context_values,
                                leadtime_window_label=leadtime_window_label,
                                window_context_key=metric_window_context_key,
                            )
                            per_model_maps: dict[str, dict[object | None, xr.DataArray]] = {}
                            base_unit: str | None = None

                            for model_name in model_values:
                                da_model = da_context.sel(model=model_name, drop=True) if model_name is not None else da_context
                                if da_model.size == 0:
                                    continue
                                da_model = _average_metric_leadtime_window(
                                    da_model,
                                    leadtime_window_values=leadtime_window_values,
                                )
                                da_model = _reduce_extra_map_dims(da_model)
                                per_model_maps[str(model_name)] = dict(
                                    _map_realization_slices(
                                        da_model,
                                        realization_mode=metric_realization_mode,
                                    )
                                )
                                if base_unit is None:
                                    base_unit = _get_da_unit(da_model) or _get_da_unit(
                                        runs.get(config.models.truth_model, xr.Dataset()),
                                        variable,
                                    )

                            if not per_model_maps:
                                continue

                            context_title_values = _grouped_map_title_context(
                                window_context_values,
                                metric_group_label=metric_group_label,
                                metric_groupby_period=metric_groupby_period,
                            )
                            context_title = _window_context_title(
                                context_title_values,
                                leadtime_unit=leadtime_unit,
                            )
                            sample_da = next(iter(next(iter(per_model_maps.values())).values()))
                            extent = _region_extent_for_plot(
                                data=sample_da,
                                context_values=window_context_values,
                                filters=filters,
                                region_extents=region_extents,
                            )
                            metric_label = _format_metric_label(metric_name, base_unit=base_unit)
                            shared_vmin, shared_vmax = _shared_map_limits(
                                *(da for per_model in per_model_maps.values() for da in per_model.values())
                            )

                            metric_folder = _get_variable_output_item_folder(
                                plot_root=plot_folder,
                                variable=variable,
                                subfolder="maps",
                                output_group=_metric_output_group(metric_type),
                                item_name=metric_name,
                            )

                            for model_name, realization_maps in per_model_maps.items():
                                for realization_value, da_sel in realization_maps.items():
                                    map_region = _resolved_single_region_label(da_sel, filters=filters)
                                    filename_parts = [
                                        metric_type,
                                        metric_name,
                                        model_name,
                                        _context_filename_suffix(window_context_values),
                                    ]
                                    save_path = metric_folder / (
                                        f"{'_'.join(filename_parts)}"
                                        f"{_realization_filename_fragment(realization_value)}_map.png"
                                    )

                                    if progress is not None and task_id is not None:
                                        progress.update(
                                            task_id,
                                            description=f"Maps | {map_region or 'unknown'} | {variable} | {metric_type} | {metric_name}",
                                        )

                                    vlimits = _configured_metric_map_limits(
                                        config=config,
                                        metric_name=metric_name,
                                        variable=variable,
                                        map_region=map_region,
                                        metric_group_label=metric_group_label,
                                        metric_groupby_period=metric_groupby_period,
                                    )
                                    if vlimits is not None:
                                        vmin, vmax = vlimits[0], vlimits[1]
                                    else:
                                        vmin, vmax = shared_vmin, shared_vmax

                                    plot_temporal_mean_map(
                                        da_sel,
                                        save_path=save_path,
                                        title=_map_title(
                                            metric_label,
                                            config.models.display_names.get(model_name, model_name),
                                            f"{context_title}{_realization_label(realization_value)}"
                                            if context_title else _realization_label(realization_value).lstrip(" |"),
                                        ),
                                        subtitle=_window_context_subtitle(
                                            da_sel,
                                            context_values=window_context_values,
                                            leadtime_unit=leadtime_unit,
                                            metric_group_label=metric_group_label,
                                            metric_groupby_period=metric_groupby_period,
                                        ),
                                        extent=extent,
                                        cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                                        cmap=_resolve_metric_map_cmap(metric_name, config=config),
                                        vmin=vmin,
                                        vmax=vmax,
                                        lon_tick_step=config.maps.lon_tick_step,
                                        lat_tick_step=config.maps.lat_tick_step,
                                    )
                                    saved_paths.append(save_path)
                                    if progress is not None and task_id is not None:
                                        progress.advance(task_id)
                                    logger.debug(
                                        f"Saved averaged metric map for metric={metric_name} "
                                        f"model={model_name} region={map_region or 'unknown'} to: {save_path}"
                                    )

                            reference_model = config.models.reference_model
                            corrected_model = config.models.corrected_model
                            difference_model = config.models.difference_model
                            if corrected_model is not None and reference_model in per_model_maps and corrected_model in per_model_maps:
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
                                    fallback_diff_vlimits = _symmetric_map_limits(
                                        xr.concat([diff_da for _, diff_da in diff_maps], dim="_diff_map")
                                    )
                                    for realization_value, diff_da in diff_maps:
                                        diff_region = _resolved_single_region_label(diff_da, filters=filters)
                                        diff_vlimits = _configured_metric_map_limits(
                                            config=config,
                                            metric_name=metric_name,
                                            variable=variable,
                                            map_region=diff_region,
                                            metric_group_label=metric_group_label,
                                            metric_groupby_period=metric_groupby_period,
                                            is_difference=True,
                                        )
                                        if diff_vlimits is not None:
                                            diff_vmin, diff_vmax = diff_vlimits
                                        else:
                                            diff_vmin, diff_vmax = fallback_diff_vlimits
                                        diff_save_path = metric_folder / (
                                            f"{metric_type}_{metric_name}_{difference_model}_{_context_filename_suffix(window_context_values)}"
                                            f"{_realization_filename_fragment(realization_value)}_map.png"
                                        )
                                        if progress is not None and task_id is not None:
                                            progress.update(
                                                task_id,
                                                description=f"Maps | {diff_region or 'unknown'} | {variable} | diff | {metric_name}",
                                            )
                                        plot_temporal_mean_map(
                                            diff_da,
                                            save_path=diff_save_path,
                                            title=_map_title(
                                                metric_label,
                                                config.models.display_names.get(difference_model, difference_model),
                                                f"{context_title}{_realization_label(realization_value)}"
                                                if context_title else _realization_label(realization_value).lstrip(" |"),
                                            ),
                                            subtitle=_window_context_subtitle(
                                                diff_da,
                                                context_values=window_context_values,
                                                leadtime_unit=leadtime_unit,
                                                metric_group_label=metric_group_label,
                                                metric_groupby_period=metric_groupby_period,
                                            ),
                                            extent=extent,
                                            cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                                            cmap="RdBu_r",
                                            vmin=diff_vmin,
                                            vmax=diff_vmax,
                                            lon_tick_step=config.maps.lon_tick_step,
                                            lat_tick_step=config.maps.lat_tick_step,
                                        )
                                        saved_paths.append(diff_save_path)
                                        if progress is not None and task_id is not None:
                                            progress.advance(task_id)
                                        logger.debug(
                                            f"Saved averaged diff map for metric={metric_name} "
                                            f"model={difference_model} region={diff_region or 'unknown'} to: {diff_save_path}"
                                        )
            continue

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
                for selection, context_values in _metric_context_selection_entries(
                    context_source,
                    base_context_dims=CONTEXT_AXES,
                ):
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
                                realization_mode=metric_realization_mode,
                            )
                        )
                        if base_unit is None:
                            base_unit = _get_da_unit(da_sel) or _get_da_unit(
                                runs.get(config.models.truth_model, xr.Dataset()),
                                variable,
                            )

                    if not per_model_maps:
                        continue

                    context_title_values = _grouped_map_title_context(
                        context_values,
                        metric_group_label=metric_group_label,
                        metric_groupby_period=metric_groupby_period,
                    )
                    context_suffix = _context_title_suffix(context_title_values, leadtime_unit=leadtime_unit)
                    sample_da = next(iter(next(iter(per_model_maps.values())).values()))
                    extent = _region_extent_for_plot(
                        data=sample_da,
                        context_values=context_values,
                        filters=filters,
                        region_extents=region_extents,
                    )
                    metric_label = _format_metric_label(metric_name, base_unit=base_unit)
                    shared_vmin, shared_vmax = _shared_map_limits(
                        *(da for per_model in per_model_maps.values() for da in per_model.values())
                    )

                    for model_name, realization_maps in per_model_maps.items():
                        metric_folder = _get_variable_output_item_folder(
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
                                f"{'_'.join(filename_parts)}"
                                f"{_realization_filename_fragment(realization_value)}_map.png"
                            )

                            if progress is not None and task_id is not None:
                                progress.update(
                                    task_id,
                                    description=f"Maps | {map_region or 'unknown'} | {variable} | {metric_type} | {metric_name}",
                                )

                            vlimits = _configured_metric_map_limits(
                                config=config,
                                metric_name=metric_name,
                                variable=variable,
                                map_region=map_region,
                                metric_group_label=metric_group_label,
                                metric_groupby_period=metric_groupby_period,
                            )
                            if vlimits is not None:
                                vmin, vmax = vlimits[0], vlimits[1]
                            else:
                                vmin, vmax = shared_vmin, shared_vmax

                            plot_temporal_mean_map(
                                da_sel,
                                save_path=save_path,
                                title=_map_title(
                                    metric_label,
                                    config.models.display_names.get(model_name, model_name),
                                    f"{context_suffix}{_realization_label(realization_value)}" if context_suffix else _realization_label(realization_value).lstrip(" |"),
                                ),
                                subtitle=_grouped_map_subtitle(
                                    da_sel,
                                    context_values=context_values,
                                    leadtime_unit=leadtime_unit,
                                    metric_group_label=metric_group_label,
                                    metric_groupby_period=metric_groupby_period,
                                ),
                                extent=extent,
                                cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                                cmap=_resolve_metric_map_cmap(metric_name, config=config),
                                vmin=vmin,
                                vmax=vmax,
                                lon_tick_step=config.maps.lon_tick_step,
                                lat_tick_step=config.maps.lat_tick_step,
                            )
                            saved_paths.append(save_path)
                            if progress is not None and task_id is not None:
                                progress.advance(task_id)
                            logger.debug(
                                f"Saved metric map for metric={metric_name} "
                                f"model={model_name} region={map_region or 'unknown'} to: {save_path}"
                            )

                    reference_model = config.models.reference_model
                    corrected_model = config.models.corrected_model
                    difference_model = config.models.difference_model
                    if reference_model in per_model_maps and corrected_model in per_model_maps:
                        metric_folder = _get_variable_output_item_folder(
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
                            fallback_diff_vlimits = _symmetric_map_limits(
                                xr.concat([diff_da for _, diff_da in diff_maps], dim="_diff_map")
                            )
                            for realization_value, diff_da in diff_maps:
                                diff_region = _resolved_single_region_label(diff_da, filters=filters)
                                diff_vlimits = _configured_metric_map_limits(
                                    config=config,
                                    metric_name=metric_name,
                                    variable=variable,
                                    map_region=diff_region,
                                    metric_group_label=metric_group_label,
                                    metric_groupby_period=metric_groupby_period,
                                    is_difference=True,
                                )
                                if diff_vlimits is not None:
                                    diff_vmin, diff_vmax = diff_vlimits
                                else:
                                    diff_vmin, diff_vmax = fallback_diff_vlimits
                                diff_save_path = metric_folder / (
                                    f"{metric_type}_{metric_name}_{difference_model}_{_context_filename_suffix(context_values)}"
                                    f"{_realization_filename_fragment(realization_value)}_map.png"
                                )
                                if progress is not None and task_id is not None:
                                    progress.update(
                                        task_id,
                                        description=f"Maps | {diff_region or 'unknown'} | {variable} | diff | {metric_name}",
                                    )
                                plot_temporal_mean_map(
                                    diff_da,
                                    save_path=diff_save_path,
                                    title=_map_title(
                                        metric_label,
                                        config.models.display_names.get(difference_model, difference_model),
                                        f"{context_suffix}{_realization_label(realization_value)}" if context_suffix else _realization_label(realization_value).lstrip(" |"),
                                    ),
                                    subtitle=_grouped_map_subtitle(
                                        diff_da,
                                        context_values=context_values,
                                        leadtime_unit=leadtime_unit,
                                        metric_group_label=metric_group_label,
                                        metric_groupby_period=metric_groupby_period,
                                    ),
                                    extent=extent,
                                    cbar_label=_map_cbar_label(variable=variable, unit=base_unit),
                                    cmap="RdBu_r",
                                    vmin=diff_vmin,
                                    vmax=diff_vmax,
                                    lon_tick_step=config.maps.lon_tick_step,
                                    lat_tick_step=config.maps.lat_tick_step,
                                )
                                saved_paths.append(diff_save_path)
                                if progress is not None and task_id is not None:
                                    progress.advance(task_id)
                                logger.debug(
                                    f"Saved diff map for metric={metric_name} "
                                    f"model={difference_model} region={diff_region or 'unknown'} to: {diff_save_path}"
                                )

    return saved_paths
