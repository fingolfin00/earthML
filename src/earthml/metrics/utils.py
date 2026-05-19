from dataclasses import dataclass
import hashlib
import json
from collections.abc import Callable
from typing import Any, Sequence, Literal
from contextlib import nullcontext
from itertools import product

from pathlib import Path

from dateutil.relativedelta import relativedelta
from datetime import datetime

from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import numpy as np
import pandas as pd
from scipy.signal import get_window
import cf_xarray
import xarray as xr
# import xskillscore as xs
# from scipy.stats import t as student_t

from .metrics.deterministic import DeterministicMetrics
from .metrics.correlation import CorrelationMetrics
from .metrics.probabilistic import ProbabilisticMetrics
from .bundles import build_standard_metric_bundle, count_standard_metric_bundle_steps
from .calculate.constants import METRIC_KIND, METRIC_SECTIONS

from ..experiments.mlbc.load import (
    load_all_exp_from_folder,
    add_ke_to_runs,
    harmonize_leadtime_int,
)
from ..experiments.mlbc.registry import MLBCExperimentMode
from ..experiments.mlbc.utils import (
    combine_masks,
    project_mask_to_reference_grid,
    select_mask_for_indexers,
)
from ..experiments.mlbc.load import (
    _discover_region_values,
    RunsByModel,
    RunsByRegion,
)


MetricLeaf = dict[str, dict[str, xr.Dataset]]
MetricsByRegion = dict[str, MetricLeaf]
ClimsByModel = dict[str, xr.Dataset | None]
ClimsByRegion = dict[str, ClimsByModel]

METRIC_GROUP_DIM = "metric_group"
METRIC_GROUPBY_PERIOD_ATTR = "metric_groupby_period"
METRIC_GROUPBY_BASIS_ATTR = "metric_groupby_basis"
METRIC_GROUPINGS_ATTR = "metric_groupings"


@dataclass(frozen=True)
class _ResolvedDimensionGrouping:
    dim: str
    mode: str
    values: tuple[object, ...] | None = None
    groups: tuple[tuple[str, tuple[object, ...]], ...] | None = None
    kinds: tuple[str, ...] | None = None
    include_all: bool = True


@dataclass(frozen=True)
class _ResolvedTimeGrouping:
    period: str
    basis: str
    include_all: bool = True


@dataclass(frozen=True)
class _ResolvedMetricGroupOperation:
    key: str
    kind: str
    label: object
    dim: str | None = None
    values: tuple[object, ...] | None = None
    mode: str | None = None
    kinds: tuple[str, ...] | None = None
    period: str | None = None
    basis: str | None = None


# ==========================================
# Standalone helper methods
# ==========================================


def build_parquet_path(
    base_folder: str,
    vars_list: Sequence[str],
    metric_names: Sequence[str] | None = None,
    diff: str = "no", # no, delta, ratio
) -> str:
    vars_tag = "_".join(sorted(map(str, vars_list))) if vars_list else "allvars"
    metrics_tag = "_".join(sorted(map(str, metric_names))) if metric_names else "allmetrics"

    return str(Path(base_folder) / f"metrics_{diff}_{vars_tag}_{metrics_tag}_parquet")


def date_diff(ymd_range: str):
    start_str, end_str = ymd_range.split('-')
    start = datetime.strptime(start_str, '%Y%m%d')
    end = datetime.strptime(end_str, '%Y%m%d')

    delta = relativedelta(end, start)
    days = (end - start).days

    return {
        "years": delta.years,
        "months": delta.months,
        "days": delta.days,
        "total_days": days,
        "total_months": delta.years * 12 + delta.months
    }


def metrics_to_df_single_region(
    metrics: MetricLeaf,
    variables: Sequence[str] | str | None = None,
    metric_names: Sequence[str] | str | None = None,
    kind: str = "scalar",
    metric_type: str = "all_dims",
    diff: str = "no",
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Convert one single-region metrics leaf returned by get_metrics(...)
    into a tidy pandas DataFrame.
    """
    ds = metrics.get(metric_type, {}).get(kind, xr.Dataset())

    base_cols = ["train_period", "model", "metric", "variable", "leadtime", METRIC_GROUP_DIM, "value"]

    if not isinstance(ds, xr.Dataset) or not ds.data_vars:
        return pd.DataFrame(columns=base_cols)

    if isinstance(metric_names, str):
        metric_names = [metric_names]
    elif metric_names is not None:
        metric_names = list(metric_names)

    if isinstance(variables, str):
        variables = [variables]
    elif variables is not None:
        variables = list(variables)

    metric_dim = "metric"

    if metric_dim not in ds.coords and metric_dim not in ds.dims:
        return pd.DataFrame(columns=base_cols)

    available_metrics = ds.coords[metric_dim].values.tolist()
    if metric_names is None:
        metric_names = list(available_metrics)
    else:
        metric_names = [m for m in metric_names if m in available_metrics]

    if not metric_names:
        return pd.DataFrame(columns=base_cols)

    available_variables = list(ds.data_vars)
    if variables is None:
        variables = available_variables
    else:
        variables = [v for v in variables if v in available_variables]

    if not variables:
        return pd.DataFrame(columns=base_cols)

    ds = ds[variables].sel({metric_dim: metric_names})

    has_model = "model" in ds.dims or "model" in ds.coords

    if diff in ("delta", "ratio"):
        if not has_model:
            raise ValueError(
                f"diff='{diff}' requires a 'model' dimension in metrics[{metric_type!r}][{kind!r}]"
            )

        model_values = ds.coords["model"].values.tolist()

        if models is None:
            if len(model_values) != 2:
                raise ValueError(
                    "When diff is 'delta' or 'ratio', exactly two models are required"
                )
            model_a, model_b = model_values
        else:
            if len(models) != 2:
                raise ValueError(
                    "When diff is 'delta' or 'ratio', models must contain exactly two model names"
                )
            model_a, model_b = models
            missing = [m for m in (model_a, model_b) if m not in model_values]
            if missing:
                raise ValueError(f"Requested diff models not found in metrics dataset: {missing}")

        ds_a = ds.sel(model=model_a)
        ds_b = ds.sel(model=model_b)
        ds_a, ds_b = xr.align(ds_a, ds_b, join="inner")

        arr_a = ds_a.to_array(dim="variable")
        arr_b = ds_b.to_array(dim="variable")

        arr = arr_b - arr_a
        if diff == "ratio":
            arr = arr / np.abs(arr_a)

        df = arr.to_dataframe(name="value").reset_index()
        df["model"] = f"{model_b}-{model_a}"

    else:
        if has_model and models is not None:
            model_values = ds.coords["model"].values.tolist()
            selected_models = [m for m in models if m in model_values]
            if not selected_models:
                return pd.DataFrame(columns=base_cols)
            ds = ds.sel(model=selected_models)

        df = ds.to_array(dim="variable").to_dataframe(name="value").reset_index()

        if not has_model:
            df["model"] = np.nan

    if metric_dim in df.columns:
        df = df.rename(columns={metric_dim: "metric"})

    for col in ["train_period", "model", "metric", "variable", "leadtime", METRIC_GROUP_DIM]:
        if col not in df.columns:
            df[col] = np.nan

    if "value" not in df.columns:
        df["value"] = np.nan

    # Keep standard columns first, then preserve any extra dims/coords such as loss
    extra_cols = [col for col in df.columns if col not in base_cols]
    df = df[base_cols + extra_cols]

    if df["train_period"].notna().any():
        df[["years", "months", "days", "total_days", "total_months"]] = (
            df["train_period"].astype(str)
            .apply(lambda x: date_diff(x) if "-" in x and len(x.split("-")) == 2 else {
                "years": np.nan,
                "months": np.nan,
                "days": np.nan,
                "total_days": np.nan,
                "total_months": np.nan,
            })
            .apply(pd.Series)
        )
    else:
        df[["years", "months", "days", "total_days", "total_months"]] = np.nan

    return df


def rechunk(da, lat_rc=None, lon_rc=None, time_rc=None, realization_rc=None):
    chunks = {}
    if time_rc is not None:
        chunks[da.earthml.guessed_dims.time] = time_rc
    if lat_rc is not None:
        chunks[da.earthml.guessed_dims.latitude] = lat_rc
    if lon_rc is not None:
        chunks[da.earthml.guessed_dims.longitude] = lon_rc

    realization_dim = da.earthml.guessed_dims.realization
    if realization_dim in da.dims and realization_rc is not None:
        chunks[realization_dim] = realization_rc

    leadtime_dim = da.earthml.guessed_dims.leadtime
    if leadtime_dim in da.dims:
        chunks[leadtime_dim] = -1  # keep together

    return da.chunk(chunks)


def _compute_metric_bundle(
    deterministic: DeterministicMetrics,
    correlation: CorrelationMetrics | None = None,
    probabilistic: ProbabilisticMetrics | None = None,
    norm: str = "std",
    clim_period: str = "month",
    metric_names: str | Sequence[str] | None = None,
    metric_sections: str | Sequence[str] = ("scalar", "map", "timeseries"),
    metric_types: str | Sequence[str] | None = None,
    metric_progress_hook: Callable[[str, str, str], None] | None = None,
    metric_mean_extra_dims: dict[str, str | Sequence[str] | None] | None = None,
) -> dict[str, dict[str, xr.Dataset]]:
    if isinstance(metric_names, str):
        metric_names = {metric_names}
    elif metric_names is not None:
        metric_names = set(metric_names)

    if isinstance(metric_sections, str):
        metric_sections = (metric_sections,)
    metric_sections = tuple(metric_sections)

    if isinstance(metric_types, str):
        metric_types = {metric_types}
    elif metric_types is not None:
        metric_types = set(metric_types)

    result = build_standard_metric_bundle(
        deterministic=deterministic,
        correlation=correlation,
        probabilistic=probabilistic,
        norm=norm,
        clim_period=clim_period,
        metric_names=metric_names,
        metric_kinds=metric_sections,
        progress_hook=metric_progress_hook,
        metric_mean_extra_dims=metric_mean_extra_dims,
    )

    filtered_result: dict[str, dict[str, xr.Dataset]] = {}

    for metric_type in METRIC_SECTIONS:
        if metric_types is not None and metric_type not in metric_types:
            continue

        filtered_result[metric_type] = {}

        for section in metric_sections:
            key = f"{section}_{metric_type}"
            filtered_result[metric_type][section] = result[key]

    return filtered_result


def _format_metric_group_label(
    *,
    period: str,
    value: object,
) -> str:
    if period == "month":
        return f"{int(value):02d}"
    if period == "dayofyear":
        return f"{int(value):03d}"
    return str(value)


def _resolve_period(period: str | None, *, param_name: str) -> str:
    valid_periods = {"month", "dayofyear", "season"}
    if period is None:
        return "month"
    if period not in valid_periods:
        raise ValueError(
            f"Unsupported {param_name}={period!r}. Expected one of {sorted(valid_periods)!r}."
        )
    return period


def _resolve_metric_groupby_basis(metric_groupby_basis: str | None) -> str:
    valid_bases = {"target_time", "reference_time"}
    if metric_groupby_basis is None:
        return "target_time"
    if metric_groupby_basis not in valid_bases:
        raise ValueError(
            f"Unsupported metric_groupby_basis={metric_groupby_basis!r}. Expected one of {sorted(valid_bases)!r}."
        )
    return metric_groupby_basis


def _normalize_group_value_sequence(values: object, *, param_name: str) -> tuple[object, ...]:
    if isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [values]

    out: list[object] = []
    for value in raw_values:
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError(f"{param_name} must contain at least one value.")
    return tuple(out)


def _normalize_metric_kind_sequence(
    kinds: object,
    *,
    param_name: str,
) -> tuple[str, ...]:
    if isinstance(kinds, str):
        raw_kinds = [kinds]
    elif isinstance(kinds, (list, tuple, set)):
        raw_kinds = list(kinds)
    else:
        raise ValueError(
            f"{param_name} must be a metric kind string or sequence of metric kinds from {METRIC_KIND!r}."
        )

    normalized: list[str] = []
    for kind in raw_kinds:
        kind_str = str(kind).strip()
        if kind_str not in METRIC_KIND:
            raise ValueError(
                f"Unsupported {param_name} entry {kind!r}. Expected one of {METRIC_KIND!r}."
            )
        if kind_str not in normalized:
            normalized.append(kind_str)
    if not normalized:
        raise ValueError(f"{param_name} must contain at least one metric kind.")
    return tuple(normalized)


def _normalize_dimension_grouping(
    dim: str,
    spec: object,
) -> _ResolvedDimensionGrouping:
    if spec == "by_value":
        return _ResolvedDimensionGrouping(dim=dim, mode="by_value")

    if isinstance(spec, (list, tuple, set)):
        return _ResolvedDimensionGrouping(
            dim=dim,
            mode="values",
            values=_normalize_group_value_sequence(spec, param_name=f"metric_groupings[{dim!r}]"),
        )

    if not isinstance(spec, dict):
        raise ValueError(
            f"Unsupported metric_groupings[{dim!r}]={spec!r}. "
            "Expected 'by_value', a sequence of values, or a mapping with 'values' or 'groups'."
        )

    has_values = "values" in spec
    has_groups = "groups" in spec
    if has_values and has_groups:
        raise ValueError(
            f"metric_groupings[{dim!r}] cannot define both 'values' and 'groups'."
        )

    if has_values:
        if "mode" in spec:
            raise ValueError(
                f"metric_groupings[{dim!r}] cannot use 'mode' together with 'values'."
            )
        return _ResolvedDimensionGrouping(
            dim=dim,
            mode="values",
            values=_normalize_group_value_sequence(
                spec["values"],
                param_name=f"metric_groupings[{dim!r}]['values']",
            ),
        )

    if has_groups:
        raw_groups = spec["groups"]
        if not isinstance(raw_groups, dict) or not raw_groups:
            raise ValueError(
                f"metric_groupings[{dim!r}]['groups'] must be a non-empty mapping of labels to values."
            )
        mode = str(spec.get("mode", "reduce"))
        if mode not in {"average", "reduce"}:
            raise ValueError(
                f"Unsupported metric_groupings[{dim!r}]['mode']={mode!r}. Expected 'average' or 'reduce'."
            )
        raw_kinds = spec.get("kinds")
        if raw_kinds is not None and mode != "reduce":
            raise ValueError(
                f"metric_groupings[{dim!r}] can only use 'kinds' when mode='reduce'."
            )
        kinds = (
            tuple(METRIC_KIND)
            if mode == "reduce" and raw_kinds is None
            else (
                _normalize_metric_kind_sequence(
                    raw_kinds,
                    param_name=f"metric_groupings[{dim!r}]['kinds']",
                )
                if raw_kinds is not None
                else None
            )
        )
        include_all = bool(spec.get("include_all", True))

        groups: list[tuple[str, tuple[object, ...]]] = []
        for raw_label, raw_values in raw_groups.items():
            label = str(raw_label).strip()
            if not label:
                raise ValueError(
                    f"metric_groupings[{dim!r}]['groups'] cannot contain empty labels."
                )
            groups.append(
                (
                    label,
                    _normalize_group_value_sequence(
                        raw_values,
                        param_name=f"metric_groupings[{dim!r}]['groups'][{label!r}]",
                    ),
                )
            )
        return _ResolvedDimensionGrouping(
            dim=dim,
            mode=mode,
            groups=tuple(groups),
            kinds=kinds,
            include_all=include_all,
        )

    raise ValueError(
        f"Unsupported metric_groupings[{dim!r}]={spec!r}. "
        "Expected 'by_value', a sequence of values, or a mapping with 'values' or 'groups'."
    )


def _normalize_time_grouping(spec: object) -> _ResolvedTimeGrouping:
    if not isinstance(spec, dict):
        raise ValueError(
            "metric_groupings['time'] must be a mapping with keys 'period' and optional 'basis'."
        )
    period = _resolve_period(
        spec.get("period"),
        param_name="metric_groupings['time']['period']",
    )
    basis = _resolve_metric_groupby_basis(spec.get("basis"))
    include_all = bool(spec.get("include_all", True))
    return _ResolvedTimeGrouping(period=period, basis=basis, include_all=include_all)


def normalize_metric_groupings(
    metric_groupings: dict[str, object] | None,
    *,
    metric_groupby_period: str | None = None,
    metric_groupby_basis: str | None = None,
) -> dict[str, _ResolvedDimensionGrouping | _ResolvedTimeGrouping]:
    normalized: dict[str, _ResolvedDimensionGrouping | _ResolvedTimeGrouping] = {}

    if metric_groupings is not None:
        if not isinstance(metric_groupings, dict):
            raise ValueError("metric_groupings must be a mapping from dimension names to grouping specs.")

        for raw_key, spec in metric_groupings.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("metric_groupings cannot contain empty dimension names.")
            if isinstance(spec, (_ResolvedDimensionGrouping, _ResolvedTimeGrouping)):
                normalized[key] = spec
                continue
            if key == "time":
                normalized[key] = _normalize_time_grouping(spec)
            else:
                normalized[key] = _normalize_dimension_grouping(key, spec)

    if metric_groupby_period is not None:
        legacy_time = _ResolvedTimeGrouping(
            period=_resolve_period(metric_groupby_period, param_name="metric_groupby_period"),
            basis=_resolve_metric_groupby_basis(metric_groupby_basis),
        )
        if "time" in normalized:
            current = normalized["time"]
            if not isinstance(current, _ResolvedTimeGrouping):
                raise ValueError("metric_groupings['time'] must use the special time grouping mapping.")
            if current != legacy_time:
                raise ValueError(
                    "metric_groupings['time'] conflicts with metric_groupby_period/metric_groupby_basis. "
                    "Use only one configuration source, or make them match."
                )
        else:
            normalized["time"] = legacy_time

    return normalized


def metric_groupings_to_serializable(
    metric_groupings: dict[str, _ResolvedDimensionGrouping | _ResolvedTimeGrouping] | None,
) -> dict[str, object]:
    if not metric_groupings:
        return {}

    serialized: dict[str, object] = {}
    for key, spec in metric_groupings.items():
        if isinstance(spec, _ResolvedTimeGrouping):
            serialized[key] = {
                "period": spec.period,
                "basis": spec.basis,
                "include_all": spec.include_all,
            }
            continue

        if spec.mode == "by_value":
            serialized[key] = "by_value"
        elif spec.mode == "values":
            serialized[key] = {"values": list(spec.values or ())}
        else:
            serialized[key] = {
                "mode": spec.mode,
                "include_all": spec.include_all,
                **({"kinds": list(spec.kinds)} if spec.kinds is not None else {}),
                "groups": {
                    label: list(values)
                    for label, values in (spec.groups or ())
                },
            }
    return serialized


def metric_groupings_signature(
    metric_groupings: dict[str, _ResolvedDimensionGrouping | _ResolvedTimeGrouping] | None,
) -> str:
    return json.dumps(
        metric_groupings_to_serializable(metric_groupings),
        sort_keys=True,
        separators=(",", ":"),
    )


def _leadtime_unit_from_data(data: xr.Dataset | xr.DataArray) -> str | None:
    for coord_name in ("leadtime",):
        if coord_name not in data.coords:
            continue
        unit = str(data[coord_name].attrs.get("unit", "")).strip()
        if unit:
            return unit
    return None


def _shift_datetime_index(
    valid_times: pd.DatetimeIndex,
    *,
    leadtime_value: object,
    leadtime_unit: str,
) -> pd.DatetimeIndex:
    try:
        leadtime_int = int(leadtime_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot derive reference_time from non-integer leadtime={leadtime_value!r}.") from exc

    if leadtime_unit == "months":
        return valid_times - pd.DateOffset(months=leadtime_int)
    if leadtime_unit == "days":
        return valid_times - pd.to_timedelta(leadtime_int, unit="D")
    if leadtime_unit == "hours":
        return valid_times - pd.to_timedelta(leadtime_int, unit="h")
    raise ValueError(
        f"Unsupported leadtime unit {leadtime_unit!r} for reference_time grouping. "
        "Expected one of {'months', 'days', 'hours'}."
    )


def _reference_time_dataarray(
    data: xr.Dataset | xr.DataArray,
    *,
    leadtime_unit: str | None = None,
    leadtime_value: object | None = None,
) -> xr.DataArray | None:
    time_dim = data.earthml.guessed_dims.time
    if time_dim is None or time_dim not in data.dims:
        return None

    if "reftime" in data.coords and data["reftime"].dims == (time_dim,):
        return data["reftime"]

    valid_times = pd.to_datetime(np.asarray(data[time_dim].values).reshape(-1), errors="coerce")
    leadtime_unit = leadtime_unit or _leadtime_unit_from_data(data)
    if not leadtime_unit:
        raise ValueError("Cannot derive reference_time grouping without a leadtime unit.")

    leadtime_dim = data.earthml.guessed_dims.leadtime
    if leadtime_dim is not None and leadtime_dim in data.dims:
        leadtime_values = data[leadtime_dim].values.tolist()
        slices: list[xr.DataArray] = []
        for leadtime_value in leadtime_values:
            shifted = _shift_datetime_index(
                pd.DatetimeIndex(valid_times),
                leadtime_value=leadtime_value,
                leadtime_unit=leadtime_unit,
            )
            slices.append(
                xr.DataArray(
                    shifted.values,
                    dims=(time_dim,),
                    coords={time_dim: data[time_dim].values},
                ).expand_dims({leadtime_dim: [leadtime_value]})
            )
        return xr.concat(slices, dim=leadtime_dim)

    resolved_leadtime_value = leadtime_value
    if resolved_leadtime_value is None and "leadtime" in data.coords:
        leadtime_values = np.asarray(data["leadtime"].values).reshape(-1)
        if leadtime_values.size == 1:
            resolved_leadtime_value = leadtime_values[0]
    if resolved_leadtime_value is None:
        raise ValueError("Cannot derive reference_time grouping without a scalar or dimensioned leadtime coordinate.")

    shifted = _shift_datetime_index(
        pd.DatetimeIndex(valid_times),
        leadtime_value=resolved_leadtime_value,
        leadtime_unit=leadtime_unit,
    )
    return xr.DataArray(
        shifted.values,
        dims=(time_dim,),
        coords={time_dim: data[time_dim].values},
    )


def _grouping_time_dataarray(
    data: xr.Dataset | xr.DataArray,
    *,
    basis: str,
    leadtime_unit: str | None = None,
    leadtime_value: object | None = None,
) -> xr.DataArray | None:
    time_dim = data.earthml.guessed_dims.time
    if time_dim is None or time_dim not in data.dims:
        return None

    if basis == "target_time":
        return data[time_dim]
    if basis == "reference_time":
        return _reference_time_dataarray(
            data,
            leadtime_unit=leadtime_unit,
            leadtime_value=leadtime_value,
        )
    raise ValueError(f"Unsupported grouping basis {basis!r}")


def _metric_group_specs_from_data(
    data: xr.Dataset | xr.DataArray,
    period: str | None,
    *,
    basis: str = "target_time",
    leadtime_unit: str | None = None,
    leadtime_value: object | None = None,
) -> list[tuple[str, object]]:
    if period is None:
        return []

    grouping_time = _grouping_time_dataarray(
        data,
        basis=basis,
        leadtime_unit=leadtime_unit,
        leadtime_value=leadtime_value,
    )
    if grouping_time is None or not hasattr(grouping_time, "dt"):
        return []

    raw_values = getattr(grouping_time.dt, period).values.reshape(-1)
    unique_values = [value.item() if isinstance(value, np.generic) else value for value in pd.unique(raw_values)]
    if period == "month":
        ordered_values = sorted(int(value) for value in unique_values if pd.notna(value))
    elif period == "dayofyear":
        ordered_values = sorted(int(value) for value in unique_values if pd.notna(value))
    elif period == "season":
        season_order = ["DJF", "MAM", "JJA", "SON"]
        ordered_values = [season for season in season_order if season in {str(value) for value in unique_values}]
    else:
        raise ValueError(f"Unsupported grouping period {period!r}")

    return [
        (_format_metric_group_label(period=period, value=value), value)
        for value in ordered_values
    ]


def slice_time_group_data(
    data: xr.Dataset | xr.DataArray | None,
    *,
    period: str | None,
    group_value: object,
    basis: str = "target_time",
    leadtime_unit: str | None = None,
    leadtime_value: object | None = None,
) -> xr.Dataset | xr.DataArray | None:
    if data is None:
        return None

    if period is None:
        return data

    grouping_time = _grouping_time_dataarray(
        data,
        basis=basis,
        leadtime_unit=leadtime_unit,
        leadtime_value=leadtime_value,
    )
    if grouping_time is None or not hasattr(grouping_time, "dt"):
        return data

    selector = getattr(grouping_time.dt, period) == group_value
    return data.where(selector, drop=True)


def _coord_values_for_dim(
    data: xr.Dataset | xr.DataArray,
    dim: str,
) -> list[object]:
    if dim not in data.dims:
        return []
    if dim in data.coords:
        return data.coords[dim].values.tolist()
    return data[dim].values.tolist()


def _resolve_existing_group_values(
    data: xr.Dataset | xr.DataArray,
    *,
    dim: str,
    requested_values: Sequence[object],
) -> tuple[object, ...]:
    available_values = _coord_values_for_dim(data, dim)
    resolved: list[object] = []
    for available_value in available_values:
        if any(str(available_value) == str(requested_value) for requested_value in requested_values):
            if available_value not in resolved:
                resolved.append(available_value)
    return tuple(resolved)


def _merge_run_filters_with_metric_groupings(
    run_filters: dict[str, object] | None,
    *,
    metric_groupings: dict[str, _ResolvedDimensionGrouping | _ResolvedTimeGrouping],
) -> dict[str, object] | None:
    if not metric_groupings:
        return run_filters

    merged = dict(run_filters or {})
    for key, spec in metric_groupings.items():
        if key == "time" or not isinstance(spec, _ResolvedDimensionGrouping):
            continue
        if spec.mode not in {"values"}:
            continue

        requested_values = list(spec.values or ())
        existing = merged.get(key)
        if existing is None:
            merged[key] = requested_values
            continue

        existing_values = existing if isinstance(existing, (list, tuple, set)) else [existing]
        existing_map = {str(value): value for value in existing_values}
        merged[key] = [
            existing_map[str(value)]
            for value in requested_values
            if str(value) in existing_map
        ]

    return merged


def _resolve_run_dims(
    truth_runs: xr.Dataset,
    *,
    metric_groupings: dict[str, _ResolvedDimensionGrouping | _ResolvedTimeGrouping],
) -> list[str]:
    default_run_dims = ("leadtime", "train_period", "region", "loss", "variant")
    run_dims = [dim for dim in default_run_dims if dim in truth_runs.dims]

    grouped_dims = {
        key
        for key, spec in metric_groupings.items()
        if key != "time"
        and isinstance(spec, _ResolvedDimensionGrouping)
        and spec.mode in {"average", "reduce"}
    }
    run_dims = [dim for dim in run_dims if dim not in grouped_dims]

    for key, spec in metric_groupings.items():
        if key == "time" or not isinstance(spec, _ResolvedDimensionGrouping):
            continue
        if spec.mode in {"by_value", "values"} and key in truth_runs.dims and key not in run_dims:
            run_dims.append(key)

    return run_dims


def _build_metric_group_operations(
    truth_data: xr.Dataset,
    *,
    metric_groupings: dict[str, _ResolvedDimensionGrouping | _ResolvedTimeGrouping],
    leadtime_unit: str | None,
    leadtime_value: object | None,
) -> list[tuple[_ResolvedMetricGroupOperation, ...]]:
    operations_by_key: list[list[_ResolvedMetricGroupOperation]] = []

    for key, spec in metric_groupings.items():
        if key == "time":
            if not isinstance(spec, _ResolvedTimeGrouping) or spec.period is None:
                continue
            time_ops = [
                *(
                    [
                        _ResolvedMetricGroupOperation(
                            key="time",
                            kind="time",
                            label="all",
                            period=spec.period,
                            basis=spec.basis,
                            values=None,
                        ),
                    ]
                    if spec.include_all else []
                ),
                *[
                _ResolvedMetricGroupOperation(
                    key="time",
                    kind="time",
                    label=label,
                    period=spec.period,
                    basis=spec.basis,
                    values=(group_value,),
                )
                for label, group_value in _metric_group_specs_from_data(
                    truth_data,
                    spec.period,
                    basis=spec.basis,
                    leadtime_unit=leadtime_unit,
                    leadtime_value=leadtime_value,
                )
                ],
            ]
            if time_ops:
                operations_by_key.append(time_ops)
            continue

        if not isinstance(spec, _ResolvedDimensionGrouping):
            continue
        if spec.mode not in {"average", "reduce"}:
            continue
        if key not in truth_data.dims:
            continue

        dim_ops: list[_ResolvedMetricGroupOperation] = []
        all_values = _coord_values_for_dim(truth_data, key)
        if spec.include_all and all_values:
            dim_ops.append(
                _ResolvedMetricGroupOperation(
                    key=key,
                    kind="dimension",
                    label="all",
                    dim=key,
                    values=tuple(all_values),
                    mode=spec.mode,
                    kinds=spec.kinds,
                )
            )
        for label, requested_values in spec.groups or ():
            resolved_values = _resolve_existing_group_values(
                truth_data,
                dim=key,
                requested_values=requested_values,
            )
            if not resolved_values:
                continue
            dim_ops.append(
                _ResolvedMetricGroupOperation(
                    key=key,
                    kind="dimension",
                    label=label,
                    dim=key,
                    values=resolved_values,
                    mode=spec.mode,
                    kinds=spec.kinds,
                )
            )
        if dim_ops:
            operations_by_key.append(dim_ops)

    if not operations_by_key:
        return [tuple()]

    return [tuple(combo) for combo in product(*operations_by_key)]


def _apply_metric_group_operation(
    data: xr.Dataset | xr.DataArray | None,
    *,
    op: _ResolvedMetricGroupOperation,
    leadtime_unit: str | None,
    leadtime_value: object | None,
) -> xr.Dataset | xr.DataArray | None:
    if data is None:
        return None

    if op.kind == "time":
        group_value = None if not op.values else op.values[0]
        if group_value is None:
            return data
        return slice_time_group_data(
            data,
            period=str(op.period),
            group_value=group_value,
            basis=str(op.basis),
            leadtime_unit=leadtime_unit,
            leadtime_value=leadtime_value,
        )

    if op.kind == "dimension":
        if op.dim is None or not op.values:
            return data
        out = data.sel({op.dim: list(op.values)}, drop=True)
        if op.mode == "average" and op.dim in out.dims:
            out = out.mean(dim=op.dim, skipna=True)
        return out

    return data


def _expand_metric_group_labels(
    ds: xr.Dataset,
    *,
    operations: Sequence[_ResolvedMetricGroupOperation],
) -> xr.Dataset:
    out = ds
    time_attrs_applied = False

    for op in operations:
        if op.kind == "time":
            out = out.expand_dims({METRIC_GROUP_DIM: [op.label]})
            out.attrs[METRIC_GROUPBY_PERIOD_ATTR] = op.period
            out.attrs[METRIC_GROUPBY_BASIS_ATTR] = op.basis
            time_attrs_applied = True
            continue

        if op.kind == "dimension" and op.dim is not None:
            out = out.expand_dims({f"{op.dim}_group": [op.label]})

    if operations:
        out.attrs[METRIC_GROUPINGS_ATTR] = metric_groupings_signature(
            {
                op.key: (
                    _ResolvedTimeGrouping(period=str(op.period), basis=str(op.basis))
                    if op.kind == "time"
                    else _ResolvedDimensionGrouping(
                        dim=str(op.dim),
                        mode=str(op.mode),
                        groups=((str(op.label), tuple(op.values or ())),),
                        kinds=op.kinds,
                    )
                )
                for op in operations
            }
        )
    elif not time_attrs_applied and METRIC_GROUPINGS_ATTR not in out.attrs:
        out.attrs[METRIC_GROUPINGS_ATTR] = metric_groupings_signature({})

    return out


def _expand_all_metric_group_labels(
    ds: xr.Dataset,
    *,
    metric_groupings: dict[str, _ResolvedDimensionGrouping | _ResolvedTimeGrouping],
) -> xr.Dataset:
    out = ds

    for key, spec in metric_groupings.items():
        if key == "time":
            if (
                isinstance(spec, _ResolvedTimeGrouping)
                and spec.period is not None
                and spec.include_all
            ):
                out = out.expand_dims({METRIC_GROUP_DIM: ["all"]})
                out.attrs[METRIC_GROUPBY_PERIOD_ATTR] = spec.period
                out.attrs[METRIC_GROUPBY_BASIS_ATTR] = spec.basis
            continue

        if (
            isinstance(spec, _ResolvedDimensionGrouping)
            and spec.mode in {"average", "reduce"}
            and spec.groups
            and spec.include_all
        ):
            out = out.expand_dims({f"{key}_group": ["all"]})

    return out


def _ensure_dimension_indexes(ds: xr.Dataset) -> xr.Dataset:
    out = ds
    for dim in out.dims:
        if dim in out.coords and dim not in out.xindexes:
            out = out.set_xindex(dim)
    return out


def _hashable_coord_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coord_signature(coord: xr.DataArray) -> tuple[object, ...]:
    values = np.asarray(coord.values)

    if values.ndim == 0:
        return (
            tuple(coord.dims),
            str(values.dtype),
            (),
            _hashable_coord_value(values.reshape(()).item()),
        )

    if values.dtype == object:
        flat_values = tuple(
            _hashable_coord_value(value)
            for value in values.reshape(-1).tolist()
        )
        return (
            tuple(coord.dims),
            str(values.dtype),
            tuple(values.shape),
            flat_values,
        )

    contiguous = np.ascontiguousarray(values)
    digest = hashlib.blake2b(
        contiguous.view(np.uint8),
        digest_size=16,
    ).hexdigest()
    return (
        tuple(coord.dims),
        str(contiguous.dtype),
        tuple(contiguous.shape),
        digest,
    )


def _concat_dims_for_runs(runs_list: Sequence[xr.Dataset]) -> tuple[str, ...]:
    preferred_dims = ("leadtime", "train_period", "loss", "variant", METRIC_GROUP_DIM)
    dynamic_group_dims = tuple(
        dim
        for dim in sorted({
            dim
            for ds in runs_list
            for dim in ds.dims
            if dim.endswith("_group") and dim != METRIC_GROUP_DIM
        })
        if dim not in preferred_dims
    )
    return preferred_dims + dynamic_group_dims

def _concat_if_possible(
    runs_list: Sequence[xr.Dataset],
    dim: str,
) -> list[xr.Dataset]:
    grouped: dict[tuple, list[xr.Dataset]] = {}

    for ds in runs_list:
        key = (
            tuple(v for v in ds.data_vars),
            tuple(d for d in ds.dims if d != dim),
            tuple(
                (name, _coord_signature(ds.coords[name]))
                for name in ds.coords
                if name != dim
            ),
        )
        grouped.setdefault(key, []).append(ds)

    out: list[xr.Dataset] = []
    for datasets in grouped.values():
        if len(datasets) == 1:
            out.append(datasets[0])
        else:
            out.append(
                xr.concat(
                    datasets,
                    dim=dim,
                    coords="minimal",
                    compat="override",
                    combine_attrs="drop_conflicts",
                )
            )
    return out

def _combine_metric_run_datasets(
    runs_list: Sequence[xr.Dataset],
) -> xr.Dataset:
    if not runs_list:
        return xr.Dataset()

    prepared = [_ensure_dimension_indexes(ds) for ds in runs_list]

    for dim in _concat_dims_for_runs(prepared):
        dim_datasets = [ds for ds in prepared if dim in ds.dims]
        other_datasets = [ds for ds in prepared if dim not in ds.dims]
        prepared = _concat_if_possible(dim_datasets, dim=dim) + other_datasets

    if len(prepared) == 1:
        return prepared[0]

    return xr.merge(
        prepared,
        join="outer",
        compat="no_conflicts",
        combine_attrs="drop_conflicts",
    )


def _metric_mean_extra_dims_from_operations(
    truth_data: xr.Dataset | xr.DataArray,
    *,
    operations: Sequence[_ResolvedMetricGroupOperation],
) -> dict[str, tuple[str, ...]] | None:
    extras: dict[str, list[str]] = {}

    for op in operations:
        if op.kind != "dimension" or op.mode != "reduce" or op.dim is None or op.kinds is None:
            continue
        if op.dim not in truth_data.dims:
            continue
        for kind in op.kinds:
            extras.setdefault(kind, [])
            if op.dim not in extras[kind]:
                extras[kind].append(op.dim)

    if not extras:
        return None

    return {
        kind: tuple(dims)
        for kind, dims in extras.items()
        if dims
    }


def _reindex_grouped_metric_output(
    ds: xr.Dataset,
    *,
    reference_truth_data: xr.Dataset,
) -> xr.Dataset:
    time_dim = reference_truth_data.earthml.guessed_dims.time
    if time_dim is None or time_dim not in reference_truth_data.dims:
        return ds
    if time_dim not in ds.dims:
        return ds
    return ds.reindex({time_dim: reference_truth_data[time_dim].values})


def _climatology_for_period(
    ds: xr.Dataset,
    period: Literal["month", "dayofyear", "season"] | None = None,
    basis: Literal["target_time", "reference_time"] = "target_time",
    leadtime_unit: str | None = None,
    leadtime_value: object | None = None,
) -> xr.Dataset:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Cannot compute climatology without a time dimension.")

    if period is None:
        return ds.mean(dim=time_dim, skipna=True)

    if basis == "target_time":
        return ds.earthml.climatology(groupby=period)

    grouping_time = _grouping_time_dataarray(
        ds,
        basis=basis,
        leadtime_unit=leadtime_unit,
        leadtime_value=leadtime_value,
    )
    if grouping_time is None or not hasattr(grouping_time, "dt"):
        raise ValueError(f"Cannot compute climatology for basis={basis!r}.")

    group_labels = getattr(grouping_time.dt, period).rename(period)

    group_values = [
        group_value
        for _, group_value in _metric_group_specs_from_data(
            ds,
            period,
            basis=basis,
            leadtime_unit=leadtime_unit,
            leadtime_value=leadtime_value,
        )
    ]

    climatology_slices: list[xr.Dataset] = []
    for group_value in group_values:
        selector = group_labels == group_value
        clim_group = ds.where(selector).mean(dim=time_dim, skipna=True)
        climatology_slices.append(clim_group.expand_dims({period: [group_value]}))

    if not climatology_slices:
        empty = ds.mean(dim=time_dim, skipna=True)
        return empty.expand_dims({period: []})

    return xr.concat(climatology_slices, dim=period)

def _apply_climatology_group_operation(
    data: xr.Dataset | xr.DataArray | None,
    *,
    op: _ResolvedMetricGroupOperation,
) -> xr.Dataset | xr.DataArray | None:
    if data is None:
        return None

    if op.kind == "time":
        if op.period is None or not op.values:
            return data

        period_dim = str(op.period)
        group_value = op.values[0]

        if period_dim not in data.dims:
            return data

        return data.sel({period_dim: [group_value]})

    if op.kind == "dimension":
        if op.dim is None or not op.values:
            return data

        out = data.sel({op.dim: list(op.values)}, drop=True)
        if op.mode == "average" and op.dim in out.dims:
            out = out.mean(dim=op.dim, skipna=True)
        return out

    return data


def _get_single_region_metrics(
    runs: RunsByModel,
    region_name: str,
    load_models: Sequence[str] = ("an", "fc", "pr"),
    run_filters: dict[str, object] | None = None,
    train_runs: RunsByModel | None = None,
    use_train_prediction_clim: bool = False, # only used if calculate_clim_from_train_period is True
    use_saved_mask: bool = True,
    external_mask_data: xr.Dataset | xr.DataArray | None = None,
    metric_names: str | Sequence[str] | None = None,
    metric_sections: str | Sequence[str] = ("scalar", "map", "timeseries"),
    metric_types: str | Sequence[str] | None = None,
    metric_groupings: dict[str, object] | None = None,
    metric_groupby_period: str | None = None,
    metric_groupby_basis: str | None = None,
    clim_period: str | None = None,
    show_progress: bool = True,
    output_clim: bool = True,
) -> tuple[
    MetricLeaf,
    ClimsByModel,
]:
    resolved_metric_groupings = normalize_metric_groupings(
        metric_groupings,
        metric_groupby_period=metric_groupby_period,
        metric_groupby_basis=metric_groupby_basis,
    )

    time_grouping = resolved_metric_groupings.get("time")
    grouping_period = time_grouping.period if isinstance(time_grouping, _ResolvedTimeGrouping) else None
    grouping_basis = time_grouping.basis if isinstance(time_grouping, _ResolvedTimeGrouping) else "target_time"
    effective_clim_period = _resolve_period(
        clim_period if clim_period is not None else grouping_period,
        param_name="clim_period",
    )
    if grouping_period is not None and effective_clim_period != grouping_period:
        raise ValueError(
            "When time grouping is enabled, clim_period must match the grouping period."
        )

    if len(load_models) < 2:
        raise ValueError("load_models must contain at least two entries: truth first, then at least one model")

    truth_model = load_models[0]
    requested_model_names = load_models[1:]

    if isinstance(metric_types, str):
        requested_metric_types = [metric_types]
    elif metric_types is not None:
        requested_metric_types = list(metric_types)
    else:
        requested_metric_types = list(METRIC_SECTIONS)

    if isinstance(metric_sections, str):
        requested_metric_sections = [metric_sections]
    else:
        requested_metric_sections = list(metric_sections)

    empty_metrics = {
        mt: {ms: xr.Dataset() for ms in requested_metric_sections}
        for mt in requested_metric_types
    }
    empty_clims = {model_name: None for model_name in load_models}

    if not runs:
        return empty_metrics, empty_clims

    if truth_model not in runs:
        raise ValueError(f"Truth model {truth_model!r} not found in region {region_name} runs")

    available_model_names = [m for m in requested_model_names if m in runs]
    if not available_model_names:
        raise ValueError(f"No requested comparison models found in region {region_name} runs")

    if isinstance(external_mask_data, xr.DataArray):
        mask_name = external_mask_data.name or "__mask__"
        external_mask_data = external_mask_data.to_dataset(name=mask_name)

    saved_mask_runs = runs.get("mask") if use_saved_mask else None

    truth_runs = runs[truth_model]
    truth_region = str(truth_runs.coords["region"].item()) if "region" in truth_runs.coords else ""
    leadtime_unit = _leadtime_unit_from_data(truth_runs)

    # These are the experiment axes created by load_all_exp_from_folder
    effective_run_filters = _merge_run_filters_with_metric_groupings(
        run_filters,
        metric_groupings=resolved_metric_groupings,
    )
    run_dims = _resolve_run_dims(
        truth_runs,
        metric_groupings=resolved_metric_groupings,
    )

    run_indexers = _build_run_indexers(truth_runs, run_dims, run_filters=effective_run_filters)

    metric_runs: dict[str, dict[str, list[xr.Dataset]]] = {
        mt: {ms: [] for ms in requested_metric_sections}
        for mt in requested_metric_types
    }
    clim_runs: dict[str, list[xr.Dataset]] = {
        model_name: []
        for model_name in load_models
    }

    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            auto_refresh=True,
            refresh_per_second=6,
            transient=False,
        )
        if show_progress else nullcontext()
    )

    with progress_cm as prog:
        run_task = prog.add_task("Computing metric bundles", total=len(run_indexers)) if prog is not None else None
        bundle_task = prog.add_task("Computing bundle metrics", total=1, visible=False) if prog is not None else None

        for indexers in run_indexers:
            indexer_str = ", ".join(f"{dim}={value}" for dim, value in indexers.items()) or "all runs"
            if prog is not None and run_task is not None:
                prog.update(run_task, description=f"Computing metrics | {indexer_str}")
            selected_leadtime_value = indexers.get("leadtime")

            truth_data = truth_runs.sel(indexers, drop=True) if indexers else truth_runs
            if not truth_data.data_vars:
                if prog is not None and run_task is not None:
                    prog.advance(run_task)
                continue

            # Skip empty truth slices
            truth_arr = truth_data.to_array()
            if not bool(truth_arr.notnull().any()):
                if prog is not None and run_task is not None:
                    prog.advance(run_task)
                continue

            model_names_this_run = []
            model_data = []
            for model_name in available_model_names:
                ds = runs[model_name]
                ds = ds.sel(indexers, drop=True) if indexers else ds
                if not ds.data_vars:
                    continue
                if not bool(ds.to_array().notnull().any()):
                    continue
                model_names_this_run.append(model_name)
                model_data.append(ds)

            if not model_data:
                if prog is not None and run_task is not None:
                    prog.advance(run_task)
                continue

            mask_data = combine_masks(
                project_mask_to_reference_grid(
                    select_mask_for_indexers(saved_mask_runs, indexers),
                    truth_data,
                ),
                project_mask_to_reference_grid(
                    select_mask_for_indexers(external_mask_data, indexers),
                    truth_data,
                ),
            )

            clim_by_model: dict[str, xr.Dataset | None] = {truth_model: None, **{m: None for m in model_names_this_run}}

            if train_runs is not None:
                if truth_model in train_runs:
                    truth_train = train_runs[truth_model]
                    truth_train = truth_train.sel(indexers, drop=True) if indexers else truth_train
                    if truth_train.data_vars and bool(truth_train.to_array().notnull().any()):
                        clim_by_model[truth_model] = _climatology_for_period(
                            ds=truth_train,
                            period=effective_clim_period,
                            basis=grouping_basis,
                            leadtime_value=selected_leadtime_value,
                            leadtime_unit=leadtime_unit,
                        )

                for model_name in model_names_this_run:
                    if model_name == "pr" and not use_train_prediction_clim:
                        clim_by_model[model_name] = clim_by_model[truth_model]
                        continue

                    if model_name in train_runs:
                        train_model = train_runs[model_name]
                        train_model = train_model.sel(indexers, drop=True) if indexers else train_model
                        if train_model.data_vars and bool(train_model.to_array().notnull().any()):
                            clim_by_model[model_name] = _climatology_for_period(
                                ds=train_model,
                                period=effective_clim_period,
                                basis=grouping_basis,
                                leadtime_value=selected_leadtime_value,
                                leadtime_unit=leadtime_unit,
                            )

                if "pr" in clim_by_model and clim_by_model["pr"] is None:
                    clim_by_model["pr"] = clim_by_model[truth_model]

            clim_data = [clim_by_model[truth_model], *[clim_by_model[m] for m in model_names_this_run]]

            metric_group_operations = _build_metric_group_operations(
                truth_data,
                metric_groupings=resolved_metric_groupings,
                leadtime_unit=leadtime_unit,
                leadtime_value=selected_leadtime_value,
            )

            for operation_idx, operations in enumerate(metric_group_operations, start=1):
                group_truth_data: xr.Dataset | xr.DataArray | None = truth_data
                group_model_data: Sequence[xr.Dataset | xr.DataArray | None] = model_data
                group_mask_data: xr.Dataset | xr.DataArray | None = mask_data
                group_clim_data: Sequence[xr.Dataset | xr.DataArray | None] = clim_data
                has_time_group = False

                for op in operations:
                    if op.kind == "time":
                        has_time_group = True
                    group_truth_data = _apply_metric_group_operation(
                        group_truth_data,
                        op=op,
                        leadtime_unit=leadtime_unit,
                        leadtime_value=selected_leadtime_value,
                    )
                    group_model_data = [
                        _apply_metric_group_operation(
                            ds,
                            op=op,
                            leadtime_unit=leadtime_unit,
                            leadtime_value=selected_leadtime_value,
                        )
                        for ds in group_model_data
                    ]
                    group_mask_data = _apply_metric_group_operation(
                        group_mask_data,
                        op=op,
                        leadtime_unit=leadtime_unit,
                        leadtime_value=selected_leadtime_value,
                    )
                    group_clim_data = [
                        _apply_climatology_group_operation(
                            ds,
                            op=op,
                        )
                        for ds in group_clim_data
                    ]

                group_time_dim = group_truth_data.earthml.guessed_dims.time
                if group_time_dim is not None and group_time_dim in group_truth_data.dims:
                    if int(group_truth_data.sizes.get(group_time_dim, 0)) == 0:
                        continue

                if output_clim:
                    # Accumulate climatologies by model
                    group_clim_by_model = dict(
                        zip([truth_model, *model_names_this_run], group_clim_data)
                    )

                    for model_name, clim_ds in group_clim_by_model.items():
                        if clim_ds is None or not clim_ds.data_vars:
                            continue

                        ds = clim_ds

                        if operations:
                            ds = _expand_metric_group_labels(
                                ds,
                                operations=operations,
                            )
                        elif resolved_metric_groupings:
                            ds = _expand_all_metric_group_labels(
                                ds,
                                metric_groupings=resolved_metric_groupings,
                            )

                        ds.attrs[METRIC_GROUPINGS_ATTR] = metric_groupings_signature(
                            resolved_metric_groupings
                        )

                        if indexers:
                            ds = ds.expand_dims({
                                dim: [value] for dim, value in indexers.items()
                            })

                        if truth_region:
                            ds = ds.assign_coords(region=truth_region)

                        ds = _ensure_dimension_indexes(ds)
                        clim_runs[model_name].append(ds)

                metric_mean_extra_dims = _metric_mean_extra_dims_from_operations(
                    group_truth_data,
                    operations=operations,
                )

                group_dm = DeterministicMetrics(
                    truth_data=group_truth_data,
                    model_data=group_model_data,
                    truth_name=truth_model,
                    model_names=model_names_this_run,
                    clim_data=group_clim_data,
                    mask_data=group_mask_data,
                )

                group_cm = CorrelationMetrics(
                    truth_data=group_truth_data,
                    model_data=group_model_data,
                    truth_name=truth_model,
                    model_names=model_names_this_run,
                    clim_data=group_clim_data,
                    mask_data=group_mask_data,
                )

                group_pm = ProbabilisticMetrics(
                    truth_data=group_truth_data,
                    model_data=group_model_data,
                    truth_name=truth_model,
                    model_names=model_names_this_run,
                    clim_data=group_clim_data,
                    mask_data=group_mask_data,
                )

                bundle_progress_hook = None
                if prog is not None and bundle_task is not None:
                    bundle_total = count_standard_metric_bundle_steps(
                        deterministic=group_dm,
                        correlation=group_cm,
                        probabilistic=group_pm,
                        metric_names=metric_names,
                        metric_kinds=metric_sections,
                    )
                    bundle_desc = (
                        f"Computing bundle metrics | {indexer_str} | "
                        f"group {operation_idx}/{len(metric_group_operations)}"
                    )
                    prog.update(
                        bundle_task,
                        description=bundle_desc,
                        total=max(bundle_total, 1),
                        completed=0,
                        visible=bundle_total > 0,
                    )

                    def _bundle_progress_hook(metric_name: str, metric_type: str, kind: str) -> None:
                        prog.update(
                            bundle_task,
                            description=f"{bundle_desc} | {metric_type}/{kind}/{metric_name}",
                        )
                        prog.advance(bundle_task)

                    bundle_progress_hook = _bundle_progress_hook

                metric_bundle = _compute_metric_bundle(
                    group_dm,
                    group_cm,
                    group_pm,
                    norm="std",
                    clim_period=effective_clim_period,
                    metric_names=metric_names,
                    metric_sections=metric_sections,
                    metric_types=metric_types,
                    metric_progress_hook=bundle_progress_hook,
                    metric_mean_extra_dims=metric_mean_extra_dims,
                )

                if prog is not None and bundle_task is not None:
                    prog.update(bundle_task, visible=False)

                for metric_type, section_dict in metric_bundle.items():
                    for section, ds in section_dict.items():
                        if ds is None or not ds.data_vars:
                            continue

                        if has_time_group:
                            ds = _reindex_grouped_metric_output(
                                ds,
                                reference_truth_data=truth_data,
                            )
                        if operations:
                            ds = _expand_metric_group_labels(
                                ds,
                                operations=operations,
                            )
                        elif resolved_metric_groupings:
                            ds = _expand_all_metric_group_labels(
                                ds,
                                metric_groupings=resolved_metric_groupings,
                            )
                        ds.attrs[METRIC_GROUPINGS_ATTR] = metric_groupings_signature(
                            resolved_metric_groupings
                        )
                        if indexers:
                            ds = ds.expand_dims({
                                dim: [value] for dim, value in indexers.items()
                            })
                        if truth_region:
                            ds = ds.assign_coords(region=truth_region)
                        ds = _ensure_dimension_indexes(ds)

                        metric_runs[metric_type][section].append(ds)

            if prog is not None and run_task is not None:
                prog.advance(run_task)

    # print(metric_runs)
    metrics: MetricLeaf = {}
    combine_steps = sum(
        1
        for section_dict in metric_runs.values()
        for runs_list in section_dict.values()
        if runs_list
    )

    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            auto_refresh=True,
            refresh_per_second=6,
            transient=False,
        )
        if show_progress else nullcontext()
    )

    with progress_cm as prog:
        combine_task = prog.add_task("Combining metric sections", total=combine_steps) if prog is not None else None
        for metric_type, section_dict in metric_runs.items():
            metrics[metric_type] = {}

            for section, runs_list in section_dict.items():
                if prog is not None and combine_task is not None and runs_list:
                    prog.update(combine_task, description=f"Combining {metric_type}/{section}")

                metrics[metric_type][section] = (
                    _combine_metric_run_datasets(runs_list)
                    if runs_list
                    else xr.Dataset()
                )

                if prog is not None and combine_task is not None and runs_list:
                    prog.advance(combine_task)

    clims: ClimsByModel = {}
    clim_combine_steps = sum(1 for runs_list in clim_runs.values() if runs_list)

    if output_clim:
        with progress_cm as prog:
            combine_task = prog.add_task("Combining climatologies", total=clim_combine_steps) if prog is not None else None
            for model_name, runs_list in clim_runs.items():
                if prog is not None and combine_task is not None and runs_list:
                    prog.update(combine_task, description=f"Combining climatologies | {model_name}")

                clims[model_name] = (
                    _combine_metric_run_datasets(runs_list)
                    if runs_list else None
                )

                if prog is not None and combine_task is not None and runs_list:
                    prog.advance(combine_task)

    return metrics, clims


def get_metrics(
    runs_by_region: RunsByRegion,
    load_models: Sequence[str] = ("an", "fc", "pr"),
    run_filters: dict[str, object] | None = None,
    train_runs: RunsByRegion | None = None,
    use_train_prediction_clim: bool = False,
    use_saved_mask: bool = True,
    external_mask_data: xr.Dataset | xr.DataArray | None = None,
    metric_names: str | Sequence[str] | None = None,
    metric_sections: str | Sequence[str] = ("scalar", "map", "timeseries"),
    metric_types: str | Sequence[str] | None = None,
    metric_groupings: dict[str, object] | None = None,
    metric_groupby_period: str | None = None,
    metric_groupby_basis: str | None = None,
    clim_period: str | None = None,
    show_progress: bool = True,
    output_clim: bool = False,
) -> tuple[
    MetricsByRegion,
    ClimsByRegion,
]:
    """
    Load runs and metrics grouped by region.

    Region is a top-level dictionary key in the returned structures rather than
    an xarray dimension, so each leaf dataset stays on a single spatial domain.
    Each metrics leaf preserves the historical single-region shape:
    ``metrics_by_region[region][metric_type][section] -> xr.Dataset``.
    """

    metrics_by_region: MetricsByRegion = {}
    clims_by_region: ClimsByRegion = {}

    for region_name, region_runs in runs_by_region.items():
        region_filters = dict(run_filters or {})
        region_filters["region"] = [region_name]

        metrics_region, clims_region = _get_single_region_metrics(
            runs=region_runs,
            region_name=region_name,
            load_models=load_models,
            run_filters=region_filters,
            train_runs=train_runs[region_name] if train_runs is not None else None,
            use_train_prediction_clim=use_train_prediction_clim,
            use_saved_mask=use_saved_mask,
            external_mask_data=external_mask_data,
            metric_names=metric_names,
            metric_sections=metric_sections,
            metric_types=metric_types,
            metric_groupings=metric_groupings,
            metric_groupby_period=metric_groupby_period,
            metric_groupby_basis=metric_groupby_basis,
            clim_period=clim_period,
            show_progress=show_progress,
            output_clim=output_clim,
        )

        metrics_by_region[str(region_name)] = metrics_region
        clims_by_region[str(region_name)] = clims_region

    return metrics_by_region, clims_by_region


def _build_run_indexers(
    truth_runs: xr.Dataset,
    run_dims: Sequence[str],
    run_filters: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    if not run_dims:
        return [{}]

    coord_values = []
    for dim in run_dims:
        values = truth_runs.coords[dim].values.tolist()
        allowed = None if run_filters is None else run_filters.get(dim)
        if allowed is not None:
            allowed_values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            keep_empty = dim == "variant"
            allowed_strings = {
                str(value)
                for value in allowed_values
                if value is not None and (keep_empty or str(value) != "")
            }
            values = [value for value in values if str(value) in allowed_strings]
        coord_values.append(values)

    if any(len(values) == 0 for values in coord_values):
        return []

    out = []

    for values in product(*coord_values):
        indexers = dict(zip(run_dims, values))
        ds = truth_runs.sel(indexers, drop=True)

        if not ds.data_vars:
            continue
        if not bool(ds.to_array().notnull().any()):
            continue

        out.append(indexers)

    return out


def save_metrics(
    metrics: MetricLeaf,
    base_folder: str,
    vars_list: Sequence[str],
    metric_names: Sequence[str] | None = None,
    models_diff: Sequence[str] = ("fc", "pr"),
    partition_cols: Sequence[str] = ("train_period", "loss", "model", "metric", "variable"),
    kind: str = "scalar",
    metric_type: str = "all_dims",
):
    """Save metrics in parquet format. Return dict of dataframes and paths."""

    out = {}

    for diff in ("no", "delta", "ratio"):
        df = metrics_to_df_single_region(
            metrics=metrics,
            variables=vars_list,
            metric_names=metric_names,
            kind=kind,
            metric_type=metric_type,
            diff=diff,
            models=models_diff if diff in ("delta", "ratio") else None,
        )

        parquet_path = build_parquet_path(base_folder, vars_list, metric_names, diff)
        Path(parquet_path).mkdir(parents=True, exist_ok=True)

        df.to_parquet(
            parquet_path,
            partition_cols=list(partition_cols),
            engine="pyarrow",
        )

        out[diff] = (df, parquet_path)

    return out
