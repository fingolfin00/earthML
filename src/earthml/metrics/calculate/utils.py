from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Sequence
from itertools import product

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...misc import Table
from ...logging import get_logger, log_renderable

from .dataclasses import CalculateMetricsConfig
from .constants import (
    METRIC_DISPLAY_NAMES,
    AXIS_DISPLAY_NAMES,
    VARIABLE_DISPLAY_NAMES,
    VARIABLE_COLORS,
    VARIABLE_SUBFOLDERS,
    OUTPUT_GROUP_SUBFOLDERS,
    CONTEXT_AXES,
)


logger = get_logger(__name__)


def _table_value(data: object) -> object:
    if data is None or not is_dataclass(data):
        return data
    return {
        field.name: Table._to_pretty(getattr(data, field.name), max_depth=4)
        for field in fields(data)
    }

def _print_user_config_table(
    config: CalculateMetricsConfig
) -> None:
    config_sections = [
        ("global", _table_value(config.global_config)),
        ("models", _table_value(config.models)),
        ("variable_colors", _table_value(config.variable_colors)),
        ("timeseries", _table_value(config.timeseries)),
        ("maps", _table_value(config.maps)),
        ("profiles", _table_value(config.profiles)),
        ("scalar_tables", _table_value(config.scalar_tables)),
        ("metric_vs_delta", _table_value(config.metric_vs_delta)),
        ("scoreboards", _table_value(config.scoreboards)),
    ]

    chunk_size = 2
    for chunk_idx in range(0, len(config_sections), chunk_size):
        chunk = {
            key: value
            for key, value in config_sections[chunk_idx:chunk_idx + chunk_size]
            if value is not None
        }
        if not chunk:
            continue
        title_suffix = f" ({chunk_idx // chunk_size + 1})"
        log_renderable(
            Table(chunk, title=f"Calculate Metrics Config{title_suffix}").table,
            logger=logger,
        )


def _load_external_mask_dataset(
    config: CalculateMetricsConfig
) -> xr.Dataset | None:
    mask_path = config.global_config.external_mask_path
    if mask_path is None:
        return None

    mask_path = Path(mask_path).expanduser()
    if not mask_path.exists():
        raise FileNotFoundError(f"External mask path not found: {mask_path}")

    if mask_path.is_dir() or mask_path.suffix == ".zarr":
        mask_ds = xr.open_zarr(mask_path)
    else:
        mask_ds = xr.open_dataset(mask_path)

    mask_variable = config.global_config.external_mask_variable
    if mask_variable is not None:
        if mask_variable not in mask_ds.data_vars:
            raise ValueError(
                f"Requested external mask variable {mask_variable!r} not found. "
                f"Available variables: {list(mask_ds.data_vars)}"
            )
        mask_ds = mask_ds[[mask_variable]]

    if not mask_ds.data_vars:
        raise ValueError(f"External mask dataset at {mask_path} contains no data variables")

    logger.info(f"Using external mask: {mask_path} (variables={list(mask_ds.data_vars)})")
    return mask_ds


def _get_da_unit(
    data: xr.Dataset | xr.DataArray,
    variable: str | None = None
) -> str | None:
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

def _get_variable_units_from_runs(
    runs: dict[str, xr.Dataset],
    variables: list[str]
) -> dict[str, str | None]:
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


def _infer_leadtime_unit_from_configs(
    exp_root: Path
) -> str:
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
            leadtime_unit = leadtime_rd.unit
        except ValueError as exc:
            raise ValueError(f"Unsupported leadtime in experiment config {cfg_path}: {leadtime_rd}") from exc

        units.add(leadtime_unit)

    if not units:
        return ""
    if len(units) > 1:
        raise ValueError(f"Multiple leadtime units found in {exp_root}: {sorted(units)}")
    return next(iter(units))


def _ordered_unique_variables(
    variables: list[str]
) -> list[str]:
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


def _infer_region_extent_from_configs(
    exp_root: Path,
    *,
    selected_region: str = "",
) -> tuple[float, float, float, float] | None:
    extents = _infer_region_extents_from_configs(
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

def _infer_region_extents_from_configs(
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


def _sanitize_filename_fragment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "all"

def _context_filename_suffix(
    context_values: dict[str, object]
) -> str:
    if not context_values:
        return "all"
    return "__".join(
        f"{field}_{_sanitize_filename_fragment('default' if field == 'variant' and str(value) == '' else str(value))}"
        for field, value in context_values.items()
    )

def _merge_filename_contexts(
    *contexts: dict[str, object]
) -> str:
    merged: dict[str, object] = {}
    for context in contexts:
        for key, value in context.items():
            if value is None or str(value) == "":
                continue
            merged[key] = value
    return "" if not merged else _context_filename_suffix(merged)


def _selection_filter_values(
    values: object,
    *,
    keep_empty: bool = False
) -> list[object]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return [
            value for value in values
            if value is not None and (keep_empty or str(value) != "")
        ]
    if not keep_empty and str(values) == "":
        return []
    return [values]

def _filters_filename_context(filters: dict | None) -> dict[str, object]:
    if not filters:
        return {}

    context: dict[str, object] = {}
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = _selection_filter_values(value, keep_empty=(key == "variant"))
            if not values:
                continue
            context[key] = (
                ("default" if key == "variant" and str(values[0]) == "" else values[0])
                if len(values) == 1
                else "-".join(
                    "default" if key == "variant" and str(item) == "" else str(item)
                    for item in values
                )
            )
            continue
        context[key] = "default" if key == "variant" and str(value) == "" else value
    return context


def _ordered_unique_variables(
    variables: list[str]
) -> list[str]:
    unique_variables: list[str] = []
    for variable in variables:
        variable_str = str(variable)
        if variable_str not in unique_variables:
            unique_variables.append(variable_str)
    return unique_variables


def _format_metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " "))

def _format_label_with_unit(
    label: str,
    unit: str | None
) -> str:
    return f"{label} [{unit}]" if unit else label

def _metric_unit(
    metric_name: str,
    base_unit: str | None
) -> str | None:
    if metric_name in {"rmse", "crmse", "mae", "bias", "error_std", "crps", "spread"}:
        return base_unit
    return None

def _format_metric_label(
    metric_name: str,
    *,
    base_unit: str | None = None
) -> str:
    return _format_label_with_unit(_format_metric_display_name(metric_name), _metric_unit(metric_name, base_unit))


def _get_variable_colors(
    *,
    variables: list[str],
    base_colors: Sequence[str] | dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    ordered_variables = _ordered_unique_variables(variables)
    if base_colors is None:
        palette = tuple(VARIABLE_COLORS.values())
    elif isinstance(base_colors, dict):
        palette = tuple(base_colors.values())
    else:
        palette = tuple(base_colors)
    if not palette:
        return {}

    color_map = {
        variable: palette[idx % len(palette)]
        for idx, variable in enumerate(ordered_variables)
    }

    if overrides:
        for variable, color in overrides.items():
            if variable in color_map:
                color_map[variable] = color
    return color_map

def _get_variable_table_color(
    *,
    variable: str,
    variables: list[str]
) -> str:
    variable_colors = _get_variable_colors(variables=variables)
    default_color = next(iter(VARIABLE_COLORS.values()), "#4c4c4c")
    if not variable_colors:
        return default_color
    return variable_colors.get(str(variable), default_color)


def _normalize_profile_axis_fields(
    x_axis: str | Sequence[str]
) -> tuple[str, ...]:
    if isinstance(x_axis, str):
        normalized = x_axis
        for delimiter in (",", "+", "|"):
            normalized = normalized.replace(delimiter, ",")
        fields = [field.strip() for field in normalized.split(",") if field.strip()]
        return tuple(fields)
    return tuple(str(field).strip() for field in x_axis if str(field).strip())


def _get_variable_plot_folder(
    *,
    plot_root: Path,
    variable: str
) -> Path:
    variable_folder = plot_root / variable
    variable_folder.mkdir(parents=True, exist_ok=True)
    return variable_folder

def _get_variable_subfolder(*, plot_root: Path, variable: str, subfolder: str) -> Path:
    if subfolder not in VARIABLE_SUBFOLDERS:
        raise ValueError(f"Unsupported variable subfolder {subfolder!r}. Expected one of {VARIABLE_SUBFOLDERS}.")
    output_folder = _get_variable_plot_folder(plot_root=plot_root, variable=variable) / subfolder
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder

def _table_output_folder(
    *,
    plot_root: Path,
    variable: str | None = None,
) -> Path:
    folder_variable = variable if variable is not None else "all_variables"
    return _get_variable_subfolder(plot_root=plot_root, variable=folder_variable, subfolder="tables")


def _format_variable_display_name(variable: str) -> str:
    return VARIABLE_DISPLAY_NAMES.get(variable, str(variable))

def _format_variable_short_name(variable: str) -> str:
    return str(variable)


def _apply_df_filters(
    df: pd.DataFrame,
    filters: dict | None
) -> pd.DataFrame:
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
    x_axis: str | Sequence[str],
) -> pd.DataFrame:
    if not filters:
        return df
    x_axis_fields = set(_normalize_profile_axis_fields(x_axis))
    effective_filters = {
        col: allowed
        for col, allowed in filters.items()
        if col not in x_axis_fields
    }
    return _apply_df_filters(df, effective_filters)

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

def _mix_colors(color_a: str, color_b: str, weight: float) -> str:
    rgb_a = np.array(mcolors.to_rgb(color_a))
    rgb_b = np.array(mcolors.to_rgb(color_b))
    mixed = (1 - weight) * rgb_a + weight * rgb_b
    return mcolors.to_hex(mixed)

def _safe_relative_change(
    values: pd.Series | pd.DataFrame,
    baseline: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    baseline_safe = baseline.where(~np.isclose(baseline, 0.0), np.nan)
    result = values / baseline_safe
    return result.where(np.isfinite(result), np.nan)

def _single_filter_value(filters: dict | None, key: str) -> str:
    if not filters or key not in filters or filters[key] is None:
        return ""
    allowed = filters[key]
    values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
    normalized = [str(value) for value in values if value is not None and str(value) != ""]
    if len(normalized) == 1:
        return normalized[0]
    return ""

def _format_axis_display_name(
    axis_name: str | Sequence[str],
    *,
    leadtime_unit: str | None = None
) -> str:
    axis_fields = _normalize_profile_axis_fields(axis_name)
    if len(axis_fields) > 1:
        return " + ".join(
            _format_axis_display_name(field, leadtime_unit=leadtime_unit if field == "leadtime" else None)
            for field in axis_fields
        )
    axis_field = axis_fields[0]
    if axis_field == "leadtime" and leadtime_unit:
        return f"Lead time [{leadtime_unit}]"
    return AXIS_DISPLAY_NAMES.get(axis_field, axis_field.replace("_", " ").title())

def _format_context_label(
    *,
    field: str,
    value: object,
    leadtime_unit: str | None = None
) -> str:
    field_name = _format_axis_display_name(field, leadtime_unit=leadtime_unit)
    if field == "leadtime" and leadtime_unit:
        return f"{field_name}={value} {leadtime_unit}"
    return f"{field_name}={value}"

def _context_title_suffix(
    context_values: dict[str, object],
    *,
    leadtime_unit: str | None = None
) -> str:
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

def _apply_xarray_filters(
    data: xr.Dataset | xr.DataArray,
    filters: dict | None
) -> xr.Dataset | xr.DataArray:
    if filters is None:
        return data

    out = data
    for dim, allowed in filters.items():
        if allowed is None or (dim not in out.dims and dim not in out.coords):
            continue
        values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        if dim in out.dims:
            out = out.sel({dim: list(values)})
            continue

        scalar_value = _scalar_coord_value(out, dim)
        if scalar_value is None:
            continue

        if any(scalar_value == value or str(scalar_value) == str(value) for value in values):
            continue

        if out.dims:
            first_dim = next(iter(out.dims))
            out = out.isel({first_dim: slice(0, 0)})
        elif isinstance(out, xr.Dataset):
            out = out.drop_vars(list(out.data_vars))
        else:
            out = out.expand_dims(__empty_filter__=[]).isel(__empty_filter__=slice(0, 0))
            out = out.drop_vars("__empty_filter__", errors="ignore")
    return out


def _scalar_coord_value(data: xr.Dataset | xr.DataArray, dim: str) -> object | None:
    if dim not in data.coords and dim not in data.dims:
        return None
    values = np.asarray(data[dim].values)
    if values.size != 1:
        return None
    value = values.reshape(-1)[0]
    return value.item() if isinstance(value, np.generic) else value

def _context_selection_entries(
    data: xr.Dataset | xr.DataArray,
    *,
    context_dims: tuple[str, ...] = CONTEXT_AXES,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    dims_present = [dim for dim in context_dims if dim in data.dims]
    if not dims_present:
        return [({}, {})]

    varying_dims = [dim for dim in dims_present if data.sizes.get(dim, 1) > 1]
    if not varying_dims:
        labels = {
            dim: _scalar_coord_value(data, dim)
            for dim in dims_present
            if _scalar_coord_value(data, dim) is not None and str(_scalar_coord_value(data, dim)) != ""
        }
        return [({}, labels)]

    entries: list[tuple[dict[str, object], dict[str, object]]] = []
    for values in product(*[data[dim].values.tolist() for dim in varying_dims]):
        selection = {dim: value for dim, value in zip(varying_dims, values)}
        labels = {}
        for dim in dims_present:
            if dim in selection:
                if str(selection[dim]) != "":
                    labels[dim] = selection[dim]
            else:
                scalar_value = _scalar_coord_value(data, dim)
                if scalar_value is not None and str(scalar_value) != "":
                    labels[dim] = scalar_value
        entries.append((selection, labels))
    return entries

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

def _members_arg(
    da: xr.DataArray,
    *,
    enable_spread: bool
) -> xr.DataArray | None:
    return da if enable_spread and _has_multi_realization(da) else None


def _get_variable_output_group_folder(
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
    output_folder = _get_variable_subfolder(
        plot_root=plot_root,
        variable=variable,
        subfolder=subfolder,
    ) / output_group
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder

def _ensure_grouped_output_folders(
    *,
    plot_root: Path,
    variable: str,
    subfolder: str
) -> dict[str, Path]:
    return {
        output_group: _get_variable_output_group_folder(
            plot_root=plot_root,
            variable=variable,
            subfolder=subfolder,
            output_group=output_group,
        )
        for output_group in OUTPUT_GROUP_SUBFOLDERS
    }


def _get_variable_output_item_folder(
    *,
    plot_root: Path,
    variable: str,
    subfolder: str,
    output_group: str,
    item_name: str,
) -> Path:
    output_folder = _get_variable_output_group_folder(
        plot_root=plot_root,
        variable=variable,
        subfolder=subfolder,
        output_group=output_group,
    ) / item_name
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder


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
