from pathlib import Path

import numpy as np
import xarray as xr

from rich.progress import Progress

from ...logging import get_logger
from ...plots import plot_temporal_mean_map

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


logger = get_logger(__name__)


MapLimitTuple = tuple[float | None, float | None]


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
    realization_mode: bool = True,
    region_extents: dict[str, tuple[float, float, float, float]] | None = None,
    progress: Progress | None = None,
    task_id: int | None = None,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_order = [model_name for model_name in load_models if model_name in runs]
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
                    total += _map_realization_count(da_sel, realization_mode=realization_mode)

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
                            per_model_maps[str(model_name)] = da_sel

                        if not per_model_maps:
                            continue

                        total += sum(
                            _map_realization_count(da_model, realization_mode=realization_mode)
                            for da_model in per_model_maps.values()
                        )

                        reference_model = config.models.reference_model
                        corrected_model = config.models.corrected_model
                        if reference_model in per_model_maps and corrected_model in per_model_maps:
                            total += _common_map_realization_count(
                                per_model_maps[reference_model],
                                per_model_maps[corrected_model],
                                realization_mode=realization_mode,
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
                            description=f"Maps | {plot_region or 'unknown'} | {variable} | field | {model_name}",
                        )
                    plot_temporal_mean_map(
                        da_plot,
                        save_path=save_path,
                        title=_map_title(
                            _format_variable_display_name(variable),
                            config.models.display_names.get(model_name, str(model_name)),
                            f"Temporal mean{_realization_label(realization_value)}",
                            _context_title_suffix(context_values, leadtime_unit=leadtime_unit),
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
                    logger.debug(f"Saved field map for model={model_name} region={plot_region or 'unknown'} to: {save_path}")

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
                                runs.get(config.models.truth_model, xr.Dataset()),
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
                                f"{'_'.join(filename_parts)}{_realization_filename_fragment(realization_value)}_map.png"
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
