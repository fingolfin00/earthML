from pathlib import Path
from typing import Literal, Sequence, cast
from dataclasses import dataclass

from datetime import datetime

import numpy as np
import xarray as xr

import earthml

from dask.diagnostics.progress import ProgressBar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import (
    BoundaryNorm,
    TwoSlopeNorm,
    Colormap,
    to_rgb,
)
from matplotlib.cm import ScalarMappable

from ..base import Settings, ClimPeriod
from ..metrics import safe_percent


PlotMode = Literal["scalar_diff_scatter"]

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from .defaults import (
    DEFAULT_IMPROVEMENT_PLOT_CONFIG,
    DEFAULT_PLOT_CONFIG,
    VARIABLE_NAMES,
    VARIABLE_UNITS,
    UNIT_CONVERSIONS,
    METRIC_IMPROVEMENT,
    METRIC_NAMES,
    METRIC_SKILL_UNITS,
    METRIC_UNITS,
    MODEL_COLORS,
    SERIES_COLORS,
    TRANSLATION_TABLE,
)


def safe_label(x: object) -> str:
    if x is None:
        return "all"
    return str(x).translate(TRANSLATION_TABLE)


def is_skill_metric(metric: str) -> bool:
    return (
        metric.endswith("_skill_clim")
        or metric.startswith("roc_")
        or metric in (
            "corr",
            "acc",
            "acc_spatial",
            "r2",
            "r2_anom",
            "spread_skill_ratio",
        )
    )

def is_skill_model(model: str) -> bool:
    return "_vs_" in model


def get_plot_config(
    var: str,
    metric: str,
    var_plot_config: dict,
) -> dict:
    if metric not in DEFAULT_PLOT_CONFIG:
        raise KeyError(f"No default plot config for metric {metric!r}")

    cfg = DEFAULT_PLOT_CONFIG[metric].copy()
    cfg.update(var_plot_config.get(var, {}).get(metric, {}))
    return cfg

def get_skill_plot_config(
    var: str,
    metric: str,
    var_plot_config: dict,
    impro_plot_config: dict,
    improvement_unit: Literal["%", "Δ"] | None = None,
) -> dict:
    if metric.endswith("_skill_clim"):
        cfg = DEFAULT_PLOT_CONFIG[metric].copy()
        cfg.update(var_plot_config.get(var, {}).get(metric, {}))
        return cfg

    unit = improvement_unit or METRIC_SKILL_UNITS[metric]
    cfg = DEFAULT_IMPROVEMENT_PLOT_CONFIG[unit].copy()

    custom_cfg = impro_plot_config.get(var, {}).get(metric, {})

    # Preferred format: {var: {metric: {"%": {...}, "Δ": {...}}}}.
    # Keep accepting the former flat per-metric format for compatibility.
    if "%" in custom_cfg or "Δ" in custom_cfg:
        cfg.update(custom_cfg.get(unit, {}))
    else:
        cfg.update(custom_cfg)

    return cfg


def lead_label(
    da: xr.DataArray,
    lead_value,
    leadtime_dim: str = "leadtime",
) -> str:
    if "seasonal_leadtime" not in da.coords:
        return str(lead_value)

    labels = da.coords["seasonal_leadtime"]

    if leadtime_dim in labels.dims:
        return str(labels.sel({leadtime_dim: lead_value}).item())

    return str(labels.item())


def select_metric(
    ds: xr.Dataset,
    *,
    var: str,
    metric: str,
    model: str,
    start_period: str | int = "all",
) -> xr.DataArray:
    da = ds[var]

    if "metric" in da.dims:
        da = da.sel(metric=metric)

    if "model" in da.dims:
        da = da.sel(model=model)

    if "start_period" in da.dims:
        da = da.sel(start_period=start_period)

    return da


def convert_to_da_list(
    ds: xr.DataArray | xr.Dataset | Sequence[xr.DataArray | xr.Dataset | None] | None,
    var: str,
) -> list[xr.DataArray | None]:
    if ds is None:
        return []

    if isinstance(ds, xr.DataArray):
        return [ds]

    if isinstance(ds, xr.Dataset):
        return [ds[var]]

    if isinstance(ds, Sequence) and not isinstance(ds, (str, bytes)):
        result: list[xr.DataArray | None] = []

        for d in ds:
            if d is None:
                result.append(None)
            elif isinstance(d, xr.Dataset):
                result.append(d[var])
            elif isinstance(d, xr.DataArray):
                result.append(d)
            else:
                raise TypeError(f"Type {type(d)} not supported.")

        return result

    raise TypeError(f"Type {type(ds)} not supported.")


def plot_profile(
    das: xr.DataArray | Sequence[xr.DataArray] | xr.Dataset | Sequence[xr.Dataset],
    *,
    var: str,
    metric: str,
    start_period: str | int,
    models: str | Sequence[str],
    out_file: Path,
    time_range: tuple[str, str],
    das_member: xr.DataArray | Sequence[xr.DataArray] | xr.Dataset | Sequence[xr.Dataset] | Sequence[None] | None = None,
    leadtime_dim: str = "leadtime",
    leadtime_unit: str = "months",
    period_dim: str = "start_date",
    realization_dim: str = "realization",
    spread: str = "std",
    plot_single_members: bool = False,
) -> None:
    if isinstance(models, str):
        models = [models]
    if len(models) != len(das):
        raise ValueError("Select same number of models and DataArrays")
    
    if das_member is not None and len(das) != len(das_member):
        raise ValueError("Select same number of DataArrays")

    plot_das = convert_to_da_list(das, var)
    if das_member is None:
        das_member = [None]*len(plot_das)
    plot_das_member = convert_to_da_list(das_member, var)

    fig, ax = plt.subplots(figsize=(12, 8))

    for model, da, da_member in zip(models, plot_das, plot_das_member, strict=True):
        color = MODEL_COLORS.get(model, None)

        if da is not None:
            da = da.reset_coords(drop=True)

            with ProgressBar():
                da = da.sel({period_dim: start_period}).compute()

            x = da[leadtime_dim].values

            ax.plot(
                x,
                da.values,
                linestyle="-",
                linewidth=1.4,
                label=f"{model} ensemble mean",
                color=color,
            )

        if da_member is not None:
            if realization_dim in da_member.dims:
                with ProgressBar():
                    da_member = da_member.sel({period_dim: start_period}).compute()

                da_member = da_member.reset_coords(drop=True)
                x = da_member[leadtime_dim].values

                if plot_single_members:
                    for i in range(da_member.sizes[realization_dim]):
                        ax.plot(
                            x,
                            da_member.isel({realization_dim: i}).values,
                            linewidth=0.6,
                            alpha=0.25,
                            color=color,
                        )

                member_mean = da_member.mean(realization_dim, skipna=True)

                if spread == "std":
                    member_std = da_member.std(realization_dim, skipna=True)
                    lower = member_mean - member_std
                    upper = member_mean + member_std
                elif spread == "minmax":
                    lower = da_member.min(realization_dim, skipna=True)
                    upper = da_member.max(realization_dim, skipna=True)
                else:
                    raise ValueError("spread must be 'std' or 'minmax'")

                ax.fill_between(
                    x,
                    lower.values,
                    upper.values,
                    alpha=0.18,
                    color=color,
                )

                ax.plot(
                    x,
                    member_mean.values,
                    linewidth=1.4,
                    linestyle="--",
                    label=f"{model} member mean",
                    color=color,
                )

    if metric in {"bias", "acc", "clim_acc", "spatial_acc"}:
        ax.axhline(0, linewidth=0.8)

    if metric in {"acc", "clim_acc", "spatial_acc"}:
        ax.set_ylim(-1, 1)

    start_period_str = "" if start_period=="all" else f" · start_period={start_period}"
    ax.set_title(f"{VARIABLE_NAMES[var]} · {METRIC_NAMES[metric]} · {safe_label(time_range)}{start_period_str}")

    ax.set_xlabel(f"{leadtime_dim} {leadtime_unit}")

    unit = VARIABLE_UNITS[var]
    unit_conversion = UNIT_CONVERSIONS.get(unit, (unit, 1.0))

    if isinstance(unit_conversion, dict):
        plot_unit, _ = unit_conversion[var]
    else:
        plot_unit, _ = unit_conversion

    ax.set_ylabel(
        f"{METRIC_NAMES[metric]} [{METRIC_UNITS[metric].format(unit=plot_unit)}]"
        if METRIC_UNITS[metric]
        else METRIC_NAMES[metric]
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(
    das: xr.DataArray | Sequence[xr.DataArray] | xr.Dataset | Sequence[xr.Dataset],
    *,
    var: str,
    metric: str,
    start_period: str | int,
    models: str | Sequence[str],
    lead_value: object,
    out_file: Path,
    time_range: tuple[str, str],
    das_member: xr.DataArray | Sequence[xr.DataArray] | xr.Dataset | Sequence[xr.Dataset] | Sequence[None] | None = None,
    leadtime_dim: str = "leadtime",
    time_dim: str = "time",
    realization_dim: str = "realization",
    spread: str = "std",
) -> None:
    if isinstance(models, str):
        models = [models]
    if len(models) != len(das):
        raise ValueError("Select same number of models and DataArrays")

    if das_member is not None and len(das) != len(das_member):
        raise ValueError("Select same number of DataArrays")

    plot_das = convert_to_da_list(das, var)
    if das_member is None:
        das_member = [None]*len(das)
    plot_das_member = convert_to_da_list(das_member, var)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    for da, da_member, model in zip(plot_das, plot_das_member, models):
        da = da.sel({leadtime_dim: lead_value}).squeeze(drop=True)

        if time_dim not in da.dims:
            print(f"Skipping timeseries: no {time_dim!r} in {da.dims}")
            return

        color = MODEL_COLORS.get(model, None)

        with ProgressBar():
            da = da.compute()
        x = da[time_dim].values

        if da_member is not None:
            da_member = da_member.sel({leadtime_dim: lead_value}).squeeze(drop=True)
            with ProgressBar():
                da_member = da_member.compute()

            if realization_dim in da_member.dims:
                for i in range(da_member.sizes[realization_dim]):
                    ax.plot(
                        x,
                        da_member.isel({realization_dim: i}).values,
                        linewidth=0.6,
                        alpha=0.25,
                        color=color,
                    )

                member_mean = da_member.mean(realization_dim)

                if spread == "std":
                    member_std = da_member.std(realization_dim)
                    lower = member_mean - member_std
                    upper = member_mean + member_std
                elif spread == "minmax":
                    lower = da_member.min(realization_dim)
                    upper = da_member.max(realization_dim)
                else:
                    raise ValueError("spread must be 'std' or 'minmax'")

                ax.fill_between(
                    x,
                    lower.values,
                    upper.values,
                    alpha=0.18,
                    color=color,
                )

                ax.plot(
                    x,
                    member_mean.values,
                    linewidth=1.4,
                    linestyle=":",
                    color=color,
                    label=f"{model} member mean",
                )

        ax.plot(
            x,
            da.values,
            linewidth=1.8,
            color=color,
            label=f"{model} ensemble mean",
        )

    if metric in {"bias", "acc", "clim_acc", "spatial_acc"}:
        ax.axhline(0, linewidth=0.8)

    if metric in {"acc", "clim_acc", "spatial_acc"}:
        ax.set_ylim(-1, 1)

    ax.set_title(f"{VARIABLE_NAMES[var]} · {METRIC_NAMES[metric]} · {safe_label(time_range)} · lead={lead_value}")
    ax.set_xlabel(time_dim)
    ax.set_ylabel(
        f"{METRIC_NAMES[metric]} [{METRIC_UNITS[metric].format(unit=UNIT_CONVERSIONS['K'][0])}]"
        if METRIC_UNITS[metric]
        else METRIC_NAMES[metric]
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def metric_style(
    var: str,
    metric: str,
    da: xr.DataArray | None = None,
    *,
    var_plot_config: dict = {},
    impro_plot_config: dict = {},
    is_skill: bool = False,
    improvement_unit: Literal["%", "Δ"] | None = None,
) -> tuple[Colormap, BoundaryNorm | TwoSlopeNorm, np.ndarray]:

    try:
        if is_skill:
            cfg = get_skill_plot_config(
                var,
                metric,
                var_plot_config,
                impro_plot_config,
                improvement_unit=improvement_unit,
            )
        else:
            cfg = get_plot_config(var, metric, var_plot_config)
    except KeyError:
        if da is None:
            raise

        vmin = float(da.quantile(0.02, skipna=True))
        vmax = float(da.quantile(0.98, skipna=True))
        ticks = np.linspace(vmin, vmax, 11)
        cmap = plt.get_cmap("viridis")
        norm = BoundaryNorm(boundaries=ticks, ncolors=cmap.N, clip=False)
        return cmap, norm, ticks

    tick_cfg = cfg["ticks"]

    if isinstance(tick_cfg, int):
        ticks = np.linspace(cfg["vmin"], cfg["vmax"], tick_cfg)
    else:
        ticks = np.asarray(tick_cfg, dtype=float)

    if is_skill:
        norm = TwoSlopeNorm(
            vmin=cfg["vmin"],
            vcenter=0,
            vmax=cfg["vmax"],
        )
    else:
        norm = BoundaryNorm(
            boundaries=ticks,
            ncolors=cfg["cmap"].N,
            clip=True,
        )

    return cfg["cmap"], norm, ticks


SQUARED_METRICS = {"mse", "mse_anom"}

def get_plot_metric_unit_and_scale(
    da: xr.DataArray,
    *,
    var: str,
    metric: str,
    var_plot_config: dict,
) -> tuple[str, float]:
    cfg = get_plot_config(var, metric, var_plot_config)

    if not cfg.get("scale_units", True):
        return "", 1.0

    unit = da.attrs.get("units") or VARIABLE_UNITS.get(var, "")

    unit_conversion = UNIT_CONVERSIONS.get(unit, (unit, 1.0))
    if isinstance(unit_conversion, dict):
        plot_unit, scale = unit_conversion[var]
    else:
        plot_unit, scale = unit_conversion

    if metric in SQUARED_METRICS:
        return f"{plot_unit}²", scale**2

    return plot_unit, scale


def plot_map(
    da: xr.DataArray,
    *,
    var: str,
    metric: str,
    model: str,
    start_period: str | int,
    lead_value: object,
    out_file: Path,
    time_range: tuple[str, str],
    plot_kind: Literal["maps", "time_lon", "time_lat"] = "maps",
    plot_type: Literal["pcolormesh", "contourf"] = "pcolormesh",
    leadtime_dim: str = "leadtime",
    leadtime_units: str = "months",
    period_dim: str = "start_month",
    clim_period: ClimPeriod | None = None,
    var_plot_config: dict | None = None,
    impro_plot_config: dict | None = None,
    force_scale: int | float | None = None,
    title_strftime: str = "%Y",
) -> None:
    var_plot_config = var_plot_config or {}
    impro_plot_config = impro_plot_config or {}

    is_skill = is_skill_model(model)

    if is_skill:
        if model.endswith("_percentage") or da.attrs.get("units") == "%":
            improvement_unit: Literal["%", "Δ"] = "%"
        elif model.endswith("_difference"):
            improvement_unit = "Δ"
        else:
            improvement_unit = METRIC_SKILL_UNITS[metric]
    else:
        improvement_unit = None

    time_dim = da.earthml.guessed_dims.time or clim_period
    lat, lon = da.earthml.guessed_dims.latitude, da.earthml.guessed_dims.longitude

    da = da.sel(
        {
            leadtime_dim: lead_value,
            period_dim: start_period,
        }
    ).squeeze(drop=True)

    if plot_kind == "maps":
        required_dims = (lat, lon)

    elif plot_kind == "time_lon":
        required_dims = (time_dim, lon)

    elif plot_kind == "time_lat":
        required_dims = (time_dim, lat)

    else:
        raise ValueError(
            f"Unsupported plot_kind={plot_kind!r}. "
            "Choose one of: 'map', 'time_lon', 'time_lat'."
        )

    missing_dims = [dim for dim in required_dims if dim not in da.dims]
    if missing_dims:
        raise ValueError(
            f"Cannot create {plot_kind!r} plot. "
            f"Missing dimensions: {missing_dims}. "
            f"Available dimensions: {da.dims}"
        )

    extra_dims = [
        dim
        for dim in da.dims
        if dim not in required_dims
    ]
    if extra_dims:
        raise ValueError(
            f"Unexpected dimensions remain before plotting: {extra_dims}. "
            f"Expected only {required_dims}, got {da.dims}."
        )

    with ProgressBar():
        da = da.transpose(*required_dims).compute()

    if is_skill:
        if improvement_unit == "%":
            plot_unit = "%"
            scale = 1.0
        else:
            plot_unit, scale = get_plot_metric_unit_and_scale(
                da,
                var=var,
                metric=metric,
                var_plot_config=var_plot_config,
            )
    else:
        plot_unit, scale = get_plot_metric_unit_and_scale(
            da,
            var=var,
            metric=metric,
            var_plot_config=var_plot_config,
        )

    if force_scale is not None:
        scale = force_scale

    da = da / scale

    if is_skill_metric(metric):
        unit_label = ""
        cb_label = METRIC_NAMES[metric]

    elif is_skill_model(model):
        unit_label = f"({plot_unit})" if plot_unit else ""
        cb_label = (
            f"{METRIC_NAMES[metric]} improvement {unit_label}"
            if improvement_unit == "%"
            else f"{METRIC_NAMES[metric]} improvement difference {unit_label}"
        )

    else:
        unit_label = f"({plot_unit})" if plot_unit else ""
        cb_label = (
            f"{METRIC_NAMES[metric]} {unit_label}"
            if unit_label
            else METRIC_NAMES[metric]
        )

    avg = float(da.mean(skipna=True).values)

    if plot_kind == "maps":
        geo_avg = float(da.earthml.geo_mean().values)
    else:
        geo_avg = None

    cmap, norm, ticks = metric_style(
        var,
        metric,
        da,
        var_plot_config=var_plot_config,
        impro_plot_config=impro_plot_config,
        is_skill=is_skill,
        improvement_unit=improvement_unit,
    )

    if plot_kind == "maps":
        fig, ax = plt.subplots(
            figsize=(7, 5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        ax = cast(GeoAxes, ax)

        lon_values = np.asarray(da[lon].values, dtype=float)
        lat_values = np.asarray(da[lat].values, dtype=float)

        finite_lon = lon_values[np.isfinite(lon_values)]
        finite_lat = lat_values[np.isfinite(lat_values)]

        if finite_lon.size == 0 or finite_lat.size == 0:
            raise ValueError("Latitude or longitude coordinates contain no finite values")

        lon_min = float(finite_lon.min())
        lon_max = float(finite_lon.max())
        lat_min = float(finite_lat.min())
        lat_max = float(finite_lat.max())

        ax.set_extent(
            [lon_min, lon_max, lat_min, lat_max],
            crs=ccrs.PlateCarree(),
        )

        if plot_type == "pcolormesh":
            im = ax.pcolormesh(
                da[lon],
                da[lat],
                da,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                shading="auto",
            )

        elif plot_type == "contourf":
            from cartopy.util import add_cyclic_point

            finite_da = da.where(np.isfinite(da))

            data_cyclic, lon_cyclic = add_cyclic_point(
                finite_da.values,
                coord=finite_da[lon].values,
                axis=finite_da.get_axis_num(lon),
            )

            im = ax.contourf(
                lon_cyclic,
                finite_da[lat].values,
                data_cyclic,
                levels=ticks,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                extend="both",
            )

        else:
            raise ValueError(
                f"Unsupported plot_type={plot_type!r}. "
                "Choose one of: 'pcolormesh', 'contourf'."
            )

        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color="gray",
            alpha=0.4,
            linestyle="--",
        )

        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

    else:
        fig, ax = plt.subplots(figsize=(9, 5.5))

        if plot_kind == "time_lon":
            x_dim = lon
            x_label = "Longitude"
        else:
            x_dim = lat
            x_label = "Latitude"

        finite_da = da.where(np.isfinite(da))

        if plot_type == "pcolormesh":
            im = ax.pcolormesh(
                finite_da[x_dim],
                finite_da[time_dim],
                finite_da,
                cmap=cmap,
                norm=norm,
                shading="auto",
            )

        elif plot_type == "contourf":
            im = ax.contourf(
                finite_da[x_dim],
                finite_da[time_dim],
                finite_da,
                levels=ticks,
                cmap=cmap,
                norm=norm,
                extend="both",
            )

        else:
            raise ValueError(
                f"Unsupported plot_type={plot_type!r}. "
                "Choose one of: 'pcolormesh', 'contourf'."
            )

        ax.set_xlabel(x_label)

        if clim_period == ClimPeriod.MONTH:
            ax.set_ylabel("Month")
            ax.set_yticks(np.arange(1, 13))
            ax.set_yticklabels(
                ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            )
        else:
            ax.set_ylabel("Time")

        ax.grid(True, alpha=0.2)

        if plot_kind == "time_lon":
            ax.set_xlim(
                float(finite_da[lon].min()),
                float(finite_da[lon].max()),
            )

        else:
            ax.set_ylim(
                finite_da[time_dim].min().values,
                finite_da[time_dim].max().values,
            )

    label_lt = lead_label(
        da,
        lead_value,
        leadtime_dim,
    )

    start_time = datetime.strptime(
        time_range[0],
        "%Y-%m-%d",
    ).strftime(title_strftime)

    end_time = datetime.strptime(
        time_range[1],
        "%Y-%m-%d",
    ).strftime(title_strftime)

    if plot_kind == "maps":
        summary = (
            f"avg={avg:.3f} {unit_label} · "
            f"geoavg={geo_avg:.3f} {unit_label}"
        )
    else:
        summary = f"avg={avg:.3f} {unit_label}"

    plot_kind_label = {
        "maps": "maps",
        "time_lon": "time-longitude",
        "time_lat": "time-latitude",
    }[plot_kind]

    title = (
        f"{VARIABLE_NAMES[var]} · {model} · {plot_kind_label} · "
        f"{start_time}-{end_time} · {period_dim}={start_period}\n"
        f"leadtime={label_lt} {leadtime_units} · {summary}"
    )

    if plot_kind == "maps":
        fig.suptitle(
            title,
            fontsize=11,
            y=0.98,
        )
    else:
        ax.set_title(
            title,
            pad=10,
        )

    cb = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.10 if plot_kind == "maps" else 0.13,
        ticks=ticks,
        boundaries=ticks,
        spacing="uniform",
    )

    cb.set_label(cb_label)
    cb.ax.tick_params(labelsize=7)

    if plot_kind == "maps":
        fig.subplots_adjust(
            top=0.84,
            bottom=0.18,
            left=0.08,
            right=0.96,
        )
    else:
        fig.subplots_adjust(
            top=0.86,
            bottom=0.18,
            left=0.10,
            right=0.96,
        )

    out_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        out_file,
        dpi=200,
    )
    plt.close(fig)


def plot_rank_histogram(
    das: Sequence[xr.DataArray | xr.Dataset],
    *,
    var: str,
    metric: str,
    models: str | Sequence[str],
    start_period: str | int,
    out_file: Path,
    time_range: tuple[str, str],
    rank_dim: str = "rank",
) -> None:
    if isinstance(models, str):
        models = [models]
    if len(models) != len(das):
        raise ValueError("Select same number of models and DataArrays")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for da, model in zip(convert_to_da_list(das, var), models):
        if rank_dim not in da.dims:
            print(f"Skipping rank histogram: no {rank_dim!r} in {da.dims}")
            return

        reduce_dims = [dim for dim in da.dims if dim != rank_dim]

        if reduce_dims:
            da = da.sum(dim=reduce_dims, skipna=True)

        with ProgressBar():
            da = da.compute()

        ax.bar(
            da[rank_dim].values,
            da.values,
            width=0.8,
            color=MODEL_COLORS.get(model, None),
            alpha=0.6,
            label=model,
        )

    start_time = datetime.strptime(time_range[0], "%Y-%m-%d").strftime("%Y")
    end_time = datetime.strptime(time_range[1], "%Y.%m-%d").strftime("%Y")
    ax.set_title(f"{VARIABLE_NAMES[var]} · {METRIC_NAMES[metric]} · {start_time}-{end_time} · start month={start_period}")
    ax.set_xlabel(rank_dim)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


LOWER_IS_BETTER = {
    "mae",
    "rmse",
    "rmse_anom",
    "nrmse_anom",
    "rmse_an_anom",
    "ens_member_rmse",
    "ens_member_rmse_anom",
    "mean_member_rmse_anom",
    "crps",
}

HIGHER_IS_BETTER = {
    "acc",
    "r2",
    "r2_anom",
}


def metric_improvement(
    fc: xr.DataArray,
    mlfc: xr.DataArray,
    metric: str,
) -> xr.DataArray:
    """
    Positive means ML forecast improved over FC.
    """
    if metric == "bias":
        return safe_percent(abs(fc) - abs(mlfc), abs(fc))

    if metric == "std_ratio":
        return safe_percent(abs(fc - 1) - abs(mlfc - 1), abs(fc - 1))

    if metric in LOWER_IS_BETTER:
        return safe_percent(fc - mlfc, fc)

    if metric in HIGHER_IS_BETTER:
        return mlfc - fc

    raise ValueError(f"No improvement rule for metric {metric!r}")


@dataclass(frozen=True)
class ScatterPoint:
    x: float
    y: float
    variable: str
    region: str
    leadtime: object
    start_month: object
    total_months: object
    experiment: str


def get_total_months(s: Settings) -> object:
    for name in ("total_months", "history_months", "input_months", "context_months"):
        if hasattr(s, name):
            return getattr(s, name)
    return "all"


def lighten(color, amount: float):
    r, g, b = to_rgb(color)
    return (
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )


def point_field(p: ScatterPoint, field: str):
    return getattr(p, field)


def plot_metric_diff_scatter(
    points: Sequence[ScatterPoint],
    *,
    forecast_metric: str,
    diff_metric: str,
    out_file: Path,
    color_by: str = "variable",
    marker_by: str = "leadtime",
    shade_by: str = "total_months",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize=(12, 12),
    cmap_name: str = "tab10",
    markers=("o", "s", "^", "D", "v", "P", "X"),
    point_size: float = 22,
    edgecolor: str = "black",
    linewidth: float = 0.3,
    shade_strength: float = 0.65,
    fit_lines: bool = True,
    fit_min_points: int = 3,
    shade_positive_x: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    if not points:
        ax.set_title(title or "No data")
        ax.set_xlabel(xlabel or f"{diff_metric} improvement")
        ax.set_ylabel(ylabel or forecast_metric)
        ax.grid(alpha=0.3)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    color_values = sorted({point_field(p, color_by) for p in points}, key=str)
    marker_values = sorted({point_field(p, marker_by) for p in points}, key=str)
    shade_values = sorted({point_field(p, shade_by) for p in points}, key=str)

    cmap = plt.get_cmap(cmap_name)

    def base_color(value):
        idx = color_values.index(value)
        return cmap(idx % cmap.N)

    for color_value in color_values:
        base = base_color(color_value)

        for shade_i, shade_value in enumerate(shade_values):
            denom = max(1, len(shade_values) - 1)
            shade = lighten(base, shade_strength * shade_i / denom)

            for marker_i, marker_value in enumerate(marker_values):
                subset = [
                    p for p in points
                    if point_field(p, color_by) == color_value
                    and point_field(p, shade_by) == shade_value
                    and point_field(p, marker_by) == marker_value
                ]

                if not subset:
                    continue

                ax.scatter(
                    [p.x for p in subset],
                    [p.y for p in subset],
                    color=shade,
                    marker=markers[marker_i % len(markers)],
                    s=point_size,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )

    if fit_lines:
        neutral_base = (0.2, 0.2, 0.2)
        linestyles = ["-", "--", ":", "-."]

        for shade_i, shade_value in enumerate(shade_values):
            denom = max(1, len(shade_values) - 1)
            line_color = lighten(neutral_base, shade_strength * shade_i / denom)

            for marker_i, marker_value in enumerate(marker_values):
                subset = [
                    p for p in points
                    if point_field(p, shade_by) == shade_value
                    and point_field(p, marker_by) == marker_value
                ]

                if len(subset) < fit_min_points:
                    continue

                x = np.asarray([p.x for p in subset], dtype=float)
                y = np.asarray([p.y for p in subset], dtype=float)

                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < fit_min_points or np.allclose(x[mask], x[mask][0]):
                    continue

                a, b = np.polyfit(x[mask], y[mask], 1)
                xs = np.linspace(x[mask].min(), x[mask].max(), 200)

                ax.plot(
                    xs,
                    a * xs + b,
                    color=line_color,
                    linestyle=linestyles[marker_i % len(linestyles)],
                    linewidth=1.0,
                    alpha=0.9,
                )

    ax.axvline(0, color="black", alpha=0.5, linewidth=0.8)

    ax.set_xlabel(xlabel or f"{METRIC_NAMES.get(diff_metric, diff_metric)} improvement")
    ax.set_ylabel(ylabel or METRIC_NAMES.get(forecast_metric, forecast_metric))

    if title:
        ax.set_title(title)

    ax.grid(alpha=0.3)

    handles: list[Line2D] = []

    handles.append(Line2D([], [], linestyle="None", label=color_by.replace("_", " ").title()))
    for value in color_values:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=base_color(value),
                markeredgecolor=edgecolor,
                label=str(value),
            )
        )

    handles.append(Line2D([], [], linestyle="None", label=shade_by.replace("_", " ").title()))
    neutral_base = (0.2, 0.2, 0.2)
    denom = max(1, len(shade_values) - 1)
    for i, value in enumerate(shade_values):
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=lighten(neutral_base, shade_strength * i / denom),
                markeredgecolor=edgecolor,
                label=str(value),
            )
        )

    handles.append(Line2D([], [], linestyle="None", label=marker_by.replace("_", " ").title()))
    for i, value in enumerate(marker_values):
        handles.append(
            Line2D(
                [0], [0],
                marker=markers[i % len(markers)],
                linestyle="None",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="black",
                label=str(value),
            )
        )

    leg = ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False)

    for txt in leg.get_texts():
        if txt.get_text() in {
            color_by.replace("_", " ").title(),
            shade_by.replace("_", " ").title(),
            marker_by.replace("_", " ").title(),
        }:
            txt.set_weight("bold")

    if shade_positive_x:
        fig.canvas.draw()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        if xmax > 0:
            ax.axvspan(0, xmax, color="red", alpha=0.08, zorder=0)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_plot_unit_and_scale(da: xr.DataArray, var: str) -> tuple[str, float]:
    unit = da.attrs.get("units") or VARIABLE_UNITS.get(var, "")

    unit_conversion = UNIT_CONVERSIONS.get(unit, (unit, 1.0))

    if isinstance(unit_conversion, dict):
        return unit_conversion.get(var, (unit, 1.0))

    return unit_conversion


def normalize_longitudes(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    if float(da[lon_name].max(skipna=True)) <= 180:
        return da

    new_lon = ((da[lon_name] + 180) % 360) - 180
    return da.assign_coords({lon_name: new_lon}).sortby(lon_name)


def prepare_map_da(da: xr.DataArray) -> xr.DataArray:
    da = da.squeeze(drop=True)
    lat, lon = "latitude", "longitude"
    da = normalize_longitudes(da, lon)
    return da.transpose(lat, lon)


def reduce_to_timeseries(
    da: xr.DataArray,
    *,
    spatial_dims: tuple[str, str] = ("latitude", "longitude"),
) -> xr.DataArray:
    reduce_dims = [dim for dim in spatial_dims if dim in da.dims]

    if len(reduce_dims) == 2:
        weights = np.cos(np.deg2rad(da[spatial_dims[0]]))
        return da.weighted(weights).mean(reduce_dims, skipna=True)

    if reduce_dims:
        return da.mean(reduce_dims, skipna=True)

    return da


def plot_field_map(
    da: xr.DataArray,
    *,
    var: str,
    title: str,
    out_file: Path,
    cmap="viridis",
    centered: bool = False,
    spatial_dims: tuple[str, str] = ("latitude", "longitude"),
) -> None:
    plot_unit, scale = get_plot_unit_and_scale(da, var)
    da = prepare_map_da(da / scale)

    lat, lon = spatial_dims[0], spatial_dims[1]

    fig, ax = plt.subplots(
        figsize=(7, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    norm = None
    if centered:
        vmax = float(np.nanmax(np.abs(da.values)))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.pcolormesh(
        da[lon],
        da[lat],
        da,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        norm=norm,
    )

    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    ax.set_title(title)

    cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.07)
    cb.set_label(f"{VARIABLE_NAMES.get(var, var.upper())} ({plot_unit})" if plot_unit else VARIABLE_NAMES.get(var, var.upper()))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"Saved map: {out_file}")
    plt.close(fig)

def plot_field_timeseries(
    *,
    series: dict[str, xr.DataArray | None],
    member_series: dict[str, xr.DataArray | None] | None,
    var: str,
    title: str,
    out_file: Path,
    time_dim: str = "time",
    spatial_dims: tuple[str, str] = ("latitude", "longitude"),
    realization_dim: str = "realization",
    spread: Literal["std", "minmax"] = "std",
    val_end: str | None = None,
    train_end: str | None = None,
    plot_single_members: bool = True,
    member_linestyle: str = "-",
    series_linestyle: str = "--",
    series_offsets: dict[str, float] | None = None,
) -> None:
    valid = {name: da for name, da in series.items() if da is not None}

    valid_members = {
        name: da
        for name, da in (member_series or {}).items()
        if da is not None and realization_dim in da.dims
    }

    if not valid and not valid_members:
        return

    first = next(iter(list(valid.values()) + list(valid_members.values())))

    plot_unit, scale = get_plot_unit_and_scale(first, var)

    ylabel = (
        f"{VARIABLE_NAMES.get(var, var.upper())} ({plot_unit})"
        if plot_unit
        else VARIABLE_NAMES.get(var, var.upper())
    )

    reduced = {
        name: reduce_to_timeseries(da / scale, spatial_dims=spatial_dims)
        for name, da in valid.items()
    }

    reduced_members = {}
    if member_series is not None:
        for name, da in member_series.items():
            if da is not None and realization_dim in da.dims:
                reduced_members[name] = reduce_to_timeseries(
                    da / scale,
                    spatial_dims=spatial_dims,
                )

    # Align everything together on the same time axis
    all_names = list(reduced.keys()) + [f"__members__{k}" for k in reduced_members]
    all_arrays = list(reduced.values()) + list(reduced_members.values())

    aligned = xr.align(*all_arrays, join="inner")

    aligned_map = dict(zip(all_names, aligned))

    reduced = {
        name: aligned_map[name]
        for name in reduced
    }

    reduced_members = {
        name: aligned_map[f"__members__{name}"]
        for name in reduced_members
    }

    fig, ax = plt.subplots(figsize=(16, 8))

    if series_offsets:
        for name, offset in series_offsets.items():
            color = SERIES_COLORS.get(name)

            ax.axhline(
                offset,
                color=color,
                linewidth=1,
                alpha=0.35,
                linestyle="--",
                zorder=0,
            )

    plot_names = list(dict.fromkeys([*reduced.keys(), *reduced_members.keys()]))

    for name in plot_names:
        color = SERIES_COLORS.get(name)
        da = reduced.get(name)
        member_da = reduced_members.get(name)

        offset = 0.0 if series_offsets is None else series_offsets.get(name, 0.0)

        if da is not None:
            x = da[time_dim].values
        elif member_da is not None:
            x = member_da[time_dim].values
        else:
            continue

        if member_da is not None:
            with ProgressBar():
                member_da = member_da.compute()

            if plot_single_members:
                for i in range(member_da.sizes[realization_dim]):
                    member_da_i = member_da.isel({realization_dim: i})

                    ax.plot(
                        x,
                        member_da_i.values + offset,
                        linestyle=member_linestyle,
                        linewidth=0.5,
                        alpha=0.2,
                        color=color,
                    )

            member_mean = member_da.mean(realization_dim, skipna=True)

            if spread == "std":
                member_std = member_da.std(realization_dim, skipna=True)
                lower = member_mean - member_std
                upper = member_mean + member_std
            else:
                lower = member_da.min(realization_dim, skipna=True)
                upper = member_da.max(realization_dim, skipna=True)

            ax.fill_between(
                x,
                lower.values + offset,
                upper.values + offset,
                alpha=0.18,
                color=color,
            )

            ax.plot(
                x,
                member_mean.values + offset,
                linestyle=member_linestyle,
                linewidth=1.4,
                color=color,
                label=f"{name} member mean",
            )

        if da is not None:
            with ProgressBar():
                da = da.compute()

            ax.plot(
                x,
                da.values + offset,
                linewidth=1.4,
                linestyle=series_linestyle,
                color=color,
                label=name,
            )

    if train_end is not None:
        ax.axvline(
            np.datetime64(train_end),
            color="red",
            linestyle="--",
            linewidth=1.3,
            label="Train end",
        )

    if val_end is not None:
        ax.axvline(
            np.datetime64(val_end),
            color="blue",
            linestyle="--",
            linewidth=1.3,
            label="Val end",
        )

    ax.set_title(title)
    ax.set_xlabel(time_dim)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if series_offsets:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())

        tick_names = [n for n in plot_names if n in series_offsets]
        ax2.set_yticks([series_offsets[n] for n in tick_names])
        tick_zeros = ["0" for n in plot_names if n in series_offsets]

        ax2.set_yticklabels(tick_zeros)

        for label, name in zip(ax2.get_yticklabels(), tick_names):
            label.set_color(SERIES_COLORS.get(name))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    print(f"Saved timeseries: {out_file}")
    plt.close(fig)
