from pathlib import Path
from typing import Sequence, cast

import numpy as np
import xarray as xr

from dask.diagnostics.progress import ProgressBar

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import (
    BoundaryNorm,
    TwoSlopeNorm,
    Colormap,
)
from matplotlib.cm import ScalarMappable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes

from ..metrics import convert_to_da_list

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
    TRANSLATION_TABLE,
)


def safe_label(x: object) -> str:
    if x is None:
        return "all"
    return str(x).translate(TRANSLATION_TABLE)


def is_skill_metric(metric: str) -> bool:
    return metric.endswith("_skill_clim")

def is_skill_model(model: str) -> bool:
    return model.endswith("_vs_fc")


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
) -> dict:
    if metric.endswith("_skill_clim"):
        cfg = DEFAULT_PLOT_CONFIG[metric].copy()
        cfg.update(var_plot_config.get(var, {}).get(metric, {}))
        return cfg

    unit = METRIC_SKILL_UNITS[metric]
    cfg = DEFAULT_IMPROVEMENT_PLOT_CONFIG[unit].copy()
    cfg.update(impro_plot_config.get(var, {}).get(metric, {}))
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

    fig, ax = plt.subplots(figsize=(12, 8))

    for model, da, da_member in zip(models, plot_das, plot_das_member):
        da = da.reset_coords(drop=True)
        print(da)

        color = MODEL_COLORS.get(model, None)

        x = da[leadtime_dim].values

        if da_member is not None:
            if realization_dim in da_member.dims:
                with ProgressBar():
                    da_member = da_member.sel({period_dim: start_period}).compute()

                da_member = da_member.reset_coords(drop=True)
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
                    linestyle="-",
                    label=f"{model} member mean",
                    color=color,
                )

        ax.plot(
            x,
            da.sel({period_dim: start_period}).values,
            linestyle="--",
            linewidth=1.4,
            label=f"{model} ensemble mean",
            color=color,
        )

    if metric in {"bias", "acc", "clim_acc", "spatial_acc"}:
        ax.axhline(0, linewidth=0.8)

    if metric in {"acc", "clim_acc", "spatial_acc"}:
        ax.set_ylim(-1, 1)

    start_period_str = "" if start_period=="all" else f" · start_period={start_period}"
    ax.set_title(f"{VARIABLE_NAMES[var]} · {METRIC_NAMES[metric]} · {safe_label(time_range)}{start_period_str}")
    ax.set_xlabel(f"{leadtime_dim} {leadtime_unit}")
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


def infer_lat_lon(da: xr.DataArray) -> tuple[str, str]:
    lat = "latitude" if "latitude" in da.dims else "lat"
    lon = "longitude" if "longitude" in da.dims else "lon"

    if lat not in da.dims or lon not in da.dims:
        raise ValueError(f"Could not infer lat/lon from dims={da.dims}")

    return lat, lon


def metric_style(
    var: str,
    metric: str,
    da: xr.DataArray | None = None,
    *,
    var_plot_config: dict = {},
    impro_plot_config: dict = {},
    is_skill: bool = False,
) -> tuple[Colormap, BoundaryNorm | TwoSlopeNorm, np.ndarray]:

    try:
        if is_skill:
            cfg = get_skill_plot_config(var, metric, var_plot_config, impro_plot_config)
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

def get_plot_unit_and_scale(
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
    plot_unit, scale = UNIT_CONVERSIONS.get(unit, (unit, 1.0))

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
    leadtime_dim: str = "leadtime",
    period_dim: str = "start_date",
    var_plot_config: dict = {},
    impro_plot_config: dict = {},
) -> None:
    is_skill = is_skill_model(model) # or is_skill_metric(metric)

    da = da.sel({leadtime_dim: lead_value, period_dim: start_period}).squeeze(drop=True)

    lat, lon = infer_lat_lon(da)
    with ProgressBar():
        da = da.transpose(lat, lon).compute()

    fig, ax = plt.subplots(
        figsize=(7, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax = cast(GeoAxes, ax)

    if is_skill:
        plot_unit = METRIC_SKILL_UNITS[metric]
        scale = 1.0
    else:
        plot_unit, scale = get_plot_unit_and_scale(
            da,
            var=var,
            metric=metric,
            var_plot_config=var_plot_config,
        )

    da = da / scale

    cmap, norm, ticks = metric_style(
        var,
        metric,
        da,
        var_plot_config=var_plot_config,
        impro_plot_config=impro_plot_config,
        is_skill=is_skill,
    )

    im = ax.pcolormesh(
        da[lon],
        da[lat],
        da,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)

    label = lead_label(da, lead_value, leadtime_dim)

    ax.set_title(
        f"{VARIABLE_NAMES[var]} · {model} · {safe_label(time_range)}\n"
        f"start month={start_period} · leadtime={label}"
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        pad=0.07,
        ticks=ticks,
        boundaries=ticks,
        spacing="uniform",
    )

    if is_skill_metric(metric):
        cb.set_label(f"{METRIC_NAMES[metric]} (%)")
    elif is_skill_model(model):
        cb.set_label(
            f"{METRIC_NAMES[metric]} improvement (%)"
            if plot_unit == "%"
            else f"{METRIC_NAMES[metric]} improvement difference"
        )
    else:
        cb.set_label(
            f"{METRIC_NAMES[metric]} ({plot_unit})"
            if plot_unit
            else METRIC_NAMES[metric]
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
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

    ax.set_title(f"{VARIABLE_NAMES[var]} · {METRIC_NAMES[metric]} · {safe_label(time_range)} · start month={start_period}")
    ax.set_xlabel(rank_dim)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
