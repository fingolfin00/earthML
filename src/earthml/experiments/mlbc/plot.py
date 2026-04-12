from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ...plots import plot_realization_timeseries


PlotSpec = dict[str, Any]


def _get_stage_plot_folder(root: Path, data_type: str) -> Path:
    stage_plot_folder = root.joinpath(data_type)
    stage_plot_folder.mkdir(parents=True, exist_ok=True)
    return stage_plot_folder


def _select_plot_var(ds: xr.Dataset, var: str) -> xr.Dataset:
    return ds[[var]] if len(ds.data_vars) > 1 else ds


def _get_plot_members(ds: xr.Dataset) -> tuple[xr.Dataset | None, str | None]:
    rdim = ds.earthml.guessed_dims.realization
    if rdim is not None and rdim in ds.dims:
        return ds, rdim
    return None, rdim


def _plot_single_timeseries(
    *,
    logger,
    ax,
    ds: xr.Dataset,
    var: str,
    time_dim: str,
    stage_kind: str,
    data_type: str,
    label: str,
    mean_label: str,
    color: str,
) -> None:
    ds_var = _select_plot_var(ds, var)
    members, rdim = _get_plot_members(ds_var)
    logger.info(
        "Plot %s %s/%s: %s realizations=%s",
        stage_kind,
        data_type,
        var,
        label,
        ds_var.sizes.get(rdim, 0) if members is not None and rdim is not None else 0,
    )
    plot_realization_timeseries(
        ds_var,
        members=members,
        x_dim=time_dim,
        ens_dim=rdim or "realization",
        ax=ax,
        x_label="Time",
        label=label,
        mean_label=mean_label,
        color=color,
        spread="minmax",
    )


def _build_residual_ds(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    var: str,
) -> xr.Dataset:
    left_var = _select_plot_var(left_ds, var)
    right_var = _select_plot_var(right_ds, var)

    rdim_left = left_var.earthml.guessed_dims.realization
    rdim_right = right_var.earthml.guessed_dims.realization

    # If the reference side is deterministic/singleton while the left side is
    # ensemble, drop the singleton realization axis so xarray broadcasts it
    # instead of intersecting realization coordinates during subtraction.
    if (
        rdim_left is not None
        and rdim_right is not None
        and rdim_left == rdim_right
        and left_var.sizes.get(rdim_left, 1) > 1
        and right_var.sizes.get(rdim_right, 1) == 1
    ):
        right_var = right_var.squeeze(rdim_right, drop=True)
        rdim_right = None

    exclude_align_dims = tuple(dim for dim in (rdim_left, rdim_right) if dim is not None)
    left_var, right_var = xr.align(left_var, right_var, join="inner", exclude=exclude_align_dims)

    return left_var - right_var


def _build_climatology_ds(ds: xr.Dataset) -> xr.Dataset:
    return ds.earthml.climatology(groupby="month")


def _build_ds_minus_climatology_ds(ds: xr.Dataset) -> xr.Dataset:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Cannot compute dataset minus climatology without a time dimension.")
    clim_ds = _build_climatology_ds(ds)
    return ds.groupby(f"{time_dim}.month") - clim_ds


def _build_anomaly_residual_ds(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    var: str,
) -> xr.Dataset:
    left_anom = _build_ds_minus_climatology_ds(left_ds)
    right_anom = _build_ds_minus_climatology_ds(right_ds)
    return _build_residual_ds(left_anom, right_anom, var=var)


def _build_anomaly_variance_ds(ds: xr.Dataset, window: int | None = None) -> xr.Dataset:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Cannot compute anomaly variance without a time dimension.")

    anom_ds = _build_ds_minus_climatology_ds(ds)
    time_len = int(anom_ds.sizes.get(time_dim, 0))
    if time_len < 3:
        raise ValueError("Need at least 3 timesteps to compute anomaly variance timeseries.")

    window = min(12, time_len) if window is None else min(window, time_len)
    min_periods = max(3, window // 2)

    return anom_ds.rolling({time_dim: window}, center=True, min_periods=min_periods).var()


def _build_anomaly_autocorr_ds(ds: xr.Dataset, nlags: int | None = None) -> xr.Dataset:
    time_dim = ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in ds.dims:
        raise ValueError("Cannot compute anomaly autocorrelation without a time dimension.")

    anom_ds = _build_ds_minus_climatology_ds(ds)
    time_len = int(anom_ds.sizes.get(time_dim, 0))
    if time_len < 3:
        raise ValueError("Need at least 3 timesteps to compute anomaly autocorrelation.")

    nlags = min(12, time_len - 1) if nlags is None else min(nlags, time_len - 1)
    if nlags < 1:
        raise ValueError("Need at least 1 valid lag to compute anomaly autocorrelation.")

    acf_vars: dict[str, xr.DataArray] = {}
    for var_name, da in anom_ds.data_vars.items():
        corr_list: list[xr.DataArray] = []
        for lag in range(nlags + 1):
            da_lag = da.shift({time_dim: lag})
            x = da
            y = da_lag
            x_mean = x.mean(dim=time_dim, skipna=True)
            y_mean = y.mean(dim=time_dim, skipna=True)
            x_anom = x - x_mean
            y_anom = y - y_mean
            cov = (x_anom * y_anom).mean(dim=time_dim, skipna=True)
            x_std = x.std(dim=time_dim, skipna=True)
            y_std = y.std(dim=time_dim, skipna=True)
            denom = x_std * y_std
            corr = xr.where(denom > 0, cov / denom, np.nan)
            corr_list.append(corr)
        acf_vars[var_name] = xr.concat(
            corr_list,
            dim=xr.IndexVariable("lag", range(nlags + 1)),
        )

    return xr.Dataset(acf_vars)


def plot_stage_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
    x_dim: str | None = None,
    x_label: str = "Time",
    filename_suffix: str = "timeseries",
) -> None:
    stage_plot_folder = _get_stage_plot_folder(plots_folder_path, data_type)
    logger.info(
        "Generate %s stage plots for %s (%s) in %s",
        stage_kind,
        data_type,
        stage,
        stage_plot_folder,
    )

    base_ds = plot_specs[0]["ds"]
    plot_dim = x_dim or base_ds.earthml.guessed_dims.time
    if plot_dim is None or any(plot_dim not in spec["ds"].dims for spec in plot_specs):
        logger.debug("Skip %s plotting for %s: no common plot dimension.", stage_kind, data_type)
        return

    common_vars = set(plot_specs[0]["ds"].data_vars)
    for spec in plot_specs[1:]:
        common_vars &= set(spec["ds"].data_vars)
    common_vars = list(common_vars)
    if not common_vars:
        logger.debug("Skip %s plotting for %s: no common variables.", stage_kind, data_type)
        return

    for var in common_vars:
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            for spec in plot_specs:
                ds_var = _select_plot_var(spec["ds"], var)
                members, rdim = _get_plot_members(ds_var)
                logger.info(
                    "Plot %s %s/%s: %s realizations=%s",
                    stage_kind,
                    data_type,
                    var,
                    spec["label"],
                    ds_var.sizes.get(rdim, 0) if members is not None and rdim is not None else 0,
                )
                plot_realization_timeseries(
                    ds_var,
                    members=members,
                    x_dim=plot_dim,
                    ens_dim=rdim or "realization",
                    ax=ax,
                    x_label=x_label,
                    label=spec["label"],
                    mean_label=spec["mean_label"],
                    color=spec["color"],
                )

            ax.legend()
            fig.tight_layout()
            fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_{filename_suffix}.png"), dpi=200)
        except Exception as exc:
            logger.warning(
                "Failed to generate %s plot for %s/%s/%s: %s",
                stage_kind,
                data_type,
                stage,
                var,
                exc,
            )
        finally:
            plt.close(fig)


def plot_stage_residual_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    data_type: str,
    stage: str,
    stage_kind: str,
    residual_label: str,
    residual_mean_label: str,
    color: str = "tab:red",
    anomaly: bool = False,
) -> None:
    stage_plot_folder = _get_stage_plot_folder(plots_folder_path, data_type)
    logger.info(
        "Generate %s residual plots for %s (%s) in %s",
        stage_kind,
        data_type,
        stage,
        stage_plot_folder,
    )

    time_dim = left_ds.earthml.guessed_dims.time
    if time_dim is None or time_dim not in left_ds.dims or time_dim not in right_ds.dims:
        logger.debug("Skip %s residual plotting for %s: no common time dimension.", stage_kind, data_type)
        return

    common_vars = [var for var in left_ds.data_vars if var in right_ds.data_vars]
    if not common_vars:
        logger.debug("Skip %s residual plotting for %s: no common variables.", stage_kind, data_type)
        return

    for var in common_vars:
        fig, ax = plt.subplots(figsize=(10, 4))
        try:
            residual_ds = (
                _build_anomaly_residual_ds(left_ds, right_ds, var=var)
                if anomaly else
                _build_residual_ds(left_ds, right_ds, var=var)
            )
            _plot_single_timeseries(
                logger=logger,
                ax=ax,
                ds=residual_ds,
                var=var,
                time_dim=time_dim,
                stage_kind=f"{stage_kind} residual",
                data_type=data_type,
                label=residual_label,
                mean_label=residual_mean_label,
                color=color,
            )
            ax.axhline(0.0, color="black", linewidth=1.0, linestyle=":")
            ax.legend()
            fig.tight_layout()
            suffix = "anomaly_residual_timeseries" if anomaly else "residual_timeseries"
            fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_{suffix}.png"), dpi=200)
        except Exception as exc:
            logger.warning(
                "Failed to generate %s residual plot for %s/%s/%s: %s",
                stage_kind,
                data_type,
                stage,
                var,
                exc,
            )
        finally:
            plt.close(fig)


def plot_stage_climatology_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
) -> None:
    clim_plot_specs = [{**spec, "ds": _build_climatology_ds(spec["ds"])} for spec in plot_specs]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=clim_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} climatology",
        x_dim="month",
        x_label="Month",
        filename_suffix="climatology_timeseries",
    )


def plot_stage_minus_climatology_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
) -> None:
    anom_plot_specs = [{**spec, "ds": _build_ds_minus_climatology_ds(spec["ds"])} for spec in plot_specs]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=anom_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} minus climatology",
        filename_suffix="minus_climatology_timeseries",
    )


def plot_stage_variance_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
    window: int | None = None,
) -> None:
    variance_plot_specs = [
        {
            **spec,
            "ds": _build_anomaly_variance_ds(spec["ds"], window=window),
        }
        for spec in plot_specs
    ]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=variance_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} anomaly variance",
        filename_suffix="variance_timeseries",
    )


def plot_stage_autocorr_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
    nlags: int | None = None,
) -> None:
    autocorr_plot_specs = [
        {
            **spec,
            "ds": _build_anomaly_autocorr_ds(spec["ds"], nlags=nlags),
        }
        for spec in plot_specs
    ]
    plot_stage_timeseries(
        logger=logger,
        plots_folder_path=plots_folder_path,
        plot_specs=autocorr_plot_specs,
        data_type=data_type,
        stage=stage,
        stage_kind=f"{stage_kind} anomaly autocorr",
        x_dim="lag",
        x_label="Lag",
        filename_suffix="autocorr_timeseries",
    )


def run_stage_plot_bundle(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    data_type: str,
    stage: str,
    stage_kind: str,
    residual_label: str,
    residual_mean_label: str,
    anomaly_residual_label: str,
    anomaly_residual_mean_label: str,
) -> None:
    shared_kwargs = dict(
        logger=logger,
        plots_folder_path=plots_folder_path,
        data_type=data_type,
        stage=stage,
        stage_kind=stage_kind,
    )

    plot_stage_timeseries(
        plot_specs=plot_specs,
        **shared_kwargs,
    )
    plot_stage_climatology_timeseries(
        plot_specs=plot_specs,
        **shared_kwargs,
    )
    plot_stage_minus_climatology_timeseries(
        plot_specs=plot_specs,
        **shared_kwargs,
    )
    plot_stage_variance_timeseries(
        plot_specs=plot_specs,
        **shared_kwargs,
    )
    plot_stage_autocorr_timeseries(
        plot_specs=plot_specs,
        **shared_kwargs,
    )
    plot_stage_residual_timeseries(
        left_ds=left_ds,
        right_ds=right_ds,
        residual_label=residual_label,
        residual_mean_label=residual_mean_label,
        **shared_kwargs,
    )
    plot_stage_residual_timeseries(
        left_ds=left_ds,
        right_ds=right_ds,
        stage_kind=f"{stage_kind} anomaly",
        residual_label=anomaly_residual_label,
        residual_mean_label=anomaly_residual_mean_label,
        anomaly=True,
        logger=logger,
        plots_folder_path=plots_folder_path,
        data_type=data_type,
        stage=stage,
    )
