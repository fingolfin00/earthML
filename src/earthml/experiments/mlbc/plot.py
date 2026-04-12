from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


def plot_stage_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
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
    time_dim = base_ds.earthml.guessed_dims.time
    if time_dim is None or any(time_dim not in spec["ds"].dims for spec in plot_specs):
        logger.debug("Skip %s plotting for %s: no common time dimension.", stage_kind, data_type)
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
                    x_dim=time_dim,
                    ens_dim=rdim or "realization",
                    ax=ax,
                    x_label="Time",
                    label=spec["label"],
                    mean_label=spec["mean_label"],
                    color=spec["color"],
                )

            ax.legend()
            fig.tight_layout()
            fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_timeseries.png"), dpi=200)
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
            residual_ds = _build_residual_ds(left_ds, right_ds, var=var)
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
            fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_residual_timeseries.png"), dpi=200)
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
