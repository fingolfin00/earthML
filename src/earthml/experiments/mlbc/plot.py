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


def plot_stage_timeseries(
    *,
    logger,
    plots_folder_path: Path,
    plot_specs: list[PlotSpec],
    data_type: str,
    stage: str,
    stage_kind: str,
    title_prefix: str = "",
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

            title = f"{var} - {data_type} - {stage}" if not title_prefix else f"{var} - {title_prefix} - {data_type} - {stage}"
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            fig.savefig(stage_plot_folder.joinpath(f"{stage}_{var}_timeseries.png"), dpi=150)
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
