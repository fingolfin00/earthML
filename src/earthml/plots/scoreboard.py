from typing import List, Optional, Sequence
from numbers import Number
from pathlib import Path
from itertools import product

from rich import print

import numpy as np
import pandas as pd
import cf_xarray
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_scoreboard(
    df,
    outer_y: dict,   # {"index": "metric", "values": [...]}
    outer_x: dict,   # {"index": "model", "values": [...]}
    inner_y: dict,   # {"index": "train_period", "values": [...]}
    inner_x: dict,   # {"index": "leadtime", "values": [...]}
    agg="mean",
    filters=None,
    metric_cmaps=None,
    metric_vlims=None,
    metric_names=None,
    metric_factor=1,
    figsize=None,
    cell_size=(0.8, 0.5),   # (width_per_score, height_per_score) in inches
    wspace=0.15,
    hspace=0.15,
    title=None,
    plt_title_fontsize=20,
    plt_outer_fontsize=14,
    plt_inner_fontsize=10,
    cbar_show=True,
    cbar_width=0.18,
    annotate=False,
    annotate_fmt="{:.2f}",
    annotate_color="auto",
    save_path=None,
    panel_select=None,
):
    """
    Generic heatmap grid from a tidy dataframe.

    The dataframe must contain a 'value' column and any columns referenced by:
        outer_y["index"], outer_x["index"], inner_y["index"], inner_x["index"]

    Parameters
    ----------
    df : pd.DataFrame
        Tidy dataframe.
    filters : dict, optional
        Fixed filters applied before plotting, e.g.
            {"train_period": "19930701-20201231"}
            {"variable": ["sst_mseloss", "sst_maskedmseloss"]}
    panel_select : dict, optional
        Per-subplot selector. Lets you collapse inner_x or inner_y only for
        specific outer panels, without aggregating across that dimension.

        Supported forms:
            {
                ("metric1", "model1"): {"inner_x": 3},
                ("metric1", "model2"): {"inner_x": None},
                "model1": {"inner_x": 3},
                "metric2": {"inner_y": "19930701-20201231"},
            }

        Rules:
        - "inner_x": value   -> keep only that one inner_x value
        - "inner_y": value   -> keep only that one inner_y value
        - None means keep all
        - if both inner_x and inner_y are given, both are reduced
    """
    metric_cmaps = metric_cmaps or {}
    metric_vlims = metric_vlims or {}
    metric_names = metric_names or {}
    filters = filters or {}
    panel_select = panel_select or {}

    data = df.copy()

    for col, val in filters.items():
        if isinstance(val, (list, tuple, set, pd.Index, np.ndarray)):
            data = data[data[col].isin(val)]
        else:
            data = data[data[col] == val]

    def _axis_values(axis, source_df):
        if axis.get("values") is not None:
            return list(axis["values"])
        return list(pd.unique(source_df[axis["index"]]))

    outer_y_vals = _axis_values(outer_y, data)
    outer_x_vals = _axis_values(outer_x, data)
    full_inner_y_vals = _axis_values(inner_y, data)
    full_inner_x_vals = _axis_values(inner_x, data)

    def _get_panel_selector(m, v):
        sel = panel_select.get((m, v))
        if sel is None:
            sel = panel_select.get(v)
        if sel is None:
            sel = panel_select.get(m)
        return sel or {}

    pivots = {m: {} for m in outer_y_vals}
    metric_values = {m: [] for m in outer_y_vals}
    panel_shapes = {}  # (m, v) -> (nrows, ncols)

    for m in outer_y_vals:
        for v in outer_x_vals:
            d = data[
                (data[outer_y["index"]] == m) &
                (data[outer_x["index"]] == v)
            ].copy()

            if d.empty:
                pivots[m][v] = None
                panel_shapes[(m, v)] = (1, 1)
                continue

            sel = _get_panel_selector(m, v)

            panel_inner_x_vals = full_inner_x_vals
            panel_inner_y_vals = full_inner_y_vals

            if sel.get("inner_x") is not None:
                panel_inner_x_vals = [sel["inner_x"]]

            if sel.get("inner_y") is not None:
                panel_inner_y_vals = [sel["inner_y"]]

            d = d[
                d[inner_y["index"]].isin(panel_inner_y_vals) &
                d[inner_x["index"]].isin(panel_inner_x_vals)
            ]

            if d.empty:
                pivots[m][v] = None
                panel_shapes[(m, v)] = (len(panel_inner_y_vals), len(panel_inner_x_vals))
                continue

            p = d.pivot_table(
                index=inner_y["index"],
                columns=inner_x["index"],
                values="value",
                aggfunc=agg,
            )

            p = p.reindex(index=panel_inner_y_vals, columns=panel_inner_x_vals)
            p = p * metric_factor

            pivots[m][v] = p
            metric_values[m].append(p.values)
            panel_shapes[(m, v)] = p.shape

    # Compute physical column widths and row heights from panel sizes
    col_units = []
    for v in outer_x_vals:
        max_cols_for_this_outer_x = max(panel_shapes[(m, v)][1] for m in outer_y_vals)
        col_units.append(max(1, max_cols_for_this_outer_x))

    row_units = []
    for m in outer_y_vals:
        max_rows_for_this_outer_y = max(panel_shapes[(m, v)][0] for v in outer_x_vals)
        row_units.append(max(1, max_rows_for_this_outer_y))

    # Optional colorbar column for each row
    width_ratios = col_units + ([cbar_width] if cbar_show else [])
    height_ratios = row_units

    if figsize is None:
        fig_w = sum(col_units) * cell_size[0] + (len(col_units) - 1) * wspace
        if cbar_show:
            fig_w += cbar_width * cell_size[0]
        fig_h = sum(row_units) * cell_size[1] + (len(row_units) - 1) * hspace
        if title:
            fig_h += 0.6
        figsize = (fig_w, fig_h)

    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=len(outer_y_vals),
        ncols=len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=wspace,
        hspace=hspace,
    )

    axes = {}

    for i, m in enumerate(outer_y_vals):
        for j, v in enumerate(outer_x_vals):
            axes[(i, j)] = fig.add_subplot(gs[i, j])

    for i, m in enumerate(outer_y_vals):
        valid_arrays = [
            arr for arr in metric_values[m]
            if arr is not None and np.size(arr) and not np.isnan(arr).all()
        ]

        if m in metric_vlims:
            vmin, vmax = metric_vlims[m]
        elif valid_arrays:
            vals = np.concatenate([a.ravel() for a in valid_arrays])
            vals = vals[~np.isnan(vals)]
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        else:
            vmin, vmax = 0.0, 1.0

        cmap = metric_cmaps.get(m, "viridis")
        im_row = None

        for j, v in enumerate(outer_x_vals):
            ax = axes[(i, j)]
            p = pivots[m][v]

            if p is None or p.empty:
                ax.text(0.5, 0.5, "NO DATA", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0 < vmax else None

            im = ax.imshow(
                p.values,
                aspect="equal",
                cmap=cmap,
                norm=norm,
                vmin=None if norm else vmin,
                vmax=None if norm else vmax,
                interpolation="none",
            )

            if annotate:
                for y in range(p.shape[0]):
                    for x in range(p.shape[1]):
                        val = p.values[y, x]
                        if np.isnan(val):
                            continue

                        if annotate_color == "auto":
                            rgba = im.cmap(im.norm(val))
                            r, g, b, _ = rgba
                            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            txt_color = "black" if luminance > 0.5 else "white"
                        else:
                            txt_color = annotate_color

                        ax.text(
                            x, y, annotate_fmt.format(val),
                            ha="center", va="center",
                            fontsize=plt_inner_fontsize,
                            color=txt_color,
                        )

            im_row = im

            if i == 0:
                labels = outer_x.get("label")
                ax.set_title(labels[j] if labels is not None else v, fontsize=plt_outer_fontsize)

            ax.set_xticks(range(len(p.columns)))
            if i == len(outer_y_vals) - 1:
                ax.set_xticklabels(list(p.columns), fontsize=plt_inner_fontsize)
            else:
                ax.set_xticklabels([])

            ax.set_yticks(range(len(p.index)))
            if j == 0:
                ax.set_yticklabels(list(p.index), fontsize=plt_inner_fontsize)
            else:
                ax.set_yticklabels([])

        if cbar_show and im_row is not None:
            divider = make_axes_locatable(axes[(i, len(outer_x_vals)-1)])
            cax = divider.append_axes("right", size="2%", pad=0.03)
            cbar = fig.colorbar(im_row, cax=cax)
            cbar.set_label(metric_names.get(m, m), fontsize=plt_inner_fontsize)
            cbar.ax.tick_params(labelsize=plt_inner_fontsize)

    if title:
        fig.suptitle(title, fontsize=plt_title_fontsize)

    fig.supxlabel(inner_x.get("label", inner_x["index"]), fontsize=plt_outer_fontsize)
    fig.supylabel(inner_y.get("label", inner_y["index"]), fontsize=plt_outer_fontsize)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close(fig)
    return fig
