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

from .utils import guess_lon_dim, guess_lat_dim, get_lonlat_coords

# Standalone plotting functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd


def metrics_to_dataframe (metrics, agg="mean"):
    """
    Flatten nested metrics dict into a tidy dataframe.

    Expected structure:
        metrics[train_period]["models"][model]["scalar"][metric] -> xarray.Dataset

    Output columns include at least:
        train_period, model, metric, variable, value
    plus any dataset dimensions (e.g. leadtime, realization) that survive aggregation.

    Parameters
    ----------
    metrics : dict
        Nested metrics dictionary.
    agg : str, callable, or None
        Reduction to apply over remaining dimensions.
        - None: keep all dataset dimensions expanded into dataframe columns
        - str/callable: reduce all non-coordinate dimensions to a scalar per
          (train_period, model, metric, variable, coordinate selection)
    """
    rows = []

    for train_period, period_block in metrics.items():
        for model, model_block in period_block.get("models", {}).items():
            for metric, ds in model_block.get("scalar", {}).items():
                for variable in ds.data_vars:
                    da = ds[variable]

                    if agg is None:
                        df = da.to_dataframe(name="value").reset_index()
                    else:
                        if callable(agg):
                            reduced = agg(da)
                        else:
                            reduced = getattr(da, agg)()

                        if hasattr(reduced, "compute"):
                            reduced = reduced.compute()

                        if np.ndim(reduced.values) == 0:
                            df = pd.DataFrame({"value": [float(reduced.values)]})
                        else:
                            df = reduced.to_dataframe(name="value").reset_index()

                    df["train_period"] = train_period
                    df["model"] = model
                    df["metric"] = metric
                    df["variable"] = variable

                    rows.append(df)

    if not rows:
        return pd.DataFrame(
            columns=["train_period", "model", "metric", "variable", "value"]
        )

    df = pd.concat(rows, ignore_index=True)

    # put metadata columns first
    preferred = ["train_period", "model", "metric", "variable"]
    other_cols = [c for c in df.columns if c not in preferred + ["value"]]
    df = df[preferred + other_cols + ["value"]]

    return df

def plot_scoreboard (
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

def plot_metric_vs_diff (
    fc_metrics_df,
    diff_metrics_df,
    *,
    forecast_metric="r2_global",
    diff_metric="nrmse_global",
    x_metric_name="diff_nrmse",
    y_metric_name="r2",
    # Fit
    fit_lines: bool = False,
    fit_kind: str = "linear",
    fit_min_points: int = 3,
    fit_alpha: float = 0.9,
    fit_lw: float = 1.0,
    fit_linestyles=None,            # optional dict: {leadtime: "-"} or None (auto)
    fit_color_mode: str = "period", # "period" or "black"
    fit_color: str = "black",       # used if fit_color_mode="black"
    # Fit CI
    fit_ci: bool = True,
    fit_ci_method: str = "bootstrap",   # "bootstrap" or "analytic"
    fit_ci_level: float = 0.95,
    fit_ci_alpha: float = 0.25,
    fit_bootstrap_iters: int = 2000,
    fit_bootstrap_seed: int | None = 0,
    # Visual
    group_cols=("total_months", "variable", "leadtime"),
    agg="mean",
    figsize=(12, 12),
    cmap_name="tab10",
    markers=("o", "s", "^", "D", "v", "P", "X"),
    point_size=20,
    edgecolor="black",
    linewidth=0.3,
    shade_strength=0.65,
    add_vline_at_0=True,
    shade_positive_x: bool = True,
    shade_positive_color: str = "red",
    shade_positive_alpha: float = 0.08,
    grid_alpha=0.3,
    xlabel=None,
    ylabel=None,
    title=None,
    # Legend
    legend=True,
    legend_labels: Sequence = None,
    legend_loc="upper left",
    legend_fontsize=9,
    leadtime_unit="",
    filters=None,  # e.g. {"variable": ["tas", "pr"], "leadtime": [1,2], "total_months":[12,24]}
):
    """
    Scatter plot: x = diff_metric (e.g., nrmse_global), y = forecast_metric (e.g., r2_global),
    grouped by (total_months, variable, leadtime). Colors encode variable, markers encode leadtime,
    and lighter shades encode total_months.

    Returns (fig, ax, merged_df).
    """

    def lighten(color, amount):
        r, g, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

    def _apply_filters(df):
        if not filters:
            return df
        out = df
        for col, allowed in filters.items():
            out = out[out[col].isin(allowed)]
        return out

    # Aggregate forecast metric
    df_fc = fc_metrics_df[fc_metrics_df["metric"] == forecast_metric].copy()
    # df_fc = _apply_filters(df_fc)

    if agg == "mean":
        df_fc = (
            df_fc.groupby(list(group_cols), as_index=False)["value"]
            .mean()
            .rename(columns={"value": y_metric_name})
        )
    elif callable(agg):
        df_fc = (
            df_fc.groupby(list(group_cols), as_index=False)["value"]
            .apply(agg)
            .rename(columns={"value": y_metric_name})
        )
    else:
        raise ValueError("agg must be 'mean' or a callable")

    # Aggregate diff metric
    df_diff = diff_metrics_df[diff_metrics_df["metric"] == diff_metric].copy()
    # df_diff = _apply_filters(df_diff)

    if agg == "mean":
        df_diff = (
            df_diff.groupby(list(group_cols), as_index=False)["value"]
            .mean()
            .rename(columns={"value": x_metric_name})
        )
    elif callable(agg):
        df_diff = (
            df_diff.groupby(list(group_cols), as_index=False)["value"]
            .apply(agg)
            .rename(columns={"value": x_metric_name})
        )
    else:
        raise ValueError("agg must be 'mean' or a callable")

    # Merge
    df = df_fc.merge(df_diff, on=list(group_cols), how="inner")
    df = _apply_filters(df)

    # Safety: if nothing to plot
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.set_title(title or "No data after filtering/merging")
        ax.set_xlabel(xlabel or x_metric_name)
        ax.set_ylabel(ylabel or y_metric_name)
        ax.grid(alpha=grid_alpha)
        return fig, ax, df

    variables = sorted(df["variable"].unique())
    leadtimes = sorted(df["leadtime"].unique())
    periods = sorted(df["total_months"].unique())

    cmap = plt.get_cmap(cmap_name)

    # Plot
    for vi, var in enumerate(variables):
        base = cmap(vi % cmap.N)
        for pi, tp in enumerate(periods):
            # amount in [0, shade_strength]
            denom = max(1, len(periods) - 1)
            amt = shade_strength * (1.0 - pi / denom)
            shade = lighten(base, amt)

            for li, lt in enumerate(leadtimes):
                dsub = df[
                    (df["variable"] == var)
                    & (df["total_months"] == tp)
                    & (df["leadtime"] == lt)
                ]
                if dsub.empty:
                    continue

                ax.scatter(
                    dsub[x_metric_name],
                    dsub[y_metric_name],
                    color=shade,
                    marker=markers[li % len(markers)],
                    s=point_size,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )

    # Fit separate lines per (total_months, leadtime), pooling all variables
    if fit_lines:
        if fit_linestyles is None:
            _ls_cycle = ["-", "--", ":", "-."]

        # Use a neutral base for period shading (fits shouldn't inherit variable colors)
        neutral_base = (0.2, 0.2, 0.2)

        for pi, tp in enumerate(periods):
            denom = max(1, (len(periods) - 1))
            amt = shade_strength * (pi / denom)

            if fit_color_mode == "period":
                line_color = lighten(neutral_base, amt)  # darker->lighter by period
            elif fit_color_mode == "black":
                line_color = fit_color
            else:
                raise ValueError("fit_color_mode must be 'period' or 'black'")

            for li, lt in enumerate(leadtimes):
                dsub = df[(df["total_months"] == tp) & (df["leadtime"] == lt)]

                if len(dsub) < fit_min_points:
                    continue

                x = dsub[x_metric_name].to_numpy()
                y = dsub[y_metric_name].to_numpy()

                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < fit_min_points:
                    continue

                x = x[mask]
                y = y[mask]

                if fit_kind == "linear":
                    # Linear regression y = a x + b
                    a, b = np.polyfit(x, y, 1)

                    xs = np.linspace(x.min(), x.max(), 200)
                    ys = a * xs + b

                    ls = (
                        _ls_cycle[li % len(_ls_cycle)]
                        if fit_linestyles is None
                        else fit_linestyles.get(lt, "-")
                    )

                    ax.plot(
                        xs, ys,
                        color=line_color,
                        linestyle=ls,
                        lw=fit_lw,
                        alpha=fit_alpha,
                        zorder=3,
                    )

                    # Bootstrap confidence interval on the mean fit
                    if fit_ci and fit_ci_method == "bootstrap":
                        rng = np.random.default_rng(fit_bootstrap_seed)
                        n = len(x)
                        if n >= 3:
                            preds = np.empty((fit_bootstrap_iters, xs.size), dtype=float)
                    
                            for bi in range(fit_bootstrap_iters):
                                idx = rng.integers(0, n, size=n)  # resample variables
                                xb = x[idx]
                                yb = y[idx]
                    
                                # guard: if all xb equal, skip
                                if np.allclose(xb, xb[0]):
                                    preds[bi, :] = np.nan
                                    continue
                    
                                ab, bb = np.polyfit(xb, yb, 1)
                                preds[bi, :] = ab * xs + bb
                    
                            lo_q = (1 - fit_ci_level) / 2
                            hi_q = 1 - lo_q
                            lo = np.nanquantile(preds, lo_q, axis=0)
                            hi = np.nanquantile(preds, hi_q, axis=0)
                    
                            ax.fill_between(xs, lo, hi, color=line_color, alpha=fit_ci_alpha, linewidth=0, zorder=2)

    # 0-line
    if add_vline_at_0:
        ax.axvline(0, color="k", alpha=0.5)

    # Labels
    ax.set_xlabel(xlabel or f"{diff_metric} (diff)")
    ax.set_ylabel(ylabel or forecast_metric)
    if title:
        ax.set_title(title)
    ax.grid(alpha=grid_alpha)

    if legend:
        handles = []
    
        def _section(title: str):
            # dummy, invisible handle used as a section header
            handles.append(Line2D([], [], linestyle="None", label=title))
    
        # 1) VARIABLES
        _section("Variable")
        var_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=cmap(vi % cmap.N),
                markeredgecolor=edgecolor,
                label=legend_labels[vi] if legend_labels else str(var),
                # label=str(var), # TODO fix this is too custom
            )
            for vi, var in enumerate(variables)
        ]
        handles.extend(var_handles)
    
        # 2) PERIODS (total_months)
        _section("Train window")
        neutral_base = (0.2, 0.2, 0.2)
        denom = max(1, (len(periods) - 1))
        period_handles = []
        for pi, tp in enumerate(periods):
            amt = shade_strength * (1.0 - pi / denom)
            shade = lighten(neutral_base, amt)
            period_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markersize=7,
                    markerfacecolor=shade,
                    markeredgecolor=edgecolor,
                    label=str(tp)+"M",
                )
            )
        handles.extend(period_handles)
    
        # 3) LEADTIME (marker for points + linestyle for fits)
        _section("Lead time")
        if fit_lines:
            if fit_linestyles is None:
                _ls_cycle = ["-", "--", ":", "-."]
                ls_for_lt = {lt: _ls_cycle[i % len(_ls_cycle)] for i, lt in enumerate(leadtimes)}
            else:
                ls_for_lt = {lt: fit_linestyles.get(lt, "-") for lt in leadtimes}
        else:
            ls_for_lt = {lt: "None" for lt in leadtimes}
    
        lt_handles = [
            Line2D(
                [0], [0],
                marker=markers[i % len(markers)],
                linestyle=ls_for_lt[lt],
                color="black",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="black",
                lw=fit_lw if fit_lines else 0,
                label=str(lt)+leadtime_unit,
            )
            for i, lt in enumerate(leadtimes)
        ]
        handles.extend(lt_handles)
    
        # Create ONE legend
        leg = ax.legend(
            handles=handles,
            loc=legend_loc,
            fontsize=legend_fontsize,
            frameon=True,
            handlelength=2.2,
            handletextpad=0.8,
            borderpad=0.8,
            labelspacing=0.6,
        )
    
        # Make section headers look like headers
        for txt in leg.get_texts():
            label = txt.get_text()
            if label in {"variable", "total_months", "leadtime"}:
                txt.set_weight("bold")
    
        ax.add_artist(leg)

    # Shade region x > 0
    if shade_positive_x:
        fig.canvas.draw()  # finalize limits without relim/autoscale
    
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    
        if xmax > 0:
            ax.axvspan(0, xmax, color=shade_positive_color, alpha=shade_positive_alpha, zorder=0)
    
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    return fig, ax, df


def _extent_from_da(da: xr.DataArray, lon_name="lon", lat_name="lat", pad_deg=2.0):
    lon = np.asarray(da[lon_name].values).ravel()
    lat = np.asarray(da[lat_name].values).ravel()

    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]

    if lon.size == 0 or lat.size == 0:
        raise ValueError("Cannot infer extent from empty lon/lat coordinates.")

    # Work in 0..360 to detect wrapped regional domains correctly
    lon360 = np.where(lon < 0, lon + 360, lon)
    lon360 = np.sort(lon360)

    lon_min = float(lon360.min())
    lon_max = float(lon360.max())

    # convert back to [-180, 180] if not crossing dateline in a problematic way
    if lon_max > 180 and lon_min > 180:
        lon_min -= 360
        lon_max -= 360
    elif lon_max > 180 and lon_min < 180:
        # keep as dateline-crossing geographic extent
        pass

    lat_min, lat_max = float(lat.min()), float(lat.max())

    lon_min -= pad_deg
    lon_max += pad_deg
    lat_min -= pad_deg
    lat_max += pad_deg

    return (lon_min, lon_max, lat_min, lat_max)

def _create_plot_grid(
    nrows: int,
    ncols: int,
    figsize_per_cell=(10, 4),
    data_projection=ccrs.PlateCarree(),
    plt_fontsize: int = 12,
    plt_projection=ccrs.PlateCarree(),  # Robinson()
    plt_coastlines: bool = True,
    plt_global_extent: bool = False,
    plt_reg_extent: Sequence[float] | None = None,  # lon_min, lon_max, lat_min, lat_max
    gridlabel_mode: str = "outer",  # "outer" or "all"
) -> dict:
    if gridlabel_mode not in {"outer", "all"}:
        raise ValueError(
            f"Invalid gridlabel_mode={gridlabel_mode!r}. Use 'outer' or 'all'."
        )

    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": plt_projection},
        squeeze=False,
    )

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]

            if plt_coastlines:
                ax.coastlines(linewidth=0.6)

            if plt_global_extent:
                ax.set_global()
            elif plt_reg_extent is not None:
                ax.set_extent(plt_reg_extent, crs=data_projection)

            gl = ax.gridlines(
                crs=data_projection,
                draw_labels=False,
                linewidth=0.35,
                linestyle="--",
                alpha=0.4,
            )
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

            show_labels = (
                gridlabel_mode == "all"
                or c == 0
                or r == nrows - 1
            )

            if show_labels:
                gl2 = ax.gridlines(
                    crs=data_projection,
                    draw_labels=True,
                    linewidth=0,
                )
                gl2.xformatter = LONGITUDE_FORMATTER
                gl2.yformatter = LATITUDE_FORMATTER
                gl2.top_labels = False
                gl2.right_labels = False

                if gridlabel_mode == "all":
                    gl2.left_labels = True
                    gl2.bottom_labels = True
                else:
                    gl2.left_labels = (c == 0)
                    gl2.bottom_labels = (r == nrows - 1)

                gl2.xlabel_style = {"size": plt_fontsize}
                gl2.ylabel_style = {"size": plt_fontsize}

    plt.close(fig)

    return fig, axes

def _create_data_panel(
    ds: xr.Dataset | xr.DataArray | List[xr.Dataset] | List[xr.DataArray],
    col_index: str | Sequence[str], # if str, look in Dataset, can't be both Sequences
    row_index: str | Sequence[str],
    var_sel: Optional[str] = None, # ignored for xr.DataArray
    extra_dim_sequence: Optional[Sequence[str]] = None, # optional sequence of names for the extra dimension (useful if ds is list)
    vars_to_be_removed: Optional[str] = "_has_var",
) -> np.ndarray:

    if col_index == "variable" and row_index == "variable":
        raise ValueError("col_index and row_index cannot be both 'variable'")
    if isinstance(ds, xr.DataArray) and (col_index == "variable" or row_index == "variable"):
        raise ValueError("Variable index not supported for xr.DataArray")

    if isinstance(ds, list):
        # Guards
        if len(ds) == 0:
            raise ValueError("Empty ds list.")
        if not all(isinstance(x, type(ds[0])) for x in ds):
            raise ValueError("All elements in ds list must be the same type.")

        if isinstance(ds[0], xr.Dataset):
            ds = xr.merge(ds) # assume different variables in datasets in list
        elif isinstance(ds[0], xr.DataArray):
            # At least one index must be "variable"
            if not (isinstance(col_index, str) and isinstance(row_index, str)):
                raise ValueError("For sequence of xr.DataArray, col_index and row_index must be strings (coord names and 'variable').")
            if col_index != "variable" and row_index != "variable":
                raise ValueError("For sequence of xr.DataArray, one of col_index and row_index must be 'variable'")
            data_dict = {(extra_dim_sequence[i] or f"var_{i}"): da for i, da in enumerate(ds)}
            ds = xr.Dataset(data_dict)
        else:
            raise ValueError(f"Unsupported dataset type in sequence: {type(ds[0])}")

    panels: list[xr.DataArray] = []
    if isinstance(ds, xr.Dataset):
        # Guards
        if isinstance(row_index, str) and row_index != "variable" and row_index not in ds.coords:
            raise KeyError(f"row_index {row_index!r} not found in Dataset coords.")
        if isinstance(col_index, str) and col_index != "variable" and col_index not in ds.coords:
            raise KeyError(f"col_index {col_index!r} not found in Dataset coords.")

        if col_index == "variable":
            if not isinstance(row_index, str) or row_index == "variable":
                raise ValueError("row_index must be a coord name (string) when col_index='variable'")
            cols = [v for v in ds.data_vars if v != vars_to_be_removed]
            rows = list(ds.coords[row_index].values)
            for r in rows:
                for v in cols:
                    panels.append(ds[v].sel({row_index: r}))

        elif row_index == "variable":
            if not isinstance(col_index, str) or col_index == "variable":
                raise ValueError("col_index must be a coord name (string) when row_index='variable'")
            cols = list(ds.coords[col_index].values)
            rows = [v for v in ds.data_vars if v != vars_to_be_removed]
            for v in rows:
                for c in cols:
                    panels.append(ds[v].sel({col_index: c}))

        else:
            # both are coord names (strings) OR you can decide to support explicit label sequences later
            if not isinstance(col_index, str) or not isinstance(row_index, str):
                raise ValueError("For Dataset, non-'variable' indices must be coord names (strings).")
            cols = list(ds.coords[col_index].values)
            rows = list(ds.coords[row_index].values)
            # Choose which variable to plot elsewhere (or require a `var` argument)
            if len(ds.data_vars) != 1 and var_sel is None:
                raise ValueError("Dataset has multiple variables; specify which one to plot with var_sel.")
            if var_sel is None:
                v = list(ds.data_vars)[0]
            else:
                if var_sel in ds.data_vars:
                    v = var_sel
                else:
                    raise ValueError(f"Variable {var_sel} not in dataset.")
            for r in rows:
                for c in cols:
                    panels.append(ds[v].sel({row_index: r, col_index: c}))

    elif isinstance(ds, xr.DataArray):
        if not (isinstance(col_index, str) and isinstance(row_index, str)):
            raise ValueError("For xr.DataArray, col_index and row_index must be coord names (strings).")
        if col_index not in ds.coords or row_index not in ds.coords:
            raise KeyError("col_index/row_index not found in DataArray coords.")

        cols = list(ds.coords[col_index].values)
        rows = list(ds.coords[row_index].values)

        for r in rows:
            for c in cols:
                panels.append(ds.sel({row_index: r, col_index: c}))

    else:
        raise ValueError(f"Unsupported dataset type: {type(ds)}")

    nrows, ncols = len(rows), len(cols)
    if ncols == 0 or nrows == 0:
        raise ValueError("No axes found in dataset.")

    # Create panel grid
    if len(panels) != nrows * ncols:
        raise ValueError(f"Expected {nrows*ncols} panels, got {len(panels)}")

    return panels, rows, cols

def _detect_limits_from_panels(
    panels: Sequence | np.ndarray,
    mode: Optional[str | Sequence] = "quantile",  # "quantile" or "minmax"
    quantile: float = 0.02,
) -> tuple[float, float]:
    panels = np.array(panels)
    flat_panels = panels.ravel()

    mins: list[float] = []
    maxs: list[float] = []

    per_panel_limits = None
    global_limits = None

    for i, v in enumerate(flat_panels):
        vv = v
        if hasattr(vv.data, "persist"):
            vv = vv.persist()

        arr = np.asarray(vv.data)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue

        if per_panel_limits is not None:
            lo, hi = per_panel_limits[i]
            mins.append(float(lo)); maxs.append(float(hi))
        elif global_limits is not None:
            lo, hi = global_limits
            mins.append(float(lo)); maxs.append(float(hi))
        else:
            if mode == "quantile":
                mins.append(float(np.quantile(arr, quantile)))
                maxs.append(float(np.quantile(arr, 1 - quantile)))
            elif mode == "minmax":
                mins.append(float(arr.min()))
                maxs.append(float(arr.max()))
            else:
                raise ValueError(f"Limits mode not supported: {mode!r}")

    if not mins or not maxs:
        raise ValueError("All selected panels contain only NaNs/non-finite values.")

    mins, maxs = np.array(mins), np.array(maxs)
    mins, maxs = np.array(mins), np.array(maxs)
    return mins.reshape(panels.shape()), maxs.reshape(panels.shape())

def _create_nparray_with_compatible_data(data, inner_data_type, rows, cols):
    if isinstance(data, Sequence) and not isinstance(data, str):
        if len(data)==len(rows):
            data_np = []
            for data_per_col in data:
                if isinstance(data_per_col, Sequence):
                    data_col = []
                    if len(data_per_col)==len(cols):
                        for data_per_row in data_per_col:
                            data_per_row = (None, None) if data_per_row is None else data_per_row # to avoid inhomogeneous array
                            data_col.append(data_per_row)
                        data_np.append(data_col)
                    else:
                        raise ValueError(f"Num of data per row {len(data_per_col)} != num panel cols {len(cols)}")
                else:
                    data_per_col = (None, None) if data_per_col is None else data_per_col # to avoid inhomogeneous array
                    data_np.append(len(rows)*[data_per_col])
        else:
            raise ValueError(f"Num of data per col {len(data)} != num panel rows {len(rows)}")
    elif isinstance(data, inner_data_type):
        data_np = len(rows)*[len(cols)*[data]]
    else:
        raise ValueError(f"Unsupported type {inner_data_type}")

    return np.array(data_np)

def _make_growing_norm(limits: Sequence[Number] | None):
    if (
        limits is None
        or (
            (isinstance(limits, Sequence) or isinstance(limits, np.ndarray))
            and len(limits)==2
            and limits[0] is None
            and limits[1] is None
        )
    ):
        return None

    if len(limits) != 2:
        raise ValueError(f"Limits length must be 2.")

    vmin, vmax = list(limits)
    if vmin < 0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        return Normalize(vmin=vmin, vmax=vmax)

def _crosses_dateline(extent) -> bool:
    lon_min, lon_max, *_ = extent
    return lon_max < lon_min

def _crop_da_to_extent(da: xr.DataArray, extent, data_projection=None) -> xr.DataArray:
    lon_name = guess_lon_dim(da)
    lat_name = guess_lat_dim(da)

    lon_min, lon_max, lat_min, lat_max = extent

    lon = da[lon_name]
    lat = da[lat_name]

    # Always crop in REAL lon coordinates, not shifted display coordinates
    lon180 = ((lon + 180) % 360) - 180

    lat_cond = (lat >= min(lat_min, lat_max)) & (lat <= max(lat_min, lat_max))

    if _crosses_dateline(extent):
        lon_cond = (lon180 >= lon_min) | (lon180 <= lon_max)
    else:
        lon_cond = (lon180 >= lon_min) & (lon180 <= lon_max)

    da = da.where(lat_cond, drop=True)
    da = da.where(lon_cond, drop=True)

    if da.size == 0:
        raise ValueError(
            f"Cropping produced empty array. "
            f"Requested extent={extent}, "
            f"lon range=({float(lon180.min())}, {float(lon180.max())}), "
            f"lat range=({float(lat.min())}, {float(lat.max())})"
        )

    return da

def _sort_lon_for_plot(da: xr.DataArray, force_wrap: bool = False) -> xr.DataArray:
    lon_name = guess_lon_dim(da)
    lon_vals = np.asarray(da[lon_name].values)

    if force_wrap:
        lon_plot = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
        da = da.assign_coords({lon_name: lon_plot}).sortby(lon_name)

    return da

def create_panel_from_data(
    ds: xr.Dataset | xr.DataArray | List[xr.Dataset] | List[xr.DataArray],
    col_index: str | Sequence[str],  # if str, look in Dataset, can't be both Sequences
    row_index: str | Sequence[str],
    var_sel: Optional[str] = None,  # ignored for xr.DataArray
    extra_dim_sequence: Optional[Sequence[str]] = None,  # optional sequence of names for the extra dimension (useful if ds is list)
    limits: Optional[str | dict | Sequence | Sequence[Sequence] | None] = None,  # "quantile" or "minmax" or dict with "rows" or "cols" keys or explicit limits
    limits_quantile: float = 0.02,
    save_path: str | Path | None = None,
    figsize_per_cell=(10, 4),
    cmap: str | Sequence[str] | Sequence[Sequence[str]] = "RdBu_r", # index 1: cols, index 2: rows
    cbar_orientation: str = "vertical",
    cbar_label: str = "",
    cbar_mode: str = "row",  # "row" or "subplot"
    gridlabel_mode: str = "outer",  # "outer" or "all"
    data_projection=ccrs.PlateCarree(),
    plt_col_titles: str | Sequence[str] | None = None,
    plt_ax_title_pre: str = "",
    plt_ax_title_suf: str = "",
    plt_fontsize: int = 12,
    plt_title_fontsize: int = 14,
    plt_projection=ccrs.PlateCarree(),  # Robinson()
    plt_coastlines: bool = True,
    plt_global_extent: bool = False,
    plt_regional_extent: Sequence[float] | None = None,  # lon_min, lon_max, lat_min, lat_max
    plt_pad_extent_deg: int = 0,
):
    if cbar_mode not in {"row", "subplot"}:
        raise ValueError(f"Invalid cbar_mode={cbar_mode!r}. Use 'row' or 'subplot'.")

    panels, rows, cols = _create_data_panel(ds, col_index, row_index, var_sel, extra_dim_sequence)
    nrows, ncols = len(rows), len(cols)

    if plt_regional_extent is None:
        ds_ref = panels[0]
        lon_name, lat_name = guess_lon_dim(ds_ref), guess_lat_dim(ds_ref)
        if isinstance(ds_ref, xr.Dataset):
            vars_ref = [v for v in ds_ref.data_vars if v != "_has_var"]
            if len(vars_ref) != 0:
                da_ref = ds_ref[vars_ref[0]]
            else:
                raise ValueError("Cannot select valid reference DataArray. Check input")
        elif isinstance(ds_ref, xr.DataArray):
            da_ref = ds_ref
        else:
            raise ValueError(f"Reference Dataset / DataArray has invalid type {type(ds_ref)}")

        extent = _extent_from_da(
            da_ref,
            lon_name=lon_name,
            lat_name=lat_name,
            pad_deg=plt_pad_extent_deg,
        )
    else:
        extent = list(plt_regional_extent)
        # lon_min, lon_max, lat_min, lat_max = plt_regional_extent
        # if lon_max < lon_min:
        #     lon_max += 360
        # if lon_min < 0:
        #     lon_min += 360
        # if lon_max < 0:
        #     lon_max += 360
        # extent = [lon_min, lon_max, lat_min, lat_max]

    fig, axes = _create_plot_grid(
        nrows,
        ncols,
        figsize_per_cell,
        data_projection,
        plt_fontsize,
        plt_projection,
        plt_coastlines,
        plt_global_extent,
        extent,
        gridlabel_mode=gridlabel_mode,
    )

    # Cmaps
    cmaps = _create_nparray_with_compatible_data(cmap, str, rows, cols)

    # Limits
    norms = np.array(len(rows)*[len(cols)*[None]]) # default auto limits
    if isinstance(limits, str): # quantile or maxmin
        norm_flatten = norms.ravel()
        mins, maxs = _detect_limits_from_panels(panels, limits, limits_quantile)
        for i,_ in enumerate(norm_flatten):
            norm_flatten[i] = _make_growing_norm((mins[i], maxs[i]))
        norms = norm_flatten.reshape(norms.shape())

    elif isinstance(limits, Sequence):
        # Could be (vmin, vmax) or list of (vmin, vmax) of length = num of cols or list of lists of length = num of rows
        if len(limits) == 2 and isinstance(limits[0], Number) and isinstance(limits[1], Number):
            norm = _make_growing_norm(limits)
            norms = np.array(len(rows)*[len(cols)*[norm]])
        else:
            limits_array = _create_nparray_with_compatible_data(limits, tuple, rows, cols)
            for r,c in product(range(nrows), range(ncols)):
                norms[r,c] = _make_growing_norm(limits_array[r,c])

    elif isinstance(limits, dict): # dict with rows or cols
        if (
            "cols" in limits
            and isinstance(limits["cols"], Sequence)
            and len(limits["cols"]) == ncols
        ):
            for r,c in product(range(nrows), range(ncols)):
                norms[r,c] = _make_growing_norm(limits["cols"][c])

        elif (
            "rows" in limits
            and isinstance(limits["rows"], Sequence)
            and len(limits["rows"]) == nrows
        ):
            for r,c in product(range(nrows), range(ncols)):
                norms[r,c] = _make_growing_norm(limits["rows"][r]) 

    elif limits is None:
        pass

    else:
        raise ValueError("Unsupported limit type: limits must be a string ('minmax'/'quantile'), a (vmin, vmax) pair, or a list (of lists) of (vmin, vmax), same dims as panel.")

    mappables, cbar_axes = [], []

    print("cmaps", cmaps)
    print("limits", norms)

    for r in range(nrows):
        row_mappable = None

        for c in range(ncols):
            ax = axes[r, c]
            p_idx = r * ncols + c

            data = panels[p_idx]

            crosses = (
                plt_regional_extent is not None
                and _crosses_dateline(plt_regional_extent)
            )

            if plt_regional_extent is not None:
                data = _crop_da_to_extent(data, plt_regional_extent)

            data = _sort_lon_for_plot(data, force_wrap=crosses)

            # set axis title
            if plt_col_titles is None:
                if extra_dim_sequence and len(extra_dim_sequence) == len(rows):
                    row_title = f"{plt_ax_title_pre}{extra_dim_sequence[r]}{plt_ax_title_suf}"
                else:
                    row_title = f"{plt_ax_title_pre}{plt_ax_title_suf}"
            elif isinstance(plt_col_titles, str):
                row_title = plt_col_titles
            elif isinstance(plt_col_titles, list):
                assert len(plt_col_titles)==ncols
                row_title = plt_col_titles[c]
            else:
                raise ValueError(f"plt_col_titles must be str, list of str or None, not {type(plt_col_titles)}")

            try:
                im = data.plot.pcolormesh(
                    ax=ax,
                    add_colorbar=False,
                    cmap=cmaps[r,c],
                    transform=data_projection,
                    norm=norms[r,c],
                )
            except Exception as e:
                ax.set_title("ERROR")
                ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                im = None

            mappables.append(im)

            col_val = data.coords[col_index].values if isinstance(col_index, str) else col_index[c]
            col_title = f" {col_index}: {col_val}"
            if plt_col_titles is None:
                row_title += col_title
            ax.set_title(row_title, fontsize=plt_title_fontsize)

            if im is not None:
                row_mappable = im

            # One colorbar per subplot
            if cbar_mode == "subplot" and im is not None:
                divider = make_axes_locatable(ax)
                if cbar_orientation == "horizontal":
                    print("Colorbar horizontal orientation not yet supported")
                else:
                    cax = divider.append_axes(
                        "right",
                        size="2.5%",
                        pad=0.04,
                        axes_class=plt.Axes,
                    )
                    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                    cbar.ax.tick_params(labelsize=plt_fontsize)
                    cbar.set_label(cbar_label, fontsize=plt_fontsize)
                    cbar_axes.append(cax)

        # One colorbar per row
        if cbar_mode == "row" and row_mappable is not None:
            last_ax = axes[r, ncols - 1]
            divider = make_axes_locatable(last_ax)
            if cbar_orientation == "horizontal":
                print("Colorbar horizontal orientation not yet supported")
            else:
                cax = divider.append_axes(
                    "right",
                    size="2.5%",
                    pad=0.04,
                    axes_class=plt.Axes,
                )
                cbar = fig.colorbar(row_mappable, cax=cax, orientation="vertical")
                cbar.ax.tick_params(labelsize=plt_fontsize)
                cbar.set_label(cbar_label, fontsize=plt_fontsize)
                cbar_axes.append(cax)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close(fig)

    return {
        "fig": fig,
        "axes": axes,
        "cbar_axes": cbar_axes,
        "mappable": mappables,
        "data": panels,
    }

def plot_ensemble_leadtime(
    ens_mean,
    members=None,
    extra=None,
    *,
    lead_dim="leadtime",
    ens_dim="realization",
    ax=None,
    label="",
    color=None,
    spread="std",          # "std" or "minmax"
    nsigma=1.0,
    plot_members=True,
    members_alpha=0.15,
    members_lw=0.8,
    members_ls="--",
    spread_alpha=0.25,
    mean_lw=2.5,
    extra_label="",
    extra_lw=2.0,
    extra_ls=":",
):
    """
    Plot ensemble mean, optionally members and spread vs lead time.

    Parameters
    ----------
    ens_mean : xarray.DataArray
        Ensemble mean with dim (lead_dim,)
    members : xarray.DataArray, optional
        Ensemble realizations with dims (lead_dim, ens_dim)
    extra : xarray.DataArray, optional
        Extra line with dim (lead_dim,)
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Ensure mean has correct dimension order
    ens_mean = ens_mean.transpose(lead_dim)
    x = ens_mean[lead_dim].values

    # Handle members if provided
    if members is not None:
        members = members.transpose(lead_dim, ens_dim)

        # Plot individual members
        if plot_members:
            y_members = members.values
            for i in range(y_members.shape[1]):
                ax.plot(
                    x,
                    y_members[:, i],
                    linestyle=members_ls,
                    linewidth=members_lw,
                    alpha=members_alpha,
                    color=color,
                )

        # Compute spread
        if spread == "std":
            sd = members.std(ens_dim)
            lower = (ens_mean - nsigma * sd).values
            upper = (ens_mean + nsigma * sd).values
        elif spread == "minmax":
            lower = members.min(ens_dim).values
            upper = members.max(ens_dim).values
        else:
            raise ValueError("spread must be 'std' or 'minmax'")

        # Plot spread
        ax.fill_between(
            x,
            lower,
            upper,
            alpha=spread_alpha,
            color=color,
            label=f"{label} spread" if label else "spread",
        )

    # Always plot mean
    ax.plot(
        x,
        ens_mean.values,
        linewidth=mean_lw,
        color=color,
        label=f"{label} ens mean" if label else "ens mean",
    )

    # Optional extra line
    if extra is not None:
        extra = extra.transpose(lead_dim)
        ax.plot(
            x,
            extra.values,
            linewidth=extra_lw,
            linestyle=extra_ls,
            color=color,
            label=f"{extra_label}" if extra_label else "extra",
        )

    ax.set_xlabel("Lead time")
    ax.grid(True, alpha=0.3)

    return ax

def quickplot (ds, varname="", folder="./", filename="rolled.png", t_idx=0):
    """Quick diagnostic plot of a 2D field in ds."""
    da = ds[varname] if isinstance(ds, xr.Dataset) else ds

    # Select time slice
    while da.ndim > 2:
        da = da.isel({da.dims[0]: t_idx})

    # Get lon/lat
    lon_coord, lat_coord = get_lonlat_coords(ds)
    lon = ds[lon_coord]
    lat = ds[lat_coord]

    plt.figure(figsize=(10,5))
    plt.pcolormesh(lon, lat, da, shading="auto")
    plt.colorbar(label=varname)
    plt.title(f"{varname}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    name, _, ext = filename.rpartition('.')
    plt.savefig(f"{folder}/{name}_{t_idx}.{ext}", dpi=150)
    plt.close()
