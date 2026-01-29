from typing import List, Optional, Sequence
from numbers import Number
from pathlib import Path

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

from .utils import guess_time_dim, guess_lon_dim, guess_lat_dim

# Standalone plotting functions

def plot_scoreboard (
    df,
    # metrics,
    # variables,
    outer_y: dict, # {"index": index_name, "values": values} metrics
    outer_x: dict, # variables
    inner_y: dict, # train_periods
    inner_x: dict, # leadtimes
    agg="mean",
    metric_cmaps=None,
    metric_vlims=None,
    metric_names=None,
    metric_factor=1,
    figsize=(24, 12),
    title=None,
    plt_title_fontsize=20,
    plt_outer_fontsize=14,
    plt_inner_fontsize=10,
    cbar_show: bool = True,
    annotate=False,
    annotate_fmt="{:.2f}",
    annotate_color="auto",   # "auto", or e.g. "black"/"white"
    save_path=None,
):
    """
    Heatmap grid with metric-specific color scales (one colorbar per row).
    """
    # Defaults
    metric_cmaps = metric_cmaps or {}
    metric_vlims = metric_vlims or {}

    # Precompute pivots
    pivots = {m: {} for m in outer_y["values"]}
    metric_values = {m: [] for m in outer_y["values"]}

    for m in outer_y["values"]: # metrics
        for v in outer_x["values"]: # variables
            d = df[(df[outer_y["index"]] == m) & (df[outer_x["index"]] == v)].copy()
            if d.empty:
                pivots[m][v] = None
                continue

            d_filtered = d.copy()

            if inner_x.get("values") is not None:
                d_filtered = d_filtered[
                    d_filtered[inner_x["index"]].isin(inner_x["values"])
                ]

            if inner_y.get("values") is not None:
                d_filtered = d_filtered[
                    d_filtered[inner_y["index"]].isin(inner_y["values"])
                ]

            p = (d_filtered.pivot_table(
                    index=inner_y["index"], # train periods (total_months)
                    columns=inner_x["index"], # leadtimes
                    values="value",
                    aggfunc=agg
                )
                .sort_index())

            p = p * metric_factor

            idx = pd.Index(p.index)
            as_num = pd.to_numeric(idx, errors="coerce")
            # If index is not numeric, try converting
            if as_num.notna().all():
                p = p.set_index(as_num)
                p.index.name = idx.name
                # Ensure descending order
                p = p.sort_index(ascending=False)

            pivots[m][v] = p
            metric_values[m].append(p.values)

    nrows, ncols = len(outer_y["values"]), len(outer_x["values"])
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True
    )

    for i, m in enumerate(outer_y["values"]): # metrics
        # Metric-specific limits
        if m in metric_vlims:
            vmin, vmax = metric_vlims[m]
        else:
            vals = np.concatenate(metric_values[m])
            vmin, vmax = np.nanmin(vals), np.nanmax(vals)

        cmap = metric_cmaps.get(m, "viridis")

        im_row = None

        for j, v in enumerate(outer_x["values"]): # variables
            ax = axes[i, j]
            p = pivots[m][v]

            if p is None:
                ax.text(0.5, 0.5, "NO DATA", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Decide normalization
            if vmin < 0 < vmax:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                norm = None
            
            im = ax.imshow(
                p.values,
                aspect="auto",
                cmap=cmap,
                norm=norm,
                vmin=None if norm else vmin,
                vmax=None if norm else vmax,
            )

            # Annotate value
            if annotate:
                for y in range(p.shape[0]):
                    for x in range(p.shape[1]):
                        val = p.values[y, x]
                        if np.isnan(val):
                            continue

                        if annotate_color == "auto":
                            # Convert the cell value to RGBA using the same norm/cmap as the image
                            rgba = im.cmap(im.norm(val))
                            r, g, b, _ = rgba
                            # Perceived luminance (sRGB)
                            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                            txt_color = "black" if luminance > 0.5 else "white"
                        else:
                            txt_color = annotate_color

                        ax.text(
                            x, y,
                            annotate_fmt.format(val),
                            ha="center", va="center",
                            fontsize=plt_inner_fontsize,
                            color=txt_color,
                        )

            im_row = im

            if i == 0:
                outer_x_title = v if "label" not in outer_x else outer_x["label"][i]
                ax.set_title(outer_x_title, fontsize=plt_outer_fontsize)
            # X ticks
            if i == nrows - 1:
                ax.set_xticks(range(len(p.columns)))
                p_col_ticks = p.columns if "tick_label" not in inner_x else inner_x["tick_label"]
                ax.set_xticklabels(p_col_ticks)
                ax.tick_params(axis='x', which='major', labelsize=plt_inner_fontsize)
            else:
                ax.set_xticks([])
            # Y ticks
            if j == 0:
                ax.set_yticks(range(len(p.index)))
                p_idx_ticks = p.index if "tick_label" not in inner_y else inner_y["tick_label"]
                ax.set_yticklabels(p_idx_ticks)
                ax.tick_params(axis='y', which='major', labelsize=plt_inner_fontsize)
            else:
                ax.set_yticks([])

        # One colorbar per metric row
        if im_row is not None:
            last_ax = axes[i, ncols - 1]
            divider = make_axes_locatable(last_ax)
            cax = divider.append_axes(
                "right",
                size="2.5%",
                pad=0.04,
                axes_class=plt.Axes,
            )
            cbar = fig.colorbar(im_row, cax=cax, orientation="vertical")
            cbar.ax.tick_params(labelsize=plt_outer_fontsize)
            # cbar_label = metric_names[m] if m in metric_names else m
            if cbar_show:
                cbar_label = metric_names.get(m, m) if metric_names else m
            else:
                cbar_label=""
            cbar.set_label(cbar_label, fontsize=plt_inner_fontsize)
            cbar.ax.tick_params(labelsize=plt_inner_fontsize)

    if title:
        fig.suptitle(title, fontsize=plt_title_fontsize)

    # Global labels
    fig.supxlabel(
        f"{inner_x["index"]}" if "label" not in inner_x else inner_x["label"],
        fontsize=plt_outer_fontsize,
    )
    fig.supylabel(
        f"{inner_y["index"]}" if "label" not in inner_y else inner_y["label"],
        fontsize=plt_outer_fontsize,
    )

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
                label=str(var).replace("_mse", ""),
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





def _extent_from_da (da: xr.DataArray, lon_name="lon", lat_name="lat", pad_deg=2.0):
    lon = da[lon_name].values
    lat = da[lat_name].values

    # reduce to 1D arrays if needed
    lon = np.asarray(lon).ravel()
    lat = np.asarray(lat).ravel()

    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]

    # handle 0..360 longitudes -> convert to -180..180 for plotting
    if lon.size and lon.min() >= 0 and lon.max() > 180:
        lon = ((lon + 180) % 360) - 180

    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # padding
    lon_min -= pad_deg
    lon_max += pad_deg
    lat_min -= pad_deg
    lat_max += pad_deg

    return (lon_min, lon_max, lat_min, lat_max)

def _create_plot_grid (
    nrows: int, ncols: int,
    figsize_per_cell=(10, 4),
    data_projection = ccrs.PlateCarree(),
    plt_fontsize: int = 12,
    plt_projection = ccrs.PlateCarree(), # Robinson()
    plt_coastlines: bool = True,
    plt_global_extent: bool = False,
    plt_reg_extent: Sequence[float] | None = None # lon_min, lon_max, lat_min, lat_max
) -> dict:

    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": plt_projection},
        squeeze=False,
    )

    # cbar_axes = []
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]

            if plt_coastlines:
                ax.coastlines(linewidth=0.6)

            if plt_global_extent:
                ax.set_global()
            else:
                if plt_reg_extent:
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

            if (c == 0) or (r == nrows - 1):
                gl2 = ax.gridlines(
                    crs=data_projection,
                    draw_labels=True,
                    linewidth=0,
                )
                gl2.xformatter = LONGITUDE_FORMATTER
                gl2.yformatter = LATITUDE_FORMATTER
                gl2.top_labels = False
                gl2.right_labels = False
                gl2.left_labels = (c == 0)
                gl2.bottom_labels = (r == nrows - 1)
                gl2.xlabel_style = {"size": plt_fontsize}
                gl2.ylabel_style = {"size": plt_fontsize}

        # One colorbar per row (use the last axis in the row)
        # last_ax = axes[r, ncols - 1]
        # divider = make_axes_locatable(last_ax)
        # cbar_axes.append(divider.append_axes(
        #     "right",
        #     size="2.5%",
        #     pad=0.04,
        #     axes_class=plt.Axes,
        # ))

    plt.close(fig)

    return fig, axes #, cbar_axes

def _create_data_panel (
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

def _detect_limits_from_panels (
    panels: Sequence | np.ndarray,
    limits: Optional[str | Sequence] = "quantile",  # "quantile" or "minmax" or explicit limits
    quantile: float = 0.02,
) -> tuple[float, float]:

    if isinstance(panels, np.ndarray):
        flat_panels = panels.ravel()
    elif isinstance(panels, Sequence) and not isinstance(panels, (str, bytes)):
        flat_panels = panels
    else:
        raise ValueError("Panels must be a sequence or numpy array.")

    mins: list[float] = []
    maxs: list[float] = []

    # Normalize limits mode once
    per_panel_limits = None
    global_limits = None
    mode = None

    if limits is None:
        mode = "quantile"
    elif isinstance(limits, str):
        mode = limits
    elif isinstance(limits, Sequence) and not isinstance(limits, (str, bytes)):
        # Could be (vmin, vmax) or list of (vmin, vmax)
        if len(limits) == 2 and not (isinstance(limits[0], Sequence) and isinstance(limits[1], Sequence)):
            global_limits = (limits[0], limits[1])
        elif len(limits) == len(flat_panels):
            per_panel_limits = limits
        else:
            raise ValueError("limits must be a string ('minmax'/'quantile'), a (vmin, vmax) pair, or a list of (vmin, vmax) per panel.")
    else:
        raise ValueError("Unsupported limits type.")

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

    vmin = float(np.min(mins))
    vmax = float(np.max(maxs))

    return vmin, vmax

def create_panel_from_data (
    ds: xr.Dataset | xr.DataArray | List[xr.Dataset] | List[xr.DataArray],
    col_index: str | Sequence[str], # if str, look in Dataset, can't be both Sequences
    row_index: str | Sequence[str],
    var_sel: Optional[str] = None, # ignored for xr.DataArray
    extra_dim_sequence: Optional[Sequence[str]] = None, # optional sequence of names for the extra dimension (useful if ds is list)
    limits: Optional[str | Sequence | dict | None] = None,  # "quantile" or "minmax" or explicit limits
    limits_quantile: float = 0.02,
    save_path: str | Path | None = None,
    figsize_per_cell=(10, 4),
    cmap: str = "RdBu_r",
    cbar_orientation: str = "vertical",
    cbar_label: str = "",
    data_projection = ccrs.PlateCarree(),
    plt_ax_title_pre: str = "",
    plt_ax_title_suf: str = "",
    plt_fontsize: int = 12,
    plt_projection = ccrs.PlateCarree(), # Robinson()
    plt_coastlines: bool = True,
    plt_global_extent: bool = False,
    plt_regional_extent: Sequence[float] | None = None, # lon_min, lon_max, lat_min, lat_max
    plt_pad_extent_deg: int = 0,
):

    panels, rows, cols = _create_data_panel(ds, col_index, row_index, var_sel, extra_dim_sequence)
    nrows, ncols = len(rows), len(cols)
    # print(nrows, ncols)

    if plt_regional_extent is None:
        ds_ref = panels[0]
        lon_name, lat_name = guess_lon_dim(ds_ref), guess_lat_dim(ds_ref)
        if isinstance(ds_ref, xr.Dataset):
            vars_ref = [v for v in ds_ref.data_vars if v != "_has_var"]
            if len(vars_ref) != 0:
                da_ref = ds_ref[vars_ref[0]]
            else:
                raise ValueError(f"Cannote select valid reference DataArray. Check input")
        elif isinstance(ds_ref, xr.DataArray):
            da_ref = ds_ref
        else:
            raise ValueError(f"Reference Dataset / DataArray has invalid type {type(ds_ref)}")

        extent = _extent_from_da(da_ref, lon_name=lon_name, lat_name=lat_name, pad_deg=plt_pad_extent_deg)
    else:
        extent = None

    fig, axes = _create_plot_grid(
        nrows, ncols,
        figsize_per_cell,
        data_projection,
        plt_fontsize, plt_projection, plt_coastlines,
        plt_global_extent, extent,
    )

    # Global limits
    if (
        limits == "quantile"
        or
        limits == "minmax"
        or (
            isinstance(limits, Sequence)
            and not isinstance(limits, (str, bytes))
            and len(limits) == 2
            and all(isinstance(x, Number) for x in limits)
        )
    ):
        vmin, vmax = _detect_limits_from_panels(panels, limits, limits_quantile)
        if vmin < 0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    mappables, cbar_axes = [], []
    for r in range(nrows):
        # Guess row title from sequence
        if extra_dim_sequence and len(extra_dim_sequence) == len(rows):
            row_title = f"{plt_ax_title_pre}{extra_dim_sequence[r]}{plt_ax_title_suf}"
        else:
            row_title = f"{plt_ax_title_pre}{plt_ax_title_suf}"
        if isinstance(limits, dict) and "rows" in limits.keys() and isinstance(limits["rows"], Sequence) and len(limits["rows"]) == nrows:
            vmin, vmax = limits["rows"][r]
            if vmin < 0 < vmax:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
        for c in range(ncols):
            if isinstance(limits, dict) and "cols" in limits.keys() and isinstance(limits["cols"], Sequence) and len(limits["cols"]) == nrows:
                vmin, vmax = limits["cols"][c]
                if vmin < 0 < vmax:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                else:
                    norm = Normalize(vmin=vmin, vmax=vmax)
            ax = axes[r,c]
            p_idx = r * ncols + c
            # print(p_idx)
            data = panels[p_idx]
            # print(data)
            try:
                im = data.plot.pcolormesh(
                    ax=ax,
                    add_colorbar=False,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    norm=norm,
                )
            except Exception as e:
                ax.set_title(f"ERROR")
                ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                im = None
            mappables.append(im)

            col_val = data.coords[col_index].values if isinstance(col_index, str) else col_index[c]
            col_title = f" {col_index}: {col_val}" 
            ax.set_title(row_title+col_title, fontsize=plt_fontsize)
        
        row_mappable = im  # keep last for row colorbar

        # One colorbar per row / col
        if row_mappable is not None:
            # print("Create cbar")
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

    return {"fig": fig, "axes": axes, "cbar_axes": cbar_axes, "mappable": mappables, "data": panels}

def plot_ensemble_leadtime (
    members,
    ens_mean,
    extra=None, # extra line
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
    Plot ensemble members, ensemble mean, and spread vs lead time.

    Parameters
    ----------
    members : xarray.DataArray
        Ensemble realizations with dims (lead_dim, ens_dim)
    ens_mean : xarray.DataArray
        Ensemble mean with dim (lead_dim,)
    extra : xarray.DataArray, optional
        Extra line with dim (lead_dim,)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # Enforce dimension order
    members = members.transpose(lead_dim, ens_dim)
    ens_mean = ens_mean.transpose(lead_dim)

    x = members[lead_dim].values

    # Plot individual members
    if plot_members:
        y_members = members.values  # (nlead, nens)
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

    # Plot spread + mean
    ax.fill_between(
        x,
        lower,
        upper,
        alpha=spread_alpha,
        color=color,
        label=f"{label} spread" if label else "spread",
    )

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
