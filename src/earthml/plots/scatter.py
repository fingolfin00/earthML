from typing import Sequence

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


def plot_metric_vs_diff(
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
    group_cols=None,
    shade_by: str = "total_months",
    shade_label: str | None = None,
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
    Scatter plot: x = diff_metric, y = forecast_metric.
    Colors encode variable, markers encode leadtime, and lighter shades encode
    the selected ``shade_by`` grouping column.

    Returns (fig, ax, merged_df).
    """

    if group_cols is None:
        group_cols = (shade_by, "variable", "leadtime")
    else:
        group_cols = tuple(group_cols)

    if shade_by not in group_cols:
        raise ValueError(f"shade_by={shade_by!r} must be included in group_cols={group_cols!r}")

    shade_label = shade_label or shade_by.replace("_", " ").title()

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
    shade_values = sorted(df[shade_by].unique())

    cmap = plt.get_cmap(cmap_name)

    # Plot
    for vi, var in enumerate(variables):
        base = cmap(vi % cmap.N)
        for pi, shade_value in enumerate(shade_values):
            # amount in [0, shade_strength]
            denom = max(1, len(shade_values) - 1)
            amt = shade_strength * (1.0 - pi / denom)
            shade = lighten(base, amt)

            for li, lt in enumerate(leadtimes):
                dsub = df[
                    (df["variable"] == var)
                    & (df[shade_by] == shade_value)
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

    # Fit separate lines per (shade_by, leadtime), pooling all variables
    if fit_lines:
        if fit_linestyles is None:
            _ls_cycle = ["-", "--", ":", "-."]

        # Use a neutral base for period shading (fits shouldn't inherit variable colors)
        neutral_base = (0.2, 0.2, 0.2)

        for pi, shade_value in enumerate(shade_values):
            denom = max(1, (len(shade_values) - 1))
            amt = shade_strength * (pi / denom)

            if fit_color_mode == "period":
                line_color = lighten(neutral_base, amt)  # darker->lighter by period
            elif fit_color_mode == "black":
                line_color = fit_color
            else:
                raise ValueError("fit_color_mode must be 'period' or 'black'")

            for li, lt in enumerate(leadtimes):
                dsub = df[(df[shade_by] == shade_value) & (df["leadtime"] == lt)]

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
    
        # 2) SHADED GROUP
        _section(shade_label)
        neutral_base = (0.2, 0.2, 0.2)
        denom = max(1, (len(shade_values) - 1))
        shade_handles = []
        for pi, shade_value in enumerate(shade_values):
            amt = shade_strength * (1.0 - pi / denom)
            shade = lighten(neutral_base, amt)
            label = f"{shade_value}M" if shade_by == "total_months" else str(shade_value)
            shade_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markersize=7,
                    markerfacecolor=shade,
                    markeredgecolor=edgecolor,
                    label=label,
                )
            )
        handles.extend(shade_handles)
    
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
            if label in {"Variable", shade_label, "Lead time"}:
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
