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
    color_by: str | Sequence[str] = "variable",
    marker_by: str | Sequence[str] = "leadtime",
    shade_by: str | Sequence[str] = "total_months",
    shade_label: str | None = None,
    agg=None,
    figsize=(12, 12),
    cmap_name="tab10",
    variable_colors: dict[str, str] | None = None,
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
    Color, marker and shade encodings are controlled by ``color_by``,
    ``marker_by`` and ``shade_by``. Each encoding can be a single column
    name or a sequence / delimited string of column names to combine into a
    single visual grouping. The scatter never aggregates points: all shared
    run-context columns are preserved during the merge unless filtered out.

    Returns (fig, ax, merged_df).
    """

    def _normalize_encoding_fields(field_spec: str | Sequence[str]) -> tuple[str, ...]:
        if isinstance(field_spec, str):
            normalized = field_spec
            for delimiter in (",", "+", "|"):
                normalized = normalized.replace(delimiter, ",")
            fields = [field.strip() for field in normalized.split(",") if field.strip()]
            return tuple(fields)
        return tuple(str(field).strip() for field in field_spec if str(field).strip())

    def _encoding_label(fields: tuple[str, ...]) -> str:
        if len(fields) == 1:
            return fields[0].replace("_", " ").title()
        return " + ".join(field.replace("_", " ").title() for field in fields)

    def _encoding_column_name(role: str, fields: tuple[str, ...]) -> str:
        if len(fields) == 1:
            return fields[0]
        return f"__{role}_{'_'.join(fields)}__"

    def _format_encoding_component(field: str, value: object) -> str:
        if field == "leadtime":
            return f"{value}{leadtime_unit}" if leadtime_unit else str(value)
        return str(value)

    def _format_encoding_value(fields: tuple[str, ...], value: object) -> str:
        if len(fields) == 1:
            return _format_encoding_component(fields[0], value)
        components = value if isinstance(value, tuple) else (value,)
        return " | ".join(
            _format_encoding_component(field, component)
            for field, component in zip(fields, components)
        )

    def _ensure_unique_keys(df: pd.DataFrame, key_cols: tuple[str, ...], label: str) -> None:
        if df.empty:
            return
        duplicated = df.duplicated(subset=list(key_cols), keep=False)
        if not duplicated.any():
            return
        duplicate_rows = df.loc[duplicated, list(key_cols)].drop_duplicates().head(5)
        sample = duplicate_rows.to_dict(orient="records")
        raise ValueError(
            f"{label} contains duplicate rows for merge keys {key_cols!r}. "
            "Scatter plots no longer aggregate automatically; please filter or "
            f"encode the extra varying dimensions explicitly. Sample duplicate keys: {sample!r}"
        )

    color_fields = _normalize_encoding_fields(color_by)
    marker_fields = _normalize_encoding_fields(marker_by)
    shade_fields = _normalize_encoding_fields(shade_by)
    if not color_fields:
        raise ValueError("color_by must reference at least one column")
    if not marker_fields:
        raise ValueError("marker_by must reference at least one column")
    if not shade_fields:
        raise ValueError("shade_by must reference at least one column")

    if agg not in (None, "none"):
        raise ValueError(
            "plot_metric_vs_diff no longer aggregates points. "
            "Use filters or visual encodings to separate additional context dimensions."
        )

    encoding_fields = [*color_fields, *marker_fields, *shade_fields]

    if group_cols is None:
        group_cols = ()
    else:
        group_cols = tuple(dict.fromkeys(group_cols))

    color_col = _encoding_column_name("color_by", color_fields)
    marker_col = _encoding_column_name("marker_by", marker_fields)
    shade_col = _encoding_column_name("shade_by", shade_fields)

    shade_label = shade_label or _encoding_label(shade_fields)
    color_label = _encoding_label(color_fields)
    marker_label = _encoding_label(marker_fields)

    def lighten(color, amount):
        r, g, b = mcolors.to_rgb(color)
        return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

    def _apply_filters(df):
        if not filters:
            return df
        out = df
        for col, allowed in filters.items():
            if allowed is None or col not in out.columns:
                continue
            values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            out = out[out[col].isin(values)]
        return out

    df_fc = fc_metrics_df[fc_metrics_df["metric"] == forecast_metric].copy()
    df_diff = diff_metrics_df[diff_metrics_df["metric"] == diff_metric].copy()

    df_fc = _apply_filters(df_fc)
    df_diff = _apply_filters(df_diff)

    excluded_merge_cols = {"metric", "value", "model"}
    shared_context_cols = tuple(
        col
        for col in df_fc.columns
        if col in df_diff.columns and col not in excluded_merge_cols
    )
    merge_cols = tuple(dict.fromkeys((*shared_context_cols, *group_cols)))

    missing_merge_fields = [field for field in encoding_fields if field not in merge_cols]
    if missing_merge_fields:
        raise ValueError(
            f"Missing required merge columns for plot encodings: {missing_merge_fields!r}. "
            f"Available merge columns={merge_cols!r}."
        )

    _ensure_unique_keys(df_fc, merge_cols, "Forecast metric dataframe")
    _ensure_unique_keys(df_diff, merge_cols, "Delta metric dataframe")

    df_fc = df_fc.rename(columns={"value": y_metric_name})
    df_diff = df_diff.rename(columns={"value": x_metric_name})
    df = df_fc.merge(
        df_diff[list(merge_cols) + [x_metric_name]],
        on=list(merge_cols),
        how="inner",
    )

    for fields, column_name in (
        (color_fields, color_col),
        (marker_fields, marker_col),
        (shade_fields, shade_col),
    ):
        if len(fields) > 1:
            df[column_name] = list(df.loc[:, list(fields)].itertuples(index=False, name=None))

    # Safety: if nothing to plot
    fig, ax = plt.subplots(figsize=figsize)
    if df.empty:
        ax.set_title(title or "No data after filtering/merging")
        ax.set_xlabel(xlabel or x_metric_name)
        ax.set_ylabel(ylabel or y_metric_name)
        ax.grid(alpha=grid_alpha)
        return fig, ax, df

    color_values = sorted(df[color_col].dropna().unique())
    marker_values = sorted(df[marker_col].dropna().unique())
    shade_values = sorted(df[shade_col].unique())

    cmap = plt.get_cmap(cmap_name)

    def _base_color_for_group(group_value, idx: int):
        if len(color_fields) == 1 and color_fields[0] == "variable" and variable_colors and str(group_value) in variable_colors:
            return variable_colors[str(group_value)]
        return cmap(idx % cmap.N)

    # Plot
    for ci, color_value in enumerate(color_values):
        base = _base_color_for_group(color_value, ci)
        for pi, shade_value in enumerate(shade_values):
            # amount in [0, shade_strength]
            denom = max(1, len(shade_values) - 1)
            amt = shade_strength * (pi / denom)
            shade = lighten(base, amt)

            for mi, marker_value in enumerate(marker_values):
                dsub = df[
                    (df[color_col] == color_value)
                    & (df[shade_col] == shade_value)
                    & (df[marker_col] == marker_value)
                ]
                if dsub.empty:
                    continue

                ax.scatter(
                    dsub[x_metric_name],
                    dsub[y_metric_name],
                    color=shade,
                    marker=markers[mi % len(markers)],
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

            for mi, marker_value in enumerate(marker_values):
                dsub = df[(df[shade_col] == shade_value) & (df[marker_col] == marker_value)]

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
                        _ls_cycle[mi % len(_ls_cycle)]
                        if fit_linestyles is None
                        else fit_linestyles.get(marker_value, "-")
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
    
        # 1) COLOR GROUP
        _section(color_label)
        color_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markersize=7,
                markerfacecolor=_base_color_for_group(group_value, ci),
                markeredgecolor=edgecolor,
                label=legend_labels[ci] if legend_labels else _format_encoding_value(color_fields, group_value),
            )
            for ci, group_value in enumerate(color_values)
        ]
        handles.extend(color_handles)
    
        # 2) SHADED GROUP
        _section(shade_label)
        neutral_base = (0.2, 0.2, 0.2)
        denom = max(1, (len(shade_values) - 1))
        shade_handles = []
        for pi, shade_value in enumerate(shade_values):
            amt = shade_strength * (1.0 - pi / denom)
            shade = lighten(neutral_base, amt)
            if len(shade_fields) == 1 and shade_fields[0] == "total_months":
                label = f"{shade_value}M"
            else:
                label = _format_encoding_value(shade_fields, shade_value)
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
    
        # 3) MARKER GROUP
        _section(marker_label)
        if fit_lines:
            if fit_linestyles is None:
                _ls_cycle = ["-", "--", ":", "-."]
                ls_for_marker = {
                    marker_value: _ls_cycle[i % len(_ls_cycle)]
                    for i, marker_value in enumerate(marker_values)
                }
            else:
                ls_for_marker = {
                    marker_value: fit_linestyles.get(marker_value, "-")
                    for marker_value in marker_values
                }
        else:
            ls_for_marker = {marker_value: "None" for marker_value in marker_values}

        marker_handles = [
            Line2D(
                [0], [0],
                marker=markers[i % len(markers)],
                linestyle=ls_for_marker[marker_value],
                color="black",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="black",
                lw=fit_lw if fit_lines else 0,
                label=_format_encoding_value(marker_fields, marker_value),
            )
            for i, marker_value in enumerate(marker_values)
        ]
        handles.extend(marker_handles)
    
        # Create ONE legend
        leg = ax.legend(
            handles=handles,
            loc=legend_loc,
            fontsize=legend_fontsize,
            frameon=True,
            framealpha=0.0,
            handlelength=2.2,
            handletextpad=0.8,
            borderpad=0.8,
            labelspacing=0.6,
        )
        leg.get_frame().set_facecolor("none")
        leg.get_frame().set_edgecolor("none")
    
        # Make section headers look like headers
        for txt in leg.get_texts():
            label = txt.get_text()
            if label in {color_label, shade_label, marker_label}:
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
