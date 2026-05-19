from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rich.style import Style
from rich.table import Table as RichTable
from rich.text import Text

from ...logging import get_logger, log_renderable
from .. import metrics_to_df_single_region

from .dataclasses import CalculateMetricsConfig
from .utils import (
    _apply_df_filters,
    _format_variable_display_name,
    _context_filename_suffix,
    _format_scalar_value,
    _mix_colors,
    _get_variable_table_color,
)
from .constants import CONTEXT_AXES


logger = get_logger(__name__)


def _build_section_summary_frame(
    *,
    metrics: dict,
    variables: list[str],
    metric_type: str,
    avg_stat: str,
    spread_stat: str,
) -> pd.DataFrame:
    df_section = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type=metric_type,
        diff="no",
    )
    if df_section.empty:
        return pd.DataFrame()

    group_cols = [col for col in df_section.columns if col not in {"value", "realization"}]

    if "realization" in df_section.columns:
        df_mean = df_section.groupby(group_cols, dropna=False, as_index=False)["value"].mean()
        df_mean["stat"] = avg_stat

        df_spread = df_section.groupby(group_cols, dropna=False, as_index=False)["value"].std(ddof=0)
        df_spread["value"] = df_spread["value"].fillna(0.0)
        df_spread["stat"] = spread_stat

        return pd.concat([df_mean, df_spread], ignore_index=True)

    df_section = df_section.copy()
    df_section["stat"] = avg_stat
    df_spread = df_section.copy()
    df_spread["value"] = 0.0
    df_spread["stat"] = spread_stat
    return pd.concat([df_section, df_spread], ignore_index=True)

def _build_metric_stat_frame(
    *,
    metrics: dict,
    variables: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    df_all_dims = _build_section_summary_frame(
        metrics=metrics,
        variables=variables,
        metric_type="all_dims",
        avg_stat="all_dims_avg",
        spread_stat="all_dims_spread",
    )
    if not df_all_dims.empty:
        frames.append(df_all_dims)

    df_spatial = _build_section_summary_frame(
        metrics=metrics,
        variables=variables,
        metric_type="spatial_mean",
        avg_stat="spatial_avg",
        spread_stat="spatial_spread",
    )
    if not df_spatial.empty:
        frames.append(df_spatial)

    df_ens = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="ensemble_mean",
        diff="no",
    )
    if not df_ens.empty:
        df_ens = df_ens.copy()
        df_ens["stat"] = "ens_avg"
        frames.append(df_ens)

    df_prob = metrics_to_df_single_region(
        metrics=metrics,
        variables=variables,
        metric_names=None,
        kind="scalar",
        metric_type="probabilistic",
        diff="no",
    )
    if not df_prob.empty:
        df_prob = df_prob.copy()
        df_prob["stat"] = "prob"
        frames.append(df_prob)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def _format_duplicate_table_key(
    *,
    key_names: tuple[str, ...],
    key_values: tuple[object, ...],
) -> str:
    parts: list[str] = []
    for name, value in zip(key_names, key_values):
        if pd.isna(value):
            value_repr = "NaN"
        else:
            value_repr = str(value)
        parts.append(f"{name}={value_repr}")
    return ", ".join(parts)

def _build_strict_scalar_table_pivot(
    df: pd.DataFrame,
    *,
    row_index: tuple[str, ...],
    column_index: tuple[str, ...],
) -> pd.DataFrame:
    key_cols = list(row_index) + list(column_index)
    duplicate_counts = (
        df.groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="count")
    )
    duplicate_counts = duplicate_counts[duplicate_counts["count"] > 1]
    if not duplicate_counts.empty:
        key_names = tuple(key_cols)
        duplicate_examples = [
            _format_duplicate_table_key(
                key_names=key_names,
                key_values=tuple(record[col] for col in key_cols),
            )
            for record in duplicate_counts.head(5).to_dict("records")
        ]
        raise ValueError(
            "Scalar table generation found duplicate cells after realization aggregation. "
            "This would require forbidden extra aggregation in calculate_metrics.py. "
            f"Duplicate keys (showing up to 5): {duplicate_examples}"
        )

    pivot = pd.pivot(
        df,
        index=list(row_index),
        columns=list(column_index),
        values="value",
    )
    return pivot.sort_index(axis=0).sort_index(axis=1)

def _metric_higher_is_better(metric: str) -> bool | None:
    mapping = {
        "corr": True,
        "clim_acc": True,
        "spatial_acc": True,
        "r2": True,
        "rmse_skill_clim": True,
        "rmse": False,
        "crmse": False,
        "mae": False,
        "bias": False,
        "error_std": False,
        "nrmse": False,
        "ncrmse": False,
        "nmae": False,
        "nbias": False,
        "crps": False,
        "spread": None,
        "variance_ratio": None,
        "spread_error_ratio": None,
    }
    return mapping.get(metric)

def _metric_gain_value(metric_name: str | None, reference_value: float, corrected_value: float) -> float:
    if metric_name is None or pd.isna(reference_value) or pd.isna(corrected_value):
        return np.nan
    if metric_name in {"bias", "nbias"}:
        return abs(reference_value) - abs(corrected_value)
    if metric_name == "variance_ratio":
        return abs(reference_value - 1.0) - abs(corrected_value - 1.0)
    if metric_name == "spread_error_ratio":
        return abs(reference_value - 1.0) - abs(corrected_value - 1.0)

    direction = _metric_higher_is_better(metric_name)
    if direction is True:
        return corrected_value - reference_value
    if direction is False:
        return reference_value - corrected_value
    return np.nan

def _metric_name_from_index_value(idx_value: object, *, index_names: list[str | None]) -> str | None:
    if "metric" in index_names:
        metric_idx = index_names.index("metric")
        if isinstance(idx_value, tuple):
            return str(idx_value[metric_idx])
        return str(idx_value)
    if isinstance(idx_value, tuple):
        return None
    return str(idx_value)

def _add_model_gain_columns(
    df: pd.DataFrame,
    config: CalculateMetricsConfig,
) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex) or "stat" not in df.columns.names or "model" not in df.columns.names:
        return df

    col_names = list(df.columns.names)
    stat_idx = col_names.index("stat")
    model_idx = col_names.index("model")
    prefix_indices = [i for i in range(len(col_names)) if i not in {stat_idx, model_idx}]
    prefix_stat_pairs = sorted({(tuple(col[i] for i in prefix_indices), col[stat_idx]) for col in df.columns})

    out = df.copy()
    gain_stats = {"all_dims_avg", "spatial_avg", "ens_avg", "prob"}
    for prefix, stat in prefix_stat_pairs:
        if stat not in gain_stats:
            continue
        reference_key = list(prefix)
        reference_key.insert(model_idx, config.models.reference_model)
        reference_key.insert(stat_idx, stat)
        corrected_key = list(prefix)
        corrected_key.insert(model_idx, config.models.corrected_model)
        corrected_key.insert(stat_idx, stat)
        gain_key = list(prefix)
        gain_key.insert(model_idx, config.models.gain_label)
        gain_key.insert(stat_idx, stat)

        reference_key = tuple(reference_key)
        corrected_key = tuple(corrected_key)
        gain_key = tuple(gain_key)

        if reference_key in out.columns and corrected_key in out.columns:
            reference_series = out[reference_key]
            corrected_series = out[corrected_key]
            gain_values = [
                _metric_gain_value(
                    _metric_name_from_index_value(out.index[row_pos], index_names=list(out.index.names)),
                    reference_series.iloc[row_pos],
                    corrected_series.iloc[row_pos],
                )
                for row_pos in range(len(out.index))
            ]
            out[gain_key] = pd.Series(gain_values, index=out.index, dtype=float)

    return out

def _order_table_columns(
    df: pd.DataFrame,
    config: CalculateMetricsConfig,
) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex) or "stat" not in df.columns.names:
        return df

    col_names = list(df.columns.names)
    stat_idx = col_names.index("stat")
    model_idx = col_names.index("model") if "model" in col_names else None

    def _sort_key(col: tuple[object, ...]) -> tuple:
        prefix = tuple("" if col[i] is None else str(col[i]) for i in range(len(col)) if i not in {stat_idx, model_idx})
        model = str(col[model_idx]) if model_idx is not None else ""
        model_order = (
            config.models.table_model_order.index(model)
            if model in config.models.table_model_order
            else len(config.models.table_model_order)
        )
        stat = str(col[stat_idx])
        stat_order = (
            config.scalar_tables.stat_order.index(stat)
            if stat in config.scalar_tables.stat_order
            else len(config.scalar_tables.stat_order)
        )
        return (*prefix, model_order, stat_order, model, stat)

    return df.loc[:, sorted(df.columns.tolist(), key=_sort_key)]

def _format_table_context_value(
    *,
    field: str,
    value: object,
    leadtime_unit: str | None = None
) -> str:
    if field == "leadtime" and leadtime_unit:
        return f"{value} {leadtime_unit}"
    return str(value)

def _table_context_subtitle(
    df: pd.DataFrame,
    *,
    fields: tuple[str, ...] = CONTEXT_AXES,
    leadtime_unit: str | None = None,
) -> str:
    parts: list[str] = []
    for field in fields:
        if field not in df.columns:
            continue
        values = [value for value in df[field].dropna().unique().tolist() if str(value) != ""]
        if not values:
            continue
        if len(values) == 1:
            parts.append(
                f"{field}: {_format_table_context_value(field=field, value=values[0], leadtime_unit=leadtime_unit)}"
            )
        else:
            joined = ", ".join(
                _format_table_context_value(field=field, value=value, leadtime_unit=leadtime_unit)
                for value in values
            )
            parts.append(f"{field}: {joined}")
    return " | ".join(parts)

def _ordered_context_filename_suffix(
    context_values: dict[str, object],
    *,
    priority_fields: tuple[str, ...] = ("train_period", "loss", "variant", "leadtime"),
) -> str:
    if not context_values:
        return "all"

    ordered_items: list[tuple[str, object]] = [
        (field, context_values[field]) for field in priority_fields if field in context_values
    ]
    ordered_items.extend(
        (field, value)
        for field, value in context_values.items()
        if field not in priority_fields
    )
    return _context_filename_suffix(dict(ordered_items))


def build_scalar_metric_tables(
    *,
    metrics: dict,
    variables: list[str],
    filters: dict | None = None,
    config: CalculateMetricsConfig,
    row_index: tuple[str, ...] | None = None,
    column_index: tuple[str, ...] | None = None,
    leadtime_unit: str | None = None,
) -> list[tuple[str, str | None, pd.DataFrame, str]]:
    df_all = _build_metric_stat_frame(metrics=metrics, variables=variables)
    df_all = _apply_df_filters(df_all, filters)
    if df_all.empty:
        return []

    effective_row_index = row_index or config.scalar_tables.row_index
    single_variable = len(variables) == 1
    effective_column_index = tuple(
        col for col in (column_index or config.scalar_tables.column_index)
        if not (single_variable and col == "variable")
    )

    context_cols = [
        col for col in CONTEXT_AXES
        if (
            col in df_all.columns
            and col not in effective_row_index
            and col not in effective_column_index
            and df_all[col].nunique(dropna=False) > 1
        )
    ]

    if context_cols:
        grouped_items = list(df_all.groupby(context_cols, dropna=False, sort=True))
    else:
        grouped_items = [((), df_all)]

    tables: list[tuple[str, str | None, pd.DataFrame, str]] = []
    for group_key, df_group in grouped_items:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        pivot = _build_strict_scalar_table_pivot(
            df_group,
            row_index=effective_row_index,
            column_index=effective_column_index,
        )
        pivot = _add_model_gain_columns(pivot, config)
        pivot = _order_table_columns(pivot, config)

        context_values: dict[str, object] = {}
        for col, value in zip(context_cols, group_key):
            if pd.isna(value):
                continue
            context_values[col] = value

        for col in CONTEXT_AXES:
            if col in context_values or col not in df_group.columns:
                continue
            values = [value for value in df_group[col].dropna().unique().tolist() if str(value) != ""]
            if len(values) == 1:
                context_values[col] = values[0]

        title = "Scalar metrics"
        if single_variable:
            title = f"{title} ({_format_variable_display_name(str(variables[0]))})"

        tables.append(
            (
                title,
                _table_context_subtitle(df_group, leadtime_unit=leadtime_unit),
                pivot,
                _ordered_context_filename_suffix(context_values),
            )
        )

    return tables

def _prettify_stat_label(stat: str) -> str:
    pretty = {
        "all_dims_avg": "all_dims_avg",
        "all_dims_spread": "all_dims_spread",
        "spatial_avg": "spatial_avg",
        "spatial_spread": "spatial_spread",
        "ens_avg": "ens_avg",
        "gain": "gain",
        "prob": "prob",
    }
    return pretty.get(stat, stat.replace("_", " "))

def _format_table_for_display(
    df: pd.DataFrame,
    *,
    stat_label_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    df_display = df.copy()
    if isinstance(df_display.columns, pd.MultiIndex):
        df_display.columns = pd.MultiIndex.from_tuples(
            [
                tuple(
                    (
                        stat_label_overrides.get(str(part), _prettify_stat_label(str(part)))
                        if stat_label_overrides is not None
                        else _prettify_stat_label(str(part))
                    ) if i == len(col) - 1 else part
                    for i, part in enumerate(col)
                )
                for col in df_display.columns
            ],
            names=df_display.columns.names,
        )
    return df_display

def _table_variables(df: pd.DataFrame) -> list[str]:
    variables: list[str] = []

    if isinstance(df.index, pd.MultiIndex) and "variable" in df.index.names:
        for value in df.index.get_level_values("variable"):
            variable = str(value)
            if variable not in variables:
                variables.append(variable)

    if isinstance(df.columns, pd.MultiIndex) and "variable" in df.columns.names:
        variable_idx = list(df.columns.names).index("variable")
        for col in df.columns:
            variable = str(col[variable_idx])
            if variable not in variables:
                variables.append(variable)

    return variables

def _variable_from_row(df: pd.DataFrame, row_pos: int) -> str | None:
    if not isinstance(df.index, pd.MultiIndex) or "variable" not in df.index.names:
        return None
    variable_idx = list(df.index.names).index("variable")
    idx_value = df.index[row_pos]
    return str(idx_value[variable_idx])


def _variable_from_column(df: pd.DataFrame, col_pos: int) -> str | None:
    if not isinstance(df.columns, pd.MultiIndex) or "variable" not in df.columns.names:
        return None
    variable_idx = list(df.columns.names).index("variable")
    col_value = df.columns[col_pos]
    return str(col_value[variable_idx])


def save_scalar_metric_table_image(
    *,
    df: pd.DataFrame,
    title: str,
    subtitle: str | None,
    image_path: Path,
    config: CalculateMetricsConfig,
    stat_label_overrides: dict[str, str] | None = None,
    base_color: str,
) -> Path:
    df_display = _format_table_for_display(df, stat_label_overrides=stat_label_overrides)
    n_rows, n_cols = df_display.shape
    table_variables = _table_variables(df)

    fig_width = max(14.0, 2.2 + 1.75 * (n_cols + df_display.index.nlevels))
    fig_height = max(2.8, 1.4 + 0.42 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(config.scalar_tables.background_color)
    ax.axis("off")

    col_labels = []
    for col in df_display.columns:
        parts = col if isinstance(col, tuple) else (col,)
        col_labels.append("\n".join(map(str, parts)))

    row_labels = [
        "\n".join(map(str, idx if isinstance(idx, tuple) else (idx,)))
        for idx in df_display.index
    ]
    cell_text = [
        [_format_scalar_value(value, config.scalar_tables.significant_digits) for value in row]
        for row in df_display.to_numpy()
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.01, 0.10, 0.98, 0.81],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    header_color = base_color
    header_text = "#fdfaf3"
    edge_color = _mix_colors(base_color, config.scalar_tables.background_color, 0.58)
    text_color = "#1b1b1b"
    gain_good = "#d7efdc"
    gain_bad = "#f6d7d2"
    gain_neutral = _mix_colors(base_color, config.scalar_tables.background_color, 0.76)

    def _resolve_variable_color(variable: str | None) -> str:
        if variable is None or not table_variables:
            return base_color
        return _get_variable_table_color(variable_name=variable, variables=table_variables)

    for (row, col), cell in table.get_celld().items():
        cell_edge_color = edge_color
        cell.set_linewidth(0.8)
        if row == 0:
            column_color = _resolve_variable_color(_variable_from_column(df, col)) if col >= 0 else header_color
            cell.set_height(cell.get_height() * 1.45)
            cell.PAD = 0.08
            cell.set_facecolor(column_color)
            cell.set_text_props(color=header_text, weight="bold")
        elif col == -1:
            row_color = _resolve_variable_color(_variable_from_row(df, row - 1))
            row_label_color = _mix_colors(row_color, config.scalar_tables.background_color, 0.50)
            cell_edge_color = _mix_colors(row_color, config.scalar_tables.background_color, 0.32)
            cell.set_facecolor(row_label_color)
            cell.set_text_props(color=text_color, weight="bold")
        else:
            row_color = _resolve_variable_color(_variable_from_row(df, row - 1))
            stripe_a = _mix_colors(row_color, config.scalar_tables.background_color, 0.78)
            stripe_b = _mix_colors(row_color, config.scalar_tables.background_color, 0.68)
            cell_edge_color = _mix_colors(row_color, config.scalar_tables.background_color, 0.42)
            base_face = stripe_a if row % 2 else stripe_b
            cell.set_facecolor(base_face)
            cell.set_text_props(color=text_color)

            if isinstance(df.columns, pd.MultiIndex):
                raw_col_key = df.columns[col]
                col_name_to_value = {
                    str(name): raw_col_key[i]
                    for i, name in enumerate(df.columns.names)
                }
                stat_name = str(col_name_to_value.get("stat", ""))
                model_name = str(col_name_to_value.get("model", ""))
                if model_name == config.models.gain_label and stat_name in {"all_dims_avg", "spatial_avg", "ens_avg", "prob"}:
                    raw_value = df.iloc[row - 1, col]
                    if pd.isna(raw_value):
                        cell.set_facecolor(gain_neutral)
                    else:
                        if raw_value > 0:
                            cell.set_facecolor(gain_good)
                        elif raw_value < 0:
                            cell.set_facecolor(gain_bad)
                        else:
                            cell.set_facecolor(gain_neutral)
                    cell.set_text_props(color=text_color, weight="bold")
        cell.set_edgecolor(cell_edge_color)

    ax.text(
        0.01,
        0.97,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=base_color,
    )
    if subtitle:
        ax.text(
            0.01,
            0.035,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            color="#5b5b5b",
        )

    fig.savefig(image_path, dpi=config.scalar_tables.image_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return image_path

def _filter_metric_rows(df: pd.DataFrame, *, metrics_keep: set[str] | None = None) -> pd.DataFrame:
    if df.empty or metrics_keep is None:
        return df
    if "metric" not in df.index.names:
        return df
    metric_values = df.index.get_level_values("metric").astype(str)
    return df.loc[metric_values.isin(metrics_keep)]

def _drop_empty_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep_mask = ~df.isna().all(axis=0)
    return df.loc[:, keep_mask]


def _flatten_label(parts: tuple[object, ...] | object) -> str:
    if not isinstance(parts, tuple):
        parts = (parts,)
    cleaned = [str(part) for part in parts if pd.notna(part) and str(part) != ""]
    return " | ".join(cleaned) if cleaned else "-"

def _rich_gain_cell(
    *,
    raw_df: pd.DataFrame,
    display_value: object,
    row_pos: int,
    col_pos: int,
    config: CalculateMetricsConfig,
) -> str | Text:
    text = _format_scalar_value(display_value, config.scalar_tables.significant_digits)
    if not isinstance(raw_df.columns, pd.MultiIndex):
        return text

    raw_col_key = raw_df.columns[col_pos]
    col_name_to_value = {
        str(name): raw_col_key[i]
        for i, name in enumerate(raw_df.columns.names)
    }
    stat_name = str(col_name_to_value.get("stat", ""))
    model_name = str(col_name_to_value.get("model", ""))
    if model_name != config.models.gain_label or stat_name not in {"all_dims_avg", "spatial_avg", "ens_avg", "prob"}:
        return text

    raw_value = raw_df.iloc[row_pos, col_pos]
    if pd.isna(raw_value):
        style = Style(color="bright_black", bold=True)
    else:
        if raw_value > 0:
            style = Style(color="green", bold=True)
        elif raw_value < 0:
            style = Style(color="red", bold=True)
        else:
            style = Style(color="bright_black", bold=True)

    return Text(text, style=style)

def _print_rich_dataframe_table(
    df: pd.DataFrame,
    *,
    title: str,
    subtitle: str | None = None,
    config: CalculateMetricsConfig,
    raw_df: pd.DataFrame | None = None,
    base_color: str | None = None,
) -> None:
    if not config.global_config.print_console_tables:
        return

    rich_table = RichTable(
        title=title,
        show_lines=False,
        footer_style="bold white",
    )
    index_labels = list(df.index.names) if isinstance(df.index, pd.MultiIndex) else [df.index.name or "row"]
    index_labels = [label if label is not None else "row" for label in index_labels]

    for label in index_labels:
        rich_table.add_column(str(label), style="magenta")
    for col in df.columns:
        rich_table.add_column(_flatten_label(col), justify="right", style="cyan")

    for row_pos, (idx, row) in enumerate(df.iterrows()):
        idx_parts = idx if isinstance(idx, tuple) else (idx,)
        row_cells = [str(part) for part in idx_parts]
        for col_pos, val in enumerate(row.tolist()):
            if raw_df is not None and base_color is not None:
                row_cells.append(
                    _rich_gain_cell(
                        raw_df=raw_df,
                        display_value=val,
                        row_pos=row_pos,
                        col_pos=col_pos,
                        config=config,
                    )
                )
            else:
                row_cells.append(_format_scalar_value(val, config.scalar_tables.significant_digits))
        rich_table.add_row(*row_cells)

    log_renderable(rich_table, logger=logger)
    if subtitle:
        logger.info(Text(subtitle, style="italic grey70"))


def save_scalar_metric_tables(
    *,
    metrics: dict,
    variables: list[str],
    output_folder: Path,
    filters: dict | None = None,
    config: CalculateMetricsConfig,
    metrics_keep: set[str] | None = None,
    filename_prefix: str = "scalar_metrics",
    title_prefix: str | None = None,
    row_index: tuple[str, ...] | None = None,
    column_index: tuple[str, ...] | None = None,
    stat_label_overrides: dict[str, str] | None = None,
    leadtime_unit: str | None = None,
    base_color: str,
) -> list[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for title, subtitle, table_df, filename_stem in build_scalar_metric_tables(
        metrics=metrics,
        variables=variables,
        filters=filters,
        config=config,
        row_index=row_index,
        column_index=column_index,
        leadtime_unit=leadtime_unit,
    ):
        if table_df.empty:
            continue

        table_df = _filter_metric_rows(table_df, metrics_keep=metrics_keep)
        if table_df.empty:
            continue
        table_df = _drop_empty_table_columns(table_df)
        if table_df.empty:
            continue

        if title_prefix:
            title = f"{title_prefix} | {title}"

        table_df_display = _format_table_for_display(
            table_df,
            stat_label_overrides=stat_label_overrides,
        )
        _print_rich_dataframe_table(
            table_df_display,
            title=title,
            subtitle=subtitle,
            config=config,
            raw_df=table_df,
            base_color=base_color,
        )

        filename_parts = [filename_prefix, filename_stem]
        base_filename = "_".join(filename_parts)
        image_path = output_folder / f"{base_filename}.png"
        save_scalar_metric_table_image(
            df=table_df,
            title=title,
            subtitle=subtitle,
            image_path=image_path,
            config=config,
            stat_label_overrides=stat_label_overrides,
            base_color=base_color,
        )
        saved_paths.append(image_path)

    return saved_paths
