from pathlib import Path
import logging
import warnings
import joblib

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from rich import print
from rich.style import Style
from rich.table import Table as RichTable
from rich.text import Text

from earthml import Dask, get_runs_and_metrics
from earthml.metrics.utils import metrics_to_df
from earthml.plots import plot_metric_vs_diff, plot_scoreboard


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Dask.*")
warnings.filterwarnings("ignore", message=".*distributed.scheduler.*")
logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
logging.getLogger("distributed.worker").setLevel(logging.ERROR)
logging.getLogger("distributed.nanny").setLevel(logging.ERROR)


#
# User knobs
# Edit these blocks first. They are grouped by output so users can quickly
# switch products on/off or retarget a plot without digging through the code.
#

GLOBAL_KNOBS = {
    # Main experiment folder to read from.
    "exp_root_folder": Path("/Users/jacopodallaglio/ML/experiments_earthML_seasonal_atmo/"),
    # All artifacts land in exp_root_folder / plot_folder_name.
    "plot_folder_name": "plots",
    # Dataset split and model loading.
    "type_data": "test",
    "load_models": ("an", "fc", "pr"),
    # Keep None to load every scalar metric available.
    "metric_names": None,
    # Model pair used for delta comparisons like pr - fc.
    "models_diff": ("fc", "pr"),
    # Global output switches.
    "enable_scalar_tables": True,
    "enable_diff_plot": True,
    "enable_leadtime_plots": True,
    "enable_scoreboard": True,
}

COMMON_PLOT_KNOBS = {
    # Applied to both the diff plot and the leadtime plots.
    "filters": None,
    # Used to color/shade runs consistently across plots.
    "shade_by": "loss",  # e.g. "loss", "total_months"
    "shade_label": "Loss",
}

DIFF_PLOT_KNOBS = {
    # x-axis: how much the ML correction changes the chosen metric.
    "diff_metric": "nrmse",
    "diff_metric_type": "ensemble",
    # y-axis: the forecast quality you want to compare against.
    "forecast_metric": "corr",
    "forecast_metric_type": "ensemble",
    "cmap": "magma",
}

LEADTIME_PLOT_KNOBS = {
    # Each metric maps to the plotting mode expected by save_metrics_vs_leadtime_plots.
    "metrics": {
        "r2": "deterministic_with_ensemble_overlay",
        "rmse": "deterministic_with_ensemble_overlay",
        "rmse_skill_clim": "deterministic_with_ensemble_overlay",
        "mae": "deterministic_with_ensemble_overlay",
        "bias": "deterministic_with_ensemble_overlay",
        "crps": "probabilistic",
        "spread": "probabilistic",
        "spread_error_ratio": "probabilistic",
    },
    "model_colors": {
        "fc": "tab:green",
        "pr": "tab:blue",
        "an": "tab:orange",
    },
}

TABLE_KNOBS = {
    # Filter rows before building summary tables.
    "filters": {
        "train_period": None,
        "loss": "mseloss",
        "leadtime": 1,
    },
    "row_index": ("metric",),
    "column_index": ("model", "variable", "stat"),
    # Tables split when these dimensions have more than one surviving value.
    "group_by": ("train_period", "loss", "leadtime"),
    "model_order": ("fc", "pr", "pr-fc"),
    "stat_order": ("avg", "spread", "ens", "prob"),
    "significant_digits": 3,
    "image_dpi": 300,
    "base_colors": ("#3c8be4", "#c9be2c", "#bb41c6"),
}

SCOREBOARD_KNOBS = {
    "mode": "relative",  # "absolute" or "relative"
    "metrics": {
        "r2": "ensemble",
        "nrmse": "ensemble",
        "crps": "probabilistic",
        "rmse_skill_clim": "ensemble",
        "spread_error_ratio": "probabilistic",
    },
    "metric_cmaps": {
        "r2": "RdBu",
        "nrmse": "RdBu_r",
        "nmae": "RdBu_r",
        "nbias": "RdBu_r",
        "rmse_skill_clim": "RdBu",
        "spread_error_ratio": "RdBu",
    },
    "metric_vlims": {
        # "r2": (-0.4, 0.4),
        # "nrmse": (-0.2, 0.2),
        # "crps": (-0.2, 0.2),
    },
    # Pick which metadata dimensions span the scoreboard grid.
    "row_axis": "variable",  # e.g. "train_period", "loss", "leadtime", "variable"
    "col_axis": "leadtime",
    # These act only when the dimension is not already on an axis.
    "filters": {
        "variable": None,  # e.g. ["t2m"]
        "loss": "mseloss",
        "train_period": None,
        "leadtime": None,
    },
    "annotate": True,
}


# Derived internal aliases. The grouped knobs above are the intended edit points.
EXP_ROOT_FOLDER = GLOBAL_KNOBS["exp_root_folder"]
PLOT_FOLDER = EXP_ROOT_FOLDER / GLOBAL_KNOBS["plot_folder_name"]

TYPE_DATA = GLOBAL_KNOBS["type_data"]
LOAD_MODELS = GLOBAL_KNOBS["load_models"]
METRIC_NAMES = GLOBAL_KNOBS["metric_names"]
MODELS_DIFF = GLOBAL_KNOBS["models_diff"]

FORECAST_METRIC = DIFF_PLOT_KNOBS["forecast_metric"]
FORECAST_METRIC_TYPE = DIFF_PLOT_KNOBS["forecast_metric_type"]
DIFF_METRIC = DIFF_PLOT_KNOBS["diff_metric"]
DIFF_METRIC_TYPE = DIFF_PLOT_KNOBS["diff_metric_type"]
DIFF_PLOT_CMAP = DIFF_PLOT_KNOBS["cmap"]

LEADTIME_METRICS = LEADTIME_PLOT_KNOBS["metrics"]
MODEL_COLORS = LEADTIME_PLOT_KNOBS["model_colors"]
PLOT_FILTERS = COMMON_PLOT_KNOBS["filters"]
SHADE_BY = COMMON_PLOT_KNOBS["shade_by"]
SHADE_LABEL = COMMON_PLOT_KNOBS["shade_label"]

SCOREBOARD_MODE = SCOREBOARD_KNOBS["mode"]
SCOREBOARD_METRICS = SCOREBOARD_KNOBS["metrics"]
SCOREBOARD_METRIC_CMAPS = SCOREBOARD_KNOBS["metric_cmaps"]
SCOREBOARD_METRIC_VLIMS = SCOREBOARD_KNOBS["metric_vlims"]
SCOREBOARD_ROW_AXIS = SCOREBOARD_KNOBS["row_axis"]
SCOREBOARD_COL_AXIS = SCOREBOARD_KNOBS["col_axis"]
SCOREBOARD_FILTERS = SCOREBOARD_KNOBS["filters"]
SCOREBOARD_ANNOTATE = SCOREBOARD_KNOBS["annotate"]

SCALAR_TABLE_IMAGE_DPI = TABLE_KNOBS["image_dpi"]
SCALAR_TABLE_BASE_COLORS = TABLE_KNOBS["base_colors"]
SCALAR_TABLE_MODEL_ORDER = TABLE_KNOBS["model_order"]
SCALAR_TABLE_STAT_ORDER = TABLE_KNOBS["stat_order"]
SCALAR_TABLE_FILTERS = TABLE_KNOBS["filters"]
SCALAR_TABLE_ROW_INDEX = TABLE_KNOBS["row_index"]
SCALAR_TABLE_COLUMN_INDEX = TABLE_KNOBS["column_index"]
SCALAR_TABLE_GROUP_BY = TABLE_KNOBS["group_by"]
SCALAR_TABLE_SIGNIFICANT_DIGITS = TABLE_KNOBS["significant_digits"]
NORMALIZED_METRICS = {"nrmse", "ncrmse", "nmae", "nbias", "r2"}


METRIC_DISPLAY_NAMES = {
    "r2": "R2",
    "rmse": "RMSE",
    "rmse_skill_clim": "RMSE Skill",
    "crmse": "CRMSE",
    "mae": "MAE",
    "bias": "Bias",
    "nrmse": "nRMSE",
    "ncrmse": "nCRMSE",
    "nmae": "nMAE",
    "nbias": "nBias",
    "crps": "CRPS",
    "spread": "Spread",
    "spread_error_ratio": "Spread/Error",
}


def build_scalar_metric_df(
    *,
    metrics,
    variables: list[str],
    metric_name: str | None,
    metric_type: str,
    diff: str,
    models: tuple[str, str] | None = None,
) -> pd.DataFrame:
    metric_names = (metric_name,) if metric_name is not None else METRIC_NAMES
    return metrics_to_df(
        metrics=metrics,
        variables=variables,
        metric_names=metric_names,
        kind="scalar",
        metric_type=metric_type,
        diff=diff,
        models=models,
    )


def get_variable_plot_folder(*, plot_root: Path, variable: str) -> Path:
    variable_folder = plot_root / variable
    variable_folder.mkdir(parents=True, exist_ok=True)
    return variable_folder


def _ordered_unique_variables(variables: list[str]) -> list[str]:
    unique_variables: list[str] = []
    for variable in variables:
        variable_str = str(variable)
        if variable_str not in unique_variables:
            unique_variables.append(variable_str)
    return unique_variables


def get_variable_colors(*, variables: list[str]) -> dict[str, str]:
    ordered_variables = _ordered_unique_variables(variables)
    return {
        variable: SCALAR_TABLE_BASE_COLORS[idx % len(SCALAR_TABLE_BASE_COLORS)]
        for idx, variable in enumerate(ordered_variables)
    }


def get_variable_table_color(*, variable: str, variables: list[str]) -> str:
    variable_colors = get_variable_colors(variables=variables)
    if not variable_colors:
        return SCALAR_TABLE_BASE_COLORS[0]
    return variable_colors.get(str(variable), SCALAR_TABLE_BASE_COLORS[0])


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


def _apply_df_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    if df.empty or not filters:
        return df

    out = df.copy()
    for col, allowed in filters.items():
        if allowed is None or col not in out.columns:
            continue
        values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        out = out[out[col].isin(values)]
    return out


def _format_scalar_value(value: object, significant_digits: int) -> str:
    if value is None or pd.isna(value):
        return "-"
    value_float = float(value)
    if np.isinf(value_float):
        return "inf" if value_float > 0 else "-inf"
    if significant_digits < 1:
        raise ValueError("significant_digits must be at least 1")
    if value_float == 0:
        return "0" if significant_digits == 1 else f"0.{('0' * (significant_digits - 1))}"

    abs_value = abs(value_float)
    exponent = int(np.floor(np.log10(abs_value)))
    if exponent >= significant_digits or exponent < -3:
        return f"{value_float:.{significant_digits - 1}e}"

    decimals = significant_digits - exponent - 1
    return f"{value_float:.{max(decimals, 0)}f}"


def _flatten_label(parts: tuple[object, ...] | object) -> str:
    if not isinstance(parts, tuple):
        parts = (parts,)
    cleaned = [str(part) for part in parts if pd.notna(part) and str(part) != ""]
    return " | ".join(cleaned) if cleaned else "-"


def _sanitize_filename_fragment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "all"


def _prettify_stat_label(stat: str) -> str:
    pretty = {
        "avg": "avg",
        "spread": "spread",
        "ens": "ens",
        "gain": "gain",
        "prob": "prob",
    }
    return pretty.get(stat, stat.replace("_", " "))


def _mix_colors(color_a: str, color_b: str, weight: float) -> str:
    rgb_a = np.array(mcolors.to_rgb(color_a))
    rgb_b = np.array(mcolors.to_rgb(color_b))
    mixed = (1 - weight) * rgb_a + weight * rgb_b
    return mcolors.to_hex(mixed)


def format_metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " "))


def _metric_name_from_index(df: pd.DataFrame, row_pos: int) -> str | None:
    idx_value = df.index[row_pos]
    if isinstance(df.index, pd.MultiIndex):
        index_names = list(df.index.names)
        if "metric" in index_names:
            metric_idx = index_names.index("metric")
            return str(idx_value[metric_idx])
    return str(idx_value[0] if isinstance(idx_value, tuple) else idx_value)


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
        "nrmse": False,
        "ncrmse": False,
        "nmae": False,
        "nbias": False,
        "crps": False,
        "spread": None,
        "spread_error_ratio": None,
    }
    return mapping.get(metric)


def _rich_gain_cell(
    *,
    raw_df: pd.DataFrame,
    display_value: object,
    row_pos: int,
    col_pos: int,
    significant_digits: int,
    base_color: str,
) -> str | Text:
    text = _format_scalar_value(display_value, significant_digits)
    if not isinstance(raw_df.columns, pd.MultiIndex):
        return text

    raw_col_key = raw_df.columns[col_pos]
    col_name_to_value = {
        str(name): raw_col_key[i]
        for i, name in enumerate(raw_df.columns.names)
    }
    stat_name = str(col_name_to_value.get("stat", ""))
    model_name = str(col_name_to_value.get("model", ""))
    if model_name != "pr-fc" or stat_name not in {"avg", "ens", "prob"}:
        return text

    metric_name = _metric_name_from_index(raw_df, row_pos)
    raw_value = raw_df.iloc[row_pos, col_pos]
    direction = _metric_higher_is_better(metric_name)
    if pd.isna(raw_value) or direction is None:
        style = Style(color="bright_black", bold=True)
    else:
        improved = raw_value > 0 if direction else raw_value < 0
        degraded = raw_value < 0 if direction else raw_value > 0
        if improved:
            style = Style(color="green", bold=True)
        elif degraded:
            style = Style(color="red", bold=True)
        else:
            style = Style(color="bright_black", bold=True)

    return Text(text, style=style)


def _print_rich_dataframe_table(
    df: pd.DataFrame,
    *,
    title: str,
    significant_digits: int,
    raw_df: pd.DataFrame | None = None,
    base_color: str | None = None,
) -> None:
    rich_table = RichTable(title=title, show_lines=False)
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
                        significant_digits=significant_digits,
                        base_color=base_color,
                    )
                )
            else:
                row_cells.append(_format_scalar_value(val, significant_digits))
        rich_table.add_row(*row_cells)

    print(rich_table)


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


def _add_model_gain_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex) or "stat" not in df.columns.names or "model" not in df.columns.names:
        return df

    col_names = list(df.columns.names)
    stat_idx = col_names.index("stat")
    model_idx = col_names.index("model")
    prefix_indices = [i for i in range(len(col_names)) if i not in {stat_idx, model_idx}]
    prefix_stat_pairs = sorted({(tuple(col[i] for i in prefix_indices), col[stat_idx]) for col in df.columns})

    out = df.copy()
    gain_stats = {"avg", "ens", "prob"}
    for prefix, stat in prefix_stat_pairs:
        if stat not in gain_stats:
            continue
        fc_key = list(prefix)
        fc_key.insert(model_idx, "fc")
        fc_key.insert(stat_idx, stat)
        pr_key = list(prefix)
        pr_key.insert(model_idx, "pr")
        pr_key.insert(stat_idx, stat)
        gain_key = list(prefix)
        gain_key.insert(model_idx, "pr-fc")
        gain_key.insert(stat_idx, stat)

        fc_key = tuple(fc_key)
        pr_key = tuple(pr_key)
        gain_key = tuple(gain_key)

        if fc_key in out.columns and pr_key in out.columns:
            out[gain_key] = out[pr_key] - out[fc_key]

    return out


def _order_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex) or "stat" not in df.columns.names:
        return df

    col_names = list(df.columns.names)
    stat_idx = col_names.index("stat")
    model_idx = col_names.index("model") if "model" in col_names else None

    def _sort_key(col: tuple[object, ...]) -> tuple:
        prefix = tuple("" if col[i] is None else str(col[i]) for i in range(len(col)) if i not in {stat_idx, model_idx})
        model = str(col[model_idx]) if model_idx is not None else ""
        model_order = (
            SCALAR_TABLE_MODEL_ORDER.index(model)
            if model in SCALAR_TABLE_MODEL_ORDER
            else len(SCALAR_TABLE_MODEL_ORDER)
        )
        stat = str(col[stat_idx])
        stat_order = (
            SCALAR_TABLE_STAT_ORDER.index(stat)
            if stat in SCALAR_TABLE_STAT_ORDER
            else len(SCALAR_TABLE_STAT_ORDER)
        )
        return (*prefix, model_order, stat_order, model, stat)

    return df.loc[:, sorted(df.columns.tolist(), key=_sort_key)]


def save_scalar_metric_table_image(
    *,
    df: pd.DataFrame,
    title: str,
    image_path: Path,
    significant_digits: int,
    dpi: int = SCALAR_TABLE_IMAGE_DPI,
    stat_label_overrides: dict[str, str] | None = None,
    base_color: str,
) -> Path:
    df_display = _format_table_for_display(df, stat_label_overrides=stat_label_overrides)
    n_rows, n_cols = df_display.shape
    table_variables = _table_variables(df)

    fig_width = max(14.0, 2.2 + 1.75 * (n_cols + df_display.index.nlevels))
    fig_height = max(2.8, 1.4 + 0.42 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#f6f1e8")
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
        [_format_scalar_value(value, significant_digits) for value in row]
        for row in df_display.to_numpy()
    ]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.01, 0.03, 0.98, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    header_color = base_color
    header_text = "#fdfaf3"
    edge_color = _mix_colors(base_color, "#ffffff", 0.58)
    text_color = "#1b1b1b"
    gain_good = "#d7efdc"
    gain_bad = "#f6d7d2"
    gain_neutral = _mix_colors(base_color, "#ffffff", 0.76)

    def _resolve_variable_color(variable: str | None) -> str:
        if variable is None or not table_variables:
            return base_color
        return get_variable_table_color(variable=variable, variables=table_variables)

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
            row_label_color = _mix_colors(row_color, "#ffffff", 0.50)
            cell_edge_color = _mix_colors(row_color, "#ffffff", 0.32)
            cell.set_facecolor(row_label_color)
            cell.set_text_props(color=text_color, weight="bold")
        else:
            row_color = _resolve_variable_color(_variable_from_row(df, row - 1))
            stripe_a = _mix_colors(row_color, "#ffffff", 0.78)
            stripe_b = _mix_colors(row_color, "#ffffff", 0.68)
            cell_edge_color = _mix_colors(row_color, "#ffffff", 0.42)
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
                if model_name == "pr-fc" and stat_name in {"avg", "ens", "prob"}:
                    metric_name = _metric_name_from_index(df, row - 1)
                    raw_value = df.iloc[row - 1, col]
                    direction = _metric_higher_is_better(metric_name)
                    if pd.isna(raw_value) or direction is None:
                        cell.set_facecolor(gain_neutral)
                    else:
                        improved = raw_value > 0 if direction else raw_value < 0
                        degraded = raw_value < 0 if direction else raw_value > 0
                        if improved:
                            cell.set_facecolor(gain_good)
                        elif degraded:
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

    fig.savefig(image_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return image_path


def print_available_scalar_metrics(*, metrics: dict) -> None:
    inventory = {}
    for metric_type, section_dict in metrics.items():
        ds_scalar = section_dict.get("scalar")
        if ds_scalar is None or "metric" not in ds_scalar.coords:
            inventory[metric_type] = []
            continue
        inventory[metric_type] = [str(metric) for metric in ds_scalar.coords["metric"].values.tolist()]

    print("Available scalar metrics:")
    for metric_type, metric_names in inventory.items():
        print(f"  {metric_type}: {metric_names or 'none'}")


def _build_deterministic_summary_frame(
    *,
    metrics: dict,
    variables: list[str],
) -> pd.DataFrame:
    df_det = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="deterministic",
        diff="no",
    )
    if df_det.empty:
        return pd.DataFrame()

    group_cols = [col for col in df_det.columns if col not in {"value", "realization"}]

    if "realization" in df_det.columns:
        df_mean = df_det.groupby(group_cols, dropna=False, as_index=False)["value"].mean()
        df_mean["stat"] = "avg"

        df_spread = df_det.groupby(group_cols, dropna=False, as_index=False)["value"].std(ddof=0)
        df_spread["value"] = df_spread["value"].fillna(0.0)
        df_spread["stat"] = "spread"

        return pd.concat([df_mean, df_spread], ignore_index=True)

    df_det = df_det.copy()
    df_det["stat"] = "avg"
    df_spread = df_det.copy()
    df_spread["value"] = 0.0
    df_spread["stat"] = "spread"
    return pd.concat([df_det, df_spread], ignore_index=True)


def _build_metric_stat_frame(
    *,
    metrics: dict,
    variables: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    df_det = _build_deterministic_summary_frame(metrics=metrics, variables=variables)
    if not df_det.empty:
        frames.append(df_det)

    df_ens = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
        metric_type="ensemble",
        diff="no",
    )
    if not df_ens.empty:
        df_ens = df_ens.copy()
        df_ens["stat"] = "ens"
        frames.append(df_ens)

    df_prob = build_scalar_metric_df(
        metrics=metrics,
        variables=variables,
        metric_name=None,
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


def build_scalar_metric_tables(
    *,
    metrics: dict,
    variables: list[str],
    filters: dict | None = None,
    row_index: tuple[str, ...] = SCALAR_TABLE_ROW_INDEX,
    column_index: tuple[str, ...] = SCALAR_TABLE_COLUMN_INDEX,
    group_by: tuple[str, ...] = SCALAR_TABLE_GROUP_BY,
) -> list[tuple[str, pd.DataFrame, str]]:
    df_all = _build_metric_stat_frame(metrics=metrics, variables=variables)
    df_all = _apply_df_filters(df_all, filters)
    if df_all.empty:
        return []

    group_cols = [
        col for col in group_by
        if col in df_all.columns and col not in row_index and col not in column_index and df_all[col].nunique(dropna=False) > 1
    ]

    if group_cols:
        grouped_items = list(df_all.groupby(group_cols, dropna=False, sort=True))
    else:
        grouped_items = [((), df_all)]

    tables: list[tuple[str, pd.DataFrame, str]] = []
    for group_key, df_group in grouped_items:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        pivot = pd.pivot_table(
            df_group,
            index=list(row_index),
            columns=list(column_index),
            values="value",
            aggfunc="mean",
            dropna=False,
        ).sort_index(axis=0).sort_index(axis=1)
        pivot = _add_model_gain_columns(pivot)
        pivot = _order_table_columns(pivot)

        title_parts = ["Scalar metrics"]
        filename_parts: list[str] = []
        for col, value in zip(group_cols, group_key):
            if pd.isna(value):
                continue
            title_parts.append(f"{col}={value}")
            filename_parts.append(f"{col}_{_sanitize_filename_fragment(str(value))}")

        tables.append((" | ".join(title_parts), pivot, "__".join(filename_parts) if filename_parts else "all"))

    return tables


def save_scalar_metric_tables(
    *,
    metrics: dict,
    variables: list[str],
    output_folder: Path,
    filters: dict | None = None,
    significant_digits: int = SCALAR_TABLE_SIGNIFICANT_DIGITS,
    metrics_keep: set[str] | None = None,
    filename_prefix: str = "scalar_metrics",
    title_prefix: str | None = None,
    row_index: tuple[str, ...] = SCALAR_TABLE_ROW_INDEX,
    column_index: tuple[str, ...] = SCALAR_TABLE_COLUMN_INDEX,
    stat_label_overrides: dict[str, str] | None = None,
    base_color: str,
) -> list[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for title, table_df, filename_stem in build_scalar_metric_tables(
        metrics=metrics,
        variables=variables,
        filters=filters,
        row_index=row_index,
        column_index=column_index,
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
            significant_digits=significant_digits,
            raw_df=table_df,
            base_color=base_color,
        )

        parquet_path = output_folder / f"{filename_prefix}_{filename_stem}.parquet"
        image_path = output_folder / f"{filename_prefix}_{filename_stem}.png"
        table_df_display.to_parquet(parquet_path)
        save_scalar_metric_table_image(
            df=table_df,
            title=title,
            image_path=image_path,
            significant_digits=significant_digits,
            stat_label_overrides=stat_label_overrides,
            base_color=base_color,
        )
        saved_paths.extend([parquet_path, image_path])

    return saved_paths


def infer_leadtime_unit_from_configs(exp_root: Path) -> str:
    units: set[str] = set()

    for cfg_path in exp_root.rglob("experiment.cfg"):
        experiment = joblib.load(cfg_path)
        config = experiment["config"]

        train_datasets = config.train_dataset
        train_datasets = train_datasets if isinstance(train_datasets, (list, tuple)) else [train_datasets]

        train_input = next((td for td in train_datasets if td.role == "input"), None)
        if train_input is None:
            continue

        source_configs = train_input.source_configs
        source_config = source_configs[0] if isinstance(source_configs, (list, tuple)) else source_configs
        leadtime_rd = source_config.leadtime

        if leadtime_rd.hours >= 1:
            units.add("hours")
        elif leadtime_rd.months >= 1:
            units.add("months")
        else:
            raise ValueError(f"Unsupported leadtime in experiment config {cfg_path}: {leadtime_rd}")

    if not units:
        return ""
    if len(units) > 1:
        raise ValueError(f"Multiple leadtime units found in {exp_root}: {sorted(units)}")
    return next(iter(units))


def infer_region_from_configs(exp_root: Path) -> str:
    regions: set[str] = set()

    for cfg_path in exp_root.rglob("experiment.cfg"):
        experiment = joblib.load(cfg_path)
        config = experiment["config"]

        train_datasets = config.train_dataset
        train_datasets = train_datasets if isinstance(train_datasets, (list, tuple)) else [train_datasets]

        train_input = next((td for td in train_datasets if td.role == "input"), None)
        if train_input is None:
            continue

        datasources = train_input.datasource
        datasources = datasources if isinstance(datasources, (list, tuple)) else [datasources]
        if not datasources:
            continue

        region = getattr(datasources[0].data_selection.region, "name", None)
        if region:
            regions.add(str(region))

    if not regions:
        return ""
    if len(regions) > 1:
        raise ValueError(f"Multiple regions found in {exp_root}: {sorted(regions)}")
    return next(iter(regions))


def save_scoreboard_plot(
    *,
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
) -> Path:
    leadtime_axis_label = f"leadtime [{leadtime_unit}]" if leadtime_unit else "leadtime"

    scoreboard_frames: list[pd.DataFrame] = []
    diff_mode = "delta" if SCOREBOARD_MODE == "relative" else "no"
    outer_x_values = ["pr-fc"] if SCOREBOARD_MODE == "relative" else ["fc", "pr"]

    for metric_name, metric_type in SCOREBOARD_METRICS.items():
        scoreboard_frames.append(
            build_scalar_metric_df(
                metrics=metrics,
                variables=variables,
                metric_name=metric_name,
                metric_type=metric_type,
                diff=diff_mode,
                models=MODELS_DIFF if diff_mode == "delta" else None,
            )
        )

    df_scoreboard = pd.concat(scoreboard_frames, ignore_index=True)

    filters = {}
    for col, val in SCOREBOARD_FILTERS.items():
        if val is None:
            continue
        if col in {SCOREBOARD_ROW_AXIS, SCOREBOARD_COL_AXIS, "metric", "model"}:
            continue
        filters[col] = val

    plot_path = plot_folder / f"scoreboard_{SCOREBOARD_MODE}_{SCOREBOARD_ROW_AXIS}_vs_{SCOREBOARD_COL_AXIS}.png"
    plot_scoreboard(
        df=df_scoreboard,
        outer_y={"index": "metric", "values": list(SCOREBOARD_METRICS.keys())},
        outer_x={"index": "model", "values": outer_x_values},
        inner_y={
            "index": SCOREBOARD_ROW_AXIS,
            "label": leadtime_axis_label if SCOREBOARD_ROW_AXIS == "leadtime" else SCOREBOARD_ROW_AXIS,
        },
        inner_x={
            "index": SCOREBOARD_COL_AXIS,
            "label": leadtime_axis_label if SCOREBOARD_COL_AXIS == "leadtime" else SCOREBOARD_COL_AXIS,
        },
        filters=filters,
        agg="mean",
        metric_cmaps=SCOREBOARD_METRIC_CMAPS,
        metric_vlims=SCOREBOARD_METRIC_VLIMS,
        annotate=SCOREBOARD_ANNOTATE,
        annotate_fmt="{:.2f}",
        annotate_color="auto",
        save_path=plot_path,
        title=f"Scoreboard ({SCOREBOARD_MODE})",
    )
    return plot_path


def save_metric_vs_diff_plot(
    *,
    df_nodiff,
    df_delta,
    plot_folder: Path,
    forecast_metric: str,
    diff_metric: str,
    leadtime_unit: str,
    region: str,
    filters: dict | None,
    shade_by: str,
    shade_label: str,
) -> Path:
    plot_folder.mkdir(parents=True, exist_ok=True)
    plot_variables = _ordered_unique_variables(
        [
            *df_nodiff.get("variable", pd.Series(dtype=object)).dropna().astype(str).tolist(),
            *df_delta.get("variable", pd.Series(dtype=object)).dropna().astype(str).tolist(),
        ]
    )
    forecast_metric_label = format_metric_display_name(forecast_metric)
    diff_metric_label = format_metric_display_name(diff_metric)
    region_suffix = f" - {region}" if region else ""

    fig, ax, merged = plot_metric_vs_diff(
        df_nodiff,
        df_delta,
        forecast_metric=forecast_metric,
        diff_metric=diff_metric,
        x_metric_name=f"delta_{diff_metric}",
        y_metric_name=forecast_metric,
        shade_by=shade_by,
        shade_label=shade_label,
        fit_lines=False,
        fit_ci=False,
        leadtime_unit=leadtime_unit,
        cmap_name=DIFF_PLOT_CMAP,
        variable_colors=get_variable_colors(variables=plot_variables),
        filters=filters,
        title=f"Forecast {forecast_metric_label} vs Δ {diff_metric_label} (ML-corrected - forecast){region_suffix}",
        xlabel=f"Δ {diff_metric_label} (ML-corrected - forecast)",
        ylabel=f"Forecast {forecast_metric_label}",
        legend_loc="lower left",
    )

    if merged.empty:
        available_y = sorted(df_nodiff["metric"].dropna().astype(str).unique())
        available_x = sorted(df_delta["metric"].dropna().astype(str).unique())
        raise ValueError(
            "Plot dataframe is empty after merging. "
            f"Requested forecast_metric={forecast_metric!r}, diff_metric={diff_metric!r}. "
            f"Available no-diff metrics={available_y}. "
            f"Available delta metrics={available_x}."
        )

    plot_path = plot_folder / f"{forecast_metric}_vs_delta_{diff_metric}.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return plot_path


def save_metrics_vs_leadtime_plots(
    *,
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    shade_by: str,
    shade_label: str,
) -> list[Path]:
    saved_paths: list[Path] = []
    model_colors = MODEL_COLORS

    def _filter_da(da):
        if not filters:
            return da
        for col, allowed in filters.items():
            if col not in da.dims and col not in da.coords:
                continue
            values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            da = da.sel({col: list(values)})
        return da

    def _reduce_da(da, keep_dims: set[str]):
        reduce_dims = [dim for dim in da.dims if dim not in keep_dims]
        if reduce_dims:
            da = da.mean(dim=reduce_dims, skipna=True)
        return da

    ds_scalar_det = metrics["deterministic"]["scalar"]
    ds_scalar_ens = metrics["ensemble"]["scalar"]
    ds_scalar_prob = metrics["probabilistic"]["scalar"]

    for variable in variables:
        variable_plot_folder = get_variable_plot_folder(plot_root=plot_folder, variable=variable)

        for metric_name, metric_type in LEADTIME_METRICS.items():
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(12, 5.5),
                squeeze=True,
            )
            ds_metric = ds_scalar_prob if metric_type == "probabilistic" else ds_scalar_ens
            da_template = ds_metric[variable].sel(metric=metric_name)
            da_template = _filter_da(da_template)
            if da_template.size == 0:
                raise ValueError(f"No data available for {metric_name!r} and variable {variable!r}")

            if shade_by in da_template.dims or shade_by in da_template.coords:
                shade_values = da_template[shade_by].values.tolist()
            else:
                shade_values = [None]

            model_values = da_template["model"].values.tolist() if "model" in da_template.dims else []

            for model_name in model_values:
                base_color = model_colors.get(model_name, None)

                for j, shade_value in enumerate(shade_values):
                    alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))
                    shade_suffix = f", {shade_label}={shade_value}" if shade_value is not None else ""

                    if metric_type == "probabilistic":
                        da_prob = ds_scalar_prob[variable].sel(model=model_name, metric=metric_name)
                        da_prob = _filter_da(da_prob)
                        if shade_value is not None and shade_by in da_prob.dims:
                            da_prob = da_prob.sel({shade_by: shade_value})
                        da_prob = _reduce_da(da_prob, {"leadtime"})
                        if da_prob.size == 0:
                            continue

                        ax.plot(
                            da_prob["leadtime"],
                            da_prob,
                            marker="o",
                            linewidth=2.0,
                            alpha=alpha,
                            color=base_color,
                            label=f"{model_name}{shade_suffix}",
                        )
                    elif metric_type == "deterministic_with_ensemble_overlay":
                        da_ens = ds_scalar_ens[variable].sel(model=model_name, metric=metric_name)
                        da_det = ds_scalar_det[variable].sel(model=model_name, metric=metric_name)
                        da_ens = _filter_da(da_ens)
                        da_det = _filter_da(da_det)
                        if shade_value is not None:
                            if shade_by in da_ens.dims:
                                da_ens = da_ens.sel({shade_by: shade_value})
                            if shade_by in da_det.dims:
                                da_det = da_det.sel({shade_by: shade_value})

                        da_ens = _reduce_da(da_ens, {"leadtime"})
                        da_det = _reduce_da(da_det, {"leadtime", "realization"})
                        if da_ens.size == 0 or da_det.size == 0:
                            continue

                        x = da_det["leadtime"].values
                        y_members = da_det.values
                        if y_members.ndim != 2:
                            continue

                        y_mean = da_det.mean(dim="realization", skipna=True).values
                        y_min = da_det.min(dim="realization", skipna=True).values
                        y_max = da_det.max(dim="realization", skipna=True).values
                        y_ens = da_ens.values

                        if not (
                            pd.notna(y_mean).any()
                            or pd.notna(y_ens).any()
                            or pd.notna(y_members).any()
                        ):
                            continue

                        for member_idx in range(y_members.shape[1]):
                            ax.plot(
                                x,
                                y_members[:, member_idx],
                                color=base_color,
                                alpha=0.06 * alpha,
                                linewidth=0.8,
                                linestyle="--",
                                marker="o",
                                markersize=2.5,
                            )

                        ax.fill_between(
                            x,
                            y_min,
                            y_max,
                            color=base_color,
                            alpha=0.12 * alpha,
                        )
                        ax.plot(
                            x,
                            y_mean,
                            color=base_color,
                            linewidth=2.0,
                            alpha=alpha,
                            marker="o",
                            markersize=5,
                            label=f"{model_name} {metric_name}{shade_suffix}",
                        )
                        ax.plot(
                            x,
                            y_ens,
                            color=base_color,
                            linewidth=2.0,
                            alpha=alpha,
                            linestyle=":",
                            marker="o",
                            markersize=5,
                            label=f"{model_name} {metric_name} ensemble{shade_suffix}",
                        )
                    else:
                        raise ValueError(f"Unsupported leadtime metric plot mode {metric_type!r}")

                ax.set_title(f"{metric_name} vs lead time ({variable})")
                ax.set_xlabel(f"Lead time ({leadtime_unit.strip()})")
                ax.set_ylabel(metric_name)
                ax.grid(True, linestyle="--", alpha=0.35)

                ymin, ymax = ax.get_ylim()
                if ymin <= 0 <= ymax:
                    ax.axhline(0.0, color="0.3", lw=1.0, ls=":")

                ax.legend(fontsize=9)

            plt.tight_layout()
            plot_path = variable_plot_folder / f"{metric_name}_vs_leadtime.png"
            fig.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved_paths.append(plot_path)

    return saved_paths


def main() -> None:
    dask_earthml = Dask(n_workers=5, memory_limit="100GiB")
    dask_earthml.start()
    client = dask_earthml.client
    base_url = "http://localhost:"
    print("Dask dashboard:", client.dashboard_link.replace("http://127.0.0.1:", base_url))

    print(f"Loading experiments from: {EXP_ROOT_FOLDER}")

    runs, metrics, _ = get_runs_and_metrics(
        exp_root=EXP_ROOT_FOLDER,
        type_data=TYPE_DATA,
        load_models=LOAD_MODELS,
        metric_names=METRIC_NAMES,
    )

    vars_from_fc = [v for v in runs["fc"].data_vars if v != "_has_var"]
    print(f"Processed vars: {vars_from_fc}")
    leadtime_unit = infer_leadtime_unit_from_configs(EXP_ROOT_FOLDER)
    region = infer_region_from_configs(EXP_ROOT_FOLDER)
    print(f"Inferred leadtime unit: {leadtime_unit or 'unknown'}")
    print(f"Inferred region: {region or 'unknown'}")
    print_available_scalar_metrics(metrics=metrics)

    if GLOBAL_KNOBS["enable_diff_plot"]:
        df_nodiff = build_scalar_metric_df(
            metrics=metrics,
            variables=vars_from_fc,
            metric_name=FORECAST_METRIC,
            metric_type=FORECAST_METRIC_TYPE,
            diff="no",
        )
        df_delta = build_scalar_metric_df(
            metrics=metrics,
            variables=vars_from_fc,
            metric_name=DIFF_METRIC,
            metric_type=DIFF_METRIC_TYPE,
            diff="delta",
            models=MODELS_DIFF,
        )

        plot_path = save_metric_vs_diff_plot(
            df_nodiff=df_nodiff,
            df_delta=df_delta,
            plot_folder=PLOT_FOLDER,
            forecast_metric=FORECAST_METRIC,
            diff_metric=DIFF_METRIC,
            leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
            region=region,
            filters=PLOT_FILTERS,
            shade_by=SHADE_BY,
            shade_label=SHADE_LABEL,
        )
        print("Saved leadtime-vs-global-metrics plot to:", plot_path)
    else:
        print("Skipping diff plot: GLOBAL_KNOBS['enable_diff_plot'] is False")

    if GLOBAL_KNOBS["enable_leadtime_plots"]:
        leadtime_plot_paths = save_metrics_vs_leadtime_plots(
            metrics=metrics,
            variables=vars_from_fc,
            plot_folder=PLOT_FOLDER,
            leadtime_unit=f" {leadtime_unit}" if leadtime_unit else "",
            filters=PLOT_FILTERS,
            shade_by=SHADE_BY,
            shade_label=SHADE_LABEL,
        )
        for leadtime_plot_path in leadtime_plot_paths:
            print("Saved metric-vs-leadtime plot to:", leadtime_plot_path)
    else:
        print("Skipping leadtime plots: GLOBAL_KNOBS['enable_leadtime_plots'] is False")

    if GLOBAL_KNOBS["enable_scalar_tables"]:
        raw_metrics = {
            metric
            for metric_group in (
                metrics.get("deterministic", {}).get("scalar"),
                metrics.get("ensemble", {}).get("scalar"),
                metrics.get("probabilistic", {}).get("scalar"),
            )
            if metric_group is not None and "metric" in metric_group.coords
            for metric in metric_group.coords["metric"].values.tolist()
        } - NORMALIZED_METRICS

        for variable in vars_from_fc:
            variable_plot_folder = get_variable_plot_folder(plot_root=PLOT_FOLDER, variable=variable)
            base_color = get_variable_table_color(variable=variable, variables=vars_from_fc)
            scalar_table_paths = save_scalar_metric_tables(
                metrics=metrics,
                variables=[variable],
                output_folder=variable_plot_folder,
                filters=SCALAR_TABLE_FILTERS,
                metrics_keep=raw_metrics,
                filename_prefix="scalar_metrics",
                base_color=base_color,
            )
            for scalar_table_path in scalar_table_paths:
                print("Saved scalar metrics table to:", scalar_table_path)

        combined_normalized_table_paths = save_scalar_metric_tables(
            metrics=metrics,
            variables=vars_from_fc,
            output_folder=PLOT_FOLDER,
            filters=SCALAR_TABLE_FILTERS,
            metrics_keep=NORMALIZED_METRICS,
            filename_prefix="normalized_metrics",
            title_prefix="Normalized metrics",
            row_index=("variable", "metric"),
            column_index=("model", "stat"),
            stat_label_overrides={"prob": ""},
            base_color=get_variable_table_color(variable=vars_from_fc[0], variables=vars_from_fc),
        )
        for normalized_table_path in combined_normalized_table_paths:
            print("Saved combined normalized metrics table to:", normalized_table_path)
    else:
        print("Skipping scalar tables: GLOBAL_KNOBS['enable_scalar_tables'] is False")

    if GLOBAL_KNOBS["enable_scoreboard"]:
        scoreboard_path = save_scoreboard_plot(
            metrics=metrics,
            variables=vars_from_fc,
            plot_folder=PLOT_FOLDER,
            leadtime_unit=leadtime_unit,
        )
        print("Saved scoreboard plot to:", scoreboard_path)
    else:
        print("Skipping scoreboard: GLOBAL_KNOBS['enable_scoreboard'] is False")


if __name__ == "__main__":
    main()
