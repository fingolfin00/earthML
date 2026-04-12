from pathlib import Path
import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from rich import print

from earthml import Dask, get_runs_and_metrics
from earthml.metrics.utils import metrics_to_df
from earthml.plots import plot_metric_vs_diff


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Dask.*")
warnings.filterwarnings("ignore", message=".*distributed.scheduler.*")
logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
logging.getLogger("distributed.worker").setLevel(logging.ERROR)
logging.getLogger("distributed.nanny").setLevel(logging.ERROR)


EXP_ROOT_FOLDER = Path("/Users/jacopodallaglio/ML/experiments_earthML_seasonal_atmo/")
PLOT_FOLDER = EXP_ROOT_FOLDER / "plots"

TYPE_DATA = "test"
LOAD_MODELS = ("an", "fc", "pr")
METRIC_NAMES = None
MODELS_DIFF = ("fc", "pr")

FORECAST_METRIC = "r2"
DIFF_METRIC = "nrmse"
LEADTIME_METRICS = {
    "r2": "deterministic",
    "rmse": "deterministic",
    "mae": "deterministic",
    "bias": "deterministic",
    "crps": "probabilistic",
}
LEADTIME_UNIT_LABEL = " months"
PLOT_FILTERS = None
SHADE_BY = "loss" # loss, total_months
SHADE_LABEL = "Loss"


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


def save_leadtime_vs_global_metrics_plot(
    *,
    df_nodiff,
    df_delta,
    plot_folder: Path,
    forecast_metric: str,
    diff_metric: str,
    leadtime_unit: str,
    filters: dict | None,
    shade_by: str,
    shade_label: str,
) -> Path:
    plot_folder.mkdir(parents=True, exist_ok=True)

    fig, ax, merged = plot_metric_vs_diff(
        df_nodiff,
        df_delta,
        forecast_metric=forecast_metric,
        diff_metric=diff_metric,
        x_metric_name=f"delta_{diff_metric}",
        y_metric_name=forecast_metric,
        shade_by=shade_by,
        shade_label=shade_label,
        fit_lines=True,
        fit_ci=False,
        leadtime_unit=leadtime_unit,
        filters=filters,
        title=f"{forecast_metric} vs delta {diff_metric}",
        xlabel=f"Delta {diff_metric}",
        ylabel=forecast_metric,
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
    metric_dfs: dict[str, pd.DataFrame],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    shade_by: str,
    shade_label: str,
) -> list[Path]:
    plot_folder.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    model_colors = {"fc": "tab:green", "pr": "tab:blue", "an": "tab:gray"}

    for metric_name, metric_df_in in metric_dfs.items():
        metric_df = metric_df_in.copy()
        if filters:
            for col, allowed in filters.items():
                metric_df = metric_df[metric_df[col].isin(allowed)]

        required_cols = {"metric", "variable", "model", "leadtime", shade_by, "value"}
        missing = required_cols.difference(metric_df.columns)
        if missing:
            raise ValueError(
                f"metrics-vs-leadtime plot for {metric_name!r} requires columns {sorted(missing)}"
            )

        if metric_df.empty:
            raise ValueError(f"No rows found for metric={metric_name!r} in no-diff dataframe")

        grouped = (
            metric_df
            .groupby(["variable", "model", shade_by, "leadtime"], as_index=False)["value"]
            .mean()
        )

        variables = sorted(grouped["variable"].dropna().astype(str).unique())
        shade_values = sorted(grouped[shade_by].dropna().tolist())
        nrows = len(variables)

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(12, max(5, 4.5 * nrows)),
            squeeze=False,
        )
        axs = axs.ravel()

        for i, variable in enumerate(variables):
            ax = axs[i]
            var_df = grouped[grouped["variable"] == variable].copy()

            for model_name in sorted(var_df["model"].dropna().astype(str).unique()):
                model_df = var_df[var_df["model"] == model_name]
                base_color = model_colors.get(model_name, None)

                for j, shade_value in enumerate(shade_values):
                    line_df = model_df[model_df[shade_by] == shade_value].sort_values("leadtime")
                    if line_df.empty:
                        continue

                    alpha = 0.35 + 0.55 * (j + 1) / max(1, len(shade_values))
                    label = f"{model_name}, {shade_label}={shade_value}"

                    ax.plot(
                        line_df["leadtime"],
                        line_df["value"],
                        marker="o",
                        linewidth=2.0,
                        alpha=alpha,
                        color=base_color,
                        label=label,
                    )

            ax.set_title(f"{metric_name} vs lead time ({variable})")
            ax.set_xlabel(f"Lead time ({leadtime_unit.strip()})")
            ax.set_ylabel(metric_name)
            ax.grid(True, linestyle="--", alpha=0.35)

            ymin, ymax = ax.get_ylim()
            if ymin <= 0 <= ymax:
                ax.axhline(0.0, color="0.3", lw=1.0, ls=":")

            ax.legend(fontsize=9)

        plt.tight_layout()
        plot_path = plot_folder / f"{metric_name}_vs_leadtime.png"
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

    df_nodiff = build_scalar_metric_df(
        metrics=metrics,
        variables=vars_from_fc,
        metric_name=FORECAST_METRIC,
        metric_type="deterministic",
        diff="no",
    )
    df_delta = build_scalar_metric_df(
        metrics=metrics,
        variables=vars_from_fc,
        metric_name=DIFF_METRIC,
        metric_type="deterministic",
        diff="delta",
        models=MODELS_DIFF,
    )

    plot_path = save_leadtime_vs_global_metrics_plot(
        df_nodiff=df_nodiff,
        df_delta=df_delta,
        plot_folder=PLOT_FOLDER,
        forecast_metric=FORECAST_METRIC,
        diff_metric=DIFF_METRIC,
        leadtime_unit=LEADTIME_UNIT_LABEL,
        filters=PLOT_FILTERS,
        shade_by=SHADE_BY,
        shade_label=SHADE_LABEL,
    )
    print("Saved leadtime-vs-global-metrics plot to:", plot_path)

    leadtime_metric_dfs = {
        metric_name: build_scalar_metric_df(
            metrics=metrics,
            variables=vars_from_fc,
            metric_name=metric_name,
            metric_type=metric_type,
            diff="no",
        )
        for metric_name, metric_type in LEADTIME_METRICS.items()
    }

    leadtime_plot_paths = save_metrics_vs_leadtime_plots(
        metric_dfs=leadtime_metric_dfs,
        plot_folder=PLOT_FOLDER,
        leadtime_unit=LEADTIME_UNIT_LABEL,
        filters=PLOT_FILTERS,
        shade_by=SHADE_BY,
        shade_label=SHADE_LABEL,
    )
    for leadtime_plot_path in leadtime_plot_paths:
        print("Saved metric-vs-leadtime plot to:", leadtime_plot_path)


if __name__ == "__main__":
    main()
