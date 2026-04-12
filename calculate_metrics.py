from pathlib import Path
import logging
import warnings

import matplotlib.pyplot as plt
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
LEADTIME_UNIT_LABEL = " months"
PLOT_FILTERS = None
SHADE_BY = "loss" # loss, total_months
SHADE_LABEL = "Loss"


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

    df_nodiff = metrics_to_df(
        metrics=metrics,
        variables=vars_from_fc,
        metric_names=METRIC_NAMES,
        kind="scalar",
        metric_type="deterministic",
        diff="no",
    )
    df_delta = metrics_to_df(
        metrics=metrics,
        variables=vars_from_fc,
        metric_names=METRIC_NAMES,
        kind="scalar",
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


if __name__ == "__main__":
    main()
