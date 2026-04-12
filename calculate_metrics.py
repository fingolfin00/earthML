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
FORECAST_METRIC_TYPE = "ensemble"
DIFF_METRIC = "nrmse"
DIFF_METRIC_TYPE = "ensemble"
LEADTIME_METRICS = {
    "r2": "deterministic_with_ensemble_overlay",
    "rmse": "deterministic_with_ensemble_overlay",
    "mae": "deterministic_with_ensemble_overlay",
    "bias": "deterministic_with_ensemble_overlay",
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
    metrics: dict,
    variables: list[str],
    plot_folder: Path,
    leadtime_unit: str,
    filters: dict | None,
    shade_by: str,
    shade_label: str,
) -> list[Path]:
    plot_folder.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    model_colors = {"fc": "tab:green", "pr": "tab:blue", "an": "tab:gray"}

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

    for metric_name, metric_type in LEADTIME_METRICS.items():
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
        if metric_type == "probabilistic":
            ds_metric = ds_scalar_prob
        else:
            ds_metric = ds_scalar_ens

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
                        raise ValueError(
                            f"Unsupported leadtime metric plot mode {metric_type!r} "
                            f"for metric {metric_name!r}"
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

    leadtime_plot_paths = save_metrics_vs_leadtime_plots(
        metrics=metrics,
        variables=vars_from_fc,
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
