import os
# Silence Intel OpenMP deprecation/info chatter emitted by downstream libraries.
os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

import logging, warnings

import matplotlib
matplotlib.use("Agg")

from earthml.metrics.calculate import CalculateMetricsRuntime


def _suppress_noisy_loader_logs() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*Dask.*")
    warnings.filterwarnings("ignore", message=".*distributed.scheduler.*")
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in divide",
        category=RuntimeWarning,
        module=r"dask\.array\.numpy_compat",
    )
    logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
    logging.getLogger("distributed.worker").setLevel(logging.ERROR)
    logging.getLogger("distributed.nanny").setLevel(logging.ERROR)

    # Keep the high-level runtime logs, but mute verbose per-source loading chatter.
    for logger_name in (
        "earthml.sources",
        "earthml.experiments.mlbc.load",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def main() -> None:
    _suppress_noisy_loader_logs()

    runtime = CalculateMetricsRuntime(
        # Globals
        dask_workers=12,
        memory_limit="100GiB",
        root_path="/Users/jacopodallaglio/ML/experiments_earthML_seasonal_atmo/",
        # exp_mode="train",
        exp_mode="test",
        metrics=None, # None means all available metrics
        log_level="info",
        print_user_config=False,
        print_console_tables=False,
        external_mask_path=None,
        external_mask_variable=None,
        disable_flox=True,
        metric_groupby_period="month", # used for climatology, anomalies and metric grouping
        # variable_colors={},
        # Filters
        leadtime=None,
        variable=("msl",),
        region=None,
        train_period=None,
        loss=("mseloss",),
        variant=("",),
        # variable=None,
        # loss=None,
        # variant=None,
        # Models
        # corrected=None,
        # Items
        timeseries=True,
        maps=True,
        profiles=True,
        tables=False,
        metric_vs_diff=False,
        scoreboard=False,
        # Items config
        profiles_x_axis="leadtime",
        profiles_enable_combined_variable=False,
        metric_vs_delta_byconfig={
            "color": "variable",
            "marker": ("region", "variant"),
            "shade": "leadtime"
        },
        scoreboard_mode="absolute",
        # Save metrics
        save_calculated_metrics=True,
        saved_metric_kinds=("scalar", "timeseries", "map"),
        saved_metric_types=("all_dims", "spatial_mean", "ensemble_mean", "probabilistic"),
        reuse_saved_metrics=True,
    )

    runtime.start()


if __name__ == "__main__":
    main()
