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
        exp_mode="test",
        metrics=None, # None means all available metrics
        log_level="info",
        print_user_config=False,
        print_console_tables=False,
        external_mask_path=None,
        external_mask_variable=None,
        disable_flox=True,
        # variable_colors={},
        # Filters
        leadtime=None,
        variable=None,
        region=None,
        train_period=None,
        loss=None,
        variant=None,
        # Items
        timeseries=False,
        maps=False,
        profiles=False,
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
    )

    runtime.start()


if __name__ == "__main__":
    main()
