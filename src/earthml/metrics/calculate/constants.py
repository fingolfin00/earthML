CALCULATE_METRICS_ITEMS = (
    "timeseries",
    "maps",
    "scalar_tables",
    "metric_vs_delta",
    "profiles",
    "scoreboards",
)

PROFILE_METRIC_MODES = (
    "all_dims",
    "spatial_mean",
    "ensemble_mean",
    "probabilistic",
    "all_dims_with_ensemble_overlay",
)

METRIC_SECTIONS = (
    "all_dims",
    "spatial_mean",
    "ensemble_mean",
    "probabilistic",
)

METRIC_KIND = (
    "scalar",
    "timeseries",
    "map",
)

SCOREBOARD_MODES = (
    "absolute",
    "relative",
)

CONTEXT_AXES = (
    "leadtime",
    "train_period",
    "loss",
    "variant",
    "variable",
    "region",
)

# TODO maybe extend
NORMALIZED_METRICS = {
    "nrmse",
    "ncrmse",
    "nmae",
    "nbias",
    "r2",
}

VARIABLE_SUBFOLDERS = ("timeseries", "maps", "profiles", "tables")
OUTPUT_GROUP_SUBFOLDERS = ("field", "all_dims", "ensemble_mean", "spatial_mean", "probabilistic")

# Display constants
VARIABLE_COLORS = {
    "t2m": "#9e1b9c",
    "msl": "#d95f02",
    "d2m": "#7570b3",
    "u10": "#e7298a",
    "v10": "#66a61e",
    "tcc": "#e6ab02",
    "sst": "#c80b0b",
    "sss": "#d7e50e",
    "ssh": "#24cd15",
    "t14d": "#1f78b4",
}

METRIC_DISPLAY_NAMES = {
    "r2": "R2",
    "rmse": "RMSE",
    "rmse_skill_clim": "RMSE vs clim Skill",
    "crmse": "CRMSE",
    "mae": "MAE",
    "bias": "Bias",
    "error_std": "Error Std",
    "variance_ratio": "Variance Ratio",
    "nrmse": "nRMSE",
    "ncrmse": "nCRMSE",
    "nmae": "nMAE",
    "nbias": "nBias",
    "crps": "CRPS",
    "spread": "Spread",
    "spread_error_ratio": "Spread/Error",
}

VARIABLE_DISPLAY_NAMES = {
    "t2m": "2m temperature",
    "d2m": "2m dewpoint temperature",
    "msl": "MSL pressure",
    "mslp": "MSL pressure",
    "u10": "10m zonal wind",
    "v10": "10m meridional wind",
    "tp": "Precipitation",
    "tcc": "Cloud cover",
    "sst": "Sea surface temperature",
    "ssh": "Sea surface height",
    "sss": "Sea surface salinity",
}

AXIS_DISPLAY_NAMES = {
    "variable": "",
    "leadtime": "Lead time",
    "loss": "Loss",
    "variant": "Variant",
    "train_period": "Train period",
    "region": "Region",
    "model": "Model",
    "metric": "Metric",
}

# Color maps
METRIC_CMAPS = {
    "bias": "RdBu_r",
    "nbias": "RdBu_r",
    "r2": "RdBu",
    "rmse_skill_clim": "RdBu",
    "spread_error_ratio": "RdBu",
}

# Profiles
PROFILE_METRICS = {
    "r2": "deterministic_with_ensemble_overlay",
    "rmse": "deterministic_with_ensemble_overlay",
    # "nrmse": "deterministic_with_ensemble_overlay",
    "rmse_skill_clim": "deterministic_with_ensemble_overlay",
    "mae": "deterministic_with_ensemble_overlay",
    # "nmae": "deterministic_with_ensemble_overlay",
    "bias": "deterministic_with_ensemble_overlay",
    "clim_acc": "deterministic_with_ensemble_overlay",
    "variance_ratio": "deterministic_with_ensemble_overlay",
    "error_std": "deterministic_with_ensemble_overlay",
    # "nbias": "deterministic_with_ensemble_overlay",
    "crps": "probabilistic",
    "spread": "probabilistic",
    "spread_error_ratio": "probabilistic",
}

PROFILE_COMBINED_VARS_METRICS = {
    "nbias": "deterministic_with_ensemble_overlay",
    "nrmse": "deterministic_with_ensemble_overlay",
    "nmae": "deterministic_with_ensemble_overlay",
    "r2": "deterministic_with_ensemble_overlay",
    "clim_acc": "deterministic_with_ensemble_overlay",
    "variance_ratio": "deterministic_with_ensemble_overlay",
}

# Metric vs delta metric
METRIC_VS_DELTA_PAIRS = [
    {
        "forecast_metric": "r2",
        "forecast_metric_type": "ensemble_mean",
        "delta_metric": "nrmse",
        "delta_metric_type": "ensemble_mean",
    },
    {
        "forecast_metric": "r2",
        "forecast_metric_type": "ensemble_mean",
        "delta_metric": "nmae",
        "delta_metric_type": "ensemble_mean",
    },
    {
        "forecast_metric": "nrmse",
        "forecast_metric_type": "ensemble_mean",
        "delta_metric": "nrmse",
        "delta_metric_type": "ensemble_mean",
    },
    {
        "forecast_metric": "clim_acc",
        "forecast_metric_type": "ensemble_mean",
        "delta_metric": "nmae",
        "delta_metric_type": "ensemble_mean",
    },
    {
        "forecast_metric": "spread_error_ratio",
        "forecast_metric_type": "probabilistic",
        "delta_metric": "crps",
        "delta_metric_type": "probabilistic",
    },
]

# Scoreboard
SCOREBOARD_METRICS = {
    "r2": "ensemble_mean",
    "nrmse": "ensemble_mean",
    "ncrmse": "ensemble_mean",
    "nmae": "ensemble_mean",
    "nbias": "ensemble_mean",
    # "crps": "probabilistic",
    "variance_ratio": "ensemble_mean",
    "rmse_skill_clim": "ensemble_mean",
    "clim_acc": "ensemble_mean",
    "spatial_acc": "ensemble_mean",
    "spread_error_ratio": "probabilistic",
}

SCOREBOARD_CMAPS = {
    "r2": "RdBu",
    "nrmse": "RdBu_r",
    "ncrmse": "RdBu_r",
    "nmae": "RdBu_r",
    "nbias": "RdBu_r",
    "variance_ratio": "RdBu_r",
    "rmse_skill_clim": "RdBu",
    "clim_acc": "RdBu",
    "spatial_acc": "RdBu",
    "spread_error_ratio": "RdBu",
    "crps": "RdBu",
}

# TODO make this limits consistent
SCOREBOARD_LIMITS = {
    "r2": (-0.4, 0.4),
    "nrmse": (-0.2, 0.2),
    "ncrmse": (-0.2, 0.2),
    "nmae": (-0.2, 0.2),
    "nbias": (-0.2, 0.2),
    "variance_ratio": (-1, 1),
    "rmse_skill_clim": (-1, 1),
    "clim_acc": (-1, 1),
    "spatial_acc": (-1, 1),
    "spread_error_ratio": (-1, 1),
    "crps": (-0.2, 0.2),
}
