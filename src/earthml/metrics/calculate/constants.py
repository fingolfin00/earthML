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

# Maps
METRIC_MAPS_CMAPS = {
    # Non-negative error magnitudes
    "rmse": "jet",
    "crmse": "jet",
    "mae": "jet",
    "error_std": "jet",
    "nrmse": "WRdY",
    "ncrmse": "WRdY",
    "nmae": "WRdY",
    "crps": "PiBRdY",
    "spread": "jet",
    # Signed metrics and zero-centered skill/correlation scores
    "bias": "PiBRdY",
    "nbias": "PiBRdY",
    "r2": "PiBRdY",
    "corr": "PiBRdY",
    "clim_acc": "PiBRdY",
    "spatial_acc": "PiBRdY",
    "rmse_skill_clim": "PiBRdY",
    # Positive ratios where 1 is ideal. Keep these sequential unless we also
    # introduce explicit centering around 1 in the map normalization.
    "variance_ratio": "cividis",
    "spread_error_ratio": "cividis",
}

METRIC_MAPS_DEFAULT_LIMITS = {
    # Non-negative error magnitudes
    "rmse": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    "crmse": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    "mae": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    "error_std": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    "nrmse": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    "ncrmse": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    "nmae": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    "crps": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    "spread": {
        "*": {
            "default": (0, None),
        },
        "msl": {
            "ConUS": (0, None),
            "Europe": (0, None),
        },
    },
    # Signed metrics and zero-centered skill/correlation scores
    "bias": {
        "*": {
            "default": (None, None),
        },
        "msl": {
            "ConUS": (None, None),
            "Europe": (None, None),
        },
    },
    "nbias": {
        "*": {
            "default": (-1, 1),
        },
        "msl": {
            "ConUS": (-1, 1),
            "Europe": (-1, 1),
        },
    },
    "r2": {
        "*": {
            "default": (-1, 1),
        },
        "msl": {
            "ConUS": (-1, 1),
            "Europe": (-1, 1),
        },
    },
    "corr": {
        "*": {
            "default": (-1, 1),
        },
        "msl": {
            "ConUS": (-1, 1),
            "Europe": (-1, 1),
        },
    },
    "clim_acc": {
        "*": {
            "default": (-1, 1),
        },
        "msl": {
            "ConUS": (-1, 1),
            "Europe": (-1, 1),
        },
    },
    "spatial_acc": {
        "*": {
            "default": (-1, 1),
        },
        "msl": {
            "ConUS": (-1, 1),
            "Europe": (-1, 1),
        },
    },
    "rmse_skill_clim": {
        "*": {
            "default": (-1, 1),
        },
        "msl": {
            "ConUS": (-1, 1),
            "Europe": (-1, 1),
        },
    },
    # Positive ratios where 1 is ideal. Keep these sequential unless we also
    # introduce explicit centering around 1 in the map normalization.
    "variance_ratio": {
        "*": {
            "default": (0, 1),
        },
        "msl": {
            "ConUS": (0, 1),
            "Europe": (0, 1),
        },
    },
    "spread_error_ratio": {
        "*": {
            "default": (0, 1),
        },
        "msl": {
            "ConUS": (0, 1),
            "Europe": (0, 1),
        },
    },
}

METRIC_MAPS_VAR_OVERRIDE_LIMITS = {
    # Non-negative error magnitudes
    "rmse": {
        "msl": {
            "ConUS": {
                "all": (0, 1000),
                "months": {
                    "01": (0, None),
                },
                "diff": (-500, 500),
            },
            "Europe": {
                "all": (0, 1600),
                "months": {
                    "01": (0, None),
                },
                "diff": (0, None),
            },
        },
    },
    "crmse": {
        "msl": {
            "ConUS": (0, 1000),
            "Europe": (0, 1600),
        },
    },
    "mae": {
        "msl": {
            "ConUS": (0, 700),
            "Europe": (0, 1200),
        },
    },
    "error_std": {
        "msl": {
            "ConUS": (0, 1000),
            "Europe": (0, 1600),
        },
    },
    "nrmse": {
        "msl": {
            "ConUS": (0, 1.2),
            "Europe": (0, 1),
        },
    },
    "ncrmse": {
        "msl": {
            "ConUS": (0, 1),
            "Europe": (0, 1.2),
        },
    },
    "nmae": {
        "msl": {
            "ConUS": (0, 0.8),
            "Europe": (0, 1),
        },
    },
    "crps": {
        "msl": {
            "ConUS": (0, 600),
            "Europe": (0, 1000),
        },
    },
    "spread": {
        "msl": {
            "ConUS": (0, 250),
            "Europe": (0, 600),
        },
    },
    # Signed metrics and zero-centered skill/correlation scores
    "bias": {
        "msl": {
            "ConUS": (-200, 200),
            "Europe": (-300, 300),
        },
    },
    "nbias": {
        "msl": {
            "ConUS": (-0.3, 0.3),
            "Europe": (-0.5, 0.5),
        },
    },
}

# Profiles
# Profile metric config accepts both scalar-only form:
#   "rmse": ("all_dims_with_ensemble_overlay", "spatial_mean")
# and richer per-source entries such as:
#   "rmse": [("all_dims", "map", ("latitude", "longitude"))]
PROFILE_METRICS = {
    # Grand mean
    "r2": [
        "all_dims_with_ensemble_overlay",
        "spatial_mean",
        ("ensemble_mean", "map", ("latitude", "longitude")),
        ("all_dims", "map", ("latitude", "longitude", "realization")),
    ],
    "rmse": [
        "all_dims_with_ensemble_overlay",
        "spatial_mean",
        ("ensemble_mean", "map", ("latitude", "longitude")),
        ("all_dims", "map", ("latitude", "longitude", "realization")),
    ],
    # "nrmse": [
    #     "all_dims_with_ensemble_overlay",
    #     "spatial_mean",
    #     ("ensemble_mean", "map", ("latitude", "longitude")),
    #     ("all_dims", "map", ("latitude", "longitude", "realization")),
    # ],
    "rmse_skill_clim": [
        "all_dims_with_ensemble_overlay",
        "spatial_mean",
        ("ensemble_mean", "map", ("latitude", "longitude")),
        ("all_dims", "map", ("latitude", "longitude", "realization")),
    ],
    "mae": [
        "all_dims_with_ensemble_overlay",
        "spatial_mean",
        ("ensemble_mean", "map", ("latitude", "longitude")),
        ("all_dims", "map", ("latitude", "longitude", "realization")),
    ],
    # "nmae": [
    #     "all_dims_with_ensemble_overlay",
    #     "spatial_mean",
    #     ("ensemble_mean", "map", ("latitude", "longitude")),
    #     ("all_dims", "map", ("latitude", "longitude", "realization")),
    # ],
    "bias": [
        "all_dims_with_ensemble_overlay",
        "spatial_mean",
        ("ensemble_mean", "map", ("latitude", "longitude")),
        ("all_dims", "map", ("latitude", "longitude", "realization")),
    ],
    "clim_acc": [
        "all_dims_with_ensemble_overlay",
        "spatial_mean",
        ("ensemble_mean", "map", ("latitude", "longitude")),
        ("all_dims", "map", ("latitude", "longitude", "realization")),
    ],
    "variance_ratio": [
        "all_dims_with_ensemble_overlay",
        "spatial_mean",
        ("ensemble_mean", "map", ("latitude", "longitude")),
        ("all_dims", "map", ("latitude", "longitude", "realization")),
    ],
    "error_std": [
        "all_dims_with_ensemble_overlay",
        "spatial_mean",
        ("ensemble_mean", "map", ("latitude", "longitude")),
        ("all_dims", "map", ("latitude", "longitude", "realization")),
    ],
    # "nbias": [
    #     "all_dims_with_ensemble_overlay",
    #     "spatial_mean",
    #     ("ensemble_mean", "map", ("latitude", "longitude")),
    #     ("all_dims", "map", ("latitude", "longitude", "realization")),
    # ],
    # Probabilistic
    "crps": [
        "probabilistic",
        ("probabilistic", "map", ("latitude", "longitude")),
    ],
    "spread": [
        "probabilistic",
        ("probabilistic", "map", ("latitude", "longitude")),
    ],
    "spread_error_ratio": [
        "probabilistic",
        ("probabilistic", "map", ("latitude", "longitude")),
    ],
}

PROFILE_COMBINED_VARS_METRICS = {
    "nbias": "all_dims_with_ensemble_overlay",
    "nrmse": "all_dims_with_ensemble_overlay",
    "nmae": "all_dims_with_ensemble_overlay",
    "r2": "all_dims_with_ensemble_overlay",
    "clim_acc": "all_dims_with_ensemble_overlay",
    "variance_ratio": "all_dims_with_ensemble_overlay",
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
