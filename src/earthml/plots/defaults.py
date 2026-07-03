import numpy as np
import matplotlib.pyplot as plt

from ..metrics import safe_percent
from .colormaps import (
    SeqWRdY,
    SeqPiBRdY,
    SeqBPi,
)


DEFAULT_PLOT_CONFIG = {
    # Bias-like
    "bias": {
        "vmin": -12,
        "vmax": 12,
        "ticks": [-12, -10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 10, 12],
        "cmap": SeqBPi,
        "scale_units": True,
    },
    "bias_anom": {
        "vmin": -12,
        "vmax": 12,
        "ticks": [-12, -10, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 10, 12],
        "cmap": SeqBPi,
        "scale_units": True,
    },

    # Absolute / squared error
    "mae": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mae_anom": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mse": {
        "vmin": 0,
        "vmax": 81,
        "ticks": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mse_anom": {
        "vmin": 0,
        "vmax": 81,
        "ticks": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "rmse": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "rmse_anom": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "nrmse": {
        "vmin": 0,
        "vmax": 2,
        "ticks": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2],
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "nrmse_anom": {
        "vmin": 0,
        "vmax": 2,
        "ticks": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2],
        "cmap": SeqWRdY,
        "scale_units": False,
    },

    # Correlation / explained variance
    "corr": {
        "vmin": -1,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "acc": {
        "vmin": -1,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "r2": {
        "vmin": -10,
        "vmax": 1,
        "ticks": [-20, -10, -5, -2, -1, -0.5, -0.4, -0.2, -0.1, 0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "r2_anom": {
        "vmin": -2,
        "vmax": 1,
        "ticks": [-2, -1, -0.8, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },

    # Standard deviation / ratios
    "fc_std": {
        "vmin": 0,
        "vmax": 10,
        "ticks": [0, 1, 2, 3, 4, 5, 6, 8, 10],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "an_std": {
        "vmin": 0,
        "vmax": 10,
        "ticks": [0, 1, 2, 3, 4, 5, 6, 8, 10],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "std_ratio": {
        "vmin": 0.1,
        "vmax": 1.9,
        "ticks": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "fc_anom_std": {
        "vmin": 0,
        "vmax": 10,
        "ticks": [0, 1, 2, 3, 4, 5, 6, 8, 10],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "an_anom_std": {
        "vmin": 0,
        "vmax": 10,
        "ticks": [0, 1, 2, 3, 4, 5, 6, 8, 10],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "std_ratio_anom": {
        "vmin": 0.1,
        "vmax": 1.9,
        "ticks": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },

    # Climatology skill
    "mse_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "mae_anom_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "mse_anom_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "rmse_anom_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "ens_member_mse_anom_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "mean_member_mse_anom_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },

    # Ensemble
    "ens_member_rmse": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mean_member_rmse": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "ens_member_rmse_anom": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mean_member_rmse_anom": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "spread": {
        "vmin": 0,
        "vmax": 10,
        "ticks": [0, 1, 2, 3, 4, 5, 6, 8, 10],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "spread_anom": {
        "vmin": 0,
        "vmax": 10,
        "ticks": [0, 1, 2, 3, 4, 5, 6, 8, 10],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "spread_skill_ratio": {
        "vmin": 0.1,
        "vmax": 1.9,
        "ticks": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "spread_anom_skill_ratio": {
        "vmin": 0.1,
        "vmax": 1.9,
        "ticks": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },

    # Probabilistic
    "crps": {
        "vmin": 0,
        "vmax": 12,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "crps_anom": {
        "vmin": 0,
        "vmax": 12,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12],
        "cmap": SeqWRdY,
        "scale_units": True,
    },

    # ROC AUC
    "roc_anom_lower": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "roc_anom_middle": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "roc_anom_upper": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
}


DEFAULT_IMPROVEMENT_PLOT_CONFIG = {
    "%": {
        "vmin": -400,
        "vmax": 100,
        "ticks": [-400, -300, -200, -100, 0, 25, 50, 75, 100],
        "cmap": plt.get_cmap("RdBu"),
    },
    "Δ": {
        "vmin": -0.5,
        "vmax": 0.5,
        "ticks": np.linspace(-0.5, 0.5, 11),
        "cmap": plt.get_cmap("RdBu"),
    },
}


METRIC_NAMES = {
    # Deterministic
    "bias": "Bias",
    "mae": "MAE",
    "mse": "MSE",
    "mse_skill_clim": "MSE Skill vs Climatology",
    "rmse": "RMSE",
    "nrmse": "Normalized RMSE",
    "r2": "R²",
    "corr": "Correlation",
    "fc_std": "Forecast STD",
    "an_std": "Analysis STD",
    "std_ratio": "STD Ratio",

    # Deterministic anomaly
    "bias_anom": "Anomaly Bias",
    "mae_anom": "Anomaly MAE",
    "mse_anom": "Anomaly MSE",
    "rmse_anom": "Anomaly RMSE",
    "nrmse_anom": "Normalized Anomaly RMSE",
    "acc": "ACC",
    "r2_anom": "Anomaly R²",
    "fc_anom_std": "Forecast Anomaly STD",
    "an_anom_std": "Analysis Anomaly STD",
    "std_ratio_anom": "Anomaly STD Ratio",

    # Anomaly skill vs climatology
    "mae_anom_skill_clim": "Anomaly MAE Skill vs Climatology",
    "mse_anom_skill_clim": "Anomaly MSE Skill vs Climatology",
    "rmse_anom_skill_clim": "Anomaly RMSE Skill vs Climatology",
    "ens_member_mse_anom_skill_clim": "Pooled Ensemble-Member Anomaly MSE Skill vs Climatology",
    "mean_member_mse_anom_skill_clim": "Mean Member Anomaly MSE Skill vs Climatology",

    # Probabilistic / ensemble
    "ens_member_rmse": "Pooled Ensemble-Member RMSE",
    "mean_member_rmse": "Mean Member RMSE",
    "spread": "Ensemble Spread",
    "spread_skill_ratio": "Spread-Skill Ratio",
    "crps": "CRPS",
    "rank_histogram": "Rank Histogram",

    # Probabilistic / ensemble anomaly
    "ens_member_rmse_anom": "Pooled Ensemble-Member Anomaly RMSE",
    "mean_member_rmse_anom": "Mean Member Anomaly RMSE",
    "spread_anom": "Anomaly Ensemble Spread",
    "spread_anom_skill_ratio": "Anomaly Spread-Skill Ratio",
    "crps_anom": "Anomaly CRPS",
    "rank_histogram_anom": "Anomaly Rank Histogram",
    "roc_anom_lower": "ROC AUC, Lower Tercile Anomaly",
    "roc_anom_middle": "ROC AUC, Middle Tercile Anomaly",
    "roc_anom_upper": "ROC AUC, Upper Tercile Anomaly",
}

METRIC_UNITS = {
    # Error metrics
    "bias": "{unit}",
    "mae": "{unit}",
    "mse": "{unit}", # squaring handled in get_plot_unit_and_scale
    "rmse": "{unit}",
    "nrmse": "",

    # Correlation / variance metrics
    "corr": "",
    "r2": "",
    "fc_std": "{unit}",
    "an_std": "{unit}",
    "std_ratio": "",

    # Anomaly error metrics
    "bias_anom": "{unit}",
    "mae_anom": "{unit}",
    "mse_anom": "{unit}", # squaring handled in get_plot_unit_and_scale
    "rmse_anom": "{unit}",
    "nrmse_anom": "",

    # Anomaly correlation / variance metrics
    "acc": "",
    "r2_anom": "",
    "fc_anom_std": "{unit}",
    "an_anom_std": "{unit}",
    "std_ratio_anom": "",

    # Skill scores
    "mse_skill_clim": "%",
    "mae_anom_skill_clim": "%",
    "mse_anom_skill_clim": "%",
    "rmse_anom_skill_clim": "%",
    "ens_member_mse_anom_skill_clim": "%",
    "mean_member_rmse_anom_skill_clim": "%",

    # Ensemble metrics
    "ens_member_rmse": "{unit}",
    "mean_member_rmse": "{unit}",
    "spread": "{unit}",
    "spread_skill_ratio": "",
    "crps": "{unit}",
    "rank_histogram": "count",

    # Ensemble anomaly metrics
    "ens_member_rmse_anom": "{unit}",
    "mean_member_rmse_anom": "{unit}",
    "spread_anom": "{unit}",
    "spread_anom_skill_ratio": "",
    "crps_anom": "{unit}",
    "rank_histogram_anom": "count",

    # ROC
    "roc_anom_lower": "",
    "roc_anom_middle": "",
    "roc_anom_upper": "",
}

LOWER_BETTER = lambda o, c: safe_percent(o - c, o)
HIGHER_BETTER = lambda o, c: c - o
TARGET_ONE = lambda o, c: safe_percent(
    abs(o - 1.0) - abs(c - 1.0),
    abs(o - 1.0),
)

METRIC_IMPROVEMENT = {
    # Bias
    "bias": lambda o, c: safe_percent(abs(o) - abs(c), abs(o)),
    "bias_anom": lambda o, c: safe_percent(abs(o) - abs(c), abs(o)),

    # Lower-is-better deterministic
    "mae": LOWER_BETTER,
    "mse": LOWER_BETTER,
    "rmse": LOWER_BETTER,
    "nrmse": LOWER_BETTER,

    "mae_anom": LOWER_BETTER,
    "mse_anom": LOWER_BETTER,
    "rmse_anom": LOWER_BETTER,
    "nrmse_anom": LOWER_BETTER,

    # Higher-is-better correlations
    "corr": HIGHER_BETTER,
    "acc": HIGHER_BETTER,
    "r2": HIGHER_BETTER,
    "r2_anom": HIGHER_BETTER,

    # Skill scores
    "mse_skill_clim": HIGHER_BETTER,
    "mae_anom_skill_clim": HIGHER_BETTER,
    "mse_anom_skill_clim": HIGHER_BETTER,
    "rmse_anom_skill_clim": HIGHER_BETTER,
    "ens_member_mse_anom_skill_clim": HIGHER_BETTER,
    "mean_member_rmse_anom_skill_clim": HIGHER_BETTER,

    # Variance metrics: target ratio = 1
    "std_ratio": TARGET_ONE,
    "std_ratio_anom": TARGET_ONE,

    # Ensemble error metrics
    "ens_member_rmse": LOWER_BETTER,
    "mean_member_rmse": LOWER_BETTER,
    "ens_member_rmse_anom": LOWER_BETTER,
    "mean_member_rmse_anom": LOWER_BETTER,

    # Probabilistic scores
    "crps": LOWER_BETTER,
    "crps_anom": LOWER_BETTER,

    # Reliability metrics: target = 1
    "spread_skill_ratio": TARGET_ONE,
    "spread_anom_skill_ratio": TARGET_ONE,

    # ROC AUC: higher is better
    "roc_anom_lower": HIGHER_BETTER,
    "roc_anom_middle": HIGHER_BETTER,
    "roc_anom_upper": HIGHER_BETTER,
}

METRIC_SKILL_UNITS = {
    # Percent improvement
    "bias": "%",
    "bias_anom": "%",

    "mae": "%",
    "mse": "%",
    "rmse": "%",
    "nrmse": "%",

    "mae_anom": "%",
    "mse_anom": "%",
    "rmse_anom": "%",
    "nrmse_anom": "%",

    "ens_member_rmse": "%",
    "mean_member_rmse": "%",
    "ens_member_rmse_anom": "%",
    "mean_member_rmse_anom": "%",

    "crps": "%",
    "crps_anom": "%",

    "std_ratio": "%",
    "std_ratio_anom": "%",
    "spread_skill_ratio": "%",
    "spread_anom_skill_ratio": "%",

    # Absolute differences
    "corr": "Δ",
    "acc": "Δ",
    "r2": "Δ",
    "r2_anom": "Δ",

    "roc_anom_lower": "Δ",
    "roc_anom_middle": "Δ",
    "roc_anom_upper": "Δ",

    # Climatology skill scores
    "mse_skill_clim": "%",
    "mae_anom_skill_clim": "%",
    "mse_anom_skill_clim": "%",
    "rmse_anom_skill_clim": "%",
    "ens_member_mse_anom_skill_clim": "%",
    "mean_member_mse_anom_skill_clim": "%",
}


VARIABLE_UNITS = {
    "mslp": "Pa",
    "t2m": "K",
    "d2m": "K",
    "sst": "K",
    "u10": "m s-1",
    "v10": "m s-1",
    "tprate": "m s-1",
}

UNIT_CONVERSIONS = {
    "Pa": ("hPa", 100.0),
    "K": ("K", 1.0),
    "m": ("m", 1.0),
    # "m s-1": ("m s$^{-1}$", 1.0),

    # plot precipitation as mm/day
    "kg m-2 s-1": ("mm day$^{-1}$", 1 / 86400.0),
    "m s-1": ("mm day$^{-1}$", 1 / (86400.0 * 1000.0)),
}

VARIABLE_NAMES = {
    "t2m": "2m Temperature",
    "d2m": "2m Dew Point Temperature",
    "sst": "Sea Surface Temperature",
    "mslp": "Mean Sea Level Pressure",
    "u10": "10m Zonal Wind",
    "v10": "10m Meridional Wind",
    "tprate": "Precipitation Rate",
}


MODEL_COLORS = {
    "fc": "tab:blue",
    "global_fc": "tab:blue",
    "time_avg_fc": "tab:purple",
    "mlfc": "tab:green",
    "ConUS_mlfc": "tab:green",
    "World_mlfc": "tab:orange",
    "ConUS_global_mlfc": "tab:green",
    "World_global_mlfc": "tab:orange",
    "ConUS_time_avg_mlfc": "tab:red",
    "World_time_avg_mlfc": "tab:brown",
}

SERIES_COLORS = {
    "Forecast": "tab:blue",
    "Corrected forecast": "tab:green",
    "Analysis": "tab:orange",
    "Forecast anomaly": "tab:blue",
    "Corrected forecast anomaly": "tab:green",
    "Analysis anomaly": "tab:orange",
}

TRANSLATION_TABLE = str.maketrans({
    " ": "",
    "/": "_",
    ".": "p",
    ",": "-",
    "'": "",
    "(": "",
    ")": "",
})
