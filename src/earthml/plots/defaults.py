import matplotlib.pyplot as plt

from .colormaps import (
    PiBRdY,
    SeqWRdY,
    SeqPiBRdY,
    SeqBPi,
    SeqBYRd,
)

DEFAULT_PLOT_CONFIG = {
    # Orography diagnostics
    "orography": {
        "vmin": 0,
        "vmax": 4000,
        "ticks": [0, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "orography_grad_mag": {
        "vmin": 0,
        "vmax": 0.1,
        "ticks": [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1],
        "cmap": SeqWRdY,
        "scale_units": False,
    },
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
        "vmin": -20,
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
    "kendall_tau": {
        "vmin": -1,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "scc": {
        "vmin": -1,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },

    "kendall_tau_anom": {
        "vmin": -1,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "scc_anom": {
        "vmin": -1,
        "vmax": 1,
        "ticks": 21,
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
        "vmin": 0.05,
        "vmax": 1/0.05,
        "ticks": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1/0.95, 1/0.9, 1/0.8, 1/0.7, 1/0.6, 1/0.5, 1/0.4, 1/0.3, 1/0.2, 1/0.1, 1/0.05],
        "cmap": PiBRdY,
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
        "vmax": 1/0.1,
        "ticks": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1/0.95, 1/0.9, 1/0.8, 1/0.7, 1/0.6, 1/0.5, 1/0.4, 1/0.3, 1/0.2, 1/0.1],
        "cmap": PiBRdY,
        "scale_units": False,
    },

    # MSE decomposition / calibration diagnostics
    "mse_bias_component": {
        "vmin": 0,
        "vmax": 81,
        "ticks": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mse_std_component": {
        "vmin": 0,
        "vmax": 81,
        "ticks": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mse_corr_component": {
        "vmin": 0,
        "vmax": 81,
        "ticks": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "crmse": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "regression_slope": {
        "vmin": 0.0,
        "vmax": 2.0,
        "ticks": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0],
        "cmap": PiBRdY,
        "scale_units": False,
    },

    # Spatial-gradient diagnostics
    "fc_grad_mag": {
        "vmin": 0,
        "vmax": 1e-4,
        "ticks": [0, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4],
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "an_grad_mag": {
        "vmin": 0,
        "vmax": 1e-4,
        "ticks": [0, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4],
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "grad_rmse": {
        "vmin": 0,
        "vmax": 1e-4,
        "ticks": [0, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4],
        "cmap": SeqWRdY,
        "scale_units": False,
    },

    "fc_anom_grad_mag": {
        "vmin": 0,
        "vmax": 1e-4,
        "ticks": [0, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4],
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "an_anom_grad_mag": {
        "vmin": 0,
        "vmax": 1e-4,
        "ticks": [0, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4],
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "grad_rmse_anom": {
        "vmin": 0,
        "vmax": 1e-4,
        "ticks": [0, 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4],
        "cmap": SeqWRdY,
        "scale_units": False,
    },

    # Anomaly MSE decomposition / calibration diagnostics
    "mse_bias_component_anom": {
        "vmin": 0,
        "vmax": 81,
        "ticks": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mse_std_component_anom": {
        "vmin": 0,
        "vmax": 81,
        "ticks": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "mse_corr_component_anom": {
        "vmin": 0,
        "vmax": 81,
        "ticks": [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "crmse_anom": {
        "vmin": 0,
        "vmax": 9,
        "ticks": [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cmap": SeqWRdY,
        "scale_units": True,
    },
    "regression_slope_anom": {
        "vmin": 0.0,
        "vmax": 2.0,
        "ticks": [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0],
        "cmap": PiBRdY,
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
    "mae_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1,
                0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "rmse_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1,
                0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
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
    "ens_member_mse_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1,
                0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "mean_member_mse_skill_clim": {
        "vmin": -4,
        "vmax": 1,
        "ticks": [-4, -3, -2, -1.5, -0.9, -0.7, -0.5, -0.3, -0.2, -0.1,
                0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
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

    # Brier score
    "brier_lower": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "brier_middle": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "brier_upper": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqWRdY,
        "scale_units": False,
    },

    "brier_anom_lower": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "brier_anom_middle": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqWRdY,
        "scale_units": False,
    },
    "brier_anom_upper": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqWRdY,
        "scale_units": False,
    },

    # ROC AUC
    "roc_lower": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "roc_middle": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
    "roc_upper": {
        "vmin": 0,
        "vmax": 1,
        "ticks": 21,
        "cmap": SeqPiBRdY,
        "scale_units": False,
    },
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
        "vmin": -100,
        "vmax": 100,
        "ticks": [-100, -90, -80, -70, -60, -50, -40, -30, -20, -15, -10, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "cmap": plt.get_cmap("RdBu"),
    },
    "Δ": {
        "vmin": -1,
        "vmax": 1,
        "ticks": [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "cmap": plt.get_cmap("RdBu"),
    },
    "normalized": {
        "vmin": -0.5,
        "vmax": 0.5,
        "ticks": [-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        "cmap": plt.get_cmap("RdBu"),
    },
}


SQUARED_METRICS = {
    "mse",
    "mse_anom",
    "mse_bias_component",
    "mse_std_component",
    "mse_corr_component",
    "mse_bias_component_anom",
    "mse_std_component_anom",
    "mse_corr_component_anom",
}


METRIC_NAMES = {
    # Orography diagnostics
    "orography": "Orography",
    "orography_grad_mag": "Orography Gradient Magnitude",
    # Deterministic
    "bias": "Bias",
    "mae": "MAE",
    "mse": "MSE",
    "mse_skill_clim": "MSE Skill vs Climatology",
    "rmse": "RMSE",
    "nrmse": "Normalized RMSE",
    "r2": "R²",
    # Correlation
    "kendall_tau": "Kendall's τ",
    "scc": "Spatial Correlation",
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
    # Anomaly correlation
    "kendall_tau_anom": "Anomaly Kendall's τ",
    "scc_anom": "Anomaly Spatial Correlation",
    "acc": "ACC",
    "r2_anom": "Anomaly R²",
    "fc_anom_std": "Forecast Anomaly STD",
    "an_anom_std": "Analysis Anomaly STD",
    "std_ratio_anom": "Anomaly STD Ratio",

    # Spatial-gradient diagnostics
    "fc_grad_mag": "Forecast Gradient Magnitude",
    "an_grad_mag": "Analysis Gradient Magnitude",
    "grad_rmse": "Gradient RMSE",

    "fc_anom_grad_mag": "Forecast Anomaly Gradient Magnitude",
    "an_anom_grad_mag": "Analysis Anomaly Gradient Magnitude",
    "grad_rmse_anom": "Anomaly Gradient RMSE",

    # MSE decomposition / calibration
    "mse_bias_component": "MSE Bias Component",
    "mse_std_component": "MSE STD Component",
    "mse_corr_component": "MSE Correlation Component",
    "crmse": "Centered RMSE",
    "regression_slope": "Regression Slope",

    # Anomaly MSE decomposition / calibration
    "mse_bias_component_anom": "Anomaly MSE Bias Component",
    "mse_std_component_anom": "Anomaly MSE STD Component",
    "mse_corr_component_anom": "Anomaly MSE Correlation Component",
    "crmse_anom": "Anomaly Centered RMSE",
    "regression_slope_anom": "Anomaly Regression Slope",

    # Anomaly skill vs climatology
    "mae_anom_skill_clim": "Anomaly MAE Skill vs Climatology",
    "mse_anom_skill_clim": "Anomaly MSE Skill vs Climatology",
    "rmse_anom_skill_clim": "Anomaly RMSE Skill vs Climatology",
    "ens_member_mse_anom_skill_clim": "Pooled Ensemble-Member Anomaly MSE Skill vs Climatology",
    "mean_member_mse_anom_skill_clim": "Mean Member Anomaly MSE Skill vs Climatology",
    "mae_skill_clim": "MAE Skill vs Climatology",
    "rmse_skill_clim": "RMSE Skill vs Climatology",
    "ens_member_mse_skill_clim": "Pooled Ensemble-Member MSE Skill vs Climatology",
    "mean_member_mse_skill_clim": "Mean Member MSE Skill vs Climatology",

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

    # Brier
    "brier_lower": "Brier Score, Lower Tercile",
    "brier_middle": "Brier Score, Middle Tercile",
    "brier_upper": "Brier Score, Upper Tercile",

    "brier_anom_lower": "Brier Score, Lower Tercile Anomaly",
    "brier_anom_middle": "Brier Score, Middle Tercile Anomaly",
    "brier_anom_upper": "Brier Score, Upper Tercile Anomaly",

    # ROC
    "roc_lower": "ROC AUC, Lower Tercile",
    "roc_middle": "ROC AUC, Middle Tercile",
    "roc_upper": "ROC AUC, Upper Tercile",

    "roc_anom_lower": "ROC AUC, Lower Tercile Anomaly",
    "roc_anom_middle": "ROC AUC, Middle Tercile Anomaly",
    "roc_anom_upper": "ROC AUC, Upper Tercile Anomaly",

    # Power spectra
    "fc_power_spectrum": "Forecast Power Spectrum",
    "an_power_spectrum": "Analysis Power Spectrum",
    "power_spectrum_ratio": "Forecast / Analysis Power Spectrum Ratio",

    "fc_isotropic_power_spectrum": "Forecast Isotropic Power Spectrum",
    "an_isotropic_power_spectrum": "Analysis Isotropic Power Spectrum",

    "fc_anom_power_spectrum": "Forecast Anomaly Power Spectrum",
    "an_anom_power_spectrum": "Analysis Anomaly Power Spectrum",
    "power_spectrum_ratio_anom": "Forecast / Analysis Anomaly Power Spectrum Ratio",

    "fc_anom_isotropic_power_spectrum": "Forecast Anomaly Isotropic Power Spectrum",
    "an_anom_isotropic_power_spectrum": "Analysis Anomaly Isotropic Power Spectrum",
}

METRIC_UNITS = {
    # Orography diagnostics
    "orography": "m",
    "orography_grad_mag": "",

    # Error metrics
    "bias": "{unit}",
    "mae": "{unit}",
    "mse": "{unit}", # squaring handled in get_plot_unit_and_scale
    "rmse": "{unit}",
    "nrmse": "",

    # Correlation / variance metrics
    "corr": "",
    "kendall_tau": "",
    "scc": "",
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
    "kendall_tau_anom": "",
    "scc_anom": "",
    "r2_anom": "",
    "fc_anom_std": "{unit}",
    "an_anom_std": "{unit}",
    "std_ratio_anom": "",

    # Spatial-gradient diagnostics
    "fc_grad_mag": "{unit} m$^{-1}$",
    "an_grad_mag": "{unit} m$^{-1}$",
    "grad_rmse": "{unit} m$^{-1}$",

    "fc_anom_grad_mag": "{unit} m$^{-1}$",
    "an_anom_grad_mag": "{unit} m$^{-1}$",
    "grad_rmse_anom": "{unit} m$^{-1}$",

    # MSE decomposition / calibration
    "mse_bias_component": "{unit}",
    "mse_std_component": "{unit}",
    "mse_corr_component": "{unit}",
    "crmse": "{unit}",
    "regression_slope": "",

    # Anomaly MSE decomposition / calibration
    "mse_bias_component_anom": "{unit}",
    "mse_std_component_anom": "{unit}",
    "mse_corr_component_anom": "{unit}",
    "crmse_anom": "{unit}",
    "regression_slope_anom": "",

    # Skill scores
    "mae_skill_clim": "",
    "rmse_skill_clim": "",
    "mse_skill_clim": "",
    "mae_anom_skill_clim": "",
    "mse_anom_skill_clim": "",
    "rmse_anom_skill_clim": "",
    "ens_member_mse_anom_skill_clim": "",
    "mean_member_rmse_anom_skill_clim": "",
    "ens_member_mse_skill_clim": "",
    "mean_member_mse_skill_clim": "",

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

    # Brier
    "brier_lower": "",
    "brier_middle": "",
    "brier_upper": "",
    "brier_anom_lower": "",
    "brier_anom_middle": "",
    "brier_anom_upper": "",

    # ROC
    "roc_lower": "",
    "roc_middle": "",
    "roc_upper": "",
    "roc_anom_lower": "",
    "roc_anom_middle": "",
    "roc_anom_upper": "",

    # Power spectrum
    "fc_power_spectrum": "",
    "an_power_spectrum": "",
    "fc_isotropic_power_spectrum": "",
    "an_isotropic_power_spectrum": "",
    "power_spectrum_ratio": "",

    "fc_anom_power_spectrum": "",
    "an_anom_power_spectrum": "",
    "fc_anom_isotropic_power_spectrum": "",
    "an_anom_isotropic_power_spectrum": "",
    "power_spectrum_ratio_anom": "",
}


VARIABLE_UNITS = {
    "mslp": "Pa",
    "t2m": "K",
    "d2m": "K",
    "sst": "K",
    "ssh": "m",
    "tcc": "fraction",
    "u10": "m s-1",
    "v10": "m s-1",
    "tprate": "m s-1 (tprate)",
}

UNIT_CONVERSIONS = {
    "Pa": ("hPa", 100.0),
    "K": ("K", 1.0),
    "m": ("m", 1.0),
    "m s-1": {
        "u10": ("m s$^{-1}$", 1.0),
        "v10": ("m s$^{-1}$", 1.0),
        "tprate": ("mm day$^{-1}$", 1 / (86400.0 * 1000.0)),
    },
    # plot precipitation as mm/day
    "kg m-2 s-1": ("mm day$^{-1}$", 1 / 86400.0),
}

VARIABLE_NAMES = {
    "t2m": "2m Temperature",
    "d2m": "2m Dew Point Temperature",
    "sst": "Sea Surface Temperature",
    "tcc": "Total Cloud Cover",
    "mslp": "Mean Sea Level Pressure",
    "u10": "10m Zonal Wind",
    "v10": "10m Meridional Wind",
    "tprate": "Precipitation Rate",
    "ssh": "Sea Surface Height",
}


MODEL_COLORS = {
    "fc": "tab:blue",
    "clim-fc": "tab:orange",

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
    "Clim-corrected forecast": "tab:gray",
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
