from typing import Literal

from .metrics import safe_percent
from .definitions import ImprovementUnit


# Absolute difference improvement.
# Positive always means corrected forecast is better.

TARGET_ZERO_DIFF = lambda o, c: abs(o) - abs(c)
LOWER_BETTER_DIFF = lambda o, c: o - c
HIGHER_BETTER_DIFF = lambda o, c: c - o
TARGET_ONE_DIFF = lambda o, c: abs(o - 1.0) - abs(c - 1.0)

METRIC_DIFFERENCE_IMPROVEMENT = {
    # Bias: target = 0
    "bias": TARGET_ZERO_DIFF,
    "bias_anom": TARGET_ZERO_DIFF,

    # Lower is better
    "mae": LOWER_BETTER_DIFF,
    "mse": LOWER_BETTER_DIFF,
    "rmse": LOWER_BETTER_DIFF,
    "nrmse": LOWER_BETTER_DIFF,

    "mae_anom": LOWER_BETTER_DIFF,
    "mse_anom": LOWER_BETTER_DIFF,
    "rmse_anom": LOWER_BETTER_DIFF,
    "nrmse_anom": LOWER_BETTER_DIFF,

    # Higher is better
    "corr": HIGHER_BETTER_DIFF,
    "acc": HIGHER_BETTER_DIFF,
    "r2": HIGHER_BETTER_DIFF,
    "r2_anom": HIGHER_BETTER_DIFF,
    "kendall_tau": HIGHER_BETTER_DIFF,
    "kendall_tau_anom": HIGHER_BETTER_DIFF,

    # Spatial-gradient errors: lower is better
    "grad_rmse": LOWER_BETTER_DIFF,
    "grad_rmse_anom": LOWER_BETTER_DIFF,

    # MSE decomposition: lower is better
    "mse_bias_component": LOWER_BETTER_DIFF,
    "mse_std_component": LOWER_BETTER_DIFF,
    "mse_corr_component": LOWER_BETTER_DIFF,
    "crmse": LOWER_BETTER_DIFF,

    "mse_bias_component_anom": LOWER_BETTER_DIFF,
    "mse_std_component_anom": LOWER_BETTER_DIFF,
    "mse_corr_component_anom": LOWER_BETTER_DIFF,
    "crmse_anom": LOWER_BETTER_DIFF,

    # Calibration: target = 1
    "regression_slope": TARGET_ONE_DIFF,
    "regression_slope_anom": TARGET_ONE_DIFF,

    # Skill scores: higher is better
    "mse_skill_clim": HIGHER_BETTER_DIFF,
    "rmse_skill_clim": HIGHER_BETTER_DIFF,
    "mae_skill_clim": HIGHER_BETTER_DIFF,

    "mse_anom_skill_clim": HIGHER_BETTER_DIFF,
    "rmse_anom_skill_clim": HIGHER_BETTER_DIFF,
    "mae_anom_skill_clim": HIGHER_BETTER_DIFF,

    "ens_member_mse_skill_clim": HIGHER_BETTER_DIFF,
    "mean_member_mse_skill_clim": HIGHER_BETTER_DIFF,
    "ens_member_mse_anom_skill_clim": HIGHER_BETTER_DIFF,
    "mean_member_mse_anom_skill_clim": HIGHER_BETTER_DIFF,

    # Variability ratios: target = 1
    "std_ratio": TARGET_ONE_DIFF,
    "std_ratio_anom": TARGET_ONE_DIFF,

    # Ensemble error metrics: lower is better
    "ens_member_rmse": LOWER_BETTER_DIFF,
    "mean_member_rmse": LOWER_BETTER_DIFF,
    "ens_member_rmse_anom": LOWER_BETTER_DIFF,
    "mean_member_rmse_anom": LOWER_BETTER_DIFF,

    # Probabilistic scores: lower is better
    "crps": LOWER_BETTER_DIFF,
    "crps_anom": LOWER_BETTER_DIFF,

    # Reliability: target = 1
    "spread_skill_ratio": TARGET_ONE_DIFF,
    "spread_anom_skill_ratio": TARGET_ONE_DIFF,

    # Brier score: lower is better
    "brier_lower": LOWER_BETTER_DIFF,
    "brier_middle": LOWER_BETTER_DIFF,
    "brier_upper": LOWER_BETTER_DIFF,

    "brier_anom_lower": LOWER_BETTER_DIFF,
    "brier_anom_middle": LOWER_BETTER_DIFF,
    "brier_anom_upper": LOWER_BETTER_DIFF,

    # ROC AUC: higher is better
    "roc_lower": HIGHER_BETTER_DIFF,
    "roc_middle": HIGHER_BETTER_DIFF,
    "roc_upper": HIGHER_BETTER_DIFF,

    "roc_anom_lower": HIGHER_BETTER_DIFF,
    "roc_anom_middle": HIGHER_BETTER_DIFF,
    "roc_anom_upper": HIGHER_BETTER_DIFF,
}

# Percentage improvement relative to distance from the optimum.

TARGET_ZERO_PERCENT = lambda o, c: safe_percent(
    abs(o) - abs(c),
    abs(o),
)
LOWER_BETTER_PERCENT = lambda o, c: safe_percent(
    o - c,
    o,
)
TARGET_ONE_PERCENT = lambda o, c: safe_percent(
    abs(o - 1.0) - abs(c - 1.0),
    abs(o - 1.0),
)

METRIC_PERCENTAGE_IMPROVEMENT = {
    # Bias: improvement relative to |bias|
    "bias": TARGET_ZERO_PERCENT,
    "bias_anom": TARGET_ZERO_PERCENT,

    # Lower is better
    "mae": LOWER_BETTER_PERCENT,
    "mse": LOWER_BETTER_PERCENT,
    "rmse": LOWER_BETTER_PERCENT,
    "nrmse": LOWER_BETTER_PERCENT,

    "mae_anom": LOWER_BETTER_PERCENT,
    "mse_anom": LOWER_BETTER_PERCENT,
    "rmse_anom": LOWER_BETTER_PERCENT,
    "nrmse_anom": LOWER_BETTER_PERCENT,

    # Spatial-gradient errors
    "grad_rmse": LOWER_BETTER_PERCENT,
    "grad_rmse_anom": LOWER_BETTER_PERCENT,

    # MSE decomposition
    "mse_bias_component": LOWER_BETTER_PERCENT,
    "mse_std_component": LOWER_BETTER_PERCENT,
    "mse_corr_component": LOWER_BETTER_PERCENT,
    "crmse": LOWER_BETTER_PERCENT,

    "mse_bias_component_anom": LOWER_BETTER_PERCENT,
    "mse_std_component_anom": LOWER_BETTER_PERCENT,
    "mse_corr_component_anom": LOWER_BETTER_PERCENT,
    "crmse_anom": LOWER_BETTER_PERCENT,

    # Calibration: improvement in distance from 1
    "regression_slope": TARGET_ONE_PERCENT,
    "regression_slope_anom": TARGET_ONE_PERCENT,

    # Variability ratios: improvement in distance from 1
    "std_ratio": TARGET_ONE_PERCENT,
    "std_ratio_anom": TARGET_ONE_PERCENT,

    # Ensemble error metrics
    "ens_member_rmse": LOWER_BETTER_PERCENT,
    "mean_member_rmse": LOWER_BETTER_PERCENT,
    "ens_member_rmse_anom": LOWER_BETTER_PERCENT,
    "mean_member_rmse_anom": LOWER_BETTER_PERCENT,

    # Probabilistic scores
    "crps": LOWER_BETTER_PERCENT,
    "crps_anom": LOWER_BETTER_PERCENT,

    # Reliability: improvement in distance from 1
    "spread_skill_ratio": TARGET_ONE_PERCENT,
    "spread_anom_skill_ratio": TARGET_ONE_PERCENT,

    # Brier score
    "brier_lower": LOWER_BETTER_PERCENT,
    "brier_middle": LOWER_BETTER_PERCENT,
    "brier_upper": LOWER_BETTER_PERCENT,

    "brier_anom_lower": LOWER_BETTER_PERCENT,
    "brier_anom_middle": LOWER_BETTER_PERCENT,
    "brier_anom_upper": LOWER_BETTER_PERCENT,

    # No percentage form for correlation / Kendall tau / R² /
    # skill scores / ROC AUC
    # Their natural improvement is an absolute difference (Δ)
}


METRIC_IMPROVEMENT_UNITS: dict[
    str,
    tuple[ImprovementUnit, ...],
] = {
    # -------------------------------------------------------------
    # Bias
    # -------------------------------------------------------------
    "bias": ("%", "Δ", "normalized"),
    "bias_anom": ("%", "Δ", "normalized"),

    # -------------------------------------------------------------
    # Deterministic errors
    # -------------------------------------------------------------
    "mae": ("%", "Δ", "normalized"),
    "mse": ("%", "Δ", "normalized"),
    "rmse": ("%", "Δ", "normalized"),

    "mae_anom": ("%", "Δ", "normalized"),
    "mse_anom": ("%", "Δ", "normalized"),
    "rmse_anom": ("%", "Δ", "normalized"),

    # Already normalized by construction.
    "nrmse": ("%", "Δ"),
    "nrmse_anom": ("%", "Δ"),

    # -------------------------------------------------------------
    # Standard deviation
    #
    # Target is the corresponding analysis standard deviation:
    #
    #   |std_fc - std_an| - |std_mlfc - std_an|
    #
    # Positive = corrected forecast is closer to analysis variability.
    # -------------------------------------------------------------
    "fc_std": ("%", "Δ", "normalized"),
    "fc_anom_std": ("%", "Δ", "normalized"),

    # -------------------------------------------------------------
    # Correlation / explained variance
    # -------------------------------------------------------------
    "corr": ("Δ",),
    "kendall_tau": ("Δ",),
    "acc": ("Δ",),
    "kendall_tau_anom": ("Δ",),
    "r2": ("Δ",),
    "r2_anom": ("Δ",),

    # -------------------------------------------------------------
    # Calibration / ratios
    # -------------------------------------------------------------
    "std_ratio": ("%", "Δ"),
    "std_ratio_anom": ("%", "Δ"),

    "regression_slope": ("%", "Δ"),
    "regression_slope_anom": ("%", "Δ"),

    "spread_skill_ratio": ("%", "Δ"),
    "spread_anom_skill_ratio": ("%", "Δ"),

    # -------------------------------------------------------------
    # MSE decomposition
    # -------------------------------------------------------------
    "mse_bias_component": ("%", "Δ", "normalized"),
    "mse_std_component": ("%", "Δ", "normalized"),
    "mse_corr_component": ("%", "Δ", "normalized"),
    "crmse": ("%", "Δ", "normalized"),

    "mse_bias_component_anom": ("%", "Δ", "normalized"),
    "mse_std_component_anom": ("%", "Δ", "normalized"),
    "mse_corr_component_anom": ("%", "Δ", "normalized"),
    "crmse_anom": ("%", "Δ", "normalized"),

    # -------------------------------------------------------------
    # Spatial gradients
    # -------------------------------------------------------------
    "grad_rmse": ("%", "Δ"),
    "grad_rmse_anom": ("%", "Δ"),

    # -------------------------------------------------------------
    # Ensemble errors
    # -------------------------------------------------------------
    "ens_member_rmse": ("%", "Δ", "normalized"),
    "mean_member_rmse": ("%", "Δ", "normalized"),

    "ens_member_rmse_anom": ("%", "Δ", "normalized"),
    "mean_member_rmse_anom": ("%", "Δ", "normalized"),

    "crps": ("%", "Δ", "normalized"),
    "crps_anom": ("%", "Δ", "normalized"),

    # -------------------------------------------------------------
    # Skill scores
    # -------------------------------------------------------------
    "mse_skill_clim": ("Δ",),
    "rmse_skill_clim": ("Δ",),
    "mae_skill_clim": ("Δ",),

    "mse_anom_skill_clim": ("Δ",),
    "rmse_anom_skill_clim": ("Δ",),
    "mae_anom_skill_clim": ("Δ",),

    "ens_member_mse_skill_clim": ("Δ",),
    "mean_member_mse_skill_clim": ("Δ",),
    "ens_member_mse_anom_skill_clim": ("Δ",),
    "mean_member_mse_anom_skill_clim": ("Δ",),

    # -------------------------------------------------------------
    # Brier score
    # -------------------------------------------------------------
    "brier_lower": ("%", "Δ"),
    "brier_middle": ("%", "Δ"),
    "brier_upper": ("%", "Δ"),

    "brier_anom_lower": ("%", "Δ"),
    "brier_anom_middle": ("%", "Δ"),
    "brier_anom_upper": ("%", "Δ"),

    # -------------------------------------------------------------
    # ROC AUC
    # -------------------------------------------------------------
    "roc_lower": ("Δ",),
    "roc_middle": ("Δ",),
    "roc_upper": ("Δ",),

    "roc_anom_lower": ("Δ",),
    "roc_anom_middle": ("Δ",),
    "roc_anom_upper": ("Δ",),
    }


NORMALIZED_IMPROVEMENT_REFERENCE: dict[str, tuple[str, int]] = {
    # Raw-field metrics: normalize by analysis standard deviation.
    "bias": ("an_std", 1),
    "mae": ("an_std", 1),
    "rmse": ("an_std", 1),
    "crmse": ("an_std", 1),

    # Forecast variability: target and normalization reference = analysis std.
    "fc_std": ("an_std", 1),

    # Squared metrics: normalize by analysis variance.
    "mse": ("an_std", 2),
    "mse_bias_component": ("an_std", 2),
    "mse_std_component": ("an_std", 2),
    "mse_corr_component": ("an_std", 2),

    # Anomaly metrics.
    "bias_anom": ("an_anom_std", 1),
    "mae_anom": ("an_anom_std", 1),
    "rmse_anom": ("an_anom_std", 1),
    "crmse_anom": ("an_anom_std", 1),

    # Forecast anomaly variability.
    "fc_anom_std": ("an_anom_std", 1),

    "mse_anom": ("an_anom_std", 2),
    "mse_bias_component_anom": ("an_anom_std", 2),
    "mse_std_component_anom": ("an_anom_std", 2),
    "mse_corr_component_anom": ("an_anom_std", 2),

    # Ensemble / probabilistic errors.
    "ens_member_rmse": ("an_std", 1),
    "mean_member_rmse": ("an_std", 1),
    "crps": ("an_std", 1),

    "ens_member_rmse_anom": ("an_anom_std", 1),
    "mean_member_rmse_anom": ("an_anom_std", 1),
    "crps_anom": ("an_anom_std", 1),
}
