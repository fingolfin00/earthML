from typing import Literal
from enum import StrEnum


MetricKind = Literal["scalar", "maps", "time_lon", "time_lat", "timeseries"]
LeadtimeAgg = Literal["single", "aggregated", "seasonal_window"]
RealizationAgg = Literal["member", "ensemble_mean"]
MetricAgg = Literal[
    "global",       # sqrt(mean(error^2))
    "spatial_avg",  # mean(point-wise RMSE)
    "spatial_rmse", # sqrt(mean(point-wise MSE))
]
ImprovementUnit = Literal["%", "Δ", "normalized"]


class Metric(StrEnum):
    # ------------------------------------------------------
    # Orography
    # ------------------------------------------------------

    OROGRAPHY = "orography"
    OROGRAPHY_GRAD_MAG = "orography_grad_mag"

    # ------------------------------------------------------
    # Deterministic
    # ------------------------------------------------------

    # Error
    BIAS = "bias"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    # Normalized
    R2 = "r2"
    NMSE = "nmse"
    NRMSE = "nrmse"
    # Temporal
    CORR = "corr"
    KENDALL_TAU = "kendall_tau"
    FC_STD = "fc_std"
    AN_STD = "an_std"
    STD_RATIO = "std_ratio"
    # MSE decomposition
    MSE_BIAS_COMPONENT = "mse_bias_component"
    MSE_STD_COMPONENT = "mse_std_component"
    MSE_CORR_COMPONENT = "mse_corr_component"
    CRMSE = "crmse"
    REGRESSION_SLOPE = "regression_slope"
    # Gradients
    FC_GRAD_MAG = "fc_grad_mag"
    AN_GRAD_MAG = "an_grad_mag"
    GRAD_RMSE = "grad_rmse"
    # Spatial only
    SCC = "spatial_corr"
    # Power spectra
    FC_POWER_SPECTRUM = "fc_power_spectrum"
    AN_POWER_SPECTRUM = "an_power_spectrum"
    FC_ISOTROPIC_POWER_SPECTRUM = "fc_isotropic_power_spectrum"
    AN_ISOTROPIC_POWER_SPECTRUM = "an_sotropic_power_spectrum"
    POWER_SPECTRUM_RATIO = "power_spectrum_ratio"

    # ------------------------------------------------------
    # Probabilistic
    # ------------------------------------------------------

    # Ens member RMSEs
    ENS_MEMBER_RMSE = "ens_member_rmse"
    MEAN_MEMBER_RMSE = "mean_member_rmse"
    # Anomaly spread
    SPREAD = "spread"
    SPREAD_SKILL_RATIO = "spread_skill_ratio"
    # CRPS
    CRPS = "crps"
    # Rank hist
    RANK_HISTOGRAM = "rank_histogram"
    # Brier's score
    BRIER_UPPER = "brier_upper"
    BRIER_MIDDLE = "brier_middle"
    BRIER_LOWER = "brier_lower"
    # ROC
    ROC_UPPER = "roc_upper"
    ROC_MIDDLE = "roc_middle"
    ROC_LOWER = "roc_lower"
    # Skills
    ENS_MEMBER_MSE_SKILL_CLIM = "ens_member_mse_skill_clim"
    MEAN_MEMBER_MSE_SKILL_CLIM = "mean_member_mse_skill_clim"

    # ------------------------------------------------------
    # Anomaly deterministic
    # ------------------------------------------------------

    # Anomaly error
    BIAS_ANOM = "bias_anom"
    MAE_ANOM = "mae_anom"
    RMSE_ANOM = "rmse_anom"
    MSE_ANOM = "mse_anom"
    # Anomaly normalized
    R2_ANOM = "r2_anom"
    NMSE_ANOM = "nmse_anom"
    NRMSE_ANOM = "nrmse_anom"
    # Anomaly temporal
    ACC = "acc"
    KENDALL_TAU_ANOM = "kendall_tau_anom"
    FC_ANOM_STD = "fc_anom_std"
    AN_ANOM_STD = "an_anom_std"
    STD_RATIO_ANOM = "std_ratio_anom"
    # Anomaly MSE decomposition
    MSE_BIAS_COMPONENT_ANOM = "mse_bias_component_anom"
    MSE_STD_COMPONENT_ANOM = "mse_std_component_anom"
    MSE_CORR_COMPONENT_ANOM = "mse_corr_component_anom"
    CRMSE_ANOM = "crmse_anom"
    REGRESSION_SLOPE_ANOM = "regression_slope_anom"
    # Gradients
    FC_ANOM_GRAD_MAG = "fc_anom_grad_mag"
    AN_ANOM_GRAD_MAG = "an_anom_grad_mag"
    GRAD_RMSE_ANOM = "grad_rmse_anom"
    # Spatial only
    SCC_ANOM = "spatial_corr_anom"
    # Anomaly power spectra
    FC_ANOM_POWER_SPECTRUM = "fc_anom_power_spectrum"
    AN_ANOM_POWER_SPECTRUM = "an_anom_power_spectrum"
    FC_ANOM_ISOTROPIC_POWER_SPECTRUM = "fc_anom_isotropic_power_spectrum"
    AN_ANOM_ISOTROPIC_POWER_SPECTRUM = "an_anom_isotropic_power_spectrum"
    POWER_SPECTRUM_RATIO_ANOM = "power_spectrum_ratio_anom"
    # Skills vs climatology
    MSE_SKILL_CLIM = "mse_skill_clim"
    RMSE_SKILL_CLIM = "mse_skill_clim"
    MAE_SKILL_CLIM = "mse_skill_clim"
    # Anomaly skills vs climatology
    MSE_ANOM_SKILL_CLIM = "mse_anom_skill_clim"
    RMSE_ANOM_SKILL_CLIM = "rmse_anom_skill_clim"
    MAE_ANOM_SKILL_CLIM = "mae_anom_skill_clim"

    # ------------------------------------------------------
    # Anomaly probabilistic
    # ------------------------------------------------------

    # Ens member anomaly RMSEs
    ENS_MEMBER_RMSE_ANOM = "ens_member_rmse_anom"
    MEAN_MEMBER_RMSE_ANOM = "mean_member_rmse_anom"
    # Anomaly spread
    SPREAD_ANOM = "spread_anom"
    SPREAD_ANOM_SKILL_RATIO = "spread_anom_skill_ratio"
    # Anomaly CRPS
    CRPS_ANOM = "crps_anom"
    # Anomaly rank hist
    RANK_HISTOGRAM_ANOM = "rank_histogram_anom"
    # Anomaly Brier's score
    BRIER_ANOM_UPPER = "brier_anom_upper"
    BRIER_ANOM_MIDDLE = "brier_anom_middle"
    BRIER_ANOM_LOWER = "brier_anom_lower"
    # Anomaly ROC
    ROC_ANOM_UPPER = "roc_anom_upper"
    ROC_ANOM_MIDDLE = "roc_anom_middle"
    ROC_ANOM_LOWER = "roc_anom_lower"
    # Skills
    ENS_MEMBER_MSE_ANOM_SKILL_CLIM = "ens_member_mse_anom_skill_clim"
    MEAN_MEMBER_MSE_ANOM_SKILL_CLIM = "mean_member_mse_anom_skill_clim"


DETERMINISTIC_METRICS: set[Metric] = {
    Metric.OROGRAPHY,
    Metric.OROGRAPHY_GRAD_MAG,

    Metric.BIAS,
    Metric.MAE,
    Metric.MSE,
    Metric.RMSE,
    Metric.NMSE,
    Metric.NRMSE,
    Metric.R2,
    Metric.CORR,
    Metric.KENDALL_TAU,
    Metric.FC_STD,
    Metric.AN_STD,
    Metric.STD_RATIO,

    Metric.MSE_BIAS_COMPONENT,
    Metric.MSE_STD_COMPONENT,
    Metric.MSE_CORR_COMPONENT,
    Metric.CRMSE,
    Metric.REGRESSION_SLOPE,

    Metric.FC_GRAD_MAG,
    Metric.AN_GRAD_MAG,
    Metric.GRAD_RMSE,

    Metric.SCC,

    Metric.FC_POWER_SPECTRUM,
    Metric.AN_POWER_SPECTRUM,
    Metric.POWER_SPECTRUM_RATIO,

    Metric.BIAS_ANOM,
    Metric.MAE_ANOM,
    Metric.MSE_ANOM,
    Metric.RMSE_ANOM,
    Metric.NMSE_ANOM,
    Metric.NRMSE_ANOM,
    Metric.ACC,
    Metric.KENDALL_TAU_ANOM,
    Metric.R2_ANOM,
    Metric.FC_ANOM_STD,
    Metric.AN_ANOM_STD,
    Metric.STD_RATIO_ANOM,

    Metric.MSE_BIAS_COMPONENT_ANOM,
    Metric.MSE_STD_COMPONENT_ANOM,
    Metric.MSE_CORR_COMPONENT_ANOM,
    Metric.CRMSE_ANOM,
    Metric.REGRESSION_SLOPE_ANOM,

    Metric.FC_ANOM_GRAD_MAG,
    Metric.AN_ANOM_GRAD_MAG,
    Metric.GRAD_RMSE_ANOM,

    Metric.SCC_ANOM,

    Metric.FC_ANOM_POWER_SPECTRUM,
    Metric.AN_ANOM_POWER_SPECTRUM,
    Metric.POWER_SPECTRUM_RATIO_ANOM,

    Metric.MSE_SKILL_CLIM,
    Metric.RMSE_SKILL_CLIM,
    Metric.MAE_SKILL_CLIM,
    Metric.MSE_ANOM_SKILL_CLIM,
    Metric.RMSE_ANOM_SKILL_CLIM,
    Metric.MAE_ANOM_SKILL_CLIM,
}

PROBABILISTIC_METRICS: set[Metric] = {
    Metric.ENS_MEMBER_RMSE,
    Metric.MEAN_MEMBER_RMSE,
    Metric.ENS_MEMBER_MSE_SKILL_CLIM,
    Metric.MEAN_MEMBER_MSE_SKILL_CLIM,

    Metric.SPREAD,
    Metric.SPREAD_SKILL_RATIO,

    Metric.CRPS,

    Metric.RANK_HISTOGRAM,

    Metric.BRIER_UPPER,
    Metric.BRIER_MIDDLE,
    Metric.BRIER_LOWER,

    Metric.ROC_UPPER,
    Metric.ROC_MIDDLE,
    Metric.ROC_LOWER,

    Metric.ENS_MEMBER_RMSE_ANOM,
    Metric.MEAN_MEMBER_RMSE_ANOM,
    Metric.ENS_MEMBER_MSE_ANOM_SKILL_CLIM,
    Metric.MEAN_MEMBER_MSE_ANOM_SKILL_CLIM,

    Metric.SPREAD_ANOM,
    Metric.SPREAD_ANOM_SKILL_RATIO,

    Metric.CRPS_ANOM,

    Metric.RANK_HISTOGRAM_ANOM,

    Metric.BRIER_ANOM_UPPER,
    Metric.BRIER_ANOM_MIDDLE,
    Metric.BRIER_ANOM_LOWER,

    Metric.ROC_ANOM_UPPER,
    Metric.ROC_ANOM_MIDDLE,
    Metric.ROC_ANOM_LOWER,
}

POWER_METRICS: set[Metric] = {
    Metric.AN_POWER_SPECTRUM,
    Metric.FC_POWER_SPECTRUM,
    Metric.POWER_SPECTRUM_RATIO,
    Metric.AN_ISOTROPIC_POWER_SPECTRUM,
    Metric.FC_ISOTROPIC_POWER_SPECTRUM,

    Metric.AN_ANOM_POWER_SPECTRUM,
    Metric.FC_ANOM_POWER_SPECTRUM,
    Metric.POWER_SPECTRUM_RATIO_ANOM,
    Metric.AN_ANOM_ISOTROPIC_POWER_SPECTRUM,
    Metric.FC_ANOM_ISOTROPIC_POWER_SPECTRUM,
}
