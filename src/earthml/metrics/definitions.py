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
    # Orography
    OROGRAPHY = "orography"
    OROGRAPHY_GRAD_MAG = "orography_grad_mag"

    # Deterministic
    BIAS = "bias"
    MAE = "mae"
    MSE = "mse"
    MSE_SKILL_CLIM = "mse_skill_clim"
    RMSE = "rmse"
    NMSE = "nmse"
    NRMSE = "nrmse"
    R2 = "r2"
    CORR = "corr"
    FC_STD = "fc_std"
    AN_STD = "an_std"
    STD_RATIO = "std_ratio"

    MSE_BIAS_COMPONENT = "mse_bias_component"
    MSE_STD_COMPONENT = "mse_std_component"
    MSE_CORR_COMPONENT = "mse_corr_component"
    CRMSE = "crmse"
    REGRESSION_SLOPE = "regression_slope"

    FC_GRAD_MAG = "fc_grad_mag"
    AN_GRAD_MAG = "an_grad_mag"
    GRAD_RMSE = "grad_rmse"

    BIAS_ANOM = "bias_anom"
    MAE_ANOM = "mae_anom"
    RMSE_ANOM = "rmse_anom"
    MSE_ANOM = "mse_anom"
    NMSE_ANOM = "nmse_anom"
    NRMSE_ANOM = "nrmse_anom"
    ACC = "acc"
    R2_ANOM = "r2_anom"
    FC_ANOM_STD = "fc_anom_std"
    AN_ANOM_STD = "an_anom_std"
    STD_RATIO_ANOM = "std_ratio_anom"

    MSE_BIAS_COMPONENT_ANOM = "mse_bias_component_anom"
    MSE_STD_COMPONENT_ANOM = "mse_std_component_anom"
    MSE_CORR_COMPONENT_ANOM = "mse_corr_component_anom"
    CRMSE_ANOM = "crmse_anom"
    REGRESSION_SLOPE_ANOM = "regression_slope_anom"

    FC_ANOM_GRAD_MAG = "fc_anom_grad_mag"
    AN_ANOM_GRAD_MAG = "an_anom_grad_mag"
    GRAD_RMSE_ANOM = "grad_rmse_anom"

    MAE_ANOM_SKILL_CLIM = "mae_anom_skill_clim"
    MSE_ANOM_SKILL_CLIM = "mse_anom_skill_clim"
    RMSE_ANOM_SKILL_CLIM = "rmse_anom_skill_clim"

    # Probabilistic
    ENS_MEMBER_RMSE = "ens_member_rmse"
    MEAN_MEMBER_RMSE = "mean_member_rmse"
    SPREAD = "spread"
    SPREAD_SKILL_RATIO = "spread_skill_ratio"
    CRPS = "crps"
    RANK_HISTOGRAM = "rank_histogram"

    ENS_MEMBER_RMSE_ANOM = "ens_member_rmse_anom"
    MEAN_MEMBER_RMSE_ANOM = "mean_member_rmse_anom"
    ENS_MEMBER_MSE_ANOM_SKILL_CLIM = "ens_member_mse_anom_skill_clim"
    MEAN_MEMBER_MSE_ANOM_SKILL_CLIM = "mean_member_mse_anom_skill_clim"
    SPREAD_ANOM = "spread_anom"
    SPREAD_ANOM_SKILL_RATIO = "spread_anom_skill_ratio"
    CRPS_ANOM = "crps_anom"
    RANK_HISTOGRAM_ANOM = "rank_histogram_anom"
    ROC_ANOM_UPPER = "roc_anom_upper"
    ROC_ANOM_MIDDLE = "roc_anom_middle"
    ROC_ANOM_LOWER = "roc_anom_lower"

DETERMINISTIC_METRICS: set[Metric] = {
    Metric.OROGRAPHY,
    Metric.OROGRAPHY_GRAD_MAG,

    Metric.BIAS,
    Metric.MAE,
    Metric.MSE,
    Metric.MSE_SKILL_CLIM,
    Metric.RMSE,
    Metric.NMSE,
    Metric.NRMSE,
    Metric.R2,
    Metric.CORR,
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

    Metric.BIAS_ANOM,
    Metric.MAE_ANOM,
    Metric.MSE_ANOM,
    Metric.RMSE_ANOM,
    Metric.NMSE_ANOM,
    Metric.NRMSE_ANOM,
    Metric.ACC,
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

    Metric.MAE_ANOM_SKILL_CLIM,
    Metric.MSE_ANOM_SKILL_CLIM,
    Metric.RMSE_ANOM_SKILL_CLIM,
}

PROBABILISTIC_METRICS: set[Metric] = {
    Metric.ENS_MEMBER_RMSE,
    Metric.MEAN_MEMBER_RMSE,
    Metric.SPREAD,
    Metric.SPREAD_SKILL_RATIO,
    Metric.CRPS,
    Metric.RANK_HISTOGRAM,

    Metric.ENS_MEMBER_RMSE_ANOM,
    Metric.MEAN_MEMBER_RMSE_ANOM,
    Metric.ENS_MEMBER_MSE_ANOM_SKILL_CLIM,
    Metric.MEAN_MEMBER_MSE_ANOM_SKILL_CLIM,
    Metric.SPREAD_ANOM,
    Metric.SPREAD_ANOM_SKILL_RATIO,
    Metric.CRPS_ANOM,
    Metric.RANK_HISTOGRAM_ANOM,
    Metric.ROC_ANOM_UPPER,
    Metric.ROC_ANOM_MIDDLE,
    Metric.ROC_ANOM_LOWER,
}
