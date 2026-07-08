from typing import Literal
from enum import StrEnum


MetricKind = Literal["scalar", "maps", "timeseries"]
LeadtimeAgg = Literal["single", "aggregated", "seasonal_window"]
RealizationAgg = Literal["member", "ensemble_mean"]
MetricAgg = Literal[
    "global",       # sqrt(mean(error^2))
    "spatial_avg",  # mean(point-wise RMSE)
    "spatial_rmse", # sqrt(mean(point-wise MSE))
]


class Metric(StrEnum):
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
