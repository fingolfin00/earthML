from typing import cast, Literal
from collections.abc import Sequence

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import kendalltau
import xskillscore as xs
import xrft

from ..base import (
    Settings,
    LeadtimeUnit,
    ClimPeriod,
    aggregate_leadtime_da,
    get_and_subset_datasets,
)

from .climatology import calculate_save_and_subset_climatologies
from .definitions import (
    MetricKind,
    LeadtimeAgg,
    MetricAgg,
    Metric,
    DETERMINISTIC_METRICS,
    PROBABILISTIC_METRICS,
)


def safe_percent(num: xr.DataArray, den: xr.DataArray) -> xr.DataArray:
    return xr.where(den != 0, 100 * num / den, np.nan)

def safe_div(num: xr.DataArray, den: xr.DataArray) -> xr.DataArray:
    return xr.where(den != 0, num / den, np.nan)


def as_metric(metric: str | Metric) -> Metric:
    return metric if isinstance(metric, Metric) else Metric(metric)

def is_deterministic(metric: str | Metric) -> bool:
    return as_metric(metric) in DETERMINISTIC_METRICS

def is_probabilistic(metric: str | Metric) -> bool:
    return as_metric(metric) in PROBABILISTIC_METRICS


EARTH_RADIUS_M = 6_371_000.0

def horizontal_gradient(
    da: xr.DataArray,
    lat_dim: str,
    lon_dim: str,
    periodic_longitude: bool | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Horizontal gradient on a regular latitude/longitude grid."""
    lat = da[lat_dim]
    lon = da[lon_dim]

    if lat.size < 3 or lon.size < 3:
        raise ValueError(
            "horizontal_gradient requires at least 3 latitude and "
            "3 longitude points."
        )

    lon_spacing = abs(lon.diff(lon_dim))
    dlon = float(lon_spacing.median())

    if not np.allclose(
        lon_spacing,
        dlon,
        rtol=1e-5,
        atol=1e-8,
    ):
        raise ValueError(
            "Horizontal gradient requires regular longitude spacing."
        )

    if periodic_longitude is None:
        lon_coverage = float(lon.max() - lon.min()) + dlon
        periodic_longitude = np.isclose(
            lon_coverage,
            360.0,
            rtol=0,
            atol=dlon * 0.1,
        )

    lat_rad = np.deg2rad(lat)

    meters_per_degree_lat = (
        EARTH_RADIUS_M * np.pi / 180.0
    )

    grad_y = da.differentiate(
        lat_dim,
        edge_order=2,
    ) / meters_per_degree_lat

    if periodic_longitude:
        forward = da.roll(
            {lon_dim: -1},
            roll_coords=False,
        )
        backward = da.roll(
            {lon_dim: 1},
            roll_coords=False,
        )

        grad_x_per_degree = (forward - backward) / (2.0 * dlon)

    else:
        grad_x_per_degree = da.differentiate(
            lon_dim,
            edge_order=2,
        )

    cos_lat = np.cos(lat_rad)

    meters_per_degree_lon = EARTH_RADIUS_M * cos_lat * np.pi / 180.0

    grad_x = xr.where(
        abs(cos_lat) > 1e-6,
        grad_x_per_degree / meters_per_degree_lon,
        np.nan,
    )

    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    return grad_x, grad_y, grad_mag


def kendall_tau(
    x: xr.DataArray,
    y: xr.DataArray,
    dim: str,
) -> xr.DataArray:
    """Xarray ufunc wrapper of SciPy Kendall's tau implementation"""
    def _kendall_tau(x, y):
        return kendalltau(
            x,
            y,
            nan_policy="omit",
        ).statistic

    return xr.apply_ufunc(
        _kendall_tau,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


def core_metrics(
    fc: xr.DataArray,
    an: xr.DataArray,
    dims: Sequence[str],
    metrics: Sequence[str],
    *,
    fc_clim: xr.DataArray | None = None,
    an_clim: xr.DataArray | None = None,
    orography: xr.DataArray | None = None,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    fair_correction: bool = False,
) -> xr.Dataset:  
    time_dim = fc.earthml.guessed_dims.time
    lat_dim = fc.earthml.guessed_dims.latitude
    lon_dim = fc.earthml.guessed_dims.longitude
    realization_dim = fc.earthml.guessed_dims.realization
    # leadtime_dim = fc.earthml.guessed_dims.leadtime

    if realization_dim in fc.dims:
        fc = fc.chunk({realization_dim: -1})
    if fc_clim is not None and realization_dim in fc_clim.dims:
        fc_clim = fc_clim.chunk({realization_dim: -1})

    if time_dim in dims:
        fc = fc.chunk({time_dim: -1})
        an = an.chunk({time_dim: -1})

    weights = cast(xr.DataArray, np.cos(np.deg2rad(fc[lat_dim])))

    # Use the common forecast-analysis valid domain.
    valid = fc.notnull() & an.notnull()

    fc = fc.where(valid)
    an = an.where(valid)

    error = fc - an

    # fairness correction for MSSS
    if fair_correction:
        n = an[time_dim].count(time_dim)
        correction = n / (n - 1)
    else:
        correction = 1

    def want(metric: Metric) -> bool:
        return metric.value in metrics

    out = xr.Dataset()

    # ------------------------------------------------------
    # Orography
    # ------------------------------------------------------

    static_orography_metrics = (
        Metric.OROGRAPHY,
        Metric.OROGRAPHY_GRAD_MAG,
    )

    if any(want(metric) for metric in static_orography_metrics):
        if orography is None:
            raise ValueError(
                "Orography is required for orography metrics."
            )

        # Only reduce dimensions that actually exist in the static field.
        oro_dims = tuple(
            d for d in dims
            if d in orography.dims
        )

        def reduce_orography(
            da: xr.DataArray,
        ) -> xr.DataArray:
            if not oro_dims:
                return da

            return da.weighted(weights).mean(oro_dims)

        if want(Metric.OROGRAPHY):
            out[Metric.OROGRAPHY.value] = reduce_orography(
                orography
            )

        if want(Metric.OROGRAPHY_GRAD_MAG):
            _, _, orography_grad_mag = horizontal_gradient(
                orography,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
            )

            out[Metric.OROGRAPHY_GRAD_MAG.value] = (
                reduce_orography(orography_grad_mag)
            )

    # ------------------------------------------------------
    # Error metrics
    # ------------------------------------------------------

    if want(Metric.BIAS):
        out[Metric.BIAS.value] = error.weighted(weights).mean(dims)

    if want(Metric.MAE):
        out[Metric.MAE.value] = abs(error).weighted(weights).mean(dims)

    if want(Metric.MSE):
        out[Metric.MSE.value] = (error ** 2).weighted(weights).mean(dims)

    if want(Metric.RMSE):
        out[Metric.RMSE.value] = np.sqrt((error ** 2).weighted(weights).mean(dims))

    # ------------------------------------------------------
    # Normalized metrics
    # ------------------------------------------------------

    if want(Metric.NMSE):
        nmse = (error ** 2).weighted(weights).mean(dims)
        an_var_total = ((an - an.weighted(weights).mean(dims)) ** 2).weighted(weights).mean(dims)
        out[Metric.NMSE.value] = nmse / an_var_total

    if want(Metric.NRMSE):
        rmse = np.sqrt((error ** 2).weighted(weights).mean(dims))
        an_std_total = np.sqrt(((an - an.weighted(weights).mean(dims)) ** 2).weighted(weights).mean(dims))
        out[Metric.NRMSE.value] = rmse / an_std_total

    if want(Metric.R2):
        sse = (error ** 2).weighted(weights).sum(dims)
        sst = ((an - an.weighted(weights).mean(dims)) ** 2).weighted(weights).sum(dims)
        out[Metric.R2.value] = 1 - sse / sst

    # ------------------------------------------------------
    # Temporal metrics and MSE decomposition
    # ------------------------------------------------------

    temporal_diagnostic_metrics = (
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
    )

    if any(want(metric) for metric in temporal_diagnostic_metrics):
        if time_dim not in dims:
            raise ValueError(
                "Temporal diagnostic metrics require the time dimension "
                "to be included in dims."
            )

        # Work in float64 for numerically stable temporal diagnostics
        fc_diag = fc.astype("float64")
        an_diag = an.astype("float64")

        # Use exactly the same valid samples for forecast and analysis
        valid = fc_diag.notnull() & an_diag.notnull()
        fc_diag = fc_diag.where(valid)
        an_diag = an_diag.where(valid)

        error_diag = fc_diag - an_diag

        # Temporal means
        fc_mean_t = fc_diag.mean(time_dim)
        an_mean_t = an_diag.mean(time_dim)

        # Centered fields
        fc_centered = fc_diag - fc_mean_t
        an_centered = an_diag - an_mean_t

        # Temporal bias
        bias_t = error_diag.mean(time_dim)

        # Population variance and covariance over time
        fc_var_t = (fc_centered ** 2).mean(time_dim)
        an_var_t = (an_centered ** 2).mean(time_dim)
        cov_t = (fc_centered * an_centered).mean(time_dim)

        fc_std_t = np.sqrt(fc_var_t)
        an_std_t = np.sqrt(an_var_t)

        # Correlation
        corr_t = safe_div(cov_t, fc_std_t * an_std_t)

        # --------------------------------------------------------------
        # MSE decomposition
        #
        # MSE = bias_component
        #     + std_component
        #     + corr_component
        #
        # where
        #
        # bias_component = bias²
        # std_component  = (sigma_fc - sigma_an)²
        # corr_component = 2 (sigma_fc sigma_an - cov)
        # --------------------------------------------------------------

        mse_bias_component_t = bias_t ** 2
        mse_std_component_t = (fc_std_t - an_std_t) ** 2
        mse_corr_component_t = 2.0 * (fc_std_t * an_std_t - cov_t)

        # Whatever remains after temporal metric is calculated
        post_temporal_dims = tuple(
            d for d in dims
            if d != time_dim
        )

        def spatial_mean(
            da: xr.DataArray,
        ) -> xr.DataArray:
            if not post_temporal_dims:
                return da
            return da.weighted(weights).mean(post_temporal_dims)

        # ------------------------------------------------------
        # Correlation and STD
        # ------------------------------------------------------

        if want(Metric.CORR):
            out[Metric.CORR.value] = spatial_mean(corr_t)

        if want(Metric.KENDALL_TAU):
            kendall = kendall_tau(
                fc_diag,
                an_diag,
                dim=time_dim,
            )

            out[Metric.KENDALL_TAU.value] = spatial_mean(kendall)

        if want(Metric.FC_STD):
            out[Metric.FC_STD.value] = spatial_mean(fc_std_t)

        if want(Metric.AN_STD):
            out[Metric.AN_STD.value] = spatial_mean(an_std_t)

        if want(Metric.STD_RATIO):
            out[Metric.STD_RATIO.value] = spatial_mean(
                safe_div(fc_std_t, an_std_t)
            )

        # ------------------------------------------------------
        # MSE decomposition metrics
        # ------------------------------------------------------

        mse_bias_component = spatial_mean(mse_bias_component_t)
        mse_std_component = spatial_mean(mse_std_component_t)
        mse_corr_component = spatial_mean(mse_corr_component_t)

        if want(Metric.MSE_BIAS_COMPONENT):
            out[Metric.MSE_BIAS_COMPONENT.value] = mse_bias_component

        if want(Metric.MSE_STD_COMPONENT):
            out[Metric.MSE_STD_COMPONENT.value] = mse_std_component

        if want(Metric.MSE_CORR_COMPONENT):
            out[Metric.MSE_CORR_COMPONENT.value] = mse_corr_component

        if want(Metric.CRMSE):
            out[Metric.CRMSE.value] = np.sqrt(mse_std_component + mse_corr_component)

        if want(Metric.REGRESSION_SLOPE):
            regression_slope_t = safe_div(cov_t, an_var_t)
            out[Metric.REGRESSION_SLOPE.value] = spatial_mean(regression_slope_t)

    # ------------------------------------------------------
    # Spatial gradients
    # ------------------------------------------------------

    spatial_gradient_metrics = (
        Metric.FC_GRAD_MAG,
        Metric.AN_GRAD_MAG,
        Metric.GRAD_RMSE,
    )

    if any(want(metric) for metric in spatial_gradient_metrics):
        fc_grad_x, fc_grad_y, fc_grad_mag = horizontal_gradient(
            fc,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        an_grad_x, an_grad_y, an_grad_mag = horizontal_gradient(
            an,
            lat_dim=lat_dim,
            lon_dim=lon_dim,
        )

        if want(Metric.FC_GRAD_MAG):
            out[Metric.FC_GRAD_MAG.value] = fc_grad_mag.weighted(weights).mean(dims)

        if want(Metric.AN_GRAD_MAG):
            out[Metric.AN_GRAD_MAG.value] = an_grad_mag.weighted(weights).mean(dims)

        if want(Metric.GRAD_RMSE):
            gradient_squared_error = (fc_grad_x - an_grad_x) ** 2 + (fc_grad_y - an_grad_y) ** 2
            out[Metric.GRAD_RMSE.value] = np.sqrt(gradient_squared_error.weighted(weights).mean(dims))

    # ------------------------------------------------------
    # Power spectrum metrics
    # Supported for 1D temporal/zonal/meridional spectra,
    # raw 2D spatial spectra, and 1D isotropic spatial spectra
    # ------------------------------------------------------

    power_metrics = (
        Metric.FC_POWER_SPECTRUM,
        Metric.AN_POWER_SPECTRUM,
        Metric.FC_ISOTROPIC_POWER_SPECTRUM,
        Metric.AN_ISOTROPIC_POWER_SPECTRUM,
        Metric.POWER_SPECTRUM_RATIO,
    )

    if any(want(metric) for metric in power_metrics):
        power_dims = tuple(
            d
            for d in dims
            if d in {time_dim, lat_dim, lon_dim}
        )

        # 1D PSD
        if power_dims in {
            (time_dim,),  # temporal
            (lat_dim,),   # meridional
            (lon_dim,),   # zonal
        }:
            if want(Metric.FC_POWER_SPECTRUM) or want(Metric.POWER_SPECTRUM_RATIO):
                fc_ps = xrft.power_spectrum(
                    fc,
                    dim=power_dims,
                    detrend="linear",
                    window=True,
                    scaling="density",
                )

                if want(Metric.FC_POWER_SPECTRUM):
                    out[Metric.FC_POWER_SPECTRUM.value] = fc_ps

            if want(Metric.AN_POWER_SPECTRUM) or want(Metric.POWER_SPECTRUM_RATIO):
                an_ps = xrft.power_spectrum(
                    an,
                    dim=power_dims,
                    detrend="linear",
                    window=True,
                    scaling="density",
                )

                if want(Metric.AN_POWER_SPECTRUM):
                    out[Metric.AN_POWER_SPECTRUM.value] = an_ps

            if want(Metric.POWER_SPECTRUM_RATIO):
                out[Metric.POWER_SPECTRUM_RATIO.value] = safe_div(fc_ps, an_ps)

        # 2D spatial field -> 1D isotropic spatial PSD
        elif set(power_dims) == {lat_dim, lon_dim}:
            if want(Metric.FC_ISOTROPIC_POWER_SPECTRUM) or want(Metric.POWER_SPECTRUM_RATIO):
                fc_iso_ps = xrft.isotropic_power_spectrum(
                    fc,
                    dim=power_dims,
                    detrend="linear",
                    window=True,
                    scaling="density",
                )

                if want(Metric.FC_ISOTROPIC_POWER_SPECTRUM):
                    out[Metric.FC_ISOTROPIC_POWER_SPECTRUM.value] = fc_iso_ps

            if want(Metric.AN_ISOTROPIC_POWER_SPECTRUM) or want(Metric.POWER_SPECTRUM_RATIO):
                an_iso_ps = xrft.isotropic_power_spectrum(
                    an,
                    dim=power_dims,
                    detrend="linear",
                    window=True,
                    scaling="density",
                )

                if want(Metric.AN_ISOTROPIC_POWER_SPECTRUM):
                    out[Metric.AN_ISOTROPIC_POWER_SPECTRUM.value] = an_iso_ps

            if want(Metric.POWER_SPECTRUM_RATIO):
                out[Metric.POWER_SPECTRUM_RATIO.value] = safe_div(fc_iso_ps, an_iso_ps)

            if want(Metric.FC_POWER_SPECTRUM):
                fc_ps = xrft.power_spectrum(
                    fc,
                    dim=power_dims,
                    detrend="linear",
                    window=True,
                    scaling="density",
                )

                if want(Metric.FC_POWER_SPECTRUM):
                    out[Metric.FC_POWER_SPECTRUM.value] = fc_ps

            if want(Metric.AN_POWER_SPECTRUM):
                an_ps = xrft.power_spectrum(
                    an,
                    dim=power_dims,
                    detrend="linear",
                    window=True,
                    scaling="density",
                )

                if want(Metric.AN_POWER_SPECTRUM):
                    out[Metric.AN_POWER_SPECTRUM.value] = an_ps

    # ------------------------------------------------------
    # Spatial only metrics
    # can produce only scalar and timeseries views
    # ------------------------------------------------------

    spatial_only_dims = (lat_dim, lon_dim)
    reduce_space = all(dim in dims for dim in spatial_only_dims)

    if want(Metric.SCC) and reduce_space:  
        fc_spatial_mean = fc.weighted(weights).mean(spatial_only_dims)
        an_spatial_mean = an.weighted(weights).mean(spatial_only_dims)

        fc_centered_spatial = fc - fc_spatial_mean
        an_centered_spatial = an - an_spatial_mean

        cov_spatial = (fc_centered_spatial * an_centered_spatial).weighted(weights).mean(spatial_only_dims)

        fc_spatial_std = np.sqrt((fc_centered_spatial ** 2).weighted(weights).mean(spatial_only_dims))
        an_spatial_std = np.sqrt((an_centered_spatial ** 2).weighted(weights).mean(spatial_only_dims))

        scc = safe_div(cov_spatial, fc_spatial_std * an_spatial_std)

        remaining_dims = [
            d
            for d in dims
            if d not in spatial_only_dims
        ]

        if remaining_dims:
            scc = scc.mean(remaining_dims)

        out[Metric.SCC.value] = scc

    # ------------------------------------------------------
    # Probabilistic and ensemble metrics
    # produced only if realization dim is present
    # ------------------------------------------------------

    if realization_dim in fc.dims:

        # --------------------------------------------------
        # RMSE pooled across all ensemble members
        # and the requested aggregation dimensions
        # --------------------------------------------------

        if want(Metric.ENS_MEMBER_RMSE):
            out[Metric.ENS_MEMBER_RMSE.value] = np.sqrt((error ** 2).weighted(weights).mean((realization_dim, *dims)))

        # --------------------------------------------------
        # Average RMSE of the individual ensemble members
        # --------------------------------------------------

        if want(Metric.MEAN_MEMBER_RMSE):
            out[Metric.MEAN_MEMBER_RMSE.value] = np.sqrt((error ** 2).weighted(weights).mean(dims)).mean(realization_dim)

        # --------------------------------------------------
        # Ensemble spread
        # --------------------------------------------------

        if want(Metric.SPREAD) or want(Metric.SPREAD_SKILL_RATIO):
            spread = fc.std(realization_dim).weighted(weights).mean(dims)
            # spread = fc.std(realization_dim).weighted(weights).mean(dims)
            if want(Metric.SPREAD):
                out[Metric.SPREAD.value] = spread

            if want(Metric.SPREAD_SKILL_RATIO):
                ens_member_rmse = np.sqrt((error ** 2).weighted(weights).mean((realization_dim, *dims)))
                out[Metric.SPREAD_SKILL_RATIO.value] = spread / ens_member_rmse

        # --------------------------------------------------
        # Ensemble scores
        # --------------------------------------------------

        # Continuous ranked probability score (CRPS)
        if want(Metric.CRPS):
            out[Metric.CRPS.value] = xs.crps_ensemble(
                observations=an.mean(dim=realization_dim),
                forecasts=fc,
                member_dim=realization_dim,
                dim=list(dims),
                weights=weights,
            )

        # --------------------------------------------------
        # Rank histogram
        # --------------------------------------------------

        if want(Metric.RANK_HISTOGRAM):
            out[Metric.RANK_HISTOGRAM.value] = xs.rank_histogram(
                observations=an.mean(dim=realization_dim),
                forecasts=fc,
                member_dim=realization_dim,
            )


        # --------------------------------------------------
        # Event-based ensemble scores
        # --------------------------------------------------

        event_based_scores = (
            Metric.BRIER_LOWER,
            Metric.BRIER_MIDDLE,
            Metric.BRIER_UPPER,
            Metric.ROC_LOWER,
            Metric.ROC_MIDDLE,
            Metric.ROC_UPPER,
        )

        if any(want(metric) for metric in event_based_scores):
            an_for_terciles = an.mean(realization_dim).chunk({time_dim: -1})
            fc_for_terciles = fc.chunk({realization_dim: -1})

            # Terciles
            q33 = an_for_terciles.quantile(1 / 3, dim=time_dim).reset_coords(drop=True)
            q67 = an_for_terciles.quantile(2 / 3, dim=time_dim).reset_coords(drop=True)

            # Events
            an_event_lower = an_for_terciles <= q33
            an_event_middle = (an_for_terciles > q33) & (an_for_terciles <= q67)
            an_event_upper = an_for_terciles > q67

            fc_event_lower = fc_for_terciles <= q33
            fc_event_middle = (fc_for_terciles > q33) & (fc_for_terciles <= q67)
            fc_event_upper = fc_for_terciles > q67

            # --------------------------------------------------
            # Brier score
            # --------------------------------------------------

            if want(Metric.BRIER_LOWER):
                out[Metric.BRIER_LOWER.value] = xs.brier_score(
                    observations=an_event_lower,
                    forecasts=fc_event_lower,
                    member_dim=realization_dim,
                    dim=list(dims),
                    fair=fair_correction,
                    weights=weights,
                )

            if want(Metric.BRIER_MIDDLE):
                out[Metric.BRIER_MIDDLE.value] = xs.brier_score(
                    observations=an_event_middle,
                    forecasts=fc_event_middle,
                    member_dim=realization_dim,
                    dim=list(dims),
                    fair=fair_correction,
                    weights=weights,
                )

            if want(Metric.BRIER_UPPER):
                out[Metric.BRIER_UPPER.value] = xs.brier_score(
                    observations=an_event_upper,
                    forecasts=fc_event_upper,
                    member_dim=realization_dim,
                    dim=list(dims),
                    fair=fair_correction,
                    weights=weights,
                )

            # --------------------------------------------------
            # Receiving operating characteristic (ROC)
            # --------------------------------------------------

            if want(Metric.ROC_LOWER):
                fc_prob_lower = fc_event_lower.mean(realization_dim)
                # print("ROC lower tercile probs calculated.")
                out[Metric.ROC_LOWER.value] = xs.roc(
                    observations=an_event_lower,
                    forecasts=fc_prob_lower,
                    dim=list(dims),
                )

            if want(Metric.ROC_MIDDLE):
                fc_prob_middle = fc_event_middle.mean(realization_dim)
                # print("ROC middle tercile probs calculated.")
                out[Metric.ROC_MIDDLE.value] = xs.roc(
                    observations=an_event_middle,
                    forecasts=fc_prob_middle,
                    dim=list(dims),
                )

            if want(Metric.ROC_UPPER):
                fc_prob_upper = fc_event_upper.mean(realization_dim)
                # print("ROC upper tercile probs calculated.")
                out[Metric.ROC_UPPER.value] = xs.roc(
                    observations=an_event_upper,
                    forecasts=fc_prob_upper,
                    dim=list(dims),
                )

    # ------------------------------------------------------
    # Anomaly metrics
    # ------------------------------------------------------

    if fc_clim is not None and an_clim is not None:
        valid_clim = fc_clim.notnull() & an_clim.notnull()

        fc_clim = fc_clim.where(valid_clim)
        an_clim = an_clim.where(valid_clim)

        fc_anom = groupby_period(fc, time_dim, clim_period) - fc_clim
        an_anom = groupby_period(an, time_dim, clim_period) - an_clim

        if time_dim in dims:
            fc_anom = fc_anom.chunk({time_dim: -1})
            an_anom = an_anom.chunk({time_dim: -1})

        # ------------------------------------------------------
        # Anomaly error metrics
        # ------------------------------------------------------

        error_anom = fc_anom - an_anom

        if want(Metric.BIAS_ANOM):
            out[Metric.BIAS_ANOM.value] = error_anom.weighted(weights).mean(dims)

        if want(Metric.MAE_ANOM):
            out[Metric.MAE_ANOM.value] = abs(error_anom).weighted(weights).mean(dims)

        if want(Metric.MSE_ANOM):
            out[Metric.MSE_ANOM.value] = (error_anom ** 2).weighted(weights).mean(dims)

        if want(Metric.RMSE_ANOM):
            out[Metric.RMSE_ANOM.value] = np.sqrt((error_anom ** 2).weighted(weights).mean(dims))

        # ------------------------------------------------------
        # Normalized anomaly metrics
        # ------------------------------------------------------

        if want(Metric.NMSE_ANOM):
            nmse_anom = (error_anom ** 2).weighted(weights).mean(dims)
            an_anom_var_total = ((an_anom - an_anom.weighted(weights).mean(dims)) ** 2).weighted(weights).mean(dims)
            out[Metric.NMSE_ANOM.value] = nmse_anom / an_anom_var_total

        if want(Metric.NRMSE_ANOM):
            rmse_anom = np.sqrt((error_anom ** 2).weighted(weights).mean(dims))
            an_anom_std_total = np.sqrt(((an_anom - an_anom.weighted(weights).mean(dims)) ** 2).weighted(weights).mean(dims))
            out[Metric.NRMSE_ANOM.value] = rmse_anom / an_anom_std_total

        if want(Metric.R2_ANOM):
            sse_anom = ((error_anom) ** 2).weighted(weights).sum(dims)
            sst_anom = ((an_anom - an_anom.weighted(weights).mean(dims)) ** 2).weighted(weights).sum(dims)
            out[Metric.R2_ANOM.value] = 1 - sse_anom / sst_anom

        # ------------------------------------------------------
        # Temporal anomaly metrics and anomaly MSE decomposition
        # ------------------------------------------------------

        anomaly_temporal_diagnostic_metrics = (
            Metric.ACC,
            Metric.KENDALL_TAU_ANOM,
            Metric.FC_ANOM_STD,
            Metric.AN_ANOM_STD,
            Metric.STD_RATIO_ANOM,
            Metric.MSE_BIAS_COMPONENT_ANOM,
            Metric.MSE_STD_COMPONENT_ANOM,
            Metric.MSE_CORR_COMPONENT_ANOM,
            Metric.CRMSE_ANOM,
            Metric.REGRESSION_SLOPE_ANOM,
        )

        if any(
            want(metric)
            for metric in anomaly_temporal_diagnostic_metrics
        ):
            if time_dim not in dims:
                raise ValueError(
                    "Anomaly temporal diagnostic metrics require the "
                    "time dimension to be included in dims."
                )

            # Work in float64 for numerically stable diagnostics
            fc_anom_diag = fc_anom.astype("float64")
            an_anom_diag = an_anom.astype("float64")

            # Use exactly the same valid samples
            valid_anom = fc_anom_diag.notnull() & an_anom_diag.notnull()

            fc_anom_diag = fc_anom_diag.where(valid_anom)
            an_anom_diag = an_anom_diag.where(valid_anom)

            error_anom_diag = fc_anom_diag - an_anom_diag

            # Temporal means
            fc_anom_mean_t = fc_anom_diag.mean(time_dim)
            an_anom_mean_t = an_anom_diag.mean(time_dim)

            # Centered anomaly fields
            fc_anom_centered = fc_anom_diag - fc_anom_mean_t
            an_anom_centered = an_anom_diag - an_anom_mean_t

            # Temporal anomaly bias
            bias_anom_t = error_anom_diag.mean(time_dim)

            # Population variance and covariance
            fc_anom_var_t = (fc_anom_centered ** 2).mean(time_dim)
            an_anom_var_t = (an_anom_centered ** 2).mean(time_dim)

            cov_anom_t = (fc_anom_centered * an_anom_centered).mean(time_dim)

            fc_anom_std_t = np.sqrt(fc_anom_var_t)
            an_anom_std_t = np.sqrt(an_anom_var_t)

            # Anomaly correlation coefficient
            acc_t = safe_div(cov_anom_t, fc_anom_std_t * an_anom_std_t)

            # ----------------------------------------------------------
            # Anomaly MSE decomposition
            #
            # MSE_anom = bias_component_anom
            #          + std_component_anom
            #          + corr_component_anom
            # ----------------------------------------------------------

            mse_bias_component_anom_t = bias_anom_t ** 2
            mse_std_component_anom_t = (fc_anom_std_t - an_anom_std_t) ** 2
            mse_corr_component_anom_t = 2.0 * (fc_anom_std_t * an_anom_std_t - cov_anom_t)

            post_temporal_dims = tuple(
                d for d in dims
                if d != time_dim
            )

            def anomaly_spatial_mean(
                da: xr.DataArray,
            ) -> xr.DataArray:
                if not post_temporal_dims:
                    return da
                return da.weighted(weights).mean(post_temporal_dims)

            # ------------------------------------------------------
            # Anomaly correlation and anomaly STD
            # ------------------------------------------------------

            if want(Metric.ACC):
                out[Metric.ACC.value] = anomaly_spatial_mean(acc_t)

            if want(Metric.KENDALL_TAU_ANOM):
                kendall_anom = kendall_tau(
                    fc_anom_diag,
                    an_anom_diag,
                    dim=time_dim,
                )

                out[Metric.KENDALL_TAU_ANOM.value] = anomaly_spatial_mean(kendall_anom)

            if want(Metric.FC_ANOM_STD):
                out[Metric.FC_ANOM_STD.value] = anomaly_spatial_mean(fc_anom_std_t)

            if want(Metric.AN_ANOM_STD):
                out[Metric.AN_ANOM_STD.value] = anomaly_spatial_mean(an_anom_std_t)

            if want(Metric.STD_RATIO_ANOM):
                out[Metric.STD_RATIO_ANOM.value] = anomaly_spatial_mean(safe_div(fc_anom_std_t, an_anom_std_t))

            # ------------------------------------------------------
            # Anomaly MSE decomposition
            # ------------------------------------------------------

            mse_bias_component_anom = anomaly_spatial_mean(mse_bias_component_anom_t)
            mse_std_component_anom = anomaly_spatial_mean(mse_std_component_anom_t)
            mse_corr_component_anom = anomaly_spatial_mean(mse_corr_component_anom_t)

            if want(Metric.MSE_BIAS_COMPONENT_ANOM):
                out[Metric.MSE_BIAS_COMPONENT_ANOM.value] = mse_bias_component_anom

            if want(Metric.MSE_STD_COMPONENT_ANOM):
                out[Metric.MSE_STD_COMPONENT_ANOM.value] = mse_std_component_anom

            if want(Metric.MSE_CORR_COMPONENT_ANOM):
                out[Metric.MSE_CORR_COMPONENT_ANOM.value] = mse_corr_component_anom

            if want(Metric.CRMSE_ANOM):
                out[Metric.CRMSE_ANOM.value] = np.sqrt(mse_std_component_anom + mse_corr_component_anom)

            if want(Metric.REGRESSION_SLOPE_ANOM):
                regression_slope_anom_t = safe_div(cov_anom_t, an_anom_var_t)

                out[Metric.REGRESSION_SLOPE_ANOM.value] = anomaly_spatial_mean(regression_slope_anom_t)

        # ------------------------------------------------------
        # Anomaly spatial gradients
        # ------------------------------------------------------

        anomaly_spatial_gradient_metrics = (
            Metric.FC_ANOM_GRAD_MAG,
            Metric.AN_ANOM_GRAD_MAG,
            Metric.GRAD_RMSE_ANOM,
        )

        if any(
            want(metric)
            for metric in anomaly_spatial_gradient_metrics
        ):
            fc_anom_grad_x, fc_anom_grad_y, fc_anom_grad_mag = horizontal_gradient(
                fc_anom,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
            )

            an_anom_grad_x, an_anom_grad_y, an_anom_grad_mag = horizontal_gradient(
                an_anom,
                lat_dim=lat_dim,
                lon_dim=lon_dim,
            )

            if want(Metric.FC_ANOM_GRAD_MAG):
                out[Metric.FC_ANOM_GRAD_MAG.value] = fc_anom_grad_mag.weighted(weights).mean(dims)

            if want(Metric.AN_ANOM_GRAD_MAG):
                out[Metric.AN_ANOM_GRAD_MAG.value] = an_anom_grad_mag.weighted(weights).mean(dims)

            if want(Metric.GRAD_RMSE_ANOM):
                gradient_anom_squared_error = (fc_anom_grad_x - an_anom_grad_x) ** 2 + (fc_anom_grad_y - an_anom_grad_y) ** 2

                out[Metric.GRAD_RMSE_ANOM.value] = np.sqrt(gradient_anom_squared_error.weighted(weights).mean(dims))

        # ------------------------------------------------------
        # Power spectrum metrics for anomalies
        # Supported for 1D temporal/zonal/meridional spectra,
        # raw 2D spatial spectra, and 1D isotropic spatial spectra
        # ------------------------------------------------------

        power_metrics_anom = (
            Metric.FC_ANOM_POWER_SPECTRUM,
            Metric.AN_ANOM_POWER_SPECTRUM,
            Metric.FC_ANOM_ISOTROPIC_POWER_SPECTRUM,
            Metric.AN_ANOM_ISOTROPIC_POWER_SPECTRUM,
            Metric.POWER_SPECTRUM_RATIO_ANOM,
        )

        if any(want(metric) for metric in power_metrics_anom):
            power_dims = tuple(
                d
                for d in dims
                if d in {time_dim, lat_dim, lon_dim}
            )

            # 1D PSD
            if power_dims in {
                (time_dim,),  # temporal
                (lat_dim,),   # meridional
                (lon_dim,),   # zonal
            }:
                if want(Metric.FC_ANOM_POWER_SPECTRUM) or want(Metric.POWER_SPECTRUM_RATIO_ANOM):
                    fc_anom_ps = xrft.power_spectrum(
                        fc_anom,
                        dim=power_dims,
                        detrend="linear",
                        window=True,
                        scaling="density",
                    )

                    if want(Metric.FC_ANOM_POWER_SPECTRUM):
                        out[Metric.FC_ANOM_POWER_SPECTRUM.value] = fc_anom_ps

                if want(Metric.AN_ANOM_POWER_SPECTRUM) or want(Metric.POWER_SPECTRUM_RATIO_ANOM):
                    an_anom_ps = xrft.power_spectrum(
                        an_anom,
                        dim=power_dims,
                        detrend="linear",
                        window=True,
                        scaling="density",
                    )

                    if want(Metric.AN_ANOM_POWER_SPECTRUM):
                        out[Metric.AN_ANOM_POWER_SPECTRUM.value] = an_anom_ps

                if want(Metric.POWER_SPECTRUM_RATIO_ANOM):
                    out[Metric.POWER_SPECTRUM_RATIO_ANOM.value] = safe_div(fc_anom_ps, an_anom_ps)

            # 2D spatial field -> 1D isotropic spatial PSD
            elif set(power_dims) == {lat_dim, lon_dim}:
                if want(Metric.FC_ANOM_ISOTROPIC_POWER_SPECTRUM) or want(Metric.POWER_SPECTRUM_RATIO_ANOM):
                    fc_anom_iso_ps = xrft.isotropic_power_spectrum(
                        fc_anom,
                        dim=power_dims,
                        detrend="linear",
                        window=True,
                        scaling="density",
                    )

                    if want(Metric.FC_ANOM_ISOTROPIC_POWER_SPECTRUM):
                        out[Metric.FC_ANOM_ISOTROPIC_POWER_SPECTRUM.value] = fc_anom_iso_ps

                if want(Metric.AN_ANOM_ISOTROPIC_POWER_SPECTRUM) or want(Metric.POWER_SPECTRUM_RATIO_ANOM):
                    an_anom_iso_ps = xrft.isotropic_power_spectrum(
                        an_anom,
                        dim=power_dims,
                        detrend="linear",
                        window=True,
                        scaling="density",
                    )

                    if want(Metric.AN_ANOM_ISOTROPIC_POWER_SPECTRUM):
                        out[Metric.AN_ANOM_ISOTROPIC_POWER_SPECTRUM.value] = an_anom_iso_ps

                if want(Metric.POWER_SPECTRUM_RATIO_ANOM):
                    out[Metric.POWER_SPECTRUM_RATIO_ANOM.value] = safe_div(fc_anom_iso_ps, an_anom_iso_ps)

                if want(Metric.FC_ANOM_POWER_SPECTRUM):
                    fc_anom_ps = xrft.power_spectrum(
                        fc_anom,
                        dim=power_dims,
                        detrend="linear",
                        window=True,
                        scaling="density",
                    )

                    if want(Metric.FC_ANOM_POWER_SPECTRUM):
                        out[Metric.FC_ANOM_POWER_SPECTRUM.value] = fc_anom_ps

                if want(Metric.AN_ANOM_POWER_SPECTRUM):
                    an_anom_ps = xrft.power_spectrum(
                        an_anom,
                        dim=power_dims,
                        detrend="linear",
                        window=True,
                        scaling="density",
                    )

                    if want(Metric.AN_ANOM_POWER_SPECTRUM):
                        out[Metric.AN_ANOM_POWER_SPECTRUM.value] = an_anom_ps

        # ------------------------------------------------------
        # Spatial only anomaly metrics
        # can produce only scalar and timeseries views
        # ------------------------------------------------------

        if want(Metric.SCC_ANOM) and reduce_space:
            fc_anom_spatial_mean = fc_anom.weighted(weights).mean(spatial_only_dims)
            an_anom_spatial_mean = an_anom.weighted(weights).mean(spatial_only_dims)

            fc_anom_centered_spatial = fc_anom - fc_anom_spatial_mean
            an_anom_centered_spatial = an_anom - an_anom_spatial_mean

            cov_spatial_anom = (fc_anom_centered_spatial * an_anom_centered_spatial).weighted(weights).mean(spatial_only_dims)

            fc_anom_spatial_std = np.sqrt((fc_anom_centered_spatial ** 2).weighted(weights).mean(spatial_only_dims))
            an_anom_spatial_std = np.sqrt((an_anom_centered_spatial ** 2).weighted(weights).mean(spatial_only_dims))

            scc_anom = safe_div(cov_spatial_anom, fc_anom_spatial_std * an_anom_spatial_std)

            remaining_dims = [
                d
                for d in dims
                if d not in spatial_only_dims
            ]

            if remaining_dims:
                scc_anom = scc_anom.mean(remaining_dims)

            out[Metric.SCC_ANOM.value] = scc_anom

        # ------------------------------------------------------
        # Skills vs climatology
        # need anomalies even for full fields
        # ------------------------------------------------------

        if want(Metric.MSE_SKILL_CLIM):
            mse = (error ** 2).weighted(weights).mean(dims)
            clim_mse_anom = ((an_anom ** 2).weighted(weights).mean(dims)) * correction
            out[Metric.MSE_SKILL_CLIM.value] = safe_div(clim_mse_anom - mse, clim_mse_anom)

        if want(Metric.RMSE_SKILL_CLIM):
            rmse = cast(xr.DataArray, np.sqrt((error ** 2).weighted(weights).mean(dims)))
            clim_rmse_anom = cast(xr.DataArray, np.sqrt((an_anom ** 2).weighted(weights).mean(dims) * correction))
            out[Metric.RMSE_SKILL_CLIM.value] = safe_div(clim_rmse_anom - rmse, clim_rmse_anom)

        if want(Metric.MAE_SKILL_CLIM):
            mae = abs(error).weighted(weights).mean(dims)
            clim_mae_anom = abs(an_anom).weighted(weights).mean(dims)
            out[Metric.MAE_SKILL_CLIM.value] = safe_div(clim_mae_anom - mae, clim_mae_anom)

        if want(Metric.MSE_ANOM_SKILL_CLIM):
            mse_anom = (error_anom ** 2).weighted(weights).mean(dims)
            clim_mse_anom = ((an_anom ** 2).weighted(weights).mean(dims)) * correction
            out[Metric.MSE_ANOM_SKILL_CLIM.value] = safe_div(clim_mse_anom - mse_anom, clim_mse_anom)

        if want(Metric.RMSE_ANOM_SKILL_CLIM):
            rmse_anom = cast(xr.DataArray, np.sqrt((error_anom ** 2).weighted(weights).mean(dims)))
            clim_rmse_anom = cast(xr.DataArray, np.sqrt((an_anom ** 2).weighted(weights).mean(dims) * correction))
            out[Metric.RMSE_ANOM_SKILL_CLIM.value] = safe_div(clim_rmse_anom - rmse_anom, clim_rmse_anom)

        if want(Metric.MAE_ANOM_SKILL_CLIM):
            mae_anom = abs(error_anom).weighted(weights).mean(dims)
            clim_mae_anom = abs(an_anom).weighted(weights).mean(dims)
            out[Metric.MAE_ANOM_SKILL_CLIM.value] = safe_div(clim_mae_anom - mae_anom, clim_mae_anom)

        # ------------------------------------------------------
        # Probabilistic and ensemble anomaly metrics
        # produced only if realization dim is present
        # ------------------------------------------------------

        if realization_dim in fc.dims:

            # --------------------------------------------------
            # Anomaly RMSE pooled across all ensemble members
            # and the requested aggregation dimensions
            # --------------------------------------------------

            if want(Metric.ENS_MEMBER_RMSE_ANOM):
                out[Metric.ENS_MEMBER_RMSE_ANOM.value] = np.sqrt((error_anom ** 2).weighted(weights).mean((realization_dim, *dims)))

            # --------------------------------------------------
            # Average anomaly RMSE of the individual ensemble members
            # --------------------------------------------------

            if want(Metric.MEAN_MEMBER_RMSE_ANOM):
                out[Metric.MEAN_MEMBER_RMSE_ANOM.value] = np.sqrt((error_anom ** 2).weighted(weights).mean(dims)).mean(realization_dim)

            # --------------------------------------------------
            # Anomaly ensemble spread
            # --------------------------------------------------

            if want(Metric.SPREAD_ANOM) or want(Metric.SPREAD_ANOM_SKILL_RATIO):
                spread = fc_anom.std(realization_dim).weighted(weights).mean(dims)
                if want(Metric.SPREAD_ANOM):
                    out[Metric.SPREAD_ANOM.value] = spread

                if want(Metric.SPREAD_ANOM_SKILL_RATIO):
                    ens_member_rmse = np.sqrt((error_anom ** 2).weighted(weights).mean((realization_dim, *dims)))
                    out[Metric.SPREAD_ANOM_SKILL_RATIO.value] = spread / ens_member_rmse

            # --------------------------------------------------
            # Anomaly continuous ranked probability score
            # --------------------------------------------------

            if want(Metric.CRPS_ANOM):
                out[Metric.CRPS_ANOM.value] = xs.crps_ensemble(
                    observations=an_anom.mean(dim=realization_dim),
                    forecasts=fc_anom,
                    member_dim=realization_dim,
                    dim=list(dims),
                    weights=weights,
                )

            # --------------------------------------------------
            # Anomaly rank histogram
            # --------------------------------------------------

            if want(Metric.RANK_HISTOGRAM_ANOM):
                out[Metric.RANK_HISTOGRAM_ANOM.value] = xs.rank_histogram(
                    observations=an_anom.mean(dim=realization_dim),
                    forecasts=fc_anom,
                    member_dim=realization_dim,
                )

            # --------------------------------------------------
            # Event-based anomaly ensemble scores
            # --------------------------------------------------

            event_based_scores = (
                Metric.BRIER_ANOM_LOWER,
                Metric.BRIER_ANOM_MIDDLE,
                Metric.BRIER_ANOM_UPPER,
                Metric.ROC_ANOM_LOWER,
                Metric.ROC_ANOM_MIDDLE,
                Metric.ROC_ANOM_UPPER,
            )

            if any(want(metric) for metric in event_based_scores):
                an_anom_for_terciles = an_anom.mean(realization_dim).chunk({time_dim: -1})
                fc_anom_for_terciles = fc_anom.chunk({realization_dim: -1})

                # Terciles
                q33_anom = an_anom_for_terciles.quantile(1 / 3, dim=time_dim).reset_coords(drop=True)
                q67_anom = an_anom_for_terciles.quantile(2 / 3, dim=time_dim).reset_coords(drop=True)

                # Events
                an_event_lower = an_anom_for_terciles <= q33_anom
                an_event_middle = (an_anom_for_terciles > q33_anom) & (an_anom_for_terciles <= q67_anom)
                an_event_upper = an_anom_for_terciles > q67_anom

                fc_event_lower = fc_anom_for_terciles <= q33_anom
                fc_event_middle = (fc_anom_for_terciles > q33_anom) & (fc_anom_for_terciles <= q67_anom)
                fc_event_upper = fc_anom_for_terciles > q67_anom

                # --------------------------------------------------
                # Anomaly Brier score
                # --------------------------------------------------

                if want(Metric.BRIER_ANOM_LOWER):
                    out[Metric.BRIER_ANOM_LOWER.value] = xs.brier_score(
                        observations=an_event_lower,
                        forecasts=fc_event_lower,
                        member_dim=realization_dim,
                        dim=list(dims),
                        fair=fair_correction,
                        weights=weights,
                    )

                if want(Metric.BRIER_ANOM_MIDDLE):
                    out[Metric.BRIER_ANOM_MIDDLE.value] = xs.brier_score(
                        observations=an_event_middle,
                        forecasts=fc_event_middle,
                        member_dim=realization_dim,
                        dim=list(dims),
                        fair=fair_correction,
                        weights=weights,
                    )

                if want(Metric.BRIER_ANOM_UPPER):
                    out[Metric.BRIER_ANOM_UPPER.value] = xs.brier_score(
                        observations=an_event_upper,
                        forecasts=fc_event_upper,
                        member_dim=realization_dim,
                        dim=list(dims),
                        fair=fair_correction,
                        weights=weights,
                    )

                # --------------------------------------------------
                # Anomaly receiving operating characteristic (ROC)
                # --------------------------------------------------

                if want(Metric.ROC_ANOM_LOWER):
                    fc_prob_lower = fc_event_lower.mean(realization_dim)
                    # print("ROC lower tercile probs calculated.")
                    out[Metric.ROC_ANOM_LOWER.value] = xs.roc(
                        observations=an_event_lower,
                        forecasts=fc_prob_lower,
                        dim=list(dims),
                    )

                if want(Metric.ROC_ANOM_MIDDLE):
                    fc_prob_middle = fc_event_middle.mean(realization_dim)
                    # print("ROC middle tercile probs calculated.")
                    out[Metric.ROC_ANOM_MIDDLE.value] = xs.roc(
                        observations=an_event_middle,
                        forecasts=fc_prob_middle,
                        dim=list(dims),
                    )

                if want(Metric.ROC_ANOM_UPPER):
                    fc_prob_upper = fc_event_upper.mean(realization_dim)
                    # print("ROC upper tercile probs calculated.")
                    out[Metric.ROC_ANOM_UPPER.value] = xs.roc(
                        observations=an_event_upper,
                        forecasts=fc_prob_upper,
                        dim=list(dims),
                    )

            # ------------------------------------------------------
            # Probabilistic skills vs climatology
            # ------------------------------------------------------

            if want(Metric.ENS_MEMBER_MSE_SKILL_CLIM):
                mse = (error ** 2).weighted(weights).mean((realization_dim, *dims))
                clim_mse_anom = ((an_anom ** 2).weighted(weights).mean(dims)) * correction
                out[Metric.ENS_MEMBER_MSE_SKILL_CLIM.value] = safe_div(clim_mse_anom - mse, clim_mse_anom)

            if want(Metric.MEAN_MEMBER_MSE_SKILL_CLIM):
                mse = ((error ** 2).weighted(weights).mean(dims)).mean(realization_dim)
                clim_mse_anom = (((an_anom ** 2).weighted(weights).mean(dims)) * correction)
                out[Metric.MEAN_MEMBER_MSE_SKILL_CLIM.value] = safe_div(clim_mse_anom - mse, clim_mse_anom)

            if want(Metric.ENS_MEMBER_MSE_ANOM_SKILL_CLIM):
                mse_anom = (error_anom ** 2).weighted(weights).mean((realization_dim, *dims))
                clim_mse_anom = ((an_anom ** 2).weighted(weights).mean(dims)) * correction
                out[Metric.ENS_MEMBER_MSE_ANOM_SKILL_CLIM.value] = safe_div(clim_mse_anom - mse_anom, clim_mse_anom)

            if want(Metric.MEAN_MEMBER_MSE_ANOM_SKILL_CLIM):
                mse_anom = ((error_anom ** 2).weighted(weights).mean(dims)).mean(realization_dim)
                clim_mse_anom = (((an_anom ** 2).weighted(weights).mean(dims)) * correction)
                out[Metric.MEAN_MEMBER_MSE_ANOM_SKILL_CLIM.value] = safe_div(clim_mse_anom - mse_anom, clim_mse_anom)

    return out


def stack_hour_clim(
    da: xr.DataArray,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
) -> xr.DataArray:
    if clim_period == ClimPeriod.DAYOFYEAR_HOUR:
        dims = ("dayofyear", "hour")
        new_dim = "dayofyear_hour"
        formatter = lambda d, h: f"{int(d):03d}_{int(h):02d}"
    elif clim_period == ClimPeriod.DAY_HOUR:
        dims = ("day", "hour")
        new_dim = "day_hour"
        formatter = lambda d, h: f"{int(d):02d}_{int(h):02d}"
    elif clim_period == ClimPeriod.MONTH_HOUR:
        dims = ("month", "hour")
        new_dim = "month_hour"
        formatter = lambda m, h: f"{int(m):02d}_{int(h):02d}"
    else:
        return da

    if not set(dims) <= set(da.dims):
        return da

    zero_dims = {d: s for d, s in da.sizes.items() if s == 0}
    if zero_dims:
        raise ValueError(
            f"Cannot stack climatology with empty dimension(s): {zero_dims}. "
            f"Full sizes: {dict(da.sizes)}"
        )

    other_dims = [d for d in da.dims if d not in dims]
    da = da.transpose(*other_dims, *dims)

    # Make the two small climatology dims single chunks before reshape.
    da = da.chunk({dims[0]: -1, dims[1]: -1})

    da = da.stack({new_dim: dims}, create_index=False)

    vals0 = da[dims[0]].values
    vals1 = da[dims[1]].values

    da = da.assign_coords({
        new_dim: [
            formatter(v0, v1)
            for v0, v1 in zip(vals0, vals1)
        ]
    })

    return da.drop_vars(dims, errors="ignore")


def _clim_group_dims(clim_period: ClimPeriod = ClimPeriod.MONTH) -> tuple[str, ...]:
    if clim_period == "dayofyear_hour":
        return ("dayofyear", "hour")
    if clim_period == "day_hour":
        return ("day", "hour")
    if clim_period == "month_hour":
        return ("month", "hour")
    return (clim_period,)

def groupby_period(
    da: xr.DataArray,
    time_dim: str,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
):
    if clim_period == "dayofyear_hour":
        return da.assign_coords(
            dayofyear_hour=(
                time_dim,
                da[time_dim].dt.dayofyear.astype(str).str.zfill(3).data
                + "_"
                + da[time_dim].dt.hour.astype(str).str.zfill(2).data
            )
        ).groupby("dayofyear_hour")

    if clim_period == "day_hour":
        return da.assign_coords(
            day_hour=(
                time_dim,
                da[time_dim].dt.day.astype(str).str.zfill(2).data
                + "_"
                + da[time_dim].dt.hour.astype(str).str.zfill(2).data
            )
        ).groupby("day_hour")

    if clim_period == "month_hour":
        return da.assign_coords(
            month_hour=(
                time_dim,
                da[time_dim].dt.month.astype(str).str.zfill(2).data
                + "_"
                + da[time_dim].dt.hour.astype(str).str.zfill(2).data
            )
        ).groupby("month_hour")

    return da.groupby(f"{time_dim}.{clim_period}")


PeriodReference = Literal["init", "valid"]


def _default_period_dim(
    period_reference: PeriodReference,
    clim_period: ClimPeriod,
) -> str:
    return f"{period_reference}_{clim_period}"


def _leadtime_offset(
    leadtime_value: int | float,
    leadtime_unit: LeadtimeUnit,
):
    """Convert a numeric leadtime to a calendar/time offset."""
    if leadtime_unit in {LeadtimeUnit.MONTHS, LeadtimeUnit.YEARS}:
        if not float(leadtime_value).is_integer():
            raise ValueError(
                f"{leadtime_unit.value} leadtime must be an integer, "
                f"got {leadtime_value}."
            )

    if leadtime_unit == LeadtimeUnit.YEARS:
        return pd.DateOffset(years=int(leadtime_value))
    if leadtime_unit == LeadtimeUnit.MONTHS:
        return pd.DateOffset(months=int(leadtime_value))
    if leadtime_unit == LeadtimeUnit.DAYS:
        return pd.Timedelta(days=leadtime_value)
    if leadtime_unit == LeadtimeUnit.HOURS:
        return pd.Timedelta(hours=leadtime_value)

    raise ValueError(f"Unsupported leadtime_unit={leadtime_unit!r}")


def _valid_time_coord(
    da: xr.DataArray,
    *,
    leadtime_dim: str,
    leadtime_unit: LeadtimeUnit,
) -> xr.DataArray:
    """Return valid time with dimensions (init_time, leadtime)."""
    time_dim = da.earthml.guessed_dims.time

    if time_dim is None or time_dim not in da.dims:
        raise ValueError("Could not determine forecast initialization-time dimension.")

    if leadtime_dim not in da.dims:
        raise ValueError(
            f"Leadtime dimension {leadtime_dim!r} is required to build valid time."
        )

    init_times = pd.DatetimeIndex(da[time_dim].values)
    leadtimes = da[leadtime_dim].values

    valid_times = np.empty(
        (len(init_times), len(leadtimes)),
        dtype="datetime64[ns]",
    )

    for i, lead in enumerate(leadtimes):
        valid_times[:, i] = (
            init_times + _leadtime_offset(lead, leadtime_unit)
        ).values

    return xr.DataArray(
        valid_times,
        dims=(time_dim, leadtime_dim),
        coords={
            time_dim: da[time_dim],
            leadtime_dim: da[leadtime_dim],
        },
        name="valid_time",
    )


def _assign_valid_time(
    fc: xr.DataArray,
    an: xr.DataArray,
    *,
    leadtime_dim: str,
    leadtime_unit: LeadtimeUnit | None,
) -> tuple[xr.DataArray, xr.DataArray]:
    if leadtime_unit is None:
        raise ValueError(
            "leadtime_unit must be provided when period_reference='valid'."
        )

    valid_time = _valid_time_coord(
        fc,
        leadtime_dim=leadtime_dim,
        leadtime_unit=leadtime_unit,
    )

    return (
        fc.assign_coords(valid_time=valid_time),
        an.assign_coords(valid_time=valid_time),
    )


def calculate_metrics(
    fc: xr.DataArray,
    an: xr.DataArray,
    dims: str | Sequence[str],
    *,
    metrics: str | Sequence[str] | None = None,
    fc_clim: xr.DataArray | None = None,
    an_clim: xr.DataArray | None = None,
    orography: xr.DataArray | None = None,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    period_reference: PeriodReference = "init",
    period_dim: str | None = None,
    periods_requested: str | Sequence[str] | None = None,
    fair_correction: bool = False,
) -> xr.Dataset:
    def _validate_dims(
        dims: str | Sequence[str] | None,
        name: str,
    ):
        if isinstance(dims, str):
            return [dims]
        elif isinstance(dims, Sequence):
            return list(dims)
        else:
            raise TypeError(
                f"Parameter {name} should be of type str or Sequence[str], "
                f"not {type(dims).__name__}"
            )

    def _validate_metrics(
        metrics: Metric | str | Sequence[Metric | str] | None,
    ) -> list[str]:
        if metrics is None:
            return [m.value for m in Metric]
        if isinstance(metrics, (str, Metric)):
            metrics = [metrics]
        return [Metric(m).value for m in metrics]

    def _gen_period_format(clim_period: str) -> str:
        return {
            "month": "02d",
            "day": "02d",
            "dayofyear": "03d",
            "year": "04d",
        }[clim_period]

    def _possible_periods(
        clim_period: str,
    ) -> tuple[list[str], range | list[tuple[int, int]]]:
        if clim_period == ClimPeriod.MONTH:
            values = range(1, 13)
        elif clim_period == ClimPeriod.DAYOFYEAR:
            values = range(1, 367)
        elif clim_period == ClimPeriod.DAY:
            values = range(1, 32)
        elif clim_period == ClimPeriod.YEAR:
            raise ValueError(
                "'year' climatology has no predefined period list. "
                "Specify periods explicitly or use 'all'."
            )
        elif clim_period == ClimPeriod.DAYOFYEAR_HOUR:
            values = [(d, h) for d in range(1, 367) for h in range(24)]
        elif clim_period == ClimPeriod.DAY_HOUR:
            values = [(d, h) for d in range(1, 32) for h in range(24)]
        elif clim_period == ClimPeriod.MONTH_HOUR:
            values = [(m, h) for m in range(1, 13) for h in range(24)]
        else:
            raise NotImplementedError(f"Unsupported clim_period={clim_period!r}")

        if clim_period == ClimPeriod.DAYOFYEAR_HOUR:
            periods = [f"{d:03d}_{h:02d}" for d, h in values]
        elif clim_period in {ClimPeriod.DAY_HOUR, ClimPeriod.MONTH_HOUR}:
            periods = [f"{d:02d}_{h:02d}" for d, h in values]
        else:
            periods = [f"{p:{_gen_period_format(clim_period)}}" for p in values]

        return periods + ["all"], values

    def _format_period(period, clim_period):
        if clim_period == ClimPeriod.DAYOFYEAR_HOUR:
            day, hour = period
            return f"{day:03d}_{hour:02d}"
        if clim_period == ClimPeriod.DAY_HOUR:
            day, hour = period
            return f"{day:02d}_{hour:02d}"
        if clim_period == ClimPeriod.MONTH_HOUR:
            month, hour = period
            return f"{month:02d}_{hour:02d}"
        return f"{period:{_gen_period_format(clim_period)}}"

    def _validate_periods(
        clim_period: ClimPeriod = ClimPeriod.MONTH,
        periods: str | Sequence[str] | None = None,
    ) -> tuple[list[str], range | list[int | tuple[int, int]]]:
        possible_periods, clim_period_range = _possible_periods(clim_period)

        if periods is None:
            return possible_periods, clim_period_range

        if isinstance(periods, str):
            periods = [periods]
        elif isinstance(periods, Sequence):
            periods = list(periods)
        else:
            raise TypeError(
                "Parameter periods should be of type str, Sequence[str], or None, "
                f"not {type(periods).__name__}"
            )

        if not periods:
            raise ValueError("periods_requested cannot be empty.")

        invalid_periods = [p for p in periods if p not in possible_periods]
        if invalid_periods:
            raise ValueError(
                f"Requested period(s) {invalid_periods} not available. "
                f"Choose one of {possible_periods}"
            )

        filtered_period_range = [
            p
            for p in clim_period_range
            if _format_period(p, clim_period) in periods
        ]

        return periods, filtered_period_range

    def _select_period(
        da: xr.DataArray,
        period,
        period_time: xr.DataArray,
        clim_period: ClimPeriod = ClimPeriod.MONTH,
    ) -> xr.DataArray:
        if clim_period == ClimPeriod.DAYOFYEAR_HOUR:
            day, hour = period
            mask = (
                (period_time.dt.dayofyear == day)
                & (period_time.dt.hour == hour)
            )
        elif clim_period == ClimPeriod.DAY_HOUR:
            day, hour = period
            mask = (
                (period_time.dt.day == day)
                & (period_time.dt.hour == hour)
            )
        elif clim_period == ClimPeriod.MONTH_HOUR:
            month, hour = period
            mask = (
                (period_time.dt.month == month)
                & (period_time.dt.hour == hour)
            )
        else:
            mask = getattr(period_time.dt, clim_period) == period

        return da.where(mask, drop=True)

    def _select_climatology_period(
        da: xr.DataArray,
        period,
        clim_period: ClimPeriod = ClimPeriod.MONTH,
    ) -> xr.DataArray:
        if clim_period == ClimPeriod.DAYOFYEAR_HOUR:
            day, hour = period
            return da.where(
                (da.dayofyear == day) & (da.hour == hour),
                drop=True,
            )
        if clim_period == ClimPeriod.DAY_HOUR:
            day, hour = period
            return da.where(
                (da.day == day) & (da.hour == hour),
                drop=True,
            )
        if clim_period == ClimPeriod.MONTH_HOUR:
            month, hour = period
            return da.where(
                (da.month == month) & (da.hour == hour),
                drop=True,
            )
        return da.where(da[clim_period] == period, drop=True)

    if period_reference not in {"init", "valid"}:
        raise ValueError(
            "period_reference must be either 'init' or 'valid', "
            f"got {period_reference!r}."
        )

    if period_dim is None:
        period_dim = _default_period_dim(period_reference, clim_period)

    valid_dims = _validate_dims(dims, "dims")
    valid_metrics = _validate_metrics(metrics)
    valid_periods, clim_period_range = _validate_periods(
        clim_period,
        periods_requested,
    )

    time_dim = fc.earthml.guessed_dims.time

    if period_reference == "init":
        period_time = fc[time_dim]
    else:
        if "valid_time" not in fc.coords:
            raise ValueError(
                "A 'valid_time' coordinate is required when "
                "period_reference='valid'."
            )
        period_time = fc["valid_time"]

    # Keep the grouping coordinate separately, then remove auxiliary coordinates
    # exactly as before before metric calculation.
    an = an.reset_coords(drop=True)
    fc = fc.reset_coords(drop=True)

    results: list[xr.Dataset] = []

    if "all" in valid_periods:
        all_dims_metrics = core_metrics(
            fc=fc,
            an=an,
            dims=valid_dims,
            metrics=valid_metrics,
            fc_clim=fc_clim,
            an_clim=an_clim,
            orography=orography,
            clim_period=clim_period,
            fair_correction=fair_correction,
        ).expand_dims({period_dim: ["all"]})

        if time_dim not in valid_dims:
            return all_dims_metrics

        results.append(all_dims_metrics)

    for period in clim_period_range:
        fc_p = _select_period(fc, period, period_time, clim_period)
        an_p = _select_period(an, period, period_time, clim_period)

        if fc_p.sizes.get(time_dim, 0) == 0:
            print(
                f"Skipping {period_dim}={_format_period(period, clim_period)}: "
                "forecast subset is empty."
            )
            continue

        if an_p.sizes.get(time_dim, 0) == 0:
            print(
                f"Skipping {period_dim}={_format_period(period, clim_period)}: "
                "analysis subset is empty."
            )
            continue

        # Existing anomaly/climatology semantics are intentionally preserved.
        # Climatologies are indexed by initialization period in core_metrics.
        # Therefore valid-time metric grouping must keep the full climatology.
        if period_reference == "init":
            fc_clim_p = (
                _select_climatology_period(fc_clim, period, clim_period)
                if fc_clim is not None
                else None
            )
            an_clim_p = (
                _select_climatology_period(an_clim, period, clim_period)
                if an_clim is not None
                else None
            )
        else:
            fc_clim_p = fc_clim
            an_clim_p = an_clim

        results.append(
            core_metrics(
                fc=fc_p,
                an=an_p,
                dims=valid_dims,
                metrics=valid_metrics,
                fc_clim=fc_clim_p,
                an_clim=an_clim_p,
                orography=orography,
                clim_period=clim_period,
                fair_correction=fair_correction,
            ).expand_dims(
                {period_dim: [_format_period(period, clim_period)]}
            )
        )

    return xr.concat(
        results,
        dim=period_dim,
        coords="different",
        compat="no_conflicts",
        combine_attrs="override",
    )

def metrics_by_lead_window(
    fc: xr.DataArray,
    an: xr.DataArray,
    dims: str | Sequence[str],
    leadtime_dim: str,
    leadtime_windows: dict[str, Sequence[int]],
    *,
    metrics: str | Sequence[str] | None = None,
    fc_clim: xr.DataArray | None = None,
    an_clim: xr.DataArray | None = None,
    orography: xr.DataArray | None = None,
    leadtime_agg_coord: str = "leadtime_seasonal",
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    period_reference: PeriodReference = "init",
    period_dim: str | None = None,
    periods_requested: str | Sequence[str] | None = None,
    leadtime_unit: LeadtimeUnit | None = None,
    align: bool = True,
    fair_correction: bool = False,
) -> xr.Dataset:
    if align:
        fc, an = xr.unify_chunks(fc, an)
        fc, an = xr.align(fc, an, join="inner")

        if fc_clim is not None and an_clim is not None:
            fc_clim, an_clim = xr.unify_chunks(fc_clim, an_clim)
            fc_clim, an_clim = xr.align(fc_clim, an_clim, join="inner")

    if period_reference == "valid":
        fc, an = _assign_valid_time(
            fc,
            an,
            leadtime_dim=leadtime_dim,
            leadtime_unit=leadtime_unit,
        )

    results: list[xr.Dataset] = []

    dims_with_lead = list(dims)
    if leadtime_dim not in dims_with_lead:
        dims_with_lead.append(leadtime_dim)

    for label, leads in leadtime_windows.items():
        fc_w = fc.sel({leadtime_dim: leads})
        an_w = an.sel({leadtime_dim: leads})

        fc_clim_w = (
            fc_clim.sel({leadtime_dim: leads})
            if fc_clim is not None
            else None
        )
        an_clim_w = (
            an_clim.sel({leadtime_dim: leads})
            if an_clim is not None
            else None
        )

        ds = calculate_metrics(
            fc=fc_w,
            an=an_w,
            dims=dims_with_lead,
            metrics=metrics,
            fc_clim=fc_clim_w,
            an_clim=an_clim_w,
            orography=orography,
            clim_period=clim_period,
            period_reference=period_reference,
            period_dim=period_dim,
            periods_requested=periods_requested,
            fair_correction=fair_correction,
        )

        fixed_vars = {}
        for name, da in ds.data_vars.items():
            if leadtime_agg_coord not in da.dims:
                da = da.expand_dims({leadtime_agg_coord: [label]})
            else:
                da = da.assign_coords({leadtime_agg_coord: [label]})
            fixed_vars[name] = da
        ds = xr.Dataset(fixed_vars, attrs=ds.attrs)

        results.append(ds)

    return xr.concat(
        results,
        dim=leadtime_agg_coord,
        coords="different",
        compat="no_conflicts",
        combine_attrs="override",
    )

def metrics_by_lead(
    fc: xr.DataArray,
    an: xr.DataArray,
    dims: str | Sequence[str],
    leadtime_dim: str,
    *,
    metrics: str | Sequence[str] | None = None,
    fc_clim: xr.DataArray | None = None,
    an_clim: xr.DataArray | None = None,
    orography: xr.DataArray | None = None,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    period_reference: PeriodReference = "init",
    period_dim: str | None = None,
    periods_requested: str | Sequence[str] | None = None,
    leadtime_unit: LeadtimeUnit | None = None,
    align: bool = True,
    fair_correction: bool = False,
) -> xr.Dataset:
    if align:
        fc, an = xr.unify_chunks(fc, an)
        fc, an = xr.align(fc, an, join="inner")

        if fc_clim is not None and an_clim is not None:
            fc_clim, an_clim = xr.unify_chunks(fc_clim, an_clim)
            fc_clim, an_clim = xr.align(fc_clim, an_clim, join="inner")

    if period_reference == "valid":
        fc, an = _assign_valid_time(
            fc,
            an,
            leadtime_dim=leadtime_dim,
            leadtime_unit=leadtime_unit,
        )

    results: list[xr.Dataset] = []

    for lead in fc[leadtime_dim].values:
        fc_l = fc.sel({leadtime_dim: lead}, drop=True)
        an_l = an.sel({leadtime_dim: lead}, drop=True)
        fc_clim_l = (
            fc_clim.sel({leadtime_dim: lead}, drop=True)
            if fc_clim is not None
            else None
        )
        an_clim_l = (
            an_clim.sel({leadtime_dim: lead}, drop=True)
            if an_clim is not None
            else None
        )

        ds = calculate_metrics(
            fc=fc_l,
            an=an_l,
            dims=dims,
            metrics=metrics,
            fc_clim=fc_clim_l,
            an_clim=an_clim_l,
            orography=orography,
            clim_period=clim_period,
            period_reference=period_reference,
            period_dim=period_dim,
            periods_requested=periods_requested,
            fair_correction=fair_correction,
        )

        ds = ds.expand_dims({leadtime_dim: [lead]})
        results.append(ds)

    return xr.concat(
        results,
        dim=leadtime_dim,
        coords="different",
        compat="no_conflicts",
        combine_attrs="override",
    )

def get_metrics(
    an: xr.Dataset,
    fc: xr.Dataset,
    var: str,
    metric_kind: MetricKind,
    leadtime_agg: LeadtimeAgg,
    realization_agg: bool,
    fc_clim: xr.Dataset | None = None,
    an_clim: xr.Dataset | None = None,
    orography_path: str | Path | None = None,
    metrics: str | Sequence[str] | None = None,
    leadtime_windows: dict[str, Sequence[int]] | None = None,
    leadtime_agg_coord: str = "leadtime_seasonal",
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    period_reference: PeriodReference = "init",
    period_dim: str | None = None,
    periods_requested: str | Sequence[str] | None = None,
    leadtime_unit: LeadtimeUnit | None = None,
    align: bool = True,
    fair_correction: bool = False,
) -> xr.Dataset:
    def _check_datasets(
        datasets: dict[str, xr.Dataset],
        reference_ds: xr.Dataset,
        reference_name: str,
        var: str,
        common_dims: Sequence[str],
    ) -> None:
        # Variable existence check
        for name, ds in datasets.items():
            if ds is not None and var not in ds.data_vars:
                raise KeyError(f"Variable '{var}' not found in {name}")

        # Dimension consistency check
        for dim in common_dims:
            if dim not in reference_ds.dims:
                raise ValueError(f"Reference dataset '{reference_name}' is missing dimension '{dim}'")

            for name, ds in datasets.items():
                if ds is None:
                    continue

                if dim not in ds.dims:
                    raise ValueError(f"Dataset '{name}' is missing dimension '{dim}'")

                if reference_ds.sizes[dim] != ds.sizes[dim]:
                    raise ValueError(
                        f"Dimension '{dim}' differs between '{reference_name}' "
                        f"({reference_ds.sizes[dim]}) and '{name}' ({ds.sizes[dim]})"
                    )

                if not np.array_equal(reference_ds[dim].values, ds[dim].values):
                    raise ValueError(
                        f"Coordinate values for dimension '{dim}' differ between "
                        f"'{reference_name}' and '{name}'"
                    )

    def _subset_clims_to_common_periods(
        fc_clim: xr.Dataset,
        an_clim: xr.Dataset,
        clim_period: ClimPeriod,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        for dim in _clim_group_dims(clim_period):
            common = np.intersect1d(
                fc_clim[dim].values,
                an_clim[dim].values,
            )

            if len(common) == 0:
                raise ValueError(
                    f"No common climatology values for dimension '{dim}'."
                )

            fc_clim = fc_clim.sel({dim: common})
            an_clim = an_clim.sel({dim: common})

        return fc_clim, an_clim

    def _rename_fc_dims_like_an(
        fc: xr.Dataset,
        an: xr.Dataset,
    ) -> xr.Dataset:
        rename = {}

        fc_lat = fc.earthml.guessed_dims.latitude
        fc_lon = fc.earthml.guessed_dims.longitude

        an_lat = an.earthml.guessed_dims.latitude
        an_lon = an.earthml.guessed_dims.longitude

        if (
            fc_lat in fc.dims
            and an_lat in an.dims
            and fc_lat != an_lat
        ):
            rename[fc_lat] = an_lat

        if (
            fc_lon in fc.dims
            and an_lon in an.dims
            and fc_lon != an_lon
        ):
            rename[fc_lon] = an_lon

        if rename:
            fc = fc.rename(rename)

        return fc

    fc = _rename_fc_dims_like_an(fc, an)

    if fc_clim is not None and an_clim is not None:
        fc_clim = _rename_fc_dims_like_an(fc_clim, an_clim)

    _check_datasets(
        datasets={
            "an": an,
            "fc": fc,
        },
        reference_ds=fc,
        reference_name="fc",
        var=var,
        common_dims=[
            fc.earthml.guessed_dims.time,
            fc.earthml.guessed_dims.latitude,
            fc.earthml.guessed_dims.longitude,
            fc.earthml.guessed_dims.leadtime,
        ],
    )

    if fc_clim is not None and fc.earthml.guessed_dims.realization is not None:
        _check_datasets(
            datasets={
                "fc": fc,
                "fc_clim": fc_clim,
            },
            reference_ds=fc,
            reference_name="fc",
            var=var,
            common_dims=[
                fc.earthml.guessed_dims.latitude,
                fc.earthml.guessed_dims.longitude,
                fc.earthml.guessed_dims.leadtime,
                fc.earthml.guessed_dims.realization,
            ],
        )

    if an_clim is not None and fc_clim is not None:
        fc_clim, an_clim = _subset_clims_to_common_periods(
            fc_clim,
            an_clim,
            clim_period,
        )

        _check_datasets(
            datasets={
                "fc_clim": fc_clim,
                "an_clim": an_clim,
            },
            reference_ds=fc_clim,
            reference_name="fc_clim",
            var=var,
            common_dims=[
                *_clim_group_dims(clim_period),
                fc.earthml.guessed_dims.latitude,
                fc.earthml.guessed_dims.longitude,
                fc.earthml.guessed_dims.leadtime,
            ],
        )

    an_da = an[var]
    fc_da = fc[var]
    an_clim_da = an_clim[var] if an_clim is not None else None
    fc_clim_da = fc_clim[var] if fc_clim is not None else None

    an_clim_da = stack_hour_clim(an_clim_da, clim_period) if an_clim_da is not None else None
    fc_clim_da = stack_hour_clim(fc_clim_da, clim_period) if fc_clim_da is not None else None

    leadtime_dim = fc_da.earthml.guessed_dims.leadtime

    requested_metrics = (
        [metrics]
        if isinstance(metrics, str)
        else list(metrics)
        if metrics is not None
        else [m.value for m in Metric]
    )

    # Orography
    needs_orography = any(
        metric in {
            Metric.OROGRAPHY.value,
            Metric.OROGRAPHY_GRAD_MAG.value,
        }
        for metric in requested_metrics
    )

    orography = None

    if needs_orography:
        if orography_path is None:
            raise ValueError(
                "orography_path must be provided when requesting "
                "'orography' or 'orography_grad_mag'."
            )

        with xr.open_zarr(
            orography_path,
            consolidated=True,
        ) as oro_ds:
            if "orography" not in oro_ds:
                raise KeyError(
                    f"'orography' not found in {orography_path}. "
                    f"Available variables: {list(oro_ds.data_vars)}"
                )

            orography = oro_ds["orography"].load()

        orography = orography.interp(
            {
                fc_da.earthml.guessed_dims.latitude:
                    fc_da[fc_da.earthml.guessed_dims.latitude],
                fc_da.earthml.guessed_dims.longitude:
                    fc_da[fc_da.earthml.guessed_dims.longitude],
            }
        )

    # Valid-time grouping must happen before collapsing individual leadtimes.
    if period_reference == "valid" and leadtime_agg == "aggregated":
        raise ValueError(
            "period_reference='valid' is not supported with "
            "leadtime_agg='aggregated' because the individual leadtimes have "
            "already been collapsed. Use individual leadtimes or "
            "leadtime_agg='seasonal_window' instead."
        )

    # Aggregate if requested
    if leadtime_agg == "aggregated" and leadtime_windows is not None:
        leadtime_dim = leadtime_agg_coord
        an_da = aggregate_leadtime_da(
            da=an_da,
            windows=leadtime_windows,
            leadtime_dim=an_da.earthml.guessed_dims.leadtime,
            leadtime_agg_coord=leadtime_agg_coord,
        )
        fc_da = aggregate_leadtime_da(
            da=fc_da,
            windows=leadtime_windows,
            leadtime_dim=fc_da.earthml.guessed_dims.leadtime,
            leadtime_agg_coord=leadtime_agg_coord,
        )
        if an_clim_da is not None:
            an_clim_da = aggregate_leadtime_da(
                da=an_clim_da,
                windows=leadtime_windows,
                leadtime_dim=an_clim_da.earthml.guessed_dims.leadtime,
                leadtime_agg_coord=leadtime_agg_coord,
            )
        if fc_clim_da is not None:
            fc_clim_da = aggregate_leadtime_da(
                da=fc_clim_da,
                windows=leadtime_windows,
                leadtime_dim=fc_clim_da.earthml.guessed_dims.leadtime,
                leadtime_agg_coord=leadtime_agg_coord,
            )

    # Ensemble mean if requested
    if realization_agg == True:
        realization_dim_an = an_da.earthml.guessed_dims.realization
        if realization_dim_an is not None:
            an_da = an_da.mean(realization_dim_an)
            an_clim_da = an_clim_da.mean(realization_dim_an) if an_clim_da is not None else None
        realization_dim_fc = fc_da.earthml.guessed_dims.realization
        if realization_dim_fc is not None:
            fc_da = fc_da.mean(realization_dim_fc)
            fc_clim_da = fc_clim_da.mean(realization_dim_fc) if fc_clim_da is not None else None

    def _metric_dims(
        kind: MetricKind,
        da: xr.DataArray
    ) -> tuple[str, ...]:
        return {
            "scalar": (da.earthml.guessed_dims.time, da.earthml.guessed_dims.latitude, da.earthml.guessed_dims.longitude),
            "maps": (da.earthml.guessed_dims.time,),
            "time_lon": (da.earthml.guessed_dims.latitude,),
            "time_lat": (da.earthml.guessed_dims.longitude,),
            "timeseries": (da.earthml.guessed_dims.latitude, da.earthml.guessed_dims.longitude),
        }[kind]

    metric_dims = _metric_dims(metric_kind, fc_da)

    if leadtime_agg == "seasonal_window":
        if leadtime_windows is None:
            raise ValueError("leadtime_windows must be provided when leadtime_agg='seasonal_window'")

        return metrics_by_lead_window(
            fc=fc_da,
            an=an_da,
            fc_clim=fc_clim_da,
            an_clim=an_clim_da,
            orography=orography,
            dims=metric_dims,
            leadtime_dim=fc_da.earthml.guessed_dims.leadtime,
            leadtime_windows=leadtime_windows,
            leadtime_agg_coord=leadtime_agg_coord,
            metrics=metrics,
            clim_period=clim_period,
            period_reference=period_reference,
            period_dim=period_dim,
            periods_requested=periods_requested,
            leadtime_unit=leadtime_unit,
            align=align,
            fair_correction=fair_correction,
        )

    return metrics_by_lead(
        fc=fc_da,
        an=an_da,
        fc_clim=fc_clim_da,
        an_clim=an_clim_da,
        orography=orography,
        dims=metric_dims,
        leadtime_dim=leadtime_dim,
        metrics=metrics,
        clim_period=clim_period,
        period_reference=period_reference,
        period_dim=period_dim,
        periods_requested=periods_requested,
        leadtime_unit=leadtime_unit,
        align=align,
        fair_correction=fair_correction,
    )


def get_scalar_metrics(
    s: Settings,
    *,
    fc_metrics: str | Sequence[str],
    mlfc_metrics: str | Sequence[str],
    metric_agg_mode: MetricAgg,
    leadtime_agg: LeadtimeAgg,
    realization_agg: bool,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    clim_rolling_window: int | None = None,
    clim_time_range: tuple[str, str] | None = None,
    leadtime_units: LeadtimeUnit = LeadtimeUnit.MONTHS,
    leadtime_agg_coord: str = "leadtime",
    force_clim_recalc: bool = False,
    period_reference: PeriodReference = "init",
    period_dim: str | None = None,
    periods_requested: Sequence[str] | None = None,
    wanted_start_periods: Sequence[str] | None = None,
    interpolate: bool = False,
    build_analysis: bool = True,
) -> tuple[xr.Dataset, xr.Dataset]:
    # Backward-compatible alias. New code should use periods_requested.
    if wanted_start_periods is not None:
        if periods_requested is not None:
            raise ValueError(
                "Use only one of periods_requested or legacy "
                "wanted_start_periods."
            )
        periods_requested = wanted_start_periods

    valid_time_range = (s.train_start, s.test_end) if time_range is None else time_range
    fc, an, mlfc = get_and_subset_datasets(
        s,
        leadtime_units=leadtime_units,
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=valid_time_range,
        interpolate=interpolate,
    )

    if mlfc is None:
        raise ValueError(f"ML-corrected forecast is not available for {s.output_name}.")

    fc_clim, an_clim, mlfc_clim = calculate_save_and_subset_climatologies(
        s,
        leadtime_units=leadtime_units,
        force=force_clim_recalc,
        clim_period=clim_period,
        rolling_window=clim_rolling_window,
        rolling_center=True,
        rolling_min_periods=1,
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=clim_time_range,
        time_start=None,
        interpolate=interpolate,
        engine="zarr",
        build_analysis=build_analysis,
        coord_rename_fc=None,
        coord_rename_an=None,
    )

    print(f"Calculating {metric_agg_mode} scalar metrics [fc={fc_metrics}, mlfc={mlfc_metrics}] for {s.output_name}")

    if metric_agg_mode == "global":
        metric_kind = "scalar"
    elif metric_agg_mode in ("spatial_avg", "spatial_rmse"):
        metric_kind = "maps"
        if metric_agg_mode == "spatial_rmse":
            if mlfc_metrics == "rmse":
                mlfc_metrics = "mse"
            elif mlfc_metrics == "rmse_anom":
                mlfc_metrics = "mse_anom"
            elif mlfc_metrics == "nrmse":
                mlfc_metrics = "nmse"
            elif mlfc_metrics == "nrmse_anom":
                mlfc_metrics = "nmse_anom"
            else:
                raise ValueError(f"metric_agg_mode={metric_agg_mode} only supports RMSE metrics.")
            if fc_metrics == "rmse":
                fc_metrics = "mse"
            elif fc_metrics == "rmse_anom":
                fc_metrics = "mse_anom"
            elif fc_metrics == "nrmse":
                fc_metrics = "nmse"
            elif fc_metrics == "nrmse_anom":
                fc_metrics = "nmse_anom"
            else:
                raise ValueError(f"metric_agg_mode={metric_agg_mode} only supports RMSE metrics.")
    else:
        raise ValueError(f"metric_agg_mode={metric_agg_mode} not available. Choose between: 'spatial_avg', 'global', 'spatial_rmse.")

    metric_scalar_fc = get_metrics(
        an=an,
        fc=fc,
        var=s.var_fc,
        metric_kind=metric_kind,
        leadtime_agg=leadtime_agg,
        realization_agg=realization_agg,
        an_clim=an_clim,
        fc_clim=fc_clim,
        metrics=fc_metrics,
        leadtime_windows=s.seasonal_leadtime_windows,
        leadtime_agg_coord=leadtime_agg_coord,
        clim_period=clim_period,
        period_reference=period_reference,
        period_dim=period_dim,
        periods_requested=periods_requested,
        leadtime_unit=leadtime_units,
    )

    metric_scalar_mlfc = get_metrics(
        an=an,
        fc=mlfc,
        var=s.var_fc,
        metric_kind=metric_kind,
        leadtime_agg=leadtime_agg,
        realization_agg=realization_agg,
        an_clim=an_clim,
        fc_clim=mlfc_clim,
        metrics=mlfc_metrics,
        leadtime_windows=s.seasonal_leadtime_windows,
        leadtime_agg_coord=leadtime_agg_coord,
        clim_period=clim_period,
        period_reference=period_reference,
        period_dim=period_dim,
        periods_requested=periods_requested,
        leadtime_unit=leadtime_units,
    )

    fc_metrics_list = [fc_metrics] if isinstance(fc_metrics, str) else list(fc_metrics)
    missing_fc_metrics = [
        metric for metric in fc_metrics_list
        if metric not in metric_scalar_fc.data_vars
    ]
    if missing_fc_metrics:
        print(f"Skipping {s.output_name}: missing forecast metrics {missing_fc_metrics!r}")
        return xr.Dataset(), xr.Dataset()

    mlfc_metrics_list = [mlfc_metrics] if isinstance(mlfc_metrics, str) else list(mlfc_metrics)
    missing_mlfc_metrics = [
        metric for metric in mlfc_metrics_list
        if metric not in metric_scalar_mlfc.data_vars
    ]
    if missing_mlfc_metrics:
        print(f"Skipping {s.output_name}: missing MLFC metrics {missing_mlfc_metrics!r}")
        return xr.Dataset(), xr.Dataset()

    if metric_agg_mode == "global":
        return metric_scalar_fc[fc_metrics_list], metric_scalar_mlfc[mlfc_metrics_list]

    if metric_agg_mode in ("spatial_avg", "spatial_rmse"):
        lat_dim = fc.earthml.guessed_dims.latitude
        lon_dim = fc.earthml.guessed_dims.longitude
        weights = np.cos(np.deg2rad(fc[lat_dim]))

        fc_metrics_da = metric_scalar_fc[fc_metrics_list].weighted(weights).mean(dim=(lat_dim, lon_dim))
        mlfc_metrics_da = metric_scalar_mlfc[mlfc_metrics_list].weighted(weights).mean(dim=(lat_dim, lon_dim))

        return fc_metrics_da, mlfc_metrics_da
