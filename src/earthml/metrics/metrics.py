from typing import cast
from collections.abc import Sequence

import numpy as np
import xarray as xr
import xskillscore as xs

from ..base import (
    Settings,
    LeadtimeUnit,
    aggregate_leadtime_da,
    get_and_subset_datasets,
)

from .climatology import calculate_save_and_subset_climatologies
from .definitions import (
    ClimPeriod,
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


def core_metrics(
    fc: xr.DataArray,
    an: xr.DataArray,
    dims: Sequence[str],
    metrics: Sequence[str],
    *,
    fc_clim: xr.DataArray | None = None,
    an_clim: xr.DataArray | None = None,
    clim_period: ClimPeriod = "month",
    fair_correction: bool = False,
) -> xr.Dataset:    
    time_dim = fc.earthml.guessed_dims.time
    lat_dim = fc.earthml.guessed_dims.latitude
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
    error = fc - an
    metric_dims_no_time = tuple(d for d in dims if d != time_dim)

    # fairness correction for MSSS
    if fair_correction:
        n = an[time_dim].count(time_dim)
        correction = n / (n - 1)
    else:
        correction = 1

    def want(metric: Metric) -> bool:
        return metric.value in metrics

    out = xr.Dataset()

    if want(Metric.BIAS):
        out[Metric.BIAS.value] = error.weighted(weights).mean(dims)

    if want(Metric.MAE):
        out[Metric.MAE.value] = abs(error).weighted(weights).mean(dims)

    if want(Metric.MSE):
        out[Metric.MSE.value] = (error ** 2).weighted(weights).mean(dims)

    if want(Metric.RMSE):
        out[Metric.RMSE.value] = np.sqrt((error ** 2).weighted(weights).mean(dims))

    if want(Metric.NMSE):
        nmse = (error ** 2).weighted(weights).mean(dims)
        an_var_total = (
            ((an - an.weighted(weights).mean(dims)) ** 2)
            .weighted(weights)
            .mean(dims)
        )
        out[Metric.NMSE.value] = nmse / an_var_total

    if want(Metric.NRMSE):
        rmse = np.sqrt((error ** 2).weighted(weights).mean(dims))
        an_std_total = np.sqrt(
            ((an - an.weighted(weights).mean(dims)) ** 2)
            .weighted(weights)
            .mean(dims)
        )
        out[Metric.NRMSE.value] = rmse / an_std_total

    if want(Metric.R2):
        sse = (error ** 2).weighted(weights).sum(dims)
        sst = ((an - an.weighted(weights).mean(dims)) ** 2).weighted(weights).sum(dims)
        out[Metric.R2.value] = 1 - sse / sst

    if want(Metric.CORR):
        out[Metric.CORR.value] = xr.corr(fc, an, dim=time_dim).weighted(weights).mean(metric_dims_no_time)

    if want(Metric.FC_STD) or want(Metric.STD_RATIO):
        fc_std = fc.std(time_dim).weighted(weights).mean(metric_dims_no_time)
        out[Metric.FC_STD.value] = fc_std

    if want(Metric.AN_STD) or want(Metric.STD_RATIO):
        an_std = an.std(time_dim).weighted(weights).mean(metric_dims_no_time)
        out[Metric.AN_STD.value] = an_std

    if want(Metric.STD_RATIO):
        out[Metric.STD_RATIO.value] = fc_std / an_std

    # Probabilistic and ensemble metrics
    if realization_dim in fc.dims:
        fc = fc.chunk({realization_dim: -1})

        if want(Metric.ENS_MEMBER_RMSE):
            out[Metric.ENS_MEMBER_RMSE.value] = np.sqrt(
                (error ** 2)
                .weighted(weights)
                .mean((realization_dim, *dims))
            )

        if want(Metric.MEAN_MEMBER_RMSE):
            out[Metric.MEAN_MEMBER_RMSE.value] = np.sqrt(
                (error ** 2)
                .weighted(weights)
                .mean(dims)
            ).mean(realization_dim)

        if want(Metric.SPREAD) or want(Metric.SPREAD_SKILL_RATIO):
            spread = fc.std(realization_dim).weighted(weights).mean(dims)
            # spread = fc.std(realization_dim).weighted(weights).mean(dims)
            if want(Metric.SPREAD):
                out[Metric.SPREAD.value] = spread

            if want(Metric.SPREAD_SKILL_RATIO):
                ens_member_rmse = np.sqrt(
                    (error ** 2)
                    .weighted(weights)
                    .mean((realization_dim, *dims))
                )
                out[Metric.SPREAD_SKILL_RATIO.value] = spread / ens_member_rmse

        if want(Metric.CRPS):
            out[Metric.CRPS.value] = xs.crps_ensemble(
                observations=an,
                forecasts=fc,
                member_dim=realization_dim,
                dim=list(dims),
                weights=weights,
            )

        if want(Metric.RANK_HISTOGRAM):
            out[Metric.RANK_HISTOGRAM.value] = xs.rank_histogram(
                observations=an,
                forecasts=fc,
                member_dim=realization_dim,
            )

    if fc_clim is not None and an_clim is not None:
        fc_anom = groupby_period(fc, time_dim, clim_period) - fc_clim
        an_anom = groupby_period(an, time_dim, clim_period) - an_clim

        if time_dim in dims:
            fc_anom = fc_anom.chunk({time_dim: -1})
            an_anom = an_anom.chunk({time_dim: -1})

        error_anom = fc_anom - an_anom

        if want(Metric.BIAS_ANOM):
            out[Metric.BIAS_ANOM.value] = error_anom.weighted(weights).mean(dims)

        if want(Metric.MAE_ANOM):
            out[Metric.MAE_ANOM.value] = abs(error_anom).weighted(weights).mean(dims)

        if want(Metric.MSE_ANOM):
            out[Metric.MSE_ANOM.value] = (error_anom ** 2).weighted(weights).mean(dims)

        if want(Metric.RMSE_ANOM):
            out[Metric.RMSE_ANOM.value] = np.sqrt((error_anom ** 2).weighted(weights).mean(dims))

        if want(Metric.ACC):
            out[Metric.ACC.value] = xr.corr(fc_anom, an_anom, dim=time_dim).weighted(weights).mean(metric_dims_no_time)

        if want(Metric.R2_ANOM):
            sse_anom = ((error_anom) ** 2).weighted(weights).sum(dims)
            sst_anom = ((an_anom - an_anom.weighted(weights).mean(dims)) ** 2).weighted(weights).sum(dims)
            out[Metric.R2_ANOM.value] = 1 - sse_anom / sst_anom

        if want(Metric.FC_ANOM_STD) or want(Metric.STD_RATIO_ANOM):
            fc_anom_std = fc_anom.std(time_dim).weighted(weights).mean(metric_dims_no_time)
            out[Metric.FC_ANOM_STD.value] = fc_anom_std

        if want(Metric.AN_ANOM_STD) or want(Metric.STD_RATIO_ANOM):
            an_anom_std = an_anom.std(time_dim).weighted(weights).mean(metric_dims_no_time)
            out[Metric.AN_ANOM_STD.value] = an_anom_std

        if want(Metric.STD_RATIO_ANOM):
            out[Metric.STD_RATIO_ANOM.value] = fc_anom_std / an_anom_std

        if want(Metric.NMSE_ANOM):
            nmse_anom = (error_anom ** 2).weighted(weights).mean(dims)
            an_anom_var_total = (
                ((an_anom - an_anom.weighted(weights).mean(dims)) ** 2)
                .weighted(weights)
                .mean(dims)
            )
            out[Metric.NMSE_ANOM.value] = nmse_anom / an_anom_var_total

        if want(Metric.NRMSE_ANOM):
            rmse_anom = np.sqrt((error_anom ** 2).weighted(weights).mean(dims))
            an_anom_std_total = np.sqrt(
                ((an_anom - an_anom.weighted(weights).mean(dims)) ** 2)
                .weighted(weights)
                .mean(dims)
            )
            out[Metric.NRMSE_ANOM.value] = rmse_anom / an_anom_std_total

        if want(Metric.MSE_SKILL_CLIM):
            mse = (error ** 2).weighted(weights).mean(dims)
            clim_mse_anom = ((an_anom ** 2).weighted(weights).mean(dims)) * correction
            out[Metric.MSE_SKILL_CLIM.value] = safe_div(
                clim_mse_anom - mse,
                clim_mse_anom,
            )

        if want(Metric.MSE_ANOM_SKILL_CLIM):
            mse_anom = (error_anom ** 2).weighted(weights).mean(dims)
            clim_mse_anom = ((an_anom ** 2).weighted(weights).mean(dims)) * correction
            out[Metric.MSE_ANOM_SKILL_CLIM.value] = safe_div(
                clim_mse_anom - mse_anom,
                clim_mse_anom,
            )

        if want(Metric.RMSE_ANOM_SKILL_CLIM):
            rmse_anom = cast(xr.DataArray, np.sqrt((error_anom ** 2).weighted(weights).mean(dims)))
            clim_rmse_anom = cast(xr.DataArray, np.sqrt((an_anom ** 2).weighted(weights).mean(dims))) * correction
            out[Metric.RMSE_ANOM_SKILL_CLIM.value] = safe_div(
                clim_rmse_anom - rmse_anom,
                clim_rmse_anom,
            )

        if want(Metric.MAE_ANOM_SKILL_CLIM):
            mae_anom = abs(error_anom).weighted(weights).mean(dims)
            clim_mae_anom = abs(an_anom).weighted(weights).mean(dims) * correction
            out[Metric.MAE_ANOM_SKILL_CLIM.value] = safe_div(
                clim_mae_anom - mae_anom,
                clim_mae_anom,
            )

        # Ensemble anomaly metrics
        if realization_dim in fc.dims:
            if want(Metric.ENS_MEMBER_RMSE_ANOM):
                out[Metric.ENS_MEMBER_RMSE_ANOM.value] = np.sqrt(
                    (error_anom ** 2)
                    .weighted(weights)
                    .mean((realization_dim, *dims))
                )

            if want(Metric.MEAN_MEMBER_RMSE_ANOM):
                out[Metric.MEAN_MEMBER_RMSE_ANOM.value] = np.sqrt(
                    (error_anom ** 2)
                    .weighted(weights)
                    .mean(dims)
                ).mean(realization_dim)

            if want(Metric.ENS_MEMBER_MSE_ANOM_SKILL_CLIM):
                mse_anom = (error_anom ** 2).weighted(weights).mean((realization_dim, *dims))
                clim_mse_anom = ((an_anom ** 2).weighted(weights).mean(dims)) * correction
                out[Metric.ENS_MEMBER_MSE_ANOM_SKILL_CLIM.value] = safe_div(
                    clim_mse_anom - mse_anom,
                    clim_mse_anom,
                )

            if want(Metric.MEAN_MEMBER_MSE_ANOM_SKILL_CLIM):
                mse_anom = ((error_anom ** 2).weighted(weights).mean(dims)).mean(realization_dim)
                clim_mse_anom = (((an_anom ** 2).weighted(weights).mean(dims)) * correction)
                out[Metric.MEAN_MEMBER_MSE_ANOM_SKILL_CLIM.value] = safe_div(
                    clim_mse_anom - mse_anom,
                    clim_mse_anom,
                )

            if want(Metric.SPREAD_ANOM) or want(Metric.SPREAD_ANOM_SKILL_RATIO):
                spread = fc_anom.std(realization_dim).weighted(weights).mean(dims)
                # spread = fc_anom.std(realization_dim).weighted(weights).mean(dims)
                if want(Metric.SPREAD_ANOM):
                    out[Metric.SPREAD_ANOM.value] = spread

                if want(Metric.SPREAD_ANOM_SKILL_RATIO):
                    ens_member_rmse = np.sqrt(
                        (error_anom ** 2)
                        .weighted(weights)
                        .mean((realization_dim, *dims))
                    )
                    out[Metric.SPREAD_ANOM_SKILL_RATIO.value] = spread / ens_member_rmse

            if want(Metric.CRPS_ANOM):
                out[Metric.CRPS_ANOM.value] = xs.crps_ensemble(
                    observations=an_anom,
                    forecasts=fc_anom,
                    member_dim=realization_dim,
                    dim=list(dims),
                    weights=weights,
                )

            if want(Metric.RANK_HISTOGRAM_ANOM):
                out[Metric.RANK_HISTOGRAM_ANOM.value] = xs.rank_histogram(
                    observations=an_anom,
                    forecasts=fc_anom,
                    member_dim=realization_dim,
                )

            if want(Metric.ROC_ANOM_LOWER) or want(Metric.ROC_ANOM_MIDDLE) or want(Metric.ROC_ANOM_UPPER):
                an_for_terciles = an_anom.chunk({time_dim: -1})
                fc_for_terciles = fc_anom.chunk({realization_dim: -1})

                q33 = an_for_terciles.quantile(1 / 3, dim=time_dim).reset_coords(drop=True)
                q67 = an_for_terciles.quantile(2 / 3, dim=time_dim).reset_coords(drop=True)

                if want(Metric.ROC_ANOM_LOWER):
                    obs_lower = an_for_terciles <= q33
                    fc_prob_lower = (fc_for_terciles <= q33).mean(realization_dim)
                    # print("ROC lower tercile probs calculated.")
                    out[Metric.ROC_ANOM_LOWER.value] = xs.roc(obs_lower, fc_prob_lower, dim=list(dims))
                if want(Metric.ROC_ANOM_MIDDLE):
                    obs_middle = (an_for_terciles > q33) & (an_for_terciles <= q67)
                    fc_prob_middle = (
                        (fc_for_terciles > q33) & (fc_for_terciles <= q67)
                    ).mean(realization_dim)
                    # print("ROC middle tercile probs calculated.")
                    out[Metric.ROC_ANOM_MIDDLE.value] = xs.roc(obs_middle, fc_prob_middle, dim=list(dims))
                if want(Metric.ROC_ANOM_UPPER):
                    obs_upper = an_for_terciles > q67
                    fc_prob_upper = (fc_for_terciles > q67).mean(realization_dim)
                    # print("ROC upper tercile probs calculated.")
                    out[Metric.ROC_ANOM_UPPER.value] = xs.roc(obs_upper, fc_prob_upper, dim=list(dims))

    return out


def stack_hour_clim(
    da: xr.DataArray,
    clim_period: ClimPeriod = "month",
) -> xr.DataArray:
    if clim_period == "dayofyear_hour":
        dims = ("dayofyear", "hour")
        new_dim = "dayofyear_hour"
        formatter = lambda d, h: f"{int(d):03d}_{int(h):02d}"
    elif clim_period == "day_hour":
        dims = ("day", "hour")
        new_dim = "day_hour"
        formatter = lambda d, h: f"{int(d):02d}_{int(h):02d}"
    elif clim_period == "month_hour":
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


def _clim_group_dims(clim_period: ClimPeriod = "month") -> tuple[str, ...]:
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
    clim_period: ClimPeriod = "month",
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


def calculate_metrics(
    fc: xr.DataArray,
    an: xr.DataArray,
    dims: str | Sequence[str],
    *,
    metrics: str | Sequence[str] | None = None,
    fc_clim: xr.DataArray | None = None,
    an_clim: xr.DataArray | None = None,
    clim_period: ClimPeriod = "month",
    period_dim: str = "start_date",
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
            raise TypeError(f"Parameter {name} should be of type str or Sequence[str], not {type(dims).__name__}")
        
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
        if clim_period == "month":
            values = range(1, 13)

        elif clim_period == "dayofyear":
            values = range(1, 367)

        elif clim_period == "day":
            values = range(1, 32)

        elif clim_period == "year":
            raise ValueError(
                "'year' climatology has no predefined period list. "
                "Specify periods explicitly or use 'all'."
            )

        elif clim_period == "dayofyear_hour":
            values = [(d, h) for d in range(1, 367) for h in range(24)]

        elif clim_period == "day_hour":
            values = [(d, h) for d in range(1, 32) for h in range(24)]

        elif clim_period == "month_hour":
            values = [(m, h) for m in range(1, 13) for h in range(24)]

        else:
            raise NotImplementedError(f"Unsupported clim_period={clim_period!r}")

        if clim_period.endswith("_hour"):
            periods = [f"{p0:03d}_{p1:02d}" for p0, p1 in values]
        else:
            periods = [f"{p:{_gen_period_format(clim_period)}}" for p in values]

        return periods + ["all"], values

    def _format_period(period, clim_period):
        if clim_period == "dayofyear_hour":
            d, h = period
            return f"{d:03d}_{h:02d}"
        elif clim_period == "day_hour":
            d, h = period
            return f"{d:02d}_{h:02d}"
        elif clim_period == "month_hour":
            m, h = period
            return f"{m:02d}_{h:02d}"
        else:
            return f"{period:{_gen_period_format(clim_period)}}"

    def _validate_periods(
        clim_period: ClimPeriod = "month",
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
                f"Parameter periods should be of type str, Sequence[str], or None, "
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

        filtered_period_range = [p for p in clim_period_range if _format_period(p, clim_period) in periods]

        return periods, filtered_period_range

    def _select_period(
        da: xr.DataArray,
        period,
        clim_period: ClimPeriod = "month",
        time_dim: str = "time",
    ) -> xr.DataArray:
        if clim_period == "dayofyear_hour":
            day, hour = period
            return da.where(
                (da[time_dim].dt.dayofyear == day)
                & (da[time_dim].dt.hour == hour),
                drop=True,
            )

        if clim_period == "day_hour":
            day, hour = period
            return da.where(
                (da[time_dim].dt.day == day)
                & (da[time_dim].dt.hour == hour),
                drop=True,
            )

        if clim_period == "month_hour":
            month, hour = period
            return da.where(
                (da[time_dim].dt.month == month)
                & (da[time_dim].dt.hour == hour),
                drop=True,
            )

        return da.where(getattr(da[time_dim].dt, clim_period) == period, drop=True)

    def _select_climatology_period(
        da: xr.DataArray,
        period,
        clim_period: ClimPeriod = "month",
    ) -> xr.DataArray:
        if clim_period == "dayofyear_hour":
            day, hour = period
            return da.where(
                (da.dayofyear == day)
                & (da.hour == hour),
                drop=True,
            )

        if clim_period == "day_hour":
            day, hour = period
            return da.where(
                (da.day == day)
                & (da.hour == hour),
                drop=True,
            )

        if clim_period == "month_hour":
            month, hour = period
            return da.where(
                (da.month == month)
                & (da.hour == hour),
                drop=True,
            )

        return da.where(da[clim_period] == period, drop=True)

    valid_dims = _validate_dims(dims, "dims")
    valid_metrics = _validate_metrics(metrics)
    valid_periods, clim_period_range = _validate_periods(clim_period, periods_requested)

    an = an.reset_coords(drop=True)
    fc = fc.reset_coords(drop=True)

    time_dim = fc.earthml.guessed_dims.time

    results: list[xr.Dataset] = []

    if "all" in valid_periods:
        all_dims_metrics = core_metrics(
            fc=fc,
            an=an,
            dims=valid_dims,
            metrics=valid_metrics,
            fc_clim=fc_clim,
            an_clim=an_clim,
            clim_period=clim_period,
            fair_correction=fair_correction,
        ).expand_dims({period_dim: ["all"]})

        if time_dim not in valid_dims:
            return all_dims_metrics

        results.append(all_dims_metrics)

    for period in clim_period_range:
        fc_p = _select_period(fc, period, clim_period, time_dim)
        an_p = _select_period(an, period, clim_period, time_dim)

        fc_clim_p = (
            _select_climatology_period(fc_clim, period, clim_period)
            if fc_clim is not None else None
        )
        an_clim_p = (
            _select_climatology_period(an_clim, period, clim_period)
            if an_clim is not None else None
        )

        results.append(
            core_metrics(
                fc=fc_p,
                an=an_p,
                dims=valid_dims,
                metrics=valid_metrics,
                fc_clim=fc_clim_p,
                an_clim=an_clim_p,
                clim_period=clim_period,
                fair_correction=fair_correction,
            ).expand_dims({period_dim: [_format_period(period, clim_period)]})
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
    leadtime_agg_coord: str = "leadtime_seasonal",
    clim_period: ClimPeriod = "month",
    period_dim: str = "start_date",
    periods_requested: str | Sequence[str] | None = None,
    align: bool = True,
    fair_correction: bool = False,
) -> xr.Dataset:
    if align:
        fc, an = xr.unify_chunks(fc, an)
        fc, an = xr.align(fc, an, join="inner")

        if fc_clim is not None and an_clim is not None:
            fc_clim, an_clim = xr.unify_chunks(fc_clim, an_clim)
            fc_clim, an_clim = xr.align(fc_clim, an_clim, join="inner")

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
            clim_period=clim_period,
            period_dim=period_dim,
            periods_requested=periods_requested,
            fair_correction=fair_correction,
        )

        # Force every metric, including CRPS, to carry the seasonal-window coordinate
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
    clim_period: ClimPeriod = "month",
    period_dim: str = "start_date",
    periods_requested: str | Sequence[str] | None = None,
    align: bool = True,
    fair_correction: bool = False,
) -> xr.Dataset:
    if align:
        fc, an = xr.unify_chunks(fc, an)
        fc, an = xr.align(fc, an, join="inner")

        if fc_clim is not None and an_clim is not None:
            fc_clim, an_clim = xr.unify_chunks(fc_clim, an_clim)
            fc_clim, an_clim = xr.align(fc_clim, an_clim, join="inner")

    results: list[xr.Dataset] = []
    for lead in fc[leadtime_dim].values:
        fc_l = fc.sel({leadtime_dim: lead}, drop=True)
        an_l = an.sel({leadtime_dim: lead}, drop=True)
        fc_clim_l = fc_clim.sel({leadtime_dim: lead}, drop=True) if fc_clim is not None else None
        an_clim_l = an_clim.sel({leadtime_dim: lead}, drop=True) if an_clim is not None else None

        ds = calculate_metrics(
            fc=fc_l,
            an=an_l,
            dims=dims,
            metrics=metrics,
            fc_clim=fc_clim_l,
            an_clim=an_clim_l,
            clim_period=clim_period,
            period_dim=period_dim,
            periods_requested=periods_requested,
            fair_correction=fair_correction,
        )

        ds = ds.expand_dims({leadtime_dim: [lead]})
        results.append(ds)

    out = xr.concat(
        results,
        dim=leadtime_dim,
        coords="different",
        compat="no_conflicts",
        combine_attrs="override",
    )

    return out


def get_metrics(
    an: xr.Dataset,
    fc: xr.Dataset,
    var: str,
    metric_kind: MetricKind,
    leadtime_agg: LeadtimeAgg,
    realization_agg: bool,
    fc_clim: xr.Dataset | None = None,
    an_clim: xr.Dataset | None = None,
    metrics: str | Sequence[str] | None = None,
    leadtime_windows: dict[str, Sequence[int]] | None = None,
    leadtime_agg_coord: str = "leadtime_seasonal",
    clim_period: ClimPeriod = "month",
    period_dim: str = "start_date",
    periods_requested: str | Sequence[str] | None = None,
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
            dims=metric_dims,
            leadtime_dim=fc_da.earthml.guessed_dims.leadtime,
            leadtime_windows=leadtime_windows,
            leadtime_agg_coord=leadtime_agg_coord,
            metrics=metrics,
            clim_period=clim_period,
            period_dim=period_dim,
            periods_requested=periods_requested,
            align=align,
            fair_correction=fair_correction,
        )

    return metrics_by_lead(
        fc=fc_da,
        an=an_da,
        fc_clim=fc_clim_da,
        an_clim=an_clim_da,
        dims=metric_dims,
        leadtime_dim=leadtime_dim,
        metrics=metrics,
        clim_period=clim_period,
        period_dim=period_dim,
        periods_requested=periods_requested,
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
    clim_period: ClimPeriod = "month",
    clim_rolling_window: int | None = None,
    clim_time_range: tuple[str, str] | None = None,
    leadtime_units: LeadtimeUnit = LeadtimeUnit.MONTHS,
    leadtime_agg_coord: str = "leadtime",
    force_clim_recalc: bool = False,
    period_dim: str = "start_months",
    wanted_start_periods: Sequence[str] | None = None,
    interpolate: bool = False,
    build_analysis: bool = True,
) -> tuple[xr.Dataset, xr.Dataset]:
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
        raise ValueError("ML-corrected forecast must be present to produce scatter plot.")

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
        period_dim=period_dim,
        periods_requested=wanted_start_periods,
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
        period_dim=period_dim,
        periods_requested=wanted_start_periods,
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
