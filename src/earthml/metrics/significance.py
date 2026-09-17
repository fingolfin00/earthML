from pathlib import Path
from collections.abc import Sequence

import numpy as np
import xarray as xr

from rich.progress import track

from ..base import (
    ClimPeriod,
    aggregate_leadtime_da,
)
from .improvement import metric_difference_improvement
from .metrics import (
    LeadtimeAgg,
    MetricKind,
    core_metrics,
    stack_hour_clim,
)


def _bootstrap_indices(
    n_samples: int,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate one paired bootstrap sample.

    block_size=1:
        Standard IID bootstrap.

    block_size>1:
        Moving-block bootstrap preserving short-range temporal
        dependence.

    The returned sample always contains exactly n_samples indices.
    """

    if n_samples < 2:
        raise ValueError(
            f"Bootstrap requires at least 2 samples, got {n_samples}."
        )

    if block_size < 1:
        raise ValueError(
            f"block_size must be >= 1, got {block_size}."
        )

    if block_size > n_samples:
        raise ValueError(
            f"block_size={block_size} exceeds "
            f"n_samples={n_samples}."
        )

    # Standard paired bootstrap.
    if block_size == 1:
        return rng.integers(
            0,
            n_samples,
            size=n_samples,
        )

    # Moving-block bootstrap.
    n_blocks = int(
        np.ceil(n_samples / block_size)
    )

    max_start = n_samples - block_size

    starts = rng.integers(
        0,
        max_start + 1,
        size=n_blocks,
    )

    indices = np.concatenate(
        [
            np.arange(
                start,
                start + block_size,
            )
            for start in starts
        ]
    )

    return indices[:n_samples]


def _paired_resample(
    *arrays: xr.DataArray,
    time_dim: str,
    indices: np.ndarray,
) -> tuple[xr.DataArray, ...]:
    """
    Apply exactly the same bootstrap indices to multiple DataArrays.

    Original time-coordinate values are preserved, including duplicate
    times introduced by bootstrap resampling. This is important for
    climatology/anomaly metrics.
    """

    indexer = xr.DataArray(
        indices,
        dims=(time_dim,),
    )

    return tuple(
        da.isel({time_dim: indexer})
        for da in arrays
    )


def _calculate_metric(
    fc: xr.DataArray,
    an: xr.DataArray,
    *,
    metric: str,
    dims: Sequence[str],
    fc_clim: xr.DataArray | None,
    an_clim: xr.DataArray | None,
    orography: xr.DataArray | None,
    clim_period: ClimPeriod,
    fair_correction: bool,
) -> xr.DataArray:
    """
    Calculate one metric through the standard metric engine.
    """

    result = core_metrics(
        fc=fc,
        an=an,
        dims=dims,
        metrics=[metric],
        fc_clim=fc_clim,
        an_clim=an_clim,
        orography=orography,
        clim_period=clim_period,
        fair_correction=fair_correction,
    )

    if metric not in result:
        raise KeyError(
            f"Metric {metric!r} was not produced by core_metrics(). "
            f"Available metrics: {list(result.data_vars)}"
        )

    return result[metric]


def bootstrap_metric_improvement(
    fc: xr.DataArray,
    mlfc: xr.DataArray,
    an: xr.DataArray,
    *,
    metric: str,
    dims: str | Sequence[str],
    time_dim: str | None = None,
    n_bootstrap: int = 1000,
    block_size: int = 1,
    confidence_level: float = 0.95,
    seed: int | None = 42,
    fc_clim: xr.DataArray | None = None,
    mlfc_clim: xr.DataArray | None = None,
    an_clim: xr.DataArray | None = None,
    orography: xr.DataArray | None = None,
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    fair_correction: bool = False,
    return_bootstrap: bool = False,
) -> xr.Dataset:
    """
    Estimate statistical significance of ML forecast improvement using
    a paired bootstrap.

    FC, MLFC and analysis are always resampled using exactly the same
    temporal indices.

    The statistic tested is the absolute difference improvement:

        improvement = improvement(metric_fc, metric_mlfc)

    where positive values always mean that MLFC is better according to
    the metric-specific improvement semantics.

    Significance is determined from a percentile bootstrap confidence
    interval:

        CI entirely > 0:
            significant improvement

        CI entirely < 0:
            significant degradation

        CI contains 0:
            not statistically significant

    Parameters
    ----------
    fc
        Original forecast.
    mlfc
        ML-corrected forecast.
    an
        Analysis / reference field.
    metric
        Metric to bootstrap.
    dims
        Dimensions reduced by the metric. For metric maps this will
        typically be only the time dimension.
    time_dim
        Bootstrap dimension. If None, inferred from the forecast.
    n_bootstrap
        Number of bootstrap replicates.
    block_size
        Number of consecutive time samples per moving block.
        block_size=1 gives an IID bootstrap.
    confidence_level
        Bootstrap confidence level, e.g. 0.95.
    seed
        Random seed. None gives non-reproducible sampling.
    fc_clim
        Original-forecast climatology when required.
    mlfc_clim
        ML-corrected-forecast climatology when required.
    an_clim
        Analysis climatology when required.
    orography
        Orography field when required by the metric.
    clim_period
        Climatology grouping period.
    fair_correction
        Pass through to core_metrics().
    return_bootstrap
        Include the full bootstrap distribution in the output.

    Returns
    -------
    xr.Dataset
        Dataset containing:

        improvement
            Original-sample improvement estimate.

        ci_lower
            Lower bootstrap confidence bound.

        ci_upper
            Upper bootstrap confidence bound.

        probability_improvement
            Fraction of bootstrap replicates with improvement > 0.

        significant
            True where the confidence interval excludes zero.

        significant_improvement
            True where the confidence interval is entirely > 0.

        significant_degradation
            True where the confidence interval is entirely < 0.

        bootstrap
            Full bootstrap distribution, only when
            return_bootstrap=True.
    """

    if isinstance(dims, str):
        dims = [dims]
    else:
        dims = list(dims)

    if n_bootstrap < 2:
        raise ValueError(
            f"n_bootstrap must be >= 2, got {n_bootstrap}."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1, "
            f"got {confidence_level}."
        )

    # -------------------------------------------------------------
    # Align paired data
    # -------------------------------------------------------------

    fc, mlfc, an = xr.align(
        fc,
        mlfc,
        an,
        join="inner",
    )

    if time_dim is None:
        time_dim = fc.earthml.guessed_dims.time

    if time_dim is None:
        raise ValueError(
            "Could not determine the time dimension."
        )

    if time_dim not in fc.dims:
        raise ValueError(
            f"time_dim={time_dim!r} is not present in FC dimensions "
            f"{fc.dims}."
        )

    if time_dim not in dims:
        raise ValueError(
            f"Bootstrap dimension {time_dim!r} must be included in "
            f"metric reduction dims {dims}."
        )

    n_samples = fc.sizes[time_dim]

    if n_samples < 2:
        raise ValueError(
            f"Bootstrap requires at least 2 samples along "
            f"{time_dim!r}, got {n_samples}."
        )

    if block_size > n_samples:
        raise ValueError(
            f"block_size={block_size} exceeds the number of "
            f"samples ({n_samples})."
        )

    # -------------------------------------------------------------
    # Original metric estimates
    # -------------------------------------------------------------

    fc_metric = _calculate_metric(
        fc,
        an,
        metric=metric,
        dims=dims,
        fc_clim=fc_clim,
        an_clim=an_clim,
        orography=orography,
        clim_period=clim_period,
        fair_correction=fair_correction,
    )

    mlfc_metric = _calculate_metric(
        mlfc,
        an,
        metric=metric,
        dims=dims,
        fc_clim=mlfc_clim,
        an_clim=an_clim,
        orography=orography,
        clim_period=clim_period,
        fair_correction=fair_correction,
    )

    improvement = metric_difference_improvement(
        fc_metric,
        mlfc_metric,
        metric,
    )

    # -------------------------------------------------------------
    # Bootstrap
    # -------------------------------------------------------------

    rng = np.random.default_rng(seed)

    bootstrap_improvements: list[xr.DataArray] = []

    for bootstrap_index in track(
        range(n_bootstrap),
        description=f"Bootstrap {metric}",
    ):
        indices = _bootstrap_indices(
            n_samples,
            block_size=block_size,
            rng=rng,
        )

        fc_boot, mlfc_boot, an_boot = _paired_resample(
            fc,
            mlfc,
            an,
            time_dim=time_dim,
            indices=indices,
        )

        fc_metric_boot = _calculate_metric(
            fc_boot,
            an_boot,
            metric=metric,
            dims=dims,
            fc_clim=fc_clim,
            an_clim=an_clim,
            orography=orography,
            clim_period=clim_period,
            fair_correction=fair_correction,
        )

        mlfc_metric_boot = _calculate_metric(
            mlfc_boot,
            an_boot,
            metric=metric,
            dims=dims,
            fc_clim=mlfc_clim,
            an_clim=an_clim,
            orography=orography,
            clim_period=clim_period,
            fair_correction=fair_correction,
        )

        improvement_boot = metric_difference_improvement(
            fc_metric_boot,
            mlfc_metric_boot,
            metric,
        )

        bootstrap_improvements.append(
            improvement_boot.expand_dims(
                bootstrap=[bootstrap_index]
            )
        )

    bootstrap = xr.concat(
        bootstrap_improvements,
        dim="bootstrap",
    )

    # -------------------------------------------------------------
    # Percentile confidence interval
    # -------------------------------------------------------------

    alpha = 1.0 - confidence_level

    ci_lower = bootstrap.quantile(
        alpha / 2.0,
        dim="bootstrap",
    )

    ci_upper = bootstrap.quantile(
        1.0 - alpha / 2.0,
        dim="bootstrap",
    )

    ci_lower = ci_lower.drop_vars("quantile")
    ci_upper = ci_upper.drop_vars("quantile")

    # -------------------------------------------------------------
    # Improvement probability and significance
    # -------------------------------------------------------------

    probability_improvement = (
        bootstrap > 0
    ).mean("bootstrap")

    significant_improvement = ci_lower > 0
    significant_degradation = ci_upper < 0

    significant = (
        significant_improvement
        | significant_degradation
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    improvement.attrs = fc_metric.attrs.copy()
    improvement.attrs["long_name"] = (
        f"{metric} improvement"
    )

    ci_lower.attrs = improvement.attrs.copy()
    ci_lower.attrs["long_name"] = (
        f"{metric} improvement "
        f"{confidence_level:.0%} CI lower"
    )

    ci_upper.attrs = improvement.attrs.copy()
    ci_upper.attrs["long_name"] = (
        f"{metric} improvement "
        f"{confidence_level:.0%} CI upper"
    )

    probability_improvement.attrs = {
        "long_name": (
            f"Probability that {metric} improves"
        ),
        "units": "",
    }

    significant.attrs = {
        "long_name": (
            f"{metric} improvement statistically significant"
        ),
        "units": "",
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
    }

    significant_improvement.attrs = {
        "long_name": (
            f"{metric} statistically significant improvement"
        ),
        "units": "",
    }

    significant_degradation.attrs = {
        "long_name": (
            f"{metric} statistically significant degradation"
        ),
        "units": "",
    }

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------

    data_vars = {
        "improvement": improvement,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "probability_improvement": probability_improvement,
        "significant": significant,
        "significant_improvement": significant_improvement,
        "significant_degradation": significant_degradation,
    }

    if return_bootstrap:
        data_vars["bootstrap"] = bootstrap

    out = xr.Dataset(data_vars)

    out.attrs.update(
        {
            "metric": metric,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "block_size": block_size,
            "bootstrap_method": (
                "iid"
                if block_size == 1
                else "moving_block"
            ),
            "seed": seed,
        }
    )

    return out

def _format_period(
    period,
    clim_period: ClimPeriod,
) -> str:
    if clim_period == ClimPeriod.DAYOFYEAR_HOUR:
        day, hour = period
        return f"{day:03d}_{hour:02d}"

    if clim_period == ClimPeriod.DAY_HOUR:
        day, hour = period
        return f"{day:02d}_{hour:02d}"

    if clim_period == ClimPeriod.MONTH_HOUR:
        month, hour = period
        return f"{month:02d}_{hour:02d}"

    formats = {
        ClimPeriod.MONTH: "02d",
        ClimPeriod.DAY: "02d",
        ClimPeriod.DAYOFYEAR: "03d",
        ClimPeriod.YEAR: "04d",
    }

    return f"{period:{formats[clim_period]}}"


def _possible_periods(
    clim_period: ClimPeriod,
):
    if clim_period == ClimPeriod.MONTH:
        return list(range(1, 13))

    if clim_period == ClimPeriod.DAY:
        return list(range(1, 32))

    if clim_period == ClimPeriod.DAYOFYEAR:
        return list(range(1, 367))

    if clim_period == ClimPeriod.DAYOFYEAR_HOUR:
        return [
            (day, hour)
            for day in range(1, 367)
            for hour in range(24)
        ]

    if clim_period == ClimPeriod.DAY_HOUR:
        return [
            (day, hour)
            for day in range(1, 32)
            for hour in range(24)
        ]

    if clim_period == ClimPeriod.MONTH_HOUR:
        return [
            (month, hour)
            for month in range(1, 13)
            for hour in range(24)
        ]

    if clim_period == ClimPeriod.YEAR:
        return []

    raise NotImplementedError(
        f"Unsupported clim_period={clim_period!r}"
    )


def _select_period(
    da: xr.DataArray,
    period,
    *,
    clim_period: ClimPeriod,
    time_dim: str,
) -> xr.DataArray:
    if clim_period == ClimPeriod.DAYOFYEAR_HOUR:
        day, hour = period
        return da.where(
            (da[time_dim].dt.dayofyear == day)
            & (da[time_dim].dt.hour == hour),
            drop=True,
        )

    if clim_period == ClimPeriod.DAY_HOUR:
        day, hour = period
        return da.where(
            (da[time_dim].dt.day == day)
            & (da[time_dim].dt.hour == hour),
            drop=True,
        )

    if clim_period == ClimPeriod.MONTH_HOUR:
        month, hour = period
        return da.where(
            (da[time_dim].dt.month == month)
            & (da[time_dim].dt.hour == hour),
            drop=True,
        )

    return da.where(
        getattr(
            da[time_dim].dt,
            str(clim_period),
        ) == period,
        drop=True,
    )


def _requested_periods(
    clim_period: ClimPeriod,
    periods_requested: str | Sequence[str] | None,
):
    possible_values = _possible_periods(clim_period)

    possible_labels = [
        _format_period(period, clim_period)
        for period in possible_values
    ]

    possible_labels.append("all")

    if periods_requested is None:
        return possible_labels, possible_values

    if isinstance(periods_requested, str):
        requested = [periods_requested]
    else:
        requested = list(periods_requested)

    invalid = [
        period
        for period in requested
        if period not in possible_labels
    ]

    if invalid:
        raise ValueError(
            f"Requested periods {invalid} are not available. "
            f"Choose from {possible_labels}."
        )

    selected_values = [
        period
        for period in possible_values
        if _format_period(period, clim_period) in requested
    ]

    return requested, selected_values


def _bootstrap_by_period(
    fc: xr.DataArray,
    mlfc: xr.DataArray,
    an: xr.DataArray,
    *,
    metric: str,
    dims: Sequence[str],
    fc_clim: xr.DataArray | None,
    mlfc_clim: xr.DataArray | None,
    an_clim: xr.DataArray | None,
    clim_period: ClimPeriod,
    period_dim: str,
    periods_requested: str | Sequence[str] | None,
    n_bootstrap: int,
    block_size: int,
    confidence_level: float,
    seed: int | None,
    fair_correction: bool,
) -> xr.Dataset:
    """
    Bootstrap one metric for all requested start periods.
    """

    time_dim = fc.earthml.guessed_dims.time

    requested, period_values = _requested_periods(
        clim_period,
        periods_requested,
    )

    results: list[xr.Dataset] = []

    # ----------------------------------------------------------
    # All cases
    # ----------------------------------------------------------

    if "all" in requested:
        result = bootstrap_metric_improvement(
            fc=fc,
            mlfc=mlfc,
            an=an,
            metric=metric,
            dims=dims,
            time_dim=time_dim,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            confidence_level=confidence_level,
            seed=seed,
            fc_clim=fc_clim,
            mlfc_clim=mlfc_clim,
            an_clim=an_clim,
            clim_period=clim_period,
            fair_correction=fair_correction,
            return_bootstrap=False,
        )

        results.append(
            result.expand_dims({
                period_dim: ["all"]
            })
        )

    # ----------------------------------------------------------
    # Individual periods
    # ----------------------------------------------------------

    for period in period_values:
        fc_p = _select_period(
            fc,
            period,
            clim_period=clim_period,
            time_dim=time_dim,
        )

        mlfc_p = _select_period(
            mlfc,
            period,
            clim_period=clim_period,
            time_dim=time_dim,
        )

        an_p = _select_period(
            an,
            period,
            clim_period=clim_period,
            time_dim=time_dim,
        )

        if fc_p.sizes.get(time_dim, 0) < 2:
            print(
                f"Skipping significance "
                f"{period_dim}="
                f"{_format_period(period, clim_period)}: "
                "not enough forecast samples."
            )
            continue

        if mlfc_p.sizes.get(time_dim, 0) < 2:
            continue

        if an_p.sizes.get(time_dim, 0) < 2:
            continue

        result = bootstrap_metric_improvement(
            fc=fc_p,
            mlfc=mlfc_p,
            an=an_p,
            metric=metric,
            dims=dims,
            time_dim=time_dim,
            n_bootstrap=n_bootstrap,
            block_size=min(
                block_size,
                fc_p.sizes[time_dim],
            ),
            confidence_level=confidence_level,
            seed=seed,
            fc_clim=fc_clim,
            mlfc_clim=mlfc_clim,
            an_clim=an_clim,
            clim_period=clim_period,
            fair_correction=fair_correction,
            return_bootstrap=False,
        )

        results.append(
            result.expand_dims({
                period_dim: [
                    _format_period(
                        period,
                        clim_period,
                    )
                ]
            })
        )

    if not results:
        return xr.Dataset()

    return xr.concat(
        results,
        dim=period_dim,
        coords="different",
        compat="no_conflicts",
        combine_attrs="override",
    )


def _bootstrap_by_lead(
    fc: xr.DataArray,
    mlfc: xr.DataArray,
    an: xr.DataArray,
    *,
    metric: str,
    dims: Sequence[str],
    leadtime_dim: str,
    fc_clim: xr.DataArray | None,
    mlfc_clim: xr.DataArray | None,
    an_clim: xr.DataArray | None,
    clim_period: ClimPeriod,
    period_dim: str,
    periods_requested: str | Sequence[str] | None,
    n_bootstrap: int,
    block_size: int,
    confidence_level: float,
    seed: int | None,
    fair_correction: bool,
) -> xr.Dataset:
    results: list[xr.Dataset] = []

    for lead in fc[leadtime_dim].values:
        fc_l = fc.sel(
            {leadtime_dim: lead},
            drop=True,
        )

        mlfc_l = mlfc.sel(
            {leadtime_dim: lead},
            drop=True,
        )

        an_l = an.sel(
            {leadtime_dim: lead},
            drop=True,
        )

        fc_clim_l = (
            fc_clim.sel(
                {leadtime_dim: lead},
                drop=True,
            )
            if fc_clim is not None
            else None
        )

        mlfc_clim_l = (
            mlfc_clim.sel(
                {leadtime_dim: lead},
                drop=True,
            )
            if mlfc_clim is not None
            else None
        )

        an_clim_l = (
            an_clim.sel(
                {leadtime_dim: lead},
                drop=True,
            )
            if an_clim is not None
            else None
        )

        result = _bootstrap_by_period(
            fc_l,
            mlfc_l,
            an_l,
            metric=metric,
            dims=dims,
            fc_clim=fc_clim_l,
            mlfc_clim=mlfc_clim_l,
            an_clim=an_clim_l,
            clim_period=clim_period,
            period_dim=period_dim,
            periods_requested=periods_requested,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            confidence_level=confidence_level,
            seed=seed,
            fair_correction=fair_correction,
        )

        if not result.data_vars:
            continue

        results.append(
            result.expand_dims({
                leadtime_dim: [lead]
            })
        )

    if not results:
        return xr.Dataset()

    return xr.concat(
        results,
        dim=leadtime_dim,
        coords="different",
        compat="no_conflicts",
        combine_attrs="override",
    )


def _bootstrap_by_lead_window(
    fc: xr.DataArray,
    mlfc: xr.DataArray,
    an: xr.DataArray,
    *,
    metric: str,
    dims: Sequence[str],
    leadtime_dim: str,
    leadtime_windows: dict[str, Sequence[int]],
    leadtime_agg_coord: str,
    fc_clim: xr.DataArray | None,
    mlfc_clim: xr.DataArray | None,
    an_clim: xr.DataArray | None,
    clim_period: ClimPeriod,
    period_dim: str,
    periods_requested: str | Sequence[str] | None,
    n_bootstrap: int,
    block_size: int,
    confidence_level: float,
    seed: int | None,
    fair_correction: bool,
) -> xr.Dataset:
    results: list[xr.Dataset] = []

    dims_with_lead = list(dims)

    if leadtime_dim not in dims_with_lead:
        dims_with_lead.append(leadtime_dim)

    for label, leads in leadtime_windows.items():
        fc_w = fc.sel({
            leadtime_dim: leads
        })

        mlfc_w = mlfc.sel({
            leadtime_dim: leads
        })

        an_w = an.sel({
            leadtime_dim: leads
        })

        fc_clim_w = (
            fc_clim.sel({
                leadtime_dim: leads
            })
            if fc_clim is not None
            else None
        )

        mlfc_clim_w = (
            mlfc_clim.sel({
                leadtime_dim: leads
            })
            if mlfc_clim is not None
            else None
        )

        an_clim_w = (
            an_clim.sel({
                leadtime_dim: leads
            })
            if an_clim is not None
            else None
        )

        result = _bootstrap_by_period(
            fc_w,
            mlfc_w,
            an_w,
            metric=metric,
            dims=dims_with_lead,
            fc_clim=fc_clim_w,
            mlfc_clim=mlfc_clim_w,
            an_clim=an_clim_w,
            clim_period=clim_period,
            period_dim=period_dim,
            periods_requested=periods_requested,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            confidence_level=confidence_level,
            seed=seed,
            fair_correction=fair_correction,
        )

        if not result.data_vars:
            continue

        result = result.expand_dims({
            leadtime_agg_coord: [label]
        })

        results.append(result)

    if not results:
        return xr.Dataset()

    return xr.concat(
        results,
        dim=leadtime_agg_coord,
        coords="different",
        compat="no_conflicts",
        combine_attrs="override",
    )


def get_metric_improvement_significance(
    an: xr.Dataset,
    fc: xr.Dataset,
    mlfc: xr.Dataset,
    *,
    var_fc: str,
    metric: str,
    var_an: str | None = None,
    metric_kind: MetricKind = "maps",
    leadtime_agg: LeadtimeAgg = "single",
    realization_agg: bool = False,
    fc_clim: xr.Dataset | None = None,
    mlfc_clim: xr.Dataset | None = None,
    an_clim: xr.Dataset | None = None,
    leadtime_windows: dict[str, Sequence[int]] | None = None,
    leadtime_agg_coord: str = "leadtime_seasonal",
    clim_period: ClimPeriod = ClimPeriod.MONTH,
    period_dim: str = "start_date",
    periods_requested: str | Sequence[str] | None = None,
    n_bootstrap: int = 1000,
    block_size: int = 1,
    confidence_level: float = 0.95,
    seed: int | None = 42,
    align: bool = True,
    fair_correction: bool = False,
) -> xr.Dataset:
    """
    Calculate bootstrap significance of FC -> MLFC metric improvement.

    This is the significance analogue of get_metrics().

    Positive improvement always means MLFC is better according to the
    metric-specific improvement semantics.

    The output has the same lead/start-period structure as the
    corresponding metric maps.

    Returned variables
    ------------------
    improvement
    ci_lower
    ci_upper
    probability_improvement
    significant
    significant_improvement
    significant_degradation
    """

    if metric_kind != "maps":
        raise NotImplementedError(
            "Improvement significance currently supports only "
            "metric_kind='maps'."
        )

    if var_an is None:
        var_an = var_fc

    # ----------------------------------------------------------
    # Validate variables
    # ----------------------------------------------------------

    if var_fc not in fc:
        raise KeyError(
            f"Variable {var_fc!r} not found in FC."
        )

    if var_fc not in mlfc:
        raise KeyError(
            f"Variable {var_fc!r} not found in MLFC."
        )

    if var_an not in an:
        raise KeyError(
            f"Variable {var_an!r} not found in analysis."
        )

    # ----------------------------------------------------------
    # Extract DataArrays
    # ----------------------------------------------------------

    fc_da = fc[var_fc]
    mlfc_da = mlfc[var_fc]
    an_da = an[var_an]

    fc_clim_da = (
        fc_clim[var_fc]
        if fc_clim is not None
        else None
    )

    mlfc_clim_da = (
        mlfc_clim[var_fc]
        if mlfc_clim is not None
        else None
    )

    an_clim_da = (
        an_clim[var_an]
        if an_clim is not None
        else None
    )

    # Same climatology representation used by get_metrics().
    fc_clim_da = (
        stack_hour_clim(
            fc_clim_da,
            clim_period,
        )
        if fc_clim_da is not None
        else None
    )

    mlfc_clim_da = (
        stack_hour_clim(
            mlfc_clim_da,
            clim_period,
        )
        if mlfc_clim_da is not None
        else None
    )

    an_clim_da = (
        stack_hour_clim(
            an_clim_da,
            clim_period,
        )
        if an_clim_da is not None
        else None
    )

    # ----------------------------------------------------------
    # Align the paired samples
    # ----------------------------------------------------------

    if align:
        fc_da, mlfc_da, an_da = xr.unify_chunks(
            fc_da,
            mlfc_da,
            an_da,
        )

        fc_da, mlfc_da, an_da = xr.align(
            fc_da,
            mlfc_da,
            an_da,
            join="inner",
        )

    # ----------------------------------------------------------
    # Lead aggregation
    # ----------------------------------------------------------

    leadtime_dim = fc_da.earthml.guessed_dims.leadtime

    if (
        leadtime_agg == "aggregated"
        and leadtime_windows is not None
    ):
        fc_da = aggregate_leadtime_da(
            da=fc_da,
            windows=leadtime_windows,
            leadtime_dim=leadtime_dim,
            leadtime_agg_coord=leadtime_agg_coord,
        )

        mlfc_da = aggregate_leadtime_da(
            da=mlfc_da,
            windows=leadtime_windows,
            leadtime_dim=leadtime_dim,
            leadtime_agg_coord=leadtime_agg_coord,
        )

        an_da = aggregate_leadtime_da(
            da=an_da,
            windows=leadtime_windows,
            leadtime_dim=leadtime_dim,
            leadtime_agg_coord=leadtime_agg_coord,
        )

        if fc_clim_da is not None:
            fc_clim_da = aggregate_leadtime_da(
                da=fc_clim_da,
                windows=leadtime_windows,
                leadtime_dim=leadtime_dim,
                leadtime_agg_coord=leadtime_agg_coord,
            )

        if mlfc_clim_da is not None:
            mlfc_clim_da = aggregate_leadtime_da(
                da=mlfc_clim_da,
                windows=leadtime_windows,
                leadtime_dim=leadtime_dim,
                leadtime_agg_coord=leadtime_agg_coord,
            )

        if an_clim_da is not None:
            an_clim_da = aggregate_leadtime_da(
                da=an_clim_da,
                windows=leadtime_windows,
                leadtime_dim=leadtime_dim,
                leadtime_agg_coord=leadtime_agg_coord,
            )

        leadtime_dim = leadtime_agg_coord

    # ----------------------------------------------------------
    # Ensemble mean if requested
    # ----------------------------------------------------------

    if realization_agg:
        fc_realization_dim = (
            fc_da.earthml.guessed_dims.realization
        )

        if fc_realization_dim is not None:
            fc_da = fc_da.mean(
                fc_realization_dim
            )

            if fc_clim_da is not None:
                fc_clim_da = fc_clim_da.mean(
                    fc_realization_dim
                )

        mlfc_realization_dim = (
            mlfc_da.earthml.guessed_dims.realization
        )

        if mlfc_realization_dim is not None:
            mlfc_da = mlfc_da.mean(
                mlfc_realization_dim
            )

            if mlfc_clim_da is not None:
                mlfc_clim_da = mlfc_clim_da.mean(
                    mlfc_realization_dim
                )

        an_realization_dim = (
            an_da.earthml.guessed_dims.realization
        )

        if an_realization_dim is not None:
            an_da = an_da.mean(
                an_realization_dim
            )

            if an_clim_da is not None:
                an_clim_da = an_clim_da.mean(
                    an_realization_dim
                )

    # ----------------------------------------------------------
    # Map metric dimensions
    # ----------------------------------------------------------

    time_dim = fc_da.earthml.guessed_dims.time

    dims = [time_dim]

    # ----------------------------------------------------------
    # Seasonal-window aggregation
    # ----------------------------------------------------------

    if leadtime_agg == "seasonal_window":
        if leadtime_windows is None:
            raise ValueError(
                "leadtime_windows must be provided when "
                "leadtime_agg='seasonal_window'."
            )

        return _bootstrap_by_lead_window(
            fc_da,
            mlfc_da,
            an_da,
            metric=metric,
            dims=dims,
            leadtime_dim=leadtime_dim,
            leadtime_windows=leadtime_windows,
            leadtime_agg_coord=leadtime_agg_coord,
            fc_clim=fc_clim_da,
            mlfc_clim=mlfc_clim_da,
            an_clim=an_clim_da,
            clim_period=clim_period,
            period_dim=period_dim,
            periods_requested=periods_requested,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            confidence_level=confidence_level,
            seed=seed,
            fair_correction=fair_correction,
        )

    # ----------------------------------------------------------
    # Single / pre-aggregated leads
    # ----------------------------------------------------------

    return _bootstrap_by_lead(
        fc_da,
        mlfc_da,
        an_da,
        metric=metric,
        dims=dims,
        leadtime_dim=leadtime_dim,
        fc_clim=fc_clim_da,
        mlfc_clim=mlfc_clim_da,
        an_clim=an_clim_da,
        clim_period=clim_period,
        period_dim=period_dim,
        periods_requested=periods_requested,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        confidence_level=confidence_level,
        seed=seed,
        fair_correction=fair_correction,
    )
