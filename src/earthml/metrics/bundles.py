from typing import Any, Sequence

import xarray as xr

from .calculate.constants import METRIC_KIND, METRIC_SECTIONS
from .metrics.deterministic import DeterministicMetrics
from .metrics.correlation import CorrelationMetrics
from .metrics.probabilistic import ProbabilisticMetrics


def _stack_metric_section(section_metrics: dict[str, xr.Dataset]) -> xr.Dataset:
    """
    Concatenate metric datasets along a new 'metric' dimension.

    Each input dataset is expected to have the same variable names
    (e.g. physical variables like sst, t2m) and compatible non-metric dims
    within a section.

    Parameters
    ----------
    section_metrics : dict[str, xr.Dataset]
        Mapping from metric name to metric dataset.

    Returns
    -------
    xr.Dataset
        Dataset with a new 'metric' dimension.
    """
    if not section_metrics:
        return xr.Dataset()

    stacked = []
    for metric_name, ds in section_metrics.items():
        if ds is None:
            continue
        stacked.append(ds.expand_dims(metric=[metric_name]))

    if not stacked:
        return xr.Dataset()

    return xr.concat(stacked, dim="metric", coords='minimal', compat='override')


def build_standard_metric_bundle(
    deterministic: DeterministicMetrics,
    correlation: CorrelationMetrics | None = None,
    probabilistic: ProbabilisticMetrics | None = None,
    norm: str = "std",
    clim_period: str = "month",
    metric_names: Sequence[str] | set[str] | None = None,
    metric_mean_extra_dims: dict[str, str | Sequence[str] | None] | None = None,
) -> dict[str, Any]:
    """
    Build a standard bundle of metrics grouped by output type.

    The returned bundle contains 12 sections in the form:
    - 'scalar_<metric_type>': metrics reduced over time and space
    - 'map_<metric_type>': metrics reduced over time
    - 'timeseries_<metric_type>': metrics reduced over latitude and longitude
    where metric_type can be:
    - all_dims
    - spatial_mean
    - ensemble_mean
    - probabilistic

    Within each section, metrics are concatenated along a new 'metric'
    dimension, while physical variables remain as dataset variables.

    Parameters
    ----------
    deterministic : DeterministicMetrics
        Deterministic metric engine.
    correlation : CorrelationMetrics or None, optional
        Correlation metric engine.
    probabilistic : ProbabilisticMetrics or None, optional
        Probabilistic metric engine.
    norm : str, optional
        Normalization mode for normalized deterministic metrics.
    clim_period : str, optional
        Climatology grouping period for anomaly correlation metrics.
    metric_names : Sequence[str] or set[str] or None, optional
        Metric names to compute. ``None`` computes all supported metrics.

    Returns
    -------
    dict[str, Any]
        Bundle with metadata and section datasets.
    """
    if correlation is not None:
        if correlation.model_names != deterministic.model_names:
            raise ValueError("correlation and deterministic metric objects must share model_names")
        if correlation.dims != deterministic.dims:
            raise ValueError("correlation and deterministic metric objects must share dims")

    if probabilistic is not None:
        if probabilistic.model_names != deterministic.model_names:
            raise ValueError("probabilistic and deterministic metric objects must share model_names")
        if probabilistic.dims != deterministic.dims:
            raise ValueError("probabilistic and deterministic metric objects must share dims")

    realization_dim = deterministic.dims.realization
    has_realization = (
        realization_dim is not None
        and any(ds.sizes.get(realization_dim, 0) > 1 for ds in deterministic.model_data)
    )
    requested_metrics = None if metric_names is None else {str(name) for name in metric_names}

    def _normalize_dims(value: str | Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    def _wants(metric_name: str) -> bool:
        return requested_metrics is None or metric_name in requested_metrics

    def _append_extra_dims(
        base_dims: tuple[str, ...],
        extra_dims: str | Sequence[str] | None,
    ) -> tuple[str, ...]:
        out = list(base_dims)
        for dim in _normalize_dims(extra_dims):
            if dim not in out:
                out.append(dim)
        return tuple(out)

    full_dims = (
        deterministic.dims.time,
        deterministic.dims.latitude,
        deterministic.dims.longitude,
    )
    ts_dims = (deterministic.dims.latitude, deterministic.dims.longitude)

    extra_dims = metric_mean_extra_dims or {}
    scalar_dims = _append_extra_dims(full_dims, extra_dims.get("scalar"))
    timeseries_dims = _append_extra_dims(ts_dims, extra_dims.get("timeseries"))
    map_dims = _append_extra_dims((deterministic.dims.time,), extra_dims.get("map"))

    reduce_dims = dict(
        zip(
            METRIC_KIND,
            (scalar_dims, timeseries_dims, map_dims),
            strict=True,
        )
    )
    section_metrics = {
        metric_type: {kind: {} for kind in reduce_dims}
        for metric_type in METRIC_SECTIONS
    }

    # ------------------
    # All dims
    # ------------------
    for kind, dims in reduce_dims.items():
        sm = section_metrics["all_dims"][kind]

        if _wants("rmse"):
            sm["rmse"] = deterministic.rmse(metric_mean_dims=dims)
        if _wants("rmse_skill_clim"):
            sm["rmse_skill_clim"] = deterministic.rmse_skill_clim(metric_mean_dims=dims, period=clim_period)
        if _wants("crmse"):
            sm["crmse"] = deterministic.crmse(metric_mean_dims=dims)
        if _wants("mae"):
            sm["mae"] = deterministic.mae(metric_mean_dims=dims)
        if _wants("bias"):
            sm["bias"] = deterministic.bias(metric_mean_dims=dims)
        if _wants("error_std"):
            sm["error_std"] = deterministic.error_std(metric_mean_dims=dims)
        if _wants("variance_ratio"):
            sm["variance_ratio"] = deterministic.variance_ratio(metric_mean_dims=dims)
        if _wants("r2"):
            sm["r2"] = deterministic.r2(metric_mean_dims=dims)
        if _wants("nrmse"):
            sm["nrmse"] = deterministic.nrmse(metric_mean_dims=dims, norm=norm)
        if _wants("ncrmse"):
            sm["ncrmse"] = deterministic.ncrmse(metric_mean_dims=dims, norm=norm)
        if _wants("nmae"):
            sm["nmae"] = deterministic.nmae(metric_mean_dims=dims, norm=norm)
        if _wants("nbias"):
            sm["nbias"] = deterministic.nbias(metric_mean_dims=dims, norm=norm)

    # ------------------
    # Spatial mean
    # ------------------
    sdims = ts_dims
    tdim = tuple(dim for dim in scalar_dims if dim not in sdims)
    sm = section_metrics["spatial_mean"]["scalar"]

    if _wants("rmse"):
        sm["rmse"] = deterministic.rmse_of_mean(sdims, tdim)
    if _wants("rmse_skill_clim"):
        sm["rmse_skill_clim"] = deterministic.rmse_skill_clim_of_mean(sdims, tdim, period=clim_period)
    if _wants("crmse"):
        sm["crmse"] = deterministic.crmse_of_mean(sdims, tdim)
    if _wants("mae"):
        sm["mae"] = deterministic.mae_of_mean(sdims, tdim)
    if _wants("bias"):
        sm["bias"] = deterministic.bias_of_mean(sdims, tdim)
    if _wants("error_std"):
        sm["error_std"] = deterministic.error_std_of_mean(sdims, tdim)
    if _wants("variance_ratio"):
        sm["variance_ratio"] = deterministic.variance_ratio_of_mean(sdims, tdim)
    if _wants("r2"):
        sm["r2"] = deterministic.r2_of_mean(sdims, tdim)
    if _wants("nrmse"):
        sm["nrmse"] = deterministic.nrmse_of_mean(sdims, tdim, norm=norm)
    if _wants("ncrmse"):
        sm["ncrmse"] = deterministic.ncrmse_of_mean(sdims, tdim, norm=norm)
    if _wants("nmae"):
        sm["nmae"] = deterministic.nmae_of_mean(sdims, tdim, norm=norm)
    if _wants("nbias"):
        sm["nbias"] = deterministic.nbias_of_mean(sdims, tdim, norm=norm)

    # ------------------
    # Ensemble mean
    # ------------------
    if has_realization:
        rdim = realization_dim

        for kind, dims in reduce_dims.items():
            sm = section_metrics["ensemble_mean"][kind]

            if _wants("rmse"):
                sm["rmse"] = deterministic.rmse_of_mean(rdim, dims)
            if _wants("rmse_skill_clim"):
                sm["rmse_skill_clim"] = deterministic.rmse_skill_clim_of_mean(rdim, dims, period=clim_period)
            if _wants("crmse"):
                sm["crmse"] = deterministic.crmse_of_mean(rdim, dims)
            if _wants("mae"):
                sm["mae"] = deterministic.mae_of_mean(rdim, dims)
            if _wants("bias"):
                sm["bias"] = deterministic.bias_of_mean(rdim, dims)
            if _wants("error_std"):
                sm["error_std"] = deterministic.error_std_of_mean(rdim, dims)
            if _wants("variance_ratio"):
                sm["variance_ratio"] = deterministic.variance_ratio_of_mean(rdim, dims)
            if _wants("r2"):
                sm["r2"] = deterministic.r2_of_mean(rdim, dims)
            if _wants("nrmse"):
                sm["nrmse"] = deterministic.nrmse_of_mean(rdim, dims, norm=norm)
            if _wants("ncrmse"):
                sm["ncrmse"] = deterministic.ncrmse_of_mean(rdim, dims, norm=norm)
            if _wants("nmae"):
                sm["nmae"] = deterministic.nmae_of_mean(rdim, dims, norm=norm)
            if _wants("nbias"):
                sm["nbias"] = deterministic.nbias_of_mean(rdim, dims, norm=norm)

    # ------------------
    # Probabilistic
    # ------------------
    if probabilistic is not None and has_realization:
        for kind, dims in reduce_dims.items():
            sm = section_metrics["probabilistic"][kind]

            if _wants("crps"):
                sm["crps"] = probabilistic.crps(dims)
            if _wants("spread"):
                sm["spread"] = probabilistic.spread(dims)
            if _wants("spread_error_ratio"):
                sm["spread_error_ratio"] = probabilistic.spread_error_ratio(dims)
            # sm["crps_skill_clim"] = probabilistic.crps_skill_clim(full_dims)

    # ------------------
    # Correlation
    # ------------------
    if correlation is not None:
        for kind, dims in reduce_dims.items():
            sm = section_metrics["all_dims"][kind]
            if _wants("corr"):
                sm["corr"] = correlation.corr(dims)
            if _wants("clim_acc"):
                sm["clim_acc"] = correlation.clim_anom_corr(dims, period=clim_period)
            if _wants("spatial_acc"):
                sm["spatial_acc"] = correlation.spatial_anom_corr(dims)

        sm = section_metrics["spatial_mean"]["scalar"]
        if _wants("corr"):
            sm["corr"] = correlation.corr_of_mean(sdims, tdim)
        if _wants("clim_acc"):
            sm["clim_acc"] = correlation.clim_anom_corr_of_mean(sdims, tdim, period=clim_period)

        if has_realization:
            rdim = realization_dim

            for kind, dims in reduce_dims.items():
                sm = section_metrics["ensemble_mean"][kind]

                if _wants("corr"):
                    sm["corr"] = correlation.corr_of_mean(rdim, dims)
                if _wants("clim_acc"):
                    sm["clim_acc"] = correlation.clim_anom_corr_of_mean(rdim, dims, period=clim_period)
                if _wants("spatial_acc"):
                    sm["spatial_acc"] = correlation.spatial_anom_corr_of_mean(rdim, dims)

    # ------------------
    # Stack
    # ------------------
    result = {}

    for metric_type in METRIC_SECTIONS:
        for kind in METRIC_KIND:
            result[f"{kind}_{metric_type}"] = _stack_metric_section(section_metrics[metric_type][kind])

    # ------------------
    # Attributes
    # ------------------
    descriptions = {
        "scalar": "Metrics reduced over time and space.",
        "map": "Metrics reduced over time; spatial fields retained.",
        "timeseries": "Metrics reduced over latitude and longitude; time series retained.",
    }

    for key, ds in result.items():
        section = key.split("_")[0]

        ds.attrs.update(
            {
                "section": section,
                "description": descriptions[section],
                "norm": norm,
                "clim_period": clim_period,
            }
        )

    return result
