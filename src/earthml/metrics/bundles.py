from typing import Any
import xarray as xr

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
) -> dict[str, Any]:
    """
    Build a standard bundle of metrics grouped by output type.

    The returned bundle contains 9 sections in the form:
    - 'scalar_<metric_type>': metrics reduced over time and space
    - 'map_<metric_type>': metrics reduced over time
    - 'timeseries_<metric_type>': metrics reduced over latitude and longitude
    where metric_type can be:
    - deterministic
    - ensemble
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

    full_dims = (
        deterministic.dims.time,
        deterministic.dims.latitude,
        deterministic.dims.longitude,
    )
    map_dims = deterministic.dims.time
    ts_dims = (deterministic.dims.latitude, deterministic.dims.longitude)

    reduce_dims = {
        "scalar": full_dims,
        "map": map_dims,
        "timeseries": ts_dims,
    }

    sections = ("deterministic", "ensemble", "probabilistic")

    section_metrics = {
        s: {"scalar": {}, "map": {}, "timeseries": {}}
        for s in sections
    }

    # ------------------
    # Deterministic
    # ------------------
    for section, dims in reduce_dims.items():
        sm = section_metrics["deterministic"][section]
        sm["rmse"] = deterministic.rmse(metric_mean_dims=dims)
        sm["rmse_skill_clim"] = deterministic.rmse_skill_clim(metric_mean_dims=dims, period=clim_period)
        sm["crmse"] = deterministic.crmse(metric_mean_dims=dims)
        sm["mae"] = deterministic.mae(metric_mean_dims=dims)
        sm["bias"] = deterministic.bias(metric_mean_dims=dims)
        sm["error_std"] = deterministic.error_std(metric_mean_dims=dims)
        sm["variance_ratio"] = deterministic.variance_ratio(metric_mean_dims=dims)
        sm["r2"] = deterministic.r2(metric_mean_dims=dims)
        sm["nrmse"] = deterministic.nrmse(metric_mean_dims=dims, norm=norm)
        sm["ncrmse"] = deterministic.ncrmse(metric_mean_dims=dims, norm=norm)
        sm["nmae"] = deterministic.nmae(metric_mean_dims=dims, norm=norm)
        sm["nbias"] = deterministic.nbias(metric_mean_dims=dims, norm=norm)

    # ------------------
    # Ensemble
    # ------------------
    if has_realization:
        rdim = realization_dim

        for section, dims in reduce_dims.items():
            sm = section_metrics["ensemble"][section]

            sm["rmse"] = deterministic.rmse_of_mean(rdim, dims)
            sm["rmse_skill_clim"] = deterministic.rmse_skill_clim_of_mean(rdim, dims, period=clim_period)
            sm["crmse"] = deterministic.crmse_of_mean(rdim, dims)
            sm["mae"] = deterministic.mae_of_mean(rdim, dims)
            sm["bias"] = deterministic.bias_of_mean(rdim, dims)
            sm["error_std"] = deterministic.error_std_of_mean(rdim, dims)
            sm["variance_ratio"] = deterministic.variance_ratio_of_mean(rdim, dims)
            sm["r2"] = deterministic.r2_of_mean(rdim, dims)
            sm["nrmse"] = deterministic.nrmse_of_mean(rdim, dims, norm=norm)
            sm["ncrmse"] = deterministic.ncrmse_of_mean(rdim, dims, norm=norm)
            sm["nmae"] = deterministic.nmae_of_mean(rdim, dims, norm=norm)
            sm["nbias"] = deterministic.nbias_of_mean(rdim, dims, norm=norm)

    # ------------------
    # Probabilistic
    # ------------------
    if probabilistic is not None and has_realization:
        for section, dims in reduce_dims.items():
            sm = section_metrics["probabilistic"][section]

            sm["crps"] = probabilistic.crps(dims)
            sm["spread"] = probabilistic.spread(dims)
            sm["spread_error_ratio"] = probabilistic.spread_error_ratio(dims)
            # sm["crps_skill_clim"] = probabilistic.crps_skill_clim(full_dims)

    # ------------------
    # Correlation
    # ------------------
    if correlation is not None:
        for section, dims in reduce_dims.items():
            sm = section_metrics["deterministic"][section]

            sm["corr"] = correlation.corr(dims)
            sm["clim_acc"] = correlation.clim_anom_corr(dims, period=clim_period)
            sm["spatial_acc"] = correlation.spatial_anom_corr(dims)

        if has_realization:
            rdim = realization_dim

            for section, dims in reduce_dims.items():
                sm = section_metrics["ensemble"][section]

                sm["corr"] = correlation.corr_of_mean(rdim, dims)
                sm["clim_acc"] = correlation.clim_anom_corr_of_mean(rdim, dims, period=clim_period)
                sm["spatial_acc"] = correlation.spatial_anom_corr_of_mean(rdim, dims)

    # ------------------
    # Stack
    # ------------------
    result = {}

    for s in sections:
        result[f"scalar_{s}"] = _stack_metric_section(section_metrics[s]["scalar"])
        result[f"map_{s}"] = _stack_metric_section(section_metrics[s]["map"])
        result[f"timeseries_{s}"] = _stack_metric_section(section_metrics[s]["timeseries"])

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
