from collections.abc import Callable
from typing import Any, Sequence

import xarray as xr

from .calculate.constants import METRIC_KIND, METRIC_SECTIONS
from .metrics.deterministic import DeterministicMetrics
from .metrics.correlation import CorrelationMetrics
from .metrics.probabilistic import ProbabilisticMetrics

MetricProgressHook = Callable[[str, str, str], None]

_DETERMINISTIC_METRICS = (
    "rmse",
    "rmse_skill_clim",
    "crmse",
    "mae",
    "bias",
    "error_std",
    "variance_ratio",
    "r2",
    "nrmse",
    "ncrmse",
    "nmae",
    "nbias",
)
_CORRELATION_METRICS = ("corr", "clim_acc", "spatial_acc")
_SPATIAL_MEAN_CORRELATION_METRICS = ("corr", "clim_acc")
_PROBABILISTIC_METRICS = ("crps", "spread", "spread_error_ratio")


def _normalize_requested_metrics(
    metric_names: Sequence[str] | set[str] | None,
) -> set[str] | None:
    if metric_names is None:
        return None
    return {str(name) for name in metric_names}


def _normalize_requested_kinds(
    metric_kinds: str | Sequence[str] | set[str] | None,
) -> tuple[str, ...]:
    if metric_kinds is None:
        return tuple(METRIC_KIND)
    if isinstance(metric_kinds, str):
        raw_kinds = (metric_kinds,)
    else:
        raw_kinds = tuple(str(kind) for kind in metric_kinds)
    return tuple(kind for kind in METRIC_KIND if kind in raw_kinds)


def _has_realization(deterministic: DeterministicMetrics) -> bool:
    realization_dim = deterministic.dims.realization
    return (
        realization_dim is not None
        and any(ds.sizes.get(realization_dim, 0) > 1 for ds in deterministic.model_data)
    )


def _count_requested(
    requested_metrics: set[str] | None,
    supported_metrics: Sequence[str],
) -> int:
    if requested_metrics is None:
        return len(supported_metrics)
    return sum(1 for metric_name in supported_metrics if metric_name in requested_metrics)


def count_standard_metric_bundle_steps(
    deterministic: DeterministicMetrics,
    correlation: CorrelationMetrics | None = None,
    probabilistic: ProbabilisticMetrics | None = None,
    metric_names: Sequence[str] | set[str] | None = None,
    metric_kinds: str | Sequence[str] | set[str] | None = None,
) -> int:
    """
    Return the number of metric computations performed by ``build_standard_metric_bundle``.
    """
    requested_metrics = _normalize_requested_metrics(metric_names)
    requested_kinds = _normalize_requested_kinds(metric_kinds)
    has_realization = _has_realization(deterministic)

    deterministic_count = _count_requested(requested_metrics, _DETERMINISTIC_METRICS)
    correlation_count = _count_requested(requested_metrics, _CORRELATION_METRICS)
    spatial_mean_correlation_count = _count_requested(
        requested_metrics,
        _SPATIAL_MEAN_CORRELATION_METRICS,
    )
    probabilistic_count = _count_requested(requested_metrics, _PROBABILISTIC_METRICS)

    steps = len(requested_kinds) * deterministic_count
    if "scalar" in requested_kinds:
        steps += deterministic_count  # spatial_mean/scalar

    if has_realization:
        steps += len(requested_kinds) * deterministic_count

    if correlation is not None:
        steps += len(requested_kinds) * correlation_count
        if "scalar" in requested_kinds:
            steps += spatial_mean_correlation_count
        if has_realization:
            steps += len(requested_kinds) * correlation_count

    if probabilistic is not None and has_realization:
        steps += len(requested_kinds) * probabilistic_count

    return steps


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
    metric_kinds: str | Sequence[str] | set[str] | None = None,
    progress_hook: MetricProgressHook | None = None,
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
    metric_kinds : str or Sequence[str] or set[str] or None, optional
        Metric output kinds to compute from ``METRIC_KIND``.
    progress_hook : callable or None, optional
        Callback invoked after each completed metric computation with
        ``(metric_name, metric_type, kind)``.

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
    has_realization = _has_realization(deterministic)
    requested_metrics = _normalize_requested_metrics(metric_names)
    requested_kinds = _normalize_requested_kinds(metric_kinds)

    def _normalize_dims(value: str | Sequence[str] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    def _wants(metric_name: str) -> bool:
        return requested_metrics is None or metric_name in requested_metrics

    def _record_metric(
        target: dict[str, xr.Dataset],
        *,
        metric_name: str,
        metric_type: str,
        kind: str,
        compute: Callable[[], xr.Dataset],
    ) -> None:
        if not _wants(metric_name):
            return
        target[metric_name] = compute()
        if progress_hook is not None:
            progress_hook(metric_name, metric_type, kind)

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
        metric_type: {kind: {} for kind in requested_kinds}
        for metric_type in METRIC_SECTIONS
    }

    # ------------------
    # All dims
    # ------------------
    for kind in requested_kinds:
        dims = reduce_dims[kind]
        sm = section_metrics["all_dims"][kind]

        _record_metric(
            sm,
            metric_name="rmse",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.rmse(metric_mean_dims=dims),
        )
        _record_metric(
            sm,
            metric_name="rmse_skill_clim",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.rmse_skill_clim(metric_mean_dims=dims, period=clim_period),
        )
        _record_metric(
            sm,
            metric_name="crmse",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.crmse(metric_mean_dims=dims),
        )
        _record_metric(
            sm,
            metric_name="mae",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.mae(metric_mean_dims=dims),
        )
        _record_metric(
            sm,
            metric_name="bias",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.bias(metric_mean_dims=dims),
        )
        _record_metric(
            sm,
            metric_name="error_std",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.error_std(metric_mean_dims=dims),
        )
        _record_metric(
            sm,
            metric_name="variance_ratio",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.variance_ratio(metric_mean_dims=dims),
        )
        _record_metric(
            sm,
            metric_name="r2",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.r2(metric_mean_dims=dims),
        )
        _record_metric(
            sm,
            metric_name="nrmse",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.nrmse(metric_mean_dims=dims, norm=norm),
        )
        _record_metric(
            sm,
            metric_name="ncrmse",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.ncrmse(metric_mean_dims=dims, norm=norm),
        )
        _record_metric(
            sm,
            metric_name="nmae",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.nmae(metric_mean_dims=dims, norm=norm),
        )
        _record_metric(
            sm,
            metric_name="nbias",
            metric_type="all_dims",
            kind=kind,
            compute=lambda dims=dims: deterministic.nbias(metric_mean_dims=dims, norm=norm),
        )

    # ------------------
    # Spatial mean
    # ------------------
    sdims = ts_dims
    tdim = tuple(dim for dim in scalar_dims if dim not in sdims)
    if "scalar" in requested_kinds:
        sm = section_metrics["spatial_mean"]["scalar"]

        _record_metric(
            sm,
            metric_name="rmse",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.rmse_of_mean(sdims, tdim),
        )
        _record_metric(
            sm,
            metric_name="rmse_skill_clim",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.rmse_skill_clim_of_mean(sdims, tdim, period=clim_period),
        )
        _record_metric(
            sm,
            metric_name="crmse",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.crmse_of_mean(sdims, tdim),
        )
        _record_metric(
            sm,
            metric_name="mae",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.mae_of_mean(sdims, tdim),
        )
        _record_metric(
            sm,
            metric_name="bias",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.bias_of_mean(sdims, tdim),
        )
        _record_metric(
            sm,
            metric_name="error_std",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.error_std_of_mean(sdims, tdim),
        )
        _record_metric(
            sm,
            metric_name="variance_ratio",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.variance_ratio_of_mean(sdims, tdim),
        )
        _record_metric(
            sm,
            metric_name="r2",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.r2_of_mean(sdims, tdim),
        )
        _record_metric(
            sm,
            metric_name="nrmse",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.nrmse_of_mean(sdims, tdim, norm=norm),
        )
        _record_metric(
            sm,
            metric_name="ncrmse",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.ncrmse_of_mean(sdims, tdim, norm=norm),
        )
        _record_metric(
            sm,
            metric_name="nmae",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.nmae_of_mean(sdims, tdim, norm=norm),
        )
        _record_metric(
            sm,
            metric_name="nbias",
            metric_type="spatial_mean",
            kind="scalar",
            compute=lambda: deterministic.nbias_of_mean(sdims, tdim, norm=norm),
        )

    # ------------------
    # Ensemble mean
    # ------------------
    if has_realization:
        rdim = realization_dim

        for kind in requested_kinds:
            dims = reduce_dims[kind]
            sm = section_metrics["ensemble_mean"][kind]

            _record_metric(
                sm,
                metric_name="rmse",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.rmse_of_mean(rdim, dims),
            )
            _record_metric(
                sm,
                metric_name="rmse_skill_clim",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.rmse_skill_clim_of_mean(rdim, dims, period=clim_period),
            )
            _record_metric(
                sm,
                metric_name="crmse",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.crmse_of_mean(rdim, dims),
            )
            _record_metric(
                sm,
                metric_name="mae",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.mae_of_mean(rdim, dims),
            )
            _record_metric(
                sm,
                metric_name="bias",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.bias_of_mean(rdim, dims),
            )
            _record_metric(
                sm,
                metric_name="error_std",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.error_std_of_mean(rdim, dims),
            )
            _record_metric(
                sm,
                metric_name="variance_ratio",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.variance_ratio_of_mean(rdim, dims),
            )
            _record_metric(
                sm,
                metric_name="r2",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.r2_of_mean(rdim, dims),
            )
            _record_metric(
                sm,
                metric_name="nrmse",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.nrmse_of_mean(rdim, dims, norm=norm),
            )
            _record_metric(
                sm,
                metric_name="ncrmse",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.ncrmse_of_mean(rdim, dims, norm=norm),
            )
            _record_metric(
                sm,
                metric_name="nmae",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.nmae_of_mean(rdim, dims, norm=norm),
            )
            _record_metric(
                sm,
                metric_name="nbias",
                metric_type="ensemble_mean",
                kind=kind,
                compute=lambda dims=dims: deterministic.nbias_of_mean(rdim, dims, norm=norm),
            )

    # ------------------
    # Probabilistic
    # ------------------
    if probabilistic is not None and has_realization:
        for kind in requested_kinds:
            dims = reduce_dims[kind]
            sm = section_metrics["probabilistic"][kind]

            _record_metric(
                sm,
                metric_name="crps",
                metric_type="probabilistic",
                kind=kind,
                compute=lambda dims=dims: probabilistic.crps(dims),
            )
            _record_metric(
                sm,
                metric_name="spread",
                metric_type="probabilistic",
                kind=kind,
                compute=lambda dims=dims: probabilistic.spread(dims),
            )
            _record_metric(
                sm,
                metric_name="spread_error_ratio",
                metric_type="probabilistic",
                kind=kind,
                compute=lambda dims=dims: probabilistic.spread_error_ratio(dims),
            )
            # sm["crps_skill_clim"] = probabilistic.crps_skill_clim(full_dims)

    # ------------------
    # Correlation
    # ------------------
    if correlation is not None:
        for kind in requested_kinds:
            dims = reduce_dims[kind]
            sm = section_metrics["all_dims"][kind]
            _record_metric(
                sm,
                metric_name="corr",
                metric_type="all_dims",
                kind=kind,
                compute=lambda dims=dims: correlation.corr(dims),
            )
            _record_metric(
                sm,
                metric_name="clim_acc",
                metric_type="all_dims",
                kind=kind,
                compute=lambda dims=dims: correlation.clim_anom_corr(dims, period=clim_period),
            )
            _record_metric(
                sm,
                metric_name="spatial_acc",
                metric_type="all_dims",
                kind=kind,
                compute=lambda dims=dims: correlation.spatial_anom_corr(dims),
            )

        if "scalar" in requested_kinds:
            sm = section_metrics["spatial_mean"]["scalar"]
            _record_metric(
                sm,
                metric_name="corr",
                metric_type="spatial_mean",
                kind="scalar",
                compute=lambda: correlation.corr_of_mean(sdims, tdim),
            )
            _record_metric(
                sm,
                metric_name="clim_acc",
                metric_type="spatial_mean",
                kind="scalar",
                compute=lambda: correlation.clim_anom_corr_of_mean(sdims, tdim, period=clim_period),
            )

        if has_realization:
            rdim = realization_dim

            for kind in requested_kinds:
                dims = reduce_dims[kind]
                sm = section_metrics["ensemble_mean"][kind]

                _record_metric(
                    sm,
                    metric_name="corr",
                    metric_type="ensemble_mean",
                    kind=kind,
                    compute=lambda dims=dims: correlation.corr_of_mean(rdim, dims),
                )
                _record_metric(
                    sm,
                    metric_name="clim_acc",
                    metric_type="ensemble_mean",
                    kind=kind,
                    compute=lambda dims=dims: correlation.clim_anom_corr_of_mean(rdim, dims, period=clim_period),
                )
                _record_metric(
                    sm,
                    metric_name="spatial_acc",
                    metric_type="ensemble_mean",
                    kind=kind,
                    compute=lambda dims=dims: correlation.spatial_anom_corr_of_mean(rdim, dims),
                )

    # ------------------
    # Stack
    # ------------------
    result = {}

    for metric_type in METRIC_SECTIONS:
        for kind in METRIC_KIND:
            if kind in requested_kinds:
                result[f"{kind}_{metric_type}"] = _stack_metric_section(section_metrics[metric_type][kind])
            else:
                result[f"{kind}_{metric_type}"] = xr.Dataset()

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
