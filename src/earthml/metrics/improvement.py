import xarray as xr

from .defaults import (
    ImprovementUnit,
    METRIC_DIFFERENCE_IMPROVEMENT,
    METRIC_PERCENTAGE_IMPROVEMENT,
    METRIC_IMPROVEMENT_UNITS,
    NORMALIZED_IMPROVEMENT_REFERENCE,
)


def get_required_improvement_metrics(
    metrics: list[str],
) -> list[str]:
    """
    Return requested metrics plus any auxiliary metrics required to
    calculate configured improvement representations.

    Example:
        rmse + normalized improvement -> also requires an_std
    """

    required = list(metrics)

    for metric in metrics:
        if metric not in METRIC_IMPROVEMENT_UNITS:
            raise KeyError(
                f"No improvement representations configured "
                f"for metric {metric!r}."
            )

        units = METRIC_IMPROVEMENT_UNITS.get(metric, ())

        if "normalized" not in units:
            continue

        if metric not in NORMALIZED_IMPROVEMENT_REFERENCE:
            raise KeyError(
                f"Metric {metric!r} enables normalized improvement "
                f"but has no NORMALIZED_IMPROVEMENT_REFERENCE."
            )

        reference_metric, _ = (
            NORMALIZED_IMPROVEMENT_REFERENCE[metric]
        )

        if reference_metric not in required:
            required.append(reference_metric)

    return required


def metric_difference_improvement(
    original: xr.DataArray,
    corrected: xr.DataArray,
    metric: str,
) -> xr.DataArray:
    """
    Absolute metric improvement.

    Positive values always mean that the corrected forecast is better.
    """

    try:
        func = METRIC_DIFFERENCE_IMPROVEMENT[metric]
    except KeyError as exc:
        raise KeyError(
            f"No difference-improvement rule configured "
            f"for metric {metric!r}."
        ) from exc

    return func(original, corrected)


def metric_percentage_improvement(
    original: xr.DataArray,
    corrected: xr.DataArray,
    metric: str,
) -> xr.DataArray:
    """
    Percentage metric improvement.

    Positive values always mean that the corrected forecast is better.

    Percentage improvement is only defined for metrics explicitly
    configured in METRIC_PERCENTAGE_IMPROVEMENT.
    """

    try:
        func = METRIC_PERCENTAGE_IMPROVEMENT[metric]
    except KeyError as exc:
        raise KeyError(
            f"No percentage-improvement rule configured "
            f"for metric {metric!r}."
        ) from exc

    return func(original, corrected)


def normalized_metric_improvement(
    original: xr.DataArray,
    corrected: xr.DataArray,
    reference: xr.DataArray,
    metric: str,
    *,
    reference_power: int = 1,
) -> xr.DataArray:
    """
    Dimensionless metric improvement normalized by a reference
    variability field.

    Examples
    --------
    RMSE:
        (RMSE_original - RMSE_corrected) / analysis_std

    MSE:
        (MSE_original - MSE_corrected) / analysis_std**2

    Positive values always mean improvement.
    """

    improvement = metric_difference_improvement(
        original,
        corrected,
        metric,
    )

    denominator = reference ** reference_power

    return (improvement / denominator).where(
        denominator > 0
    )


def build_metric_improvement(
    original: xr.DataArray,
    corrected: xr.DataArray,
    *,
    metric: str,
    improvement_unit: ImprovementUnit,
    reference: xr.DataArray | None = None,
    reference_power: int = 1,
) -> xr.DataArray:
    """
    Build one improvement representation for two metric DataArrays.

    Parameters
    ----------
    original
        Metric from the baseline/original forecast.
    corrected
        Metric from the corrected forecast.
    metric
        Metric name.
    improvement_unit
        One of "%", "Δ", or "normalized".
    reference
        Reference variability field required for normalized
        improvement.
    reference_power
        Power applied to the reference field.

    Returns
    -------
    xr.DataArray
        Improvement field.
    """

    original, corrected = xr.align(
        original,
        corrected,
        join="exact",
    )

    if improvement_unit == "%":
        improvement = metric_percentage_improvement(
            original,
            corrected,
            metric,
        )

        units = "%"

    elif improvement_unit == "Δ":
        improvement = metric_difference_improvement(
            original,
            corrected,
            metric,
        )

        units = original.attrs.get("units", "")

    elif improvement_unit == "normalized":
        if reference is None:
            raise ValueError(
                f"Normalized improvement for {metric!r} "
                f"requires a reference DataArray."
            )

        original, corrected, reference = xr.align(
            original,
            corrected,
            reference,
            join="exact",
        )

        improvement = normalized_metric_improvement(
            original,
            corrected,
            reference,
            metric,
            reference_power=reference_power,
        )

        units = ""

    else:
        raise ValueError(
            f"Unsupported improvement representation "
            f"{improvement_unit!r}. "
            f"Expected one of '%', 'Δ', or 'normalized'."
        )

    improvement.attrs = original.attrs.copy()
    improvement.attrs["units"] = units
    improvement.attrs["long_name"] = (
        f"{metric} improvement ({improvement_unit})"
    )

    return improvement


def build_metric_improvements(
    baseline_ds: xr.Dataset,
    target_ds: xr.Dataset,
    *,
    metric: str,
    baseline_model: str,
    target_model: str,
) -> dict[str, xr.DataArray]:
    """
    Build all configured improvement representations for one metric.

    Representations are taken from METRIC_IMPROVEMENT_UNITS.

    Parameters
    ----------
    baseline_ds : xr.Dataset
        Dataset containing metrics for the reference model.
    target_ds : xr.Dataset
        Dataset containing metrics for the model being compared
        against the baseline.
    metric : str
        Metric to compare.
    baseline_model : str
        Name of the reference model.
    target_model : str
        Name of the target model.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping from plotting model name to improvement DataArray.

        Examples:
            mlfc_vs_fc_percentage
            mlfc_vs_fc_difference
            mlfc_vs_fc_normalized

            clim-fc_vs_fc_percentage
            clim-fc_vs_fc_difference
            clim-fc_vs_fc_normalized
    """
    if metric not in baseline_ds:
        raise KeyError(
            f"Metric {metric!r} is missing from baseline model "
            f"{baseline_model!r}."
        )

    if metric not in target_ds:
        raise KeyError(
            f"Metric {metric!r} is missing from target model "
            f"{target_model!r}."
        )

    if metric not in METRIC_IMPROVEMENT_UNITS:
        raise KeyError(
            f"No improvement representations configured "
            f"for metric {metric!r}."
        )

    baseline, target = xr.align(
        baseline_ds[metric],
        target_ds[metric],
        join="exact",
    )

    improvement_suffix = {
        "%": "percentage",
        "Δ": "difference",
        "normalized": "normalized",
    }

    result: dict[str, xr.DataArray] = {}

    for improvement_unit in METRIC_IMPROVEMENT_UNITS[metric]:
        reference = None
        reference_power = 1

        if improvement_unit == "normalized":
            if metric not in NORMALIZED_IMPROVEMENT_REFERENCE:
                raise KeyError(
                    f"No normalized-improvement reference "
                    f"configured for metric {metric!r}."
                )

            reference_metric, reference_power = (
                NORMALIZED_IMPROVEMENT_REFERENCE[metric]
            )

            if reference_metric not in baseline_ds:
                raise KeyError(
                    f"Normalized improvement for {metric!r} "
                    f"requires baseline reference metric "
                    f"{reference_metric!r}, but it is missing "
                    f"from model {baseline_model!r}."
                )

            reference = baseline_ds[reference_metric]

        improvement = build_metric_improvement(
            baseline,
            target,
            metric=metric,
            improvement_unit=improvement_unit,
            reference=reference,
            reference_power=reference_power,
        )

        suffix = improvement_suffix[improvement_unit]

        model_name = (
            f"{target_model}_vs_{baseline_model}_{suffix}"
        )

        result[model_name] = improvement

    return result
