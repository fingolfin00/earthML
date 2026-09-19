import xarray as xr

from .defaults import (
    ImprovementUnit,
    METRIC_DIFFERENCE_IMPROVEMENT,
    METRIC_PERCENTAGE_IMPROVEMENT,
    METRIC_IMPROVEMENT_UNITS,
    NORMALIZED_IMPROVEMENT_REFERENCE,
)


STD_REFERENCE_METRICS = {
    "fc_std": "an_std",
    "fc_anom_std": "an_anom_std",
}


def get_required_improvement_metrics(
    metrics: list[str],
) -> list[str]:
    """
    Return requested metrics plus any auxiliary metrics required to
    calculate configured improvement representations.

    Examples
    --------
    rmse + normalized improvement:
        -> also requires an_std

    fc_std improvement:
        -> also requires an_std

    fc_anom_std improvement:
        -> also requires an_anom_std
    """

    required = list(metrics)

    for metric in metrics:

        # Not every plotted metric needs to support model improvement.
        # Examples: an_std, an_anom_std.
        units = METRIC_IMPROVEMENT_UNITS.get(metric)

        if units is None:
            continue

        # ----------------------------------------------------------
        # Standard-deviation improvement reference
        # ----------------------------------------------------------

        if metric in STD_REFERENCE_METRICS:
            reference_metric = STD_REFERENCE_METRICS[metric]

            if reference_metric not in required:
                required.append(reference_metric)

        # ----------------------------------------------------------
        # Normalized improvement reference
        # ----------------------------------------------------------

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
    *,
    reference: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Absolute metric improvement.

    Positive values always mean that the corrected forecast is better.

    For forecast-standard-deviation metrics, improvement means reduction
    in absolute deviation from the corresponding analysis standard
    deviation.
    """

    if metric in STD_REFERENCE_METRICS:
        if reference is None:
            raise ValueError(
                f"Improvement for {metric!r} requires "
                f"{STD_REFERENCE_METRICS[metric]!r}."
            )

        original, corrected, reference = xr.align(
            original,
            corrected,
            reference,
            join="exact",
        )

        return (
            abs(original - reference)
            - abs(corrected - reference)
        )

    try:
        func = METRIC_DIFFERENCE_IMPROVEMENT[metric]

    except KeyError as exc:
        raise KeyError(
            f"No difference-improvement rule configured "
            f"for metric {metric!r}."
        ) from exc

    return func(
        original,
        corrected,
    )


def metric_percentage_improvement(
    original: xr.DataArray,
    corrected: xr.DataArray,
    metric: str,
    *,
    reference: xr.DataArray | None = None,
) -> xr.DataArray:
    """
    Percentage metric improvement.

    Positive values always mean that the corrected forecast is better.

    For forecast-standard-deviation metrics:

        100 * (
            |std_original - std_analysis|
            - |std_corrected - std_analysis|
        ) / |std_original - std_analysis|
    """

    if metric in STD_REFERENCE_METRICS:
        if reference is None:
            raise ValueError(
                f"Improvement for {metric!r} requires "
                f"{STD_REFERENCE_METRICS[metric]!r}."
            )

        original, corrected, reference = xr.align(
            original,
            corrected,
            reference,
            join="exact",
        )

        original_error = abs(
            original - reference
        )

        corrected_error = abs(
            corrected - reference
        )

        return (
            100
            * (original_error - corrected_error)
            / original_error
        ).where(
            original_error > 0
        )

    try:
        func = METRIC_PERCENTAGE_IMPROVEMENT[metric]

    except KeyError as exc:
        raise KeyError(
            f"No percentage-improvement rule configured "
            f"for metric {metric!r}."
        ) from exc

    return func(
        original,
        corrected,
    )


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

    Forecast std:
        (
            |std_original - std_analysis|
            - |std_corrected - std_analysis|
        ) / std_analysis

    Positive values always mean improvement.
    """

    improvement = metric_difference_improvement(
        original,
        corrected,
        metric,
        reference=(
            reference
            if metric in STD_REFERENCE_METRICS
            else None
        ),
    )

    denominator = (
        reference ** reference_power
    )

    return (
        improvement / denominator
    ).where(
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
    """

    original, corrected = xr.align(
        original,
        corrected,
        join="exact",
    )

    if reference is not None:
        original, corrected, reference = xr.align(
            original,
            corrected,
            reference,
            join="exact",
        )

    if improvement_unit == "%":

        improvement = metric_percentage_improvement(
            original,
            corrected,
            metric,
            reference=reference,
        )

        units = "%"

    elif improvement_unit == "Δ":

        improvement = metric_difference_improvement(
            original,
            corrected,
            metric,
            reference=reference,
        )

        units = original.attrs.get(
            "units",
            "",
        )

    elif improvement_unit == "normalized":

        if reference is None:
            raise ValueError(
                f"Normalized improvement for {metric!r} "
                f"requires a reference DataArray."
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

    improvement.attrs = (
        original.attrs.copy()
    )

    improvement.attrs["units"] = units

    improvement.attrs["long_name"] = (
        f"{metric} improvement "
        f"({improvement_unit})"
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
        return {}

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

    result: dict[
        str,
        xr.DataArray,
    ] = {}

    for improvement_unit in (
        METRIC_IMPROVEMENT_UNITS[metric]
    ):

        reference = None
        reference_power = 1

        # ----------------------------------------------------------
        # Standard deviation reference
        # ----------------------------------------------------------

        if metric in STD_REFERENCE_METRICS:

            reference_metric = (
                STD_REFERENCE_METRICS[metric]
            )

            if reference_metric not in baseline_ds:
                raise KeyError(
                    f"Improvement for {metric!r} requires "
                    f"baseline reference metric "
                    f"{reference_metric!r}, but it is missing "
                    f"from model {baseline_model!r}."
                )

            reference = (
                baseline_ds[reference_metric]
            )

        # ----------------------------------------------------------
        # Normalized reference
        # ----------------------------------------------------------

        if improvement_unit == "normalized":

            if metric not in NORMALIZED_IMPROVEMENT_REFERENCE:
                raise KeyError(
                    f"No normalized-improvement reference "
                    f"configured for metric {metric!r}."
                )

            (
                reference_metric,
                reference_power,
            ) = NORMALIZED_IMPROVEMENT_REFERENCE[
                metric
            ]

            if reference_metric not in baseline_ds:
                raise KeyError(
                    f"Normalized improvement for {metric!r} "
                    f"requires baseline reference metric "
                    f"{reference_metric!r}, but it is missing "
                    f"from model {baseline_model!r}."
                )

            reference = (
                baseline_ds[reference_metric]
            )

        improvement = (
            build_metric_improvement(
                baseline,
                target,
                metric=metric,
                improvement_unit=improvement_unit,
                reference=reference,
                reference_power=reference_power,
            )
        )

        suffix = (
            improvement_suffix[
                improvement_unit
            ]
        )

        model_name = (
            f"{target_model}_vs_"
            f"{baseline_model}_"
            f"{suffix}"
        )

        result[model_name] = improvement

    return result
