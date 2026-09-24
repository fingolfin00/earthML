import torch

from .metrics import (
    expand_mask_to,
    masked_bias,
    masked_crmse,
    masked_mean,
    masked_mse,
    masked_std,
    masked_std_ratio,
    masked_temporal_corr,
    masked_geo_bias,
    masked_geo_crmse,
    masked_geo_mean,
    masked_geo_mse,
    masked_geo_std,
    masked_geo_std_ratio,
)


def _geo_map_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
) -> torch.Tensor:
    """
    Latitude-weighted mean of a spatial map.

    Accepts:
        (H, W)
        (C, H, W)

    and promotes it to (N, C, H, W) for masked_geo_mean().
    """
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)

    elif x.ndim == 3:
        x = x.unsqueeze(0)
        mask = mask.unsqueeze(0)

    else:
        raise ValueError(
            f"Expected map shaped (H, W) or (C, H, W), "
            f"got {tuple(x.shape)}"
        )

    return masked_geo_mean(
        x,
        mask,
        latitudes,
    )


def _safe_abs_max(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    values = x[
        mask & torch.isfinite(x)
    ]

    if values.numel() == 0:
        return torch.tensor(
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )

    return values.abs().max()


def diagnostics(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    latitudes: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, dict[str, float]]:
    """
    Compute end-of-run diagnostics for normalized prediction tensors.

    Expected tensor shape:
        preds, target: (N, C, H, W)
        baseline:      (N, C, H, W), optional
        latitudes:     (H,)

    `baseline`, when supplied, must be expressed in exactly the same
    normalized target space as `preds` and `target`.

    Examples:
        analysis target:
            baseline = normalized original forecast

        residual target:
            baseline = zeros_like(target)

    Domain-wide quantities are latitude weighted using cosine-latitude
    weights.

    Per-grid quantities are computed over the sample dimension N.
    Latitude weighting is applied only when reducing those maps to a
    domain-wide scalar summary.
    """

    if preds.shape != target.shape:
        raise ValueError(
            f"preds and target must have the same shape, "
            f"got {preds.shape} and {target.shape}"
        )

    if preds.ndim != 4:
        raise ValueError(
            "diagnostics expects tensors shaped (N, C, H, W)."
        )

    if baseline is not None and baseline.shape != target.shape:
        raise ValueError(
            f"baseline and target must have the same shape, "
            f"got {baseline.shape} and {target.shape}"
        )

    if latitudes.numel() != preds.shape[-2]:
        raise ValueError(
            f"latitudes length {latitudes.numel()} "
            f"does not match H={preds.shape[-2]}"
        )

    valid = expand_mask_to(
        preds,
        mask,
    )

    # ==========================================================
    # Global geographic diagnostics
    # ==========================================================

    model_mse = masked_geo_mse(
        preds,
        target,
        valid,
        latitudes,
    )

    baseline_mse: torch.Tensor | None = None
    skill_vs_baseline: torch.Tensor | None = None

    if baseline is not None:
        baseline_mse = masked_geo_mse(
            baseline,
            target,
            valid,
            latitudes,
        )

        skill_vs_baseline = (
            1.0
            - model_mse
            / baseline_mse.clamp_min(eps)
        )

    bias = masked_geo_bias(
        preds,
        target,
        valid,
        latitudes,
    )

    crmse = masked_geo_crmse(
        preds,
        target,
        valid,
        latitudes,
    )

    target_mean = masked_geo_mean(
        target,
        valid,
        latitudes,
    )

    prediction_mean = masked_geo_mean(
        preds,
        valid,
        latitudes,
    )

    target_std = masked_geo_std(
        target,
        valid,
        latitudes,
    )

    prediction_std = masked_geo_std(
        preds,
        valid,
        latitudes,
    )

    std_ratio = masked_geo_std_ratio(
        preds,
        target,
        valid,
        latitudes,
    )

    # ==========================================================
    # Per-grid diagnostics
    #
    # Reduce only over N.
    #
    # Result shape:
    #     (C, H, W)
    # ==========================================================

    reduce_dim = 0

    valid_count_map = valid.sum(
        dim=reduce_dim
    )

    valid_grid_cells = (
        valid_count_map > 0
    )

    model_mse_map = masked_mse(
        preds,
        target,
        valid,
        dim=reduce_dim,
    )

    baseline_mse_map: torch.Tensor | None = None

    if baseline is not None:
        baseline_mse_map = masked_mse(
            baseline,
            target,
            valid,
            dim=reduce_dim,
        )

    bias_map = masked_bias(
        preds,
        target,
        valid,
        dim=reduce_dim,
    )

    crmse_map = masked_crmse(
        preds,
        target,
        valid,
        dim=reduce_dim,
    )

    target_mean_map = masked_mean(
        target,
        valid,
        dim=reduce_dim,
    )

    prediction_mean_map = masked_mean(
        preds,
        valid,
        dim=reduce_dim,
    )

    target_std_map = masked_std(
        target,
        valid,
        dim=reduce_dim,
    )

    prediction_std_map = masked_std(
        preds,
        valid,
        dim=reduce_dim,
    )

    std_ratio_map = masked_std_ratio(
        preds,
        target,
        valid,
        dim=reduce_dim,
    )

    temporal_corr_map = masked_temporal_corr(
        preds,
        target,
        valid,
        dim=reduce_dim,
        eps=eps,
    )

    # ==========================================================
    # Baseline spatial improvement
    # ==========================================================

    improved_grid_fraction: torch.Tensor | None = None
    improved_area_fraction: torch.Tensor | None = None

    if baseline_mse_map is not None:
        improved_grid_cells = (
            (model_mse_map < baseline_mse_map)
            & valid_grid_cells
        )

        improved_grid_fraction = (
            improved_grid_cells.sum()
            / valid_grid_cells.sum().clamp_min(1)
        )

        improved_area_fraction = _geo_map_mean(
            improved_grid_cells.to(
                dtype=preds.dtype
            ),
            valid_grid_cells,
            latitudes,
        )

    # ==========================================================
    # Spatially aggregated map diagnostics
    # ==========================================================

    bias_map_abs_mean = _geo_map_mean(
        bias_map.abs(),
        (
            valid_grid_cells
            & torch.isfinite(bias_map)
        ),
        latitudes,
    )

    bias_map_abs_max = _safe_abs_max(
        bias_map,
        valid_grid_cells,
    )

    crmse_map_mean = _geo_map_mean(
        crmse_map,
        (
            valid_grid_cells
            & torch.isfinite(crmse_map)
        ),
        latitudes,
    )

    target_mean_map_abs_mean = _geo_map_mean(
        target_mean_map.abs(),
        (
            valid_grid_cells
            & torch.isfinite(target_mean_map)
        ),
        latitudes,
    )

    prediction_mean_map_abs_mean = _geo_map_mean(
        prediction_mean_map.abs(),
        (
            valid_grid_cells
            & torch.isfinite(prediction_mean_map)
        ),
        latitudes,
    )

    target_std_map_mean = _geo_map_mean(
        target_std_map,
        (
            valid_grid_cells
            & torch.isfinite(target_std_map)
        ),
        latitudes,
    )

    prediction_std_map_mean = _geo_map_mean(
        prediction_std_map,
        (
            valid_grid_cells
            & torch.isfinite(prediction_std_map)
        ),
        latitudes,
    )

    std_ratio_map_mean = _geo_map_mean(
        std_ratio_map,
        (
            valid_grid_cells
            & torch.isfinite(std_ratio_map)
        ),
        latitudes,
    )

    temporal_corr_map_mean = _geo_map_mean(
        temporal_corr_map,
        (
            valid_grid_cells
            & torch.isfinite(temporal_corr_map)
        ),
        latitudes,
    )

    # ==========================================================
    # Output
    # ==========================================================

    overall = {
        "MSE": float(model_mse),
        "Bias": float(bias),
        "cRMSE": float(crmse),
    }

    spatial = {
        "Bias map |mean|": float(
            bias_map_abs_mean
        ),
        "Bias map max |.|": float(
            bias_map_abs_max
        ),
        "cRMSE map mean": float(
            crmse_map_mean
        ),
        "Target mean-map |mean|": float(
            target_mean_map_abs_mean
        ),
        "Prediction mean-map |mean|": float(
            prediction_mean_map_abs_mean
        ),
        "Target std-map mean": float(
            target_std_map_mean
        ),
        "Prediction std-map mean": float(
            prediction_std_map_mean
        ),
        "Std-ratio map mean": float(
            std_ratio_map_mean
        ),
        "Temporal CC map mean": float(
            temporal_corr_map_mean
        ),
    }

    if baseline_mse is not None:
        overall["Baseline MSE"] = float(
            baseline_mse
        )

    if skill_vs_baseline is not None:
        overall["Skill vs baseline"] = float(
            skill_vs_baseline
        )

    if improved_grid_fraction is not None:
        spatial["Improved grid cells [%]"] = (
            100.0
            * float(improved_grid_fraction)
        )

    if improved_area_fraction is not None:
        spatial["Improved area [%]"] = (
            100.0
            * float(improved_area_fraction)
        )

    return {
        "Overall": overall,
        "Distribution": {
            "Target mean": float(target_mean),
            "Prediction mean": float(prediction_mean),
            "Target std": float(target_std),
            "Prediction std": float(prediction_std),
            "Std ratio": float(std_ratio),
        },
        "Spatial": spatial,
    }
