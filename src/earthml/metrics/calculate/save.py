import hashlib
from pathlib import Path
from typing import Sequence

import xarray as xr

from .dataclasses import CalculateMetricsSaveConfig


def _normalized_dataset_attrs(attrs: dict[str, object] | None) -> dict[str, str]:
    if not attrs:
        return {}
    return {
        str(key): str(value)
        for key, value in attrs.items()
        if value is not None
    }


def _sanitize_fragment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text).strip("_") or "all"


def _normalize_metric_names(metric_names: str | Sequence[str] | None) -> tuple[str, ...]:
    if metric_names is None:
        return ()
    if isinstance(metric_names, str):
        return (metric_names,)
    return tuple(sorted({str(name) for name in metric_names}))


def _metric_name_fragment(metric_names: str | Sequence[str] | None) -> str:
    normalized = _normalize_metric_names(metric_names)
    if not normalized:
        return "metrics_all"
    return f"metrics_{'-'.join(_sanitize_fragment(name) for name in normalized)}"


def _model_fragment(model_names: Sequence[str]) -> str:
    normalized = [_sanitize_fragment(str(name)) for name in model_names if name is not None]
    return f"models_{'-'.join(normalized)}" if normalized else "models_none"


def _build_metric_file_stem(
    *,
    kind: str,
    filename_context_suffix: str,
    model_names: Sequence[str],
    metric_names: str | Sequence[str] | None,
    clim_period: str,
    metric_groupby_period: str | None,
    metric_groupby_basis: str,
    include_group_dim: bool,
) -> str:
    fragments = [
        _sanitize_fragment(kind),
        _sanitize_fragment(filename_context_suffix or "all"),
        _model_fragment(model_names),
        _metric_name_fragment(metric_names),
        f"clim_{_sanitize_fragment(clim_period)}",
        f"groupperiod_{_sanitize_fragment(metric_groupby_period or 'none')}",
        f"groupbasis_{_sanitize_fragment(metric_groupby_basis)}",
        f"groupdim_{'yes' if include_group_dim else 'no'}",
    ]
    readable = "__".join(fragment for fragment in fragments if fragment)
    digest = hashlib.sha1(readable.encode("utf-8")).hexdigest()[:10]
    return f"{readable}__{digest}"


def _saved_metric_path(
    *,
    output_root: Path,
    config: CalculateMetricsSaveConfig,
    metric_type: str,
    kind: str,
    file_stem: str,
) -> Path:
    return output_root / config.output_subfolder / metric_type / f"{file_stem}.nc"


def load_metric_datasets(
    *,
    output_root: Path,
    config: CalculateMetricsSaveConfig,
    filename_context_suffix: str,
    model_names: Sequence[str],
    metric_names: str | Sequence[str] | None,
    clim_period: str,
    metric_groupby_period: str | None,
    metric_groupby_basis: str,
    include_group_dim: bool,
) -> dict[str, dict[str, xr.Dataset]] | None:
    """Load a complete saved metric bundle, or return None if any file is missing."""

    metrics: dict[str, dict[str, xr.Dataset]] = {}

    for metric_type in config.metric_types:
        metrics[metric_type] = {}
        for kind in config.kinds:
            file_stem = _build_metric_file_stem(
                kind=kind,
                filename_context_suffix=filename_context_suffix,
                model_names=model_names,
                metric_names=metric_names,
                clim_period=clim_period,
                metric_groupby_period=metric_groupby_period,
                metric_groupby_basis=metric_groupby_basis,
                include_group_dim=include_group_dim,
            )
            path = _saved_metric_path(
                output_root=output_root,
                config=config,
                metric_type=metric_type,
                kind=kind,
                file_stem=file_stem,
            )
            if not path.exists():
                return None
            metrics[metric_type][kind] = xr.load_dataset(path)

    return metrics


def save_metric_datasets(
    *,
    metrics: dict[str, dict[str, xr.Dataset]],
    output_root: Path,
    config: CalculateMetricsSaveConfig,
    filename_context_suffix: str,
    model_names: Sequence[str],
    metric_names: str | Sequence[str] | None,
    clim_period: str,
    metric_groupby_period: str | None,
    metric_groupby_basis: str,
    include_group_dim: bool,
    dataset_attrs: dict[str, object] | None = None,
) -> list[Path]:
    """Persist calculated metric datasets to disk and return saved paths."""

    saved_paths: list[Path] = []
    extra_attrs = _normalized_dataset_attrs(dataset_attrs)

    for metric_type in config.metric_types:
        section_dict = metrics.get(metric_type, {})
        for kind in config.kinds:
            ds = section_dict.get(kind)
            if not isinstance(ds, xr.Dataset):
                continue
            if not config.include_empty and not config.reuse_existing and not ds.data_vars:
                continue

            file_stem = _build_metric_file_stem(
                kind=kind,
                filename_context_suffix=filename_context_suffix,
                model_names=model_names,
                metric_names=metric_names,
                clim_period=clim_period,
                metric_groupby_period=metric_groupby_period,
                metric_groupby_basis=metric_groupby_basis,
                include_group_dim=include_group_dim,
            )
            output_path = _saved_metric_path(
                output_root=output_root,
                config=config,
                metric_type=metric_type,
                kind=kind,
                file_stem=file_stem,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            ds_to_save = ds
            if extra_attrs:
                ds_to_save = ds.copy(deep=False)
                ds_to_save.attrs = {
                    **ds.attrs,
                    **extra_attrs,
                }

            ds_to_save.to_netcdf(output_path)
            saved_paths.append(output_path)

    return saved_paths
