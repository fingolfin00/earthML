from typing import Any, Sequence
from contextlib import nullcontext
from itertools import product

from pathlib import Path

from dateutil.relativedelta import relativedelta
from datetime import datetime

from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

import numpy as np
import pandas as pd
from scipy.signal import get_window
import cf_xarray
import xarray as xr
# import xskillscore as xs
# from scipy.stats import t as student_t

from .metrics.deterministic import DeterministicMetrics
from .metrics.correlation import CorrelationMetrics
from .metrics.probabilistic import ProbabilisticMetrics
from .bundles import build_standard_metric_bundle
from .calculate.constants import METRIC_SECTIONS

from ..experiments.mlbc.load import (
    load_all_exp_from_folder,
    add_ke_to_runs,
    harmonize_leadtime_int,
)
from ..experiments.mlbc.registry import MLBCExperimentMode
from ..experiments.mlbc.utils import (
    apply_mask_to_dataset,
    combine_masks,
    project_mask_to_reference_grid,
    select_mask_for_indexers,
)


RunsByModel = dict[str, xr.Dataset]
RunsByRegion = dict[str, RunsByModel]
MetricLeaf = dict[str, dict[str, xr.Dataset]]
MetricsByRegion = dict[str, MetricLeaf]
ClimTuple = tuple[xr.Dataset | None, ...]
ClimsByRegion = dict[str, ClimTuple]


# ==========================================
# Standalone helper methods
# ==========================================


def build_parquet_path(
    base_folder: str,
    vars_list: Sequence[str],
    metric_names: Sequence[str] | None = None,
    diff: str = "no", # no, delta, ratio
) -> str:
    vars_tag = "_".join(sorted(map(str, vars_list))) if vars_list else "allvars"
    metrics_tag = "_".join(sorted(map(str, metric_names))) if metric_names else "allmetrics"

    return str(Path(base_folder) / f"metrics_{diff}_{vars_tag}_{metrics_tag}_parquet")


def date_diff(ymd_range: str):
    start_str, end_str = ymd_range.split('-')
    start = datetime.strptime(start_str, '%Y%m%d')
    end = datetime.strptime(end_str, '%Y%m%d')

    delta = relativedelta(end, start)
    days = (end - start).days

    return {
        "years": delta.years,
        "months": delta.months,
        "days": delta.days,
        "total_days": days,
        "total_months": delta.years * 12 + delta.months
    }


def _normalize_region_values(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        out: list[str] = []
        for value in values:
            if value is None or str(value) == "":
                continue
            value_str = str(value)
            if value_str not in out:
                out.append(value_str)
        return out

    value_str = str(values)
    return [] if value_str == "" else [value_str]


def _discover_region_values(
    *,
    exp_root: str | Path,
    exp_mode: MLBCExperimentMode,
    load_models: Sequence[str],
    variables: Sequence[str] | None,
    run_filters: dict[str, object] | None,
    show_progress: bool,
) -> list[str]:
    explicit_regions = _normalize_region_values((run_filters or {}).get("region"))
    if explicit_regions:
        return explicit_regions

    _, grouped_sizes = load_all_exp_from_folder(
        exp_root=exp_root,
        exp_mode=exp_mode,
        load_train_preds=(exp_mode == "train"),
        load_models=load_models,
        variables=variables,
        run_filters=run_filters,
        only_sizes=True,
        show_progress=False,
    )

    regions: list[str] = []
    for size_map in grouped_sizes.values():
        for (_, _, region, _, _) in size_map:
            region_str = str(region)
            if region_str not in regions:
                regions.append(region_str)
    return regions


def metrics_to_df_single_region(
    metrics: MetricLeaf,
    variables: list[str] | str | None = None,
    metric_names: list[str] | str | None = None,
    kind: str = "scalar",
    metric_type: str = "all_dims",
    diff: str = "no",
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Convert one single-region metrics leaf returned by get_runs_and_metrics(...)
    into a tidy pandas DataFrame.
    """
    ds = metrics.get(metric_type, {}).get(kind, xr.Dataset())

    base_cols = ["train_period", "model", "metric", "variable", "leadtime", "value"]

    if not isinstance(ds, xr.Dataset) or not ds.data_vars:
        return pd.DataFrame(columns=base_cols)

    if isinstance(metric_names, str):
        metric_names = [metric_names]
    elif metric_names is not None:
        metric_names = list(metric_names)

    if isinstance(variables, str):
        variables = [variables]
    elif variables is not None:
        variables = list(variables)

    metric_dim = "metric"

    if metric_dim not in ds.coords and metric_dim not in ds.dims:
        return pd.DataFrame(columns=base_cols)

    available_metrics = ds.coords[metric_dim].values.tolist()
    if metric_names is None:
        metric_names = list(available_metrics)
    else:
        metric_names = [m for m in metric_names if m in available_metrics]

    if not metric_names:
        return pd.DataFrame(columns=base_cols)

    available_variables = list(ds.data_vars)
    if variables is None:
        variables = available_variables
    else:
        variables = [v for v in variables if v in available_variables]

    if not variables:
        return pd.DataFrame(columns=base_cols)

    ds = ds[variables].sel({metric_dim: metric_names})

    has_model = "model" in ds.dims or "model" in ds.coords

    if diff in ("delta", "ratio"):
        if not has_model:
            raise ValueError(
                f"diff='{diff}' requires a 'model' dimension in metrics[{metric_type!r}][{kind!r}]"
            )

        model_values = ds.coords["model"].values.tolist()

        if models is None:
            if len(model_values) != 2:
                raise ValueError(
                    "When diff is 'delta' or 'ratio', exactly two models are required"
                )
            model_a, model_b = model_values
        else:
            if len(models) != 2:
                raise ValueError(
                    "When diff is 'delta' or 'ratio', models must contain exactly two model names"
                )
            model_a, model_b = models
            missing = [m for m in (model_a, model_b) if m not in model_values]
            if missing:
                raise ValueError(f"Requested diff models not found in metrics dataset: {missing}")

        ds_a = ds.sel(model=model_a)
        ds_b = ds.sel(model=model_b)
        ds_a, ds_b = xr.align(ds_a, ds_b, join="inner")

        arr_a = ds_a.to_array(dim="variable")
        arr_b = ds_b.to_array(dim="variable")

        arr = arr_b - arr_a
        if diff == "ratio":
            arr = arr / np.abs(arr_a)

        df = arr.to_dataframe(name="value").reset_index()
        df["model"] = f"{model_b}-{model_a}"

    else:
        if has_model and models is not None:
            model_values = ds.coords["model"].values.tolist()
            selected_models = [m for m in models if m in model_values]
            if not selected_models:
                return pd.DataFrame(columns=base_cols)
            ds = ds.sel(model=selected_models)

        df = ds.to_array(dim="variable").to_dataframe(name="value").reset_index()

        if not has_model:
            df["model"] = np.nan

    if metric_dim in df.columns:
        df = df.rename(columns={metric_dim: "metric"})

    for col in ["train_period", "model", "metric", "variable", "leadtime"]:
        if col not in df.columns:
            df[col] = np.nan

    if "value" not in df.columns:
        df["value"] = np.nan

    # Keep standard columns first, then preserve any extra dims/coords such as loss
    extra_cols = [col for col in df.columns if col not in base_cols]
    df = df[base_cols + extra_cols]

    if df["train_period"].notna().any():
        df[["years", "months", "days", "total_days", "total_months"]] = (
            df["train_period"].astype(str)
            .apply(lambda x: date_diff(x) if "-" in x and len(x.split("-")) == 2 else {
                "years": np.nan,
                "months": np.nan,
                "days": np.nan,
                "total_days": np.nan,
                "total_months": np.nan,
            })
            .apply(pd.Series)
        )
    else:
        df[["years", "months", "days", "total_days", "total_months"]] = np.nan

    return df


def rechunk(da, lat_rc=None, lon_rc=None, time_rc=None, realization_rc=None):
    chunks = {}
    if time_rc is not None:
        chunks[da.earthml.guessed_dims.time] = time_rc
    if lat_rc is not None:
        chunks[da.earthml.guessed_dims.latitude] = lat_rc
    if lon_rc is not None:
        chunks[da.earthml.guessed_dims.longitude] = lon_rc

    realization_dim = da.earthml.guessed_dims.realization
    if realization_dim in da.dims and realization_rc is not None:
        chunks[realization_dim] = realization_rc

    leadtime_dim = da.earthml.guessed_dims.leadtime
    if leadtime_dim in da.dims:
        chunks[leadtime_dim] = -1  # keep together

    return da.chunk(chunks)


def _compute_metric_bundle(
    deterministic: DeterministicMetrics,
    correlation: CorrelationMetrics | None = None,
    probabilistic: ProbabilisticMetrics | None = None,
    norm: str = "std",
    clim_period: str = "month",
    metric_names: str | Sequence[str] | None = None,
    metric_sections: str | Sequence[str] = ("scalar", "map", "timeseries"),
    metric_types: str | Sequence[str] | None = None,
) -> dict[str, dict[str, xr.Dataset]]:
    if isinstance(metric_names, str):
        metric_names = {metric_names}
    elif metric_names is not None:
        metric_names = set(metric_names)

    if isinstance(metric_sections, str):
        metric_sections = (metric_sections,)
    metric_sections = tuple(metric_sections)

    if isinstance(metric_types, str):
        metric_types = {metric_types}
    elif metric_types is not None:
        metric_types = set(metric_types)

    result = build_standard_metric_bundle(
        deterministic=deterministic,
        correlation=correlation,
        probabilistic=probabilistic,
        norm=norm,
        clim_period=clim_period,
    )

    filtered_result: dict[str, dict[str, xr.Dataset]] = {}

    for metric_type in METRIC_SECTIONS:
        if metric_types is not None and metric_type not in metric_types:
            continue

        filtered_result[metric_type] = {}

        for section in metric_sections:
            key = f"{section}_{metric_type}"
            ds = result[key]

            if metric_names is not None and "metric" in ds.coords:
                keep_metrics = [name for name in ds.metric.values if name in metric_names]
                ds = ds.sel(metric=keep_metrics) if keep_metrics else xr.Dataset(attrs=ds.attrs)

            filtered_result[metric_type][section] = ds

    return filtered_result


def _get_single_region_runs_and_metrics(
    exp_root: str | Path,
    exp_mode: MLBCExperimentMode = "test",
    load_models: Sequence[str] = ("an", "fc", "pr"),
    variables: Sequence[str] | None = None,
    run_filters: dict[str, object] | None = None,
    calculate_clim_from_train_period: bool = False,
    use_train_prediction_clim: bool = False,  # only used if calculate_clim_from_train_period is True
    use_saved_mask: bool = True,
    external_mask_data: xr.Dataset | xr.DataArray | None = None,
    apply_external_mask_to_runs: bool = False,
    metric_names: str | Sequence[str] | None = None,
    metric_sections: str | Sequence[str] = ("scalar", "map", "timeseries"),
    metric_types: str | Sequence[str] | None = None,
    show_progress: bool = True,
) -> tuple[
    RunsByModel,
    MetricLeaf,
    ClimTuple,
]:
    models = list(load_models)
    if len(models) < 2:
        raise ValueError("load_models must contain at least two entries: truth first, then at least one model")

    truth_model = models[0]
    requested_model_names = models[1:]

    loaded_runs, _ = load_all_exp_from_folder(
        exp_root=exp_root,
        exp_mode=exp_mode,
        load_train_preds=(exp_mode == "train"),
        load_models=models,
        variables=variables,
        run_filters=run_filters,
        show_progress=show_progress,
    )

    if isinstance(metric_types, str):
        requested_metric_types = [metric_types]
    elif metric_types is not None:
        requested_metric_types = list(metric_types)
    else:
        requested_metric_types = list(METRIC_SECTIONS)

    if isinstance(metric_sections, str):
        requested_metric_sections = [metric_sections]
    else:
        requested_metric_sections = list(metric_sections)

    empty_metrics = {
        mt: {ms: xr.Dataset() for ms in requested_metric_sections}
        for mt in requested_metric_types
    }

    if not loaded_runs:
        return {}, empty_metrics, tuple(None for _ in models)

    # Add KE to each loaded model dataset independently
    for model_name in models:
        if model_name in loaded_runs:
            loaded_runs[model_name] = add_ke_to_runs(
                loaded_runs[model_name],
                dataset_keys=None,   # no "model" dim here
            )

    if truth_model not in loaded_runs:
        raise ValueError(f"Truth model {truth_model!r} not found in loaded runs")

    available_model_names = [m for m in requested_model_names if m in loaded_runs]
    if not available_model_names:
        raise ValueError("No requested comparison models found in loaded runs")

    if isinstance(external_mask_data, xr.DataArray):
        mask_name = external_mask_data.name or "__mask__"
        external_mask_data = external_mask_data.to_dataset(name=mask_name)

    saved_mask_runs = loaded_runs.get("mask") if use_saved_mask else None

    train_loaded_runs = None
    if calculate_clim_from_train_period:
        train_loaded_runs, _ = load_all_exp_from_folder(
            exp_root=exp_root,
            exp_mode="train",
            load_train_preds=use_train_prediction_clim,
            load_models=models,
            variables=variables,
            run_filters=run_filters,
            show_progress=show_progress,
        )

        for model_name in models:
            if model_name in train_loaded_runs:
                train_loaded_runs[model_name] = add_ke_to_runs(
                    train_loaded_runs[model_name],
                    dataset_keys=None,
                )

    runs = loaded_runs

    truth_runs = loaded_runs[truth_model]
    truth_region = str(truth_runs.coords["region"].item()) if "region" in truth_runs.coords else ""

    # These are the experiment axes created by load_all_exp_from_folder
    run_dims = [dim for dim in ("leadtime", "train_period", "region", "loss", "variant") if dim in truth_runs.dims]

    run_indexers = _build_run_indexers(truth_runs, run_dims, run_filters=run_filters)

    metric_runs: dict[str, dict[str, list[xr.Dataset]]] = {
        mt: {ms: [] for ms in requested_metric_sections}
        for mt in requested_metric_types
    }

    last_clim_by_model = {model_name: None for model_name in models}
    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            auto_refresh=True,
            refresh_per_second=6,
            transient=False,
        )
        if show_progress else nullcontext()
    )

    with progress_cm as prog:
        run_task = prog.add_task("Computing metric bundles", total=len(run_indexers)) if show_progress else None

        for indexers in run_indexers:
            if show_progress:
                indexer_str = ", ".join(f"{dim}={value}" for dim, value in indexers.items()) or "all runs"
                prog.update(run_task, description=f"Computing metrics | {indexer_str}")

            truth_data = truth_runs.sel(indexers, drop=True) if indexers else truth_runs
            if not truth_data.data_vars:
                if show_progress:
                    prog.advance(run_task)
                continue

            # Skip empty truth slices
            truth_arr = truth_data.to_array()
            if not bool(truth_arr.notnull().any()):
                if show_progress:
                    prog.advance(run_task)
                continue

            model_names_this_run = []
            model_data = []
            for model_name in available_model_names:
                ds = loaded_runs[model_name]
                ds = ds.sel(indexers, drop=True) if indexers else ds
                if not ds.data_vars:
                    continue
                if not bool(ds.to_array().notnull().any()):
                    continue
                model_names_this_run.append(model_name)
                model_data.append(ds)

            if not model_data:
                if show_progress:
                    prog.advance(run_task)
                continue

            mask_data = combine_masks(
                project_mask_to_reference_grid(
                    select_mask_for_indexers(saved_mask_runs, indexers),
                    truth_data,
                ),
                project_mask_to_reference_grid(
                    select_mask_for_indexers(external_mask_data, indexers),
                    truth_data,
                ),
            )

            clim_by_model = {truth_model: None, **{m: None for m in model_names_this_run}}

            if calculate_clim_from_train_period and train_loaded_runs is not None:
                if truth_model in train_loaded_runs:
                    truth_train = train_loaded_runs[truth_model]
                    truth_train = truth_train.sel(indexers, drop=True) if indexers else truth_train
                    if truth_train.data_vars and bool(truth_train.to_array().notnull().any()):
                        clim_by_model[truth_model] = truth_train.earthml.climatology(groupby="month")

                for model_name in model_names_this_run:
                    if model_name == "pr" and not use_train_prediction_clim:
                        clim_by_model[model_name] = clim_by_model[truth_model]
                        continue

                    if model_name in train_loaded_runs:
                        train_model = train_loaded_runs[model_name]
                        train_model = train_model.sel(indexers, drop=True) if indexers else train_model
                        if train_model.data_vars and bool(train_model.to_array().notnull().any()):
                            clim_by_model[model_name] = train_model.earthml.climatology(groupby="month")

                if "pr" in clim_by_model and clim_by_model["pr"] is None:
                    clim_by_model["pr"] = clim_by_model[truth_model]

            clim_data = [clim_by_model[truth_model], *[clim_by_model[m] for m in model_names_this_run]]

            # Harmonize leadtime coordinates across truth / models / mask
            to_harmonize = [truth_data, *model_data]
            has_mask = mask_data is not None and "leadtime" in mask_data.coords
            if has_mask:
                to_harmonize.append(mask_data)

            try:
                harmonized = harmonize_leadtime_int(*to_harmonize, coord="leadtime", method="int_source")
                truth_data = harmonized[0]
                model_data = harmonized[1:1 + len(model_data)]
                if has_mask:
                    mask_data = harmonized[-1]
            except Exception:
                # If leadtime is already aligned or cannot be normalized this way, continue as-is.
                pass

            # Same idea for climatologies if they exist and carry leadtime
            valid_clim = [ds for ds in clim_data if ds is not None and "leadtime" in ds.coords]
            if len(valid_clim) >= 2:
                try:
                    harmonized_clim = harmonize_leadtime_int(*valid_clim, coord="leadtime", method="int_source")
                    it = iter(harmonized_clim)
                    clim_data = [
                        next(it) if ds is not None and "leadtime" in ds.coords else ds
                        for ds in clim_data
                    ]
                except Exception:
                    pass

            dm = DeterministicMetrics(
                truth_data=truth_data,
                model_data=model_data,
                truth_name=truth_model,
                model_names=model_names_this_run,
                clim_data=clim_data,
                mask_data=mask_data,
            )

            cm = CorrelationMetrics(
                truth_data=truth_data,
                model_data=model_data,
                truth_name=truth_model,
                model_names=model_names_this_run,
                clim_data=clim_data,
                mask_data=mask_data,
            )

            pm = ProbabilisticMetrics(
                truth_data=truth_data,
                model_data=model_data,
                truth_name=truth_model,
                model_names=model_names_this_run,
                clim_data=clim_data,
                mask_data=mask_data,
            )

            metric_bundle = _compute_metric_bundle(
                dm,
                cm,
                pm,
                norm="std",
                clim_period="month",
                metric_names=metric_names,
                metric_sections=metric_sections,
                metric_types=metric_types,
            )

            for metric_type, section_dict in metric_bundle.items():
                for section, ds in section_dict.items():
                    if ds is None or not ds.data_vars:
                        continue

                    if indexers:
                        ds = ds.expand_dims({
                            dim: [value] for dim, value in indexers.items()
                        })
                    if truth_region:
                        ds = ds.assign_coords(region=truth_region)

                    metric_runs[metric_type][section].append(ds)

            last_clim_by_model.update(clim_by_model)

            if show_progress:
                prog.advance(run_task)

    metrics: MetricLeaf = {}
    combine_steps = sum(
        1
        for section_dict in metric_runs.values()
        for runs_list in section_dict.values()
        if runs_list
    )

    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            auto_refresh=True,
            refresh_per_second=6,
            transient=False,
        )
        if show_progress else nullcontext()
    )

    with progress_cm as prog:
        combine_task = prog.add_task("Combining metric sections", total=combine_steps) if show_progress else None

        for metric_type, section_dict in metric_runs.items():
            metrics[metric_type] = {}

            for section, runs_list in section_dict.items():
                if show_progress and runs_list:
                    prog.update(combine_task, description=f"Combining {metric_type}/{section}")

                metrics[metric_type][section] = (
                    xr.combine_by_coords(runs_list, combine_attrs="drop_conflicts")
                    if runs_list
                    else xr.Dataset()
                )

                if show_progress and runs_list:
                    prog.advance(combine_task)

    runs_for_return = runs
    if apply_external_mask_to_runs and external_mask_data is not None:
        # Keep metric computation on the original loaded runs and only mask the
        # returned field datasets for downstream consumers such as plotting.
        runs_for_return = {
            model_name: ds if model_name == "mask" else apply_mask_to_dataset(ds, external_mask_data)
            for model_name, ds in runs.items()
        }

    return runs_for_return, metrics, tuple(last_clim_by_model.get(model_name) for model_name in models)


def get_runs_and_metrics(
    exp_root: str | Path,
    exp_mode: MLBCExperimentMode = "test",
    load_models: Sequence[str] = ("an", "fc", "pr"),
    variables: Sequence[str] | None = None,
    run_filters: dict[str, object] | None = None,
    calculate_clim_from_train_period: bool = False,
    use_train_prediction_clim: bool = False,
    use_saved_mask: bool = True,
    external_mask_data: xr.Dataset | xr.DataArray | None = None,
    apply_external_mask_to_runs: bool = False,
    metric_names: str | Sequence[str] | None = None,
    metric_sections: str | Sequence[str] = ("scalar", "map", "timeseries"),
    metric_types: str | Sequence[str] | None = None,
    show_progress: bool = True,
) -> tuple[
    RunsByRegion,
    MetricsByRegion,
    ClimsByRegion,
]:
    """
    Load runs and metrics grouped by region.

    Region is a top-level dictionary key in the returned structures rather than
    an xarray dimension, so each leaf dataset stays on a single spatial domain.
    Each metrics leaf preserves the historical single-region shape:
    ``metrics_by_region[region][metric_type][section] -> xr.Dataset``.
    """
    region_values = _discover_region_values(
        exp_root=exp_root,
        exp_mode=exp_mode,
        load_models=load_models,
        variables=variables,
        run_filters=run_filters,
        show_progress=show_progress,
    )

    if not region_values:
        return {}, {}, {}

    runs_by_region: RunsByRegion = {}
    metrics_by_region: MetricsByRegion = {}
    clims_by_region: ClimsByRegion = {}

    for region_name in region_values:
        region_filters = dict(run_filters or {})
        region_filters["region"] = [region_name]

        runs_region, metrics_region, clims_region = _get_single_region_runs_and_metrics(
            exp_root=exp_root,
            exp_mode=exp_mode,
            load_models=load_models,
            variables=variables,
            run_filters=region_filters,
            calculate_clim_from_train_period=calculate_clim_from_train_period,
            use_train_prediction_clim=use_train_prediction_clim,
            use_saved_mask=use_saved_mask,
            external_mask_data=external_mask_data,
            apply_external_mask_to_runs=apply_external_mask_to_runs,
            metric_names=metric_names,
            metric_sections=metric_sections,
            metric_types=metric_types,
            show_progress=show_progress,
        )

        if not runs_region:
            continue

        runs_by_region[str(region_name)] = runs_region
        metrics_by_region[str(region_name)] = metrics_region
        clims_by_region[str(region_name)] = clims_region

    return runs_by_region, metrics_by_region, clims_by_region


def _build_run_indexers(
    truth_runs: xr.Dataset,
    run_dims: Sequence[str],
    run_filters: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    if not run_dims:
        return [{}]

    coord_values = []
    for dim in run_dims:
        values = truth_runs.coords[dim].values.tolist()
        allowed = None if run_filters is None else run_filters.get(dim)
        if allowed is not None:
            allowed_values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            keep_empty = dim == "variant"
            allowed_strings = {
                str(value)
                for value in allowed_values
                if value is not None and (keep_empty or str(value) != "")
            }
            values = [value for value in values if str(value) in allowed_strings]
        coord_values.append(values)

    if any(len(values) == 0 for values in coord_values):
        return []

    out = []

    for values in product(*coord_values):
        indexers = dict(zip(run_dims, values))
        ds = truth_runs.sel(indexers, drop=True)

        if not ds.data_vars:
            continue
        if not bool(ds.to_array().notnull().any()):
            continue

        out.append(indexers)

    return out


def save_metrics(
    metrics: MetricLeaf,
    base_folder: str,
    vars_list: Sequence[str],
    metric_names: Sequence[str] | None = None,
    models_diff: Sequence[str] = ("fc", "pr"),
    partition_cols: Sequence[str] = ("train_period", "loss", "model", "metric", "variable"),
    kind: str = "scalar",
    metric_type: str = "all_dims",
):
    """Save metrics in parquet format. Return dict of dataframes and paths."""

    out = {}

    for diff in ("no", "delta", "ratio"):
        df = metrics_to_df_single_region(
            metrics=metrics,
            variables=vars_list,
            metric_names=metric_names,
            kind=kind,
            metric_type=metric_type,
            diff=diff,
            models=models_diff if diff in ("delta", "ratio") else None,
        )

        parquet_path = build_parquet_path(base_folder, vars_list, metric_names, diff)
        Path(parquet_path).mkdir(parents=True, exist_ok=True)

        df.to_parquet(
            parquet_path,
            partition_cols=list(partition_cols),
            engine="pyarrow",
        )

        out[diff] = (df, parquet_path)

    return out
