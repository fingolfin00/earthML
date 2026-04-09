from typing import Any, Sequence

from pathlib import Path
from dateutil.relativedelta import relativedelta
from datetime import datetime

from rich import print

import numpy as np
import pandas as pd
from scipy.signal import get_window
import cf_xarray
import xarray as xr
# import xskillscore as xs
# from scipy.stats import t as student_t

from .deterministic import DeterministicMetrics
from .correlation import CorrelationMetrics
from .probabilistic import ProbabilisticMetrics
from .bundles import build_standard_metric_bundle

from ..experiments.mlbc.load import load_exp, add_ke_to_runs


# ==========================================
# Standalone helper methods
# ==========================================

def _get_ds_from_metrics(
    metrics: dict,
    tp: str, model: str,
    kind: str,
    metric_name: str
) -> xr.Dataset:
    return metrics[tp]["models"][model][kind].get(metric_name, None)

def _get_da_from_metrics(
    metrics: dict,
    tp: str,
    model: str,
    kind: str,
    metric_name: str,
    model_a: str | None,
    model_b: str | None,
    variable: str,
    diff: str = "no", # diff options: no, delta, ratio
) -> xr.DataArray | None:
    if diff in ("delta", "ratio"):
        ds_a = _get_ds_from_metrics(metrics, tp, model_a, kind, metric_name)
        ds_b = _get_ds_from_metrics(metrics, tp, model_b, kind, metric_name)
        if ds_a is None or ds_b is None or variable not in ds_a or variable not in ds_b:
            return None
        ds_a, ds_b = xr.align(ds_a, ds_b, join="inner")
        if diff == "delta":
            da = ds_b[variable] - ds_a[variable]
        elif diff == "ratio":
            da = ( ds_b[variable] - ds_a[variable] ) / np.abs(ds_a[variable])
    else:
        ds = _get_ds_from_metrics(metrics, tp, model, kind, metric_name)
        if ds is None or variable not in ds:
            return None
        da = ds[variable]

    # print(list(da.coords))
    # keep leadtime
    for dim in list(da.dims):
        if dim != "leadtime" and da.sizes.get(dim, 1) == 1:
            da = da.squeeze(dim, drop=True)

    if hasattr(da.data, "persist"):
        da = da.persist() # make it work if Dask-based

    return da


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

def metrics_to_df(
    metrics_dict: dict,
    variables: list[str] | str | None = None,
    metric_names: list[str] | str | None = None,
    kind: str = "scalar",
    metric_type: str = "deterministic",
    diff: str = "no", # no, delta, ratio
    models: Sequence[str] | None = None, # list/tuple
):
    """Convert metrics dict to a pandas DataFrame."""

    if isinstance(metric_names, str):
        metric_names = [metric_names]
    elif metric_names is not None:
        metric_names = list(metric_names)

    if isinstance(variables, str):
        variables = [variables]
    elif variables is not None:
        variables = list(variables)

    section_key = f"{kind}_{metric_type}"

    first_ds = None
    for res in metrics_dict.values():
        ds = res.get(section_key)
        if isinstance(ds, xr.Dataset) and ds:
            first_ds = ds
            break

    if first_ds is None:
        raise ValueError(
            f"No metrics dataset found for section='{section_key}'. Check metrics_dict structure."
        )

    if metric_names is None:
        metric_names = list(first_ds.coords["metric"].values) if "metric" in first_ds.coords else []
    if variables is None:
        variables = list(first_ds.data_vars)

    if not metric_names:
        raise ValueError(f"No metric_names found under kind='{kind}'. Check metrics_dict structure.")
    if not variables:
        raise ValueError("No variables inferred. Check metrics_dict structure.")

    frames = []
    for tp, res in metrics_dict.items():
        ds = res.get(section_key)
        if ds is None or not ds.data_vars:
            continue

        available_metrics = set(ds.coords["metric"].values.tolist()) if "metric" in ds.coords else set()
        available_vars = [var for var in variables if var in ds.data_vars]
        selected_metrics = [metric for metric in metric_names if metric in available_metrics]
        if not available_vars or not selected_metrics:
            continue

        ds = ds[available_vars].sel(metric=selected_metrics)

        if diff in ("delta", "ratio"):
            if models is None:
                if "model" not in ds.coords or ds.sizes.get("model", 0) != 2:
                    raise ValueError("When diff is 'delta' or 'ratio', exactly two models are required")
                model_a, model_b = ds.coords["model"].values.tolist()
            else:
                if len(models) != 2:
                    raise ValueError("When diff is 'delta' or 'ratio', models must contain exactly two model names")
                model_a, model_b = models

            ds_a = ds.sel(model=model_a)
            ds_b = ds.sel(model=model_b)

            a = ds_a.to_array(dim="variable")
            b = ds_b.to_array(dim="variable")

            # print("A:", a)
            # print("A dtype:", a.dtype)
            # print("B:", b)
            # print("B dtype:", b.dtype)

            # print("Computing A...")
            a_loaded = a.compute()

            # print("Computing B...")
            b_loaded = b.compute()

            arr = b_loaded - a_loaded
            if diff == "ratio":
                arr = arr / np.abs(a_loaded)

            df_tp = arr.to_dataframe(name="value").reset_index()
            df_tp["model"] = f"{model_b}-{model_a}"
        else:
            if "model" in ds.coords:
                selected_models = ds.coords["model"].values.tolist() if models is None else [m for m in models if m in ds.coords["model"].values]
                if not selected_models:
                    continue
                ds = ds.sel(model=selected_models)

            df_tp = ds.to_array(dim="variable").to_dataframe(name="value").reset_index()

        if "leadtime" not in df_tp.columns:
            df_tp["leadtime"] = np.nan

        df_tp["train_period"] = tp
        frames.append(df_tp[["train_period", "model", "metric", "variable", "leadtime", "value"]])

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["train_period", "model", "metric", "variable", "leadtime", "value"]
    )

    # Add diff date
    df[["years", "months", "days", "total_days", "total_months"]] = (
        df["train_period"].astype(str)
            .apply(date_diff)
            .apply(pd.Series)
    )

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

def preprocess_datasets_for_metric(
    datasets: Sequence[xr.Dataset | None],
    *,
    name: str = "",
    leadtimes: Sequence[Any] | None = None,
    align_join_strategy: str = "inner",
    rechunk_factor: int = 1,
    allowed_extra_dims: Sequence[str] = ("missed_time",),
) -> tuple[list[xr.Dataset | None], dict[str, Any]]:
    """
    Generic preprocessing for metric calculation.

    Parameters
    ----------
    datasets
        Sequence of xr.Dataset or None. Output order matches input order.
    name
        Run name for logging/debugging.
    leadtimes
        Optional expected leadtime sequence. Its length is checked and, when
        provided, it is assigned as the canonical leadtime coordinate before
        dataset alignment.
    align_join_strategy
        Join strategy passed to xr.align.
    rechunk_factor
        If > 1, rechunk lat/lon to about 1 / rechunk_factor of the reference size.
    allowed_extra_dims
        Additional dims/coords to preserve beyond canonical dims.

    Returns
    -------
    processed, meta
        processed: list[xr.Dataset | None]
        meta: dict with canonical dim names and chunk info
    """
    if not datasets:
        raise ValueError("datasets must not be empty")

    ds_list = list(datasets)

    for i,ds in enumerate(ds_list):
        ds_list[i] = ds.earthml.normalize_dims_and_coords()

    time_dim, lat_dim, lon_dim, realization_dim, leadtime_dim = ds_list[0].earthml.guessed_dims






    # Optional leadtime length check and coordinate harmonization.
    if leadtimes is not None:
        expected_n = len(leadtimes)
        actual = {}
        for i, ds in enumerate(ds_list):
            if ds is None:
                continue
            if leadtime_dim not in ds.sizes:
                raise ValueError(
                    f"Dataset at index {i} does not contain leadtime dim {leadtime_dim!r}"
                )
            actual[i] = ds.sizes[leadtime_dim]

        bad = {i: n for i, n in actual.items() if n != expected_n}
        if bad:
            raise ValueError(
                f"Leadtime length mismatch for run {name!r}: "
                f"{bad}, expected {expected_n}"
            )

    leadtime_info = {
        i: (
            ds[leadtime_dim].values
            if ds is not None and leadtime_dim in ds.coords
            else "N/A"
        )
        for i, ds in enumerate(ds_list)
    }
    print(f"Calculate metrics for run {name}; leadtimes: {leadtime_info}")

    allowed_dims = {
        time_dim,
        lat_dim,
        lon_dim,
        realization_dim,
        leadtime_dim,
        *allowed_extra_dims,
    }
    print(f"Allowed dims: {allowed_dims}")

    # Remove unwanted dims/coords
    ds_list = [
        ds.earthml.remove_dims_and_coords(allowed_dims) if ds is not None else None
        for ds in ds_list
    ]

    # Align only when necessary
    non_null = [ds for ds in ds_list if ds is not None]
    if len(non_null) > 1:
        aligned = xr.align(*non_null, join=align_join_strategy)
        it = iter(aligned)
        ds_list = [next(it) if ds is not None else None for ds in ds_list]

        # Apply shared valid mask only when there is more than one dataset
        non_null = [ds for ds in ds_list if ds is not None]
        valid_mask = np.isfinite(non_null[0].to_array()).all("variable")
        for ds in non_null[1:]:
            valid_mask = valid_mask & np.isfinite(ds.to_array()).all("variable")
        ds_list = [ds.where(valid_mask) if ds is not None else None for ds in ds_list]

    # Rechunk only when requested
    lat_rc = lon_rc = None
    if rechunk_factor > 1:
        ref = ds_list[0] # use first dataset as reference for rechunking
        if ref is None:
            raise RuntimeError("Reference dataset became None unexpectedly")

        lat_rc = max(1, ref.sizes[lat_dim] // rechunk_factor)
        lon_rc = max(1, ref.sizes[lon_dim] // rechunk_factor)

        print("Rechunking...")
        ds_list = [
            rechunk(ds, lat_rc=lat_rc, lon_rc=lon_rc) if ds is not None else None
            for ds in ds_list
        ]

    meta = {
        "dims": {
            "time": time_dim,
            "lat": lat_dim,
            "lon": lon_dim,
            "realization": realization_dim,
            "leadtime": leadtime_dim,
        },
        "allowed_dims": allowed_dims,
        "lat_rc": lat_rc,
        "lon_rc": lon_rc,
    }

    return ds_list, meta

# TODO: allow calculating only some metrics
def get_runs_and_metrics(
    var_names: str | Sequence[str] | dict | Sequence[dict], var_suffix: str | Sequence[str],
    region: str,
    exp_suffix_root: str | Sequence[str], exp_name: str, exp_root: str | Path,
    input_provider: str | Sequence[str], target_provider: str | Sequence[str], # same dims as var_names
    leadtimes: tuple, leadtime_unit: str,
    train_periods: Sequence[str], test_period: str,
    type_data: str = "test", models: Sequence[str] = ("an", "fc", "pr"),
    calculate_clim_from_train_period: bool = False,
    use_train_prediction_clim: bool = False, # only used if calculate_clim_from_train_period is True, if True requires train_preds available
    use_saved_mask: bool = True,
    align_join_strategy: str = "inner",
    rechunk_factor: int = 1, # default no lat/lon rechunking, set to e.g. 4 to reduce memory usage by rechunking to 1/4 of original lat/lon chunk size
) -> Sequence[dict]:

    models = list(models)
    if len(models) < 2:
        raise ValueError("models must contain at least two entries: truth first, then at least one model")

    var_names = [var_names] if isinstance(var_names, (str, dict)) else list(var_names)
    var_suffix = [var_suffix] if isinstance(var_suffix, str) else list(var_suffix)
    exp_suffix_root = [exp_suffix_root] if isinstance(exp_suffix_root, str) else list(exp_suffix_root)
    input_provider = [input_provider] if isinstance(input_provider, str) else list(input_provider)
    target_provider = [target_provider] if isinstance(target_provider, str) else list(target_provider)

    exp_suffix = []
    for suf in var_suffix:
        for suf_root in exp_suffix_root:
            exp_suffix.append(f"{suf}{suf_root}")
    # print(f"Exp suffix: {exp_suffix}")

    # Sanity check, providers must align with var_names
    if not (len(input_provider) == len(target_provider) == len(var_names)):
        raise ValueError("input_provider, target_provider and var_names must have the same length")

    vars_dict = {}

    # For each variable (paired with its providers), iterate over the paired suffixes
    for var, ips, tps in zip(var_names, input_provider, target_provider):
        for exp_suf in exp_suffix:
            if isinstance(var, str) and isinstance(ips, str) and isinstance(tps, str):
                var_fc, var_an = var, var
            elif isinstance(var, dict):
                var_fc, var_an = var['fc'], var['an']
            else:
                continue

            vars_dict[f"{var_fc}{exp_suf}"] = {
                "exp_var": {"fc": var_fc, "an": var_an},
                "region": region,
                "exp_suffix": exp_suf,
                "input_provider": ips,
                "target_provider": tps,
            }

    experiments = {
        "name": exp_name,
        "vars": vars_dict,
        "leadtimes": leadtimes,
        "leadtime_unit": leadtime_unit,
        "train_periods": train_periods,
        "test_period": test_period,
        "root": exp_root,
    }

    runs, _ = load_exp(
        exp_cfg=experiments,
        type_data=type_data,
        load_train_preds=True if type_data=="train" else False,
        load_models=models,
    )
    runs = add_ke_to_runs(runs, suffixes=var_suffix)

    if calculate_clim_from_train_period:
        train_runs, _ = load_exp(
            exp_cfg=experiments,
            type_data="train",
            load_train_preds=use_train_prediction_clim,
            load_models=models,
        )
        train_runs = add_ke_to_runs(train_runs, suffixes=var_suffix)

    truth_model = models[0]
    selected_model_names = models[1:]

    metrics = {}
    clim_by_model = {truth_model: None, **{model_name: None for model_name in selected_model_names}}

    for name, run in runs.items():
        an = run[truth_model]
        selected_models = [run[model_name] for model_name in selected_model_names]
        mask = run["mask"] if use_saved_mask else None

        processed, _ = preprocess_datasets_for_metric(
            datasets=[an, *selected_models, mask],
            name=name,
            leadtimes=leadtimes,
            reference_index=1,
            align_join_strategy=align_join_strategy,
            rechunk_factor=rechunk_factor,
        )

        an_m = processed[0]
        processed_models = processed[1:-1]
        mask_m = processed[-1]

        candidate_data = list(zip(processed_models, selected_model_names))
        data_models = [ds for ds, _ in candidate_data if ds is not None]
        data_model_names = [nm for ds, nm in candidate_data if ds is not None]

        if calculate_clim_from_train_period:
            an_train = train_runs[name][truth_model]
            train_model_names = list(selected_model_names)
            if not use_train_prediction_clim:
                train_model_names = [model_name for model_name in train_model_names if model_name != "pr"]
            train_model_data = [train_runs[name][model_name] for model_name in train_model_names]

            processed_train, _ = preprocess_datasets_for_metric(
                datasets=[an_train, *train_model_data],
                name=f"{name}_train",
                leadtimes=leadtimes,
                reference_index=1,
                align_join_strategy=align_join_strategy,
                rechunk_factor=rechunk_factor,
            )
            an_train_m = processed_train[0]
            an_clim_ts = an_train_m.earthml.climatology(groupby="month")
            clim_by_model[truth_model] = an_clim_ts
            for model_name, train_ds in zip(train_model_names, processed_train[1:]):
                clim_by_model[model_name] = train_ds.climatology(groupby="month")
            if "pr" in clim_by_model and clim_by_model["pr"] is None:
                clim_by_model["pr"] = an_clim_ts

        runs[name][truth_model] = an_m
        for model_name, model_ds in zip(selected_model_names, processed_models):
            runs[name][model_name] = model_ds
        runs[name]["mask"] = mask_m

        clim_data = [clim_by_model[truth_model], *[clim_by_model[model_name] for model_name in data_model_names]]

        dm = DeterministicMetrics(
            truth_data=an_m,
            model_data=data_models,
            truth_name=truth_model,
            model_names=data_model_names,
            clim_data=clim_data,
            mask_data=mask_m,
        )
        cm = CorrelationMetrics(
            truth_data=an_m,
            model_data=data_models,
            truth_name=truth_model,
            model_names=data_model_names,
            clim_data=clim_data,
            mask_data=mask_m,
        )
        pm = ProbabilisticMetrics(
            truth_data=an_m,
            model_data=data_models,
            truth_name=truth_model,
            model_names=data_model_names,
            clim_data=clim_data,
            mask_data=mask_m,
        )

        metrics[name] = build_standard_metric_bundle(dm, cm, pm, norm="std", clim_period="month")

    return runs, metrics, tuple(clim_by_model.get(model_name) for model_name in models)

def save_metrics (
    metrics: dict,
    base_folder: str,
    vars_list: Sequence[str],
    metric_names: Sequence[str] | None = None, # None -> save all available metrics
    models_diff: Sequence[str] = ("fc", "pr"), # used only for diff
    partition_cols: Sequence[str] = ("train_period", "model", "metric", "variable"), # no leadtime
):
    """Save metrics in pandas dataframe format to parquet files. Return dict of dataframes and paths."""

    out = {}

    for diff in ("no", "delta", "ratio"):
        df = metrics_to_df(
            metrics,
            variables=vars_list,
            metric_names=metric_names,
            diff=diff,
            models=models_diff if diff in ("delta", "ratio") else None,
        )

        parquet_path = build_parquet_path(base_folder, vars_list, metric_names, diff)

        # Ensure output directory exists
        Path(parquet_path).mkdir(parents=True, exist_ok=True)

        df.to_parquet(
            parquet_path,
            partition_cols=list(partition_cols),
            engine="pyarrow",
        )

        out[diff] = (df, parquet_path)

    return out
