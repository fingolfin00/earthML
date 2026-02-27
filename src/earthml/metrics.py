from typing import List, Any, Literal, Callable, Optional, Sequence
from pathlib import Path
from itertools import product

from rich import print
from rich.pretty import pprint
from rich.console import Console

import numpy as np
import pandas as pd
from scipy.signal import get_window
import cf_xarray
import xarray as xr
# import xskillscore as xs
# from scipy.stats import t as student_t

from .utils import guess_time_dim, guess_lon_dim, guess_lat_dim, guess_realization_dim, guess_leadtime_dim, date_diff, load_exp, add_ke_to_runs, remove_unwanted_dims_and_coords, geo_weights, geo_avg

MetricFn = Callable[[xr.Dataset, xr.Dataset], xr.Dataset]
FinalFn  = Callable[[xr.Dataset], xr.Dataset]

# Standalone helper methods

def _get_ds_from_metrics (
    metrics: dict,
    tp: str, model: str,
    kind: str,
    metric_name: str
) -> xr.Dataset:
    return metrics[tp]["models"][model][kind].get(metric_name, None)

def _get_da_from_metrics (
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

def build_parquet_path (
    base_folder: str,
    vars_list: Sequence[str],
    metric_names: Sequence[str] | None = None,
    diff: str = "no", # no, delta, ratio
) -> str:
    vars_tag = "_".join(sorted(map(str, vars_list))) if vars_list else "allvars"
    metrics_tag = "_".join(sorted(map(str, metric_names))) if metric_names else "allmetrics"

    return str(Path(base_folder) / f"metrics_{diff}_{vars_tag}_{metrics_tag}_parquet")

def infer_metric_names (metrics_dict: dict, kind: str) -> list[str]:
    # grab first tp/model/kind and return keys there
    for _, res in metrics_dict.items():
        for _, md in res["models"].items():
            if kind in md:
                return list(md[kind].keys())
    return []

def infer_variables (metrics_dict: dict, kind: str, metric_name: str | None = None) -> list[str]:
    for tp, res in metrics_dict.items():
        for model, md in res.get("models", {}).items():
            kind_dict = md.get(kind, {})
            if not isinstance(kind_dict, dict) or not kind_dict:
                continue

            # Infer from first metric dataset if metric_name is not provided
            m = metric_name or next(iter(kind_dict.keys()))
            ds = kind_dict.get(m)
            if ds is None:
                continue

            return list(ds.data_vars)
    return []

def metrics_to_df (
    metrics_dict: dict,
    variables: list[str] | str | None = None,
    metric_names: list[str] | str | None = None,
    kind: str = "scalar",
    diff: str = "no", # no, delta, ratio
    models: Sequence[str] | None = None, # list/tuple
):
    """Convert metrics dict to a pandas DataFrame."""

    if metric_names is None:
        metric_names = infer_metric_names(metrics_dict, kind)
    elif isinstance(metric_names, str):
        metric_names = [metric_names]
    else:
        metric_names = list(metric_names)

    if variables is None:
        variables = infer_variables(metrics_dict, kind, metric_names[0] if metric_names else None)
    elif isinstance(variables, str):
        variables = [variables]
    else:
        variables = list(variables)

    if not metric_names:
        raise ValueError(f"No metric_names found under kind='{kind}'. Check metrics_dict structure.")
    if not variables:
        raise ValueError("No variables inferred. Check metrics_dict structure.")

    rows = []
    for tp, res in metrics_dict.items():
        models_tp = list(res["models"].keys()) if models is None else list(models)

        if diff in ("delta", "ratio"):
            if len(models_tp) != 2:
                raise ValueError("When diff=True, models must contain exactly two model names")
            model_a, model_b = models_tp
            model_iter = [f"{model_b}-{model_a}"]
        else:
            model_iter = models_tp
            model_a = model_b = None  # not used

        for model_name in model_iter:
            for m in metric_names:
                for var in variables:
                    da = _get_da_from_metrics(metrics_dict, tp, model_name, kind, m, model_a, model_b, var, diff)
                    if da is None:
                        continue

                    if da.ndim == 0:
                        rows.append({
                            "train_period": tp,
                            "model": model_name,
                            "metric": m,
                            "variable": var,
                            "leadtime": np.nan,
                            "value": float(da.values) if np.isfinite(da.values) else np.nan,
                        })
                    else:
                        df_da = da.to_dataframe(name="value").reset_index()

                        if "leadtime" not in df_da.columns:
                            df_da["leadtime"] = np.nan

                        df_da["train_period"] = tp
                        df_da["model"] = model_name
                        df_da["metric"] = m
                        df_da["variable"] = var

                        rows.extend(
                            df_da[["train_period", "model", "metric", "variable", "leadtime", "value"]]
                            .to_dict("records")
                        )

    df = pd.DataFrame(rows)

    # Add diff date
    df[["years", "months", "days", "total_days", "total_months"]] = (
        df["train_period"].astype(str)
            .apply(date_diff)
            .apply(pd.Series)
    )

    return df

def rechunk (da, lat_rc, lon_rc, time_rc=1, realization_rc=1):
    chunks = {guess_time_dim(da): time_rc, guess_lat_dim(da): lat_rc, guess_lon_dim(da): lon_rc}
    realization_dim = guess_realization_dim(da)
    if realization_dim in da.dims:
        chunks[realization_dim] = realization_rc
    leadtime_dim = guess_leadtime_dim(da)
    if leadtime_dim in da.dims:
        chunks[leadtime_dim] = len(da[leadtime_dim]) # keep leadtime together
    return da.chunk(chunks)

def rename_dim_and_coord (ds: xr.Dataset, src: str, target: str) -> xr.Dataset:
    # Rename dimension
    if src in ds.dims:
        ds = ds.rename_dims({src: target})
    # Rename coordinate or variable with the same name
    if (src in ds.coords) or (src in ds.data_vars):
        ds = ds.rename_vars({src: target})
    return ds

def normalize_reference (reference: xr.Dataset) -> xr.Dataset:
    # Standardize ensemble + leadtime names on the reference itself
    r = reference

    src_real = guess_realization_dim(r)
    if src_real is not None and src_real != "realization":
        r = rename_dim_and_coord(r, src_real, "realization")

    src_lead = guess_leadtime_dim(r)
    # print(f"Reference dataset leadtime dimension: {src_lead}")
    # print(f"Reference dataset right before renaming dims: {r.dims}")
    if src_lead is not None and src_lead != "leadtime":
        r = rename_dim_and_coord(r, src_lead, "leadtime")

    return r

def get_and_rename_dim (reference: xr.Dataset, data: List[xr.Dataset]):
    # Normalize reference first
    r = normalize_reference(reference)

    # Decide final target names (match reference for time/lat/lon; standard for the others)
    target_time = guess_time_dim(r)
    target_lat  = guess_lat_dim(r)
    target_lon  = guess_lon_dim(r)

    target_dims = (target_time, target_lat, target_lon, "realization", "leadtime")

    out: List[xr.Dataset] = []
    for d in data:
        ds = d

        # Guess source names on the current ds
        src_time = guess_time_dim(ds)
        src_lat  = guess_lat_dim(ds)
        src_lon  = guess_lon_dim(ds)
        src_real = guess_realization_dim(ds)
        src_lead = guess_leadtime_dim(ds)

        src_dims = (src_time, src_lat, src_lon, src_real, src_lead)

        for target, src in zip(target_dims, src_dims):
            if target is None or src is None or src == target:
                continue
            ds = rename_dim_and_coord(ds, src, target)

        out.append(ds)

    return r, out, target_dims  # (time, lat, lon, realization, leadtime)

def _is_int_leadtime (ds, coord="leadtime"):
    if coord not in ds.coords:
        return False
    return np.issubdtype(ds[coord].dtype, np.integer)

def harmonize_leadtime_int (*datasets, coord="leadtime", fallback="range"):
    """
    Make all datasets share the same integer leadtime coordinate.
    - If any dataset already has int leadtime, that coord is used as the target.
    - Otherwise fallback:
        * "range": set leadtime = 1..N based on the first dataset
        * "days": convert timedelta leadtime to integer days (requires timedelta)
    Returns: list of datasets in same order as input.
    """
    dsets = list(datasets)

    # Pick target integer leadtime if present
    int_sources = [ds for ds in dsets if _is_int_leadtime(ds, coord)]
    if int_sources:
        target = int_sources[0][coord].astype("int64")
    else:
        # No integer leadtime anywhere -> choose fallback
        if coord not in dsets[0].coords:
            raise ValueError(f"No coord {coord!r} found in the first dataset.")
        n = dsets[0].sizes.get(coord, dsets[0][coord].size)

        if fallback == "range":
            target = xr.DataArray(np.arange(1, n + 1), dims=coord, coords={coord: np.arange(1, n + 1)})
        elif fallback == "days":
            lt0 = dsets[0][coord]
            if not np.issubdtype(lt0.dtype, np.timedelta64):
                raise ValueError("fallback='days' requires timedelta64 leadtime.")
            days = (lt0 / np.timedelta64(1, "D")).astype("int64")
            target = xr.DataArray(days.values, dims=coord, coords={coord: days.values})
        else:
            raise ValueError("fallback must be 'range' or 'days'.")

    out = []
    for ds in dsets:
        if coord not in ds.coords:
            raise ValueError(f"Dataset missing coord {coord!r}.")
        if ds.sizes.get(coord, ds[coord].size) != target.size:
            raise ValueError(f"Leadtime length mismatch: {ds.sizes.get(coord)} vs {target.size}")

        ds2 = ds.assign_coords({coord: target})
        out.append(ds2)

    return out

# TODO: allow calculating only some metrics
def get_runs_and_metrics (
    var_names: str | Sequence[str] | dict | Sequence[dict], var_suffix: str | Sequence[str],
    region: str,
    exp_suffix_root: str | Sequence[str], exp_name: str, exp_root: str | Path,
    input_provider: str, target_provider: str,
    leadtimes: tuple, leadtime_unit: str,
    train_periods: Sequence[str], test_period: str,
    type_data: str = "test", models: Sequence[str] = ("an", "fc", "pr"),
    align_join_strategy: str = "inner",
    rechunk_factor: int = 1, # default no lat/lon rechunking, set to e.g. 4 to reduce memory usage by rechunking to 1/4 of original lat/lon chunk size
) -> Sequence[dict]:

    var_names = [var_names] if isinstance(var_names, str) or isinstance(var_names, dict) else var_names
    var_suffix = [var_suffix] if isinstance(var_suffix, str) else var_suffix
    exp_suffix_root = [exp_suffix_root] if isinstance(exp_suffix_root, str) else exp_suffix_root

    exp_suffix = []
    for suf in var_suffix:
        for suf_root in exp_suffix_root:
            exp_suffix.append(f"{suf_root}{suf}")
    # print(f"Exp suffix: {exp_suffix}")

    vars_dict = {}
    for var, (var_suf, exp_suf) in product(var_names, zip(var_suffix, exp_suffix)):
        if isinstance(var, str):
            var_fc, var_an = var, var
        elif isinstance(var, dict):
            var_fc, var_an = var['fc'], var['an']

        vars_dict[f"{var_fc}{var_suf}"] = {
                "exp_var": {"fc": var_fc, "an": var_an},
                "region": region,
                "exp_suffix": exp_suf,
            }

    experiments = {
        "name": exp_name,
        "input_provider": input_provider,
        "target_provider": target_provider,
        "vars": vars_dict,
        "leadtimes": leadtimes,
        "leadtime_unit": leadtime_unit,
        "train_periods": train_periods,
        "test_period": test_period,
        "root": exp_root,
    }

    runs, _ = load_exp(
        exp_root=exp_root,
        exp_cfg=experiments,
        type_data=type_data,
    )

    truth_model = models[0] # an
    data_model_a, data_model_b = models[1], models[2] # fc, pr

    runs = add_ke_to_runs(runs, suffixes=var_suffix)

    metrics = {}
    for name, run in runs.items():
        an, fc, pr = run[truth_model], run[data_model_a], run[data_model_b] # dict of datasets per leadtime
        # print(an_lt)

        # Rename all dims using fc as reference
        fc, renamed_ds, dims = get_and_rename_dim(fc, [an, pr] if pr is not None else [an])
        an, pr = renamed_ds[0], renamed_ds[1] if pr is not None else None
        time_dim, lat_dim, lon_dim, realization_dim, leadtime_dim = dims

        assert len(an[leadtime_dim].values) == len(fc[leadtime_dim].values) == len(pr[leadtime_dim].values if pr is not None else [None]*len(an[leadtime_dim].values)) == len(leadtimes), f"Length mismatch: an {len(an[leadtime_dim].values)}, fc {len(fc[leadtime_dim].values)}, pr {len(pr[leadtime_dim].values if pr is not None else [None]*len(an[leadtime_dim].values))}, leadtimes {len(leadtimes)}"

        print(f"Calculate metrics for run {name} and leadtimes fc {fc[leadtime_dim].values}, leadtime an {an[leadtime_dim].values if an is not None else 'N/A'}, leadtime pr {pr[leadtime_dim].values if pr is not None else 'N/A'}")

        # print(f"Before renaming an dims: {an.dims}, fc dims: {fc.dims}, pr dims: {pr.dims if pr is not None else 'N/A'}")
        allowed_dims = {
            time_dim,
            lat_dim,
            lon_dim,
            realization_dim,
            leadtime_dim,
            "missed_time",
        }
        print(f"Allowed dims: {allowed_dims}")
        # print(f"Before harmonizing and removing extra dims an dims: {an.dims}, fc dims: {fc.dims}, pr dims: {pr.dims if pr is not None else 'N/A'}")

        if pr is None:
            an, fc = harmonize_leadtime_int(an, fc, coord="leadtime", fallback="range")
            # print(f"After harmonizing and before removing extra dims an dims: {an.dims}, fc dims: {fc.dims}, pr dims: {pr.dims if pr is not None else 'N/A'}")
            an, fc = remove_unwanted_dims_and_coords(an, allowed_dims), remove_unwanted_dims_and_coords(fc, allowed_dims)
        else:
            an, fc, pr = harmonize_leadtime_int(an, fc, pr, coord="leadtime", fallback="range")
            an, fc, pr = remove_unwanted_dims_and_coords(an, allowed_dims), remove_unwanted_dims_and_coords(fc, allowed_dims), remove_unwanted_dims_and_coords(pr, allowed_dims)
        # print(f"Before aligning an dims: {an.dims}, fc dims: {fc.dims}, pr dims: {pr.dims if pr is not None else 'N/A'}")
        # print(an, fc, pr)

        # Align masks
        if pr is None:
            # print(fc[leadtime_dim])
            # print(an[leadtime_dim])
            fc_a, an_a = xr.align(fc, an, join=align_join_strategy) # inner: intersection of coords, outer: union of coords
            valid_mask = np.isfinite(fc_a) & np.isfinite(an_a)
            fc_m = fc_a.where(valid_mask)
            an_m = an_a.where(valid_mask)
            pr_m = None
            data_models = [fc_m]
            data_model_names = [data_model_a]
        else:
            fc_a, pr_a, an_a = xr.align(fc, pr, an, join=align_join_strategy)
            valid_mask = np.isfinite(fc_a) & np.isfinite(pr_a) & np.isfinite(an_a)
            fc_m = fc_a.where(valid_mask)
            pr_m = pr_a.where(valid_mask)
            an_m = an_a.where(valid_mask)
            # print("Valid fraction an:", np.isfinite(an).mean().values)
            # print("Valid fraction fc:", np.isfinite(fc).mean().values)
            # print("Valid fraction pr:", np.isfinite(pr).mean().values)
            # print("Valid fraction intersection:", valid_mask.mean().values)
            data_models = [fc_m, pr_m]
            data_model_names = [data_model_a, data_model_b]

        lat_rc, lon_rc = an_m.sizes[lat_dim] // rechunk_factor, an_m.sizes[lon_dim] // rechunk_factor
        print("Rechunking...")
        # print(f"Before rechunking an dims: {an_m.dims}, fc dims: {fc_m.dims}, pr dims: {pr_m.dims if pr_m is not None else 'N/A'}")

        if pr is not None:
            an_m, fc_m, pr_m = rechunk(an_m, lat_rc, lon_rc), rechunk(fc_m, lat_rc, lon_rc), rechunk(pr_m, lat_rc, lon_rc)
        else:
            an_m, fc_m = rechunk(an_m, lat_rc, lon_rc), rechunk(fc_m, lat_rc, lon_rc)
            pr_m = None
        # print(f"After rechunking an dims: {an_m.dims}, fc dims: {fc_m.dims}, pr dims: {pr_m.dims if pr_m is not None else 'N/A'}")

        runs[name][truth_model], runs[name][data_model_a], runs[name][data_model_b] = an_m, fc_m, pr_m if pr is not None else (an_m, fc_m, None)

        m = Metrics(truth=an_m, data=data_models, truth_name=truth_model, data_name=data_model_names)
        metrics[name] = m.compute_all_metrics(geo_weighted=True) #, eps=1e-12)

    return runs, metrics # runs not aligned

# Classes

class Metrics:

    def __init__ (self, truth: xr.Dataset, data: xr.Dataset | List[xr.Dataset], truth_name: str , data_name: str | List[str]):
        self.truth = truth
        self.data = data if isinstance(data, list) else [data]

        # Drop _has_var
        truth_sel_vars = [v for v in self.truth.data_vars if v != '_has_var']
        self.truth = self.truth[truth_sel_vars]
        sel_data = []
        for d in self.data:
            data_sel_vars = [v for v in d.data_vars if v != '_has_var']
            sel_data.append(d[data_sel_vars])
        self.data = sel_data

        # Align and mask
        datasets = [self.truth] + self.data
        # ds_aligned = xr.align(*datasets, join="outer")
        ds_aligned = xr.align(*datasets, join="inner")

        masks = [np.isfinite(ds.to_array()).all("variable") for ds in ds_aligned]

        valid = masks[0]
        for m in masks[1:]:
            valid = valid & m

        ds_masked = [ds.where(valid) for ds in ds_aligned]
        self.truth = ds_masked[0]
        self.data = ds_masked[1:]

        self.truth_name = truth_name
        self.data_name = data_name if isinstance(data_name, list) else [data_name]

        self.truth, self.data, dims = self._get_and_rename_dim()
        self.time_dim, self.lat_dim, self.lon_dim, self.realization_dim, self.leadtime_dim = dims

        print("Metrics initialized with truth and data shapes:")
        print(truth_name, self.truth.dims)
        for d, n in zip(self.data, self.data_name):
            print(n, d.dims)

    def _get_and_rename_dim (self):
        return get_and_rename_dim(self.truth, self.data)

    @staticmethod
    def save (
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

    # Metrics
    def _generic_metric (
        self,
        truth: xr.Dataset,
        data: List[xr.Dataset],
        parent_metric_fn: MetricFn,
        final_metric_fn: Optional[FinalFn] = None,
        order: Literal["3d", "2d", "1d"] = "1d",
        geo_weighted: bool = True,
    ) -> list[xr.Dataset]:

        out: list[xr.Dataset] = []

        for d in data:
            metric = parent_metric_fn(truth, d)

            if order == "1d":
                if geo_weighted:
                    val = geo_avg(metric, self.lat_dim, self.lon_dim).mean(dim=self.time_dim, skipna=True)
                else:
                    val = metric.mean(dim=(self.time_dim, self.lat_dim, self.lon_dim), skipna=True)
            elif order == "2d":
                val = metric.mean(dim=self.time_dim, skipna=True)
            elif order == "3d":
                val = metric
            else:
                raise ValueError(f"Invalid order: {order}")

            if final_metric_fn is not None:
                val = final_metric_fn(val)

            # print("_generic_metric value:", parent_metric_fn, final_metric_fn, order, val)
            out.append(val)

        return out

    def mae (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> List[xr.Dataset]:
        return self._generic_metric(self.truth, self.data, lambda truth, pred: abs(pred - truth), order=order, geo_weighted=geo_weighted)

    @staticmethod
    def _err_field (truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
        return pred - truth # negative if pred is underestimating truth

    def err (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> List[xr.Dataset]:
        # 2d and 1d cases are bias
        return self._generic_metric(self.truth, self.data, self._err_field, order=order, geo_weighted=geo_weighted)

    def abs_bias (self, order: Literal["2d", "1d"] = "1d", geo_weighted: bool = True) -> list[xr.Dataset]:
        # compute err first, then abs using xarray .abs()
        err_list = self.err(order=order, geo_weighted=geo_weighted)
        return self._generic_metric(self.truth, self.data, self._err_field, order=order, geo_weighted=geo_weighted, final_metric_fn=np.abs)
        # return [abs(x) for x in self.err(order=order, geo_weighted=geo_weighted)]

    def std_of_errs (
        self,
        order: Literal["3d", "2d", "1d"] = "1d",
        geo_weighted: bool = True,
    ) -> list[xr.Dataset]:
        """
        Standard deviation of errors (pred - truth).

        Definitions
        -----------
        err = pred - truth

        - order="2d" or "3d":
            std_map = std_t( err )   # std over the time dimension
            Returns a 2D (lat, lon) map (plus any extra dims, e.g. leadtime).

        - order="1d":
            std_map as above, then:
              * if geo_weighted: area-weighted mean over lat/lon
              * else: plain mean over lat/lon

            Returns a scalar per variable (and any remaining non-spatial dims).

        Notes
        -----
        - "3d" is treated as an alias for "2d" here, because the natural
          std-of-errors quantity is a time-reduced field.
        """
        out: list[xr.Dataset] = []

        for d in self.data:
            # error field: same shape as truth
            err = self._err_field(self.truth, d)

            # std over time dimension
            std_map = err.std(dim=self.time_dim, skipna=True)

            if order in ("2d", "3d"):
                # map of standard deviation of errors
                out.append(std_map)
            elif order == "1d":
                # global scalar (mean of std_map over space)
                if geo_weighted:
                    out.append(geo_avg(std_map, self.lat_dim, self.lon_dim))
                else:
                    out.append(std_map.mean(dim=(self.lat_dim, self.lon_dim), skipna=True))
            else:
                raise ValueError(f"Invalid order: {order}")

        return out

    def stderr (
        self,
        order: Literal["3d", "2d", "1d"] = "1d",
        geo_weighted: bool = True,
    ) -> list[xr.Dataset]:
        """
        Standard error of the time-mean error (bias).

        err = pred - truth

        Per-grid, the standard error is:
            SE(lat, lon, ...) = std_t(err) / sqrt(N_t(lat, lon, ...))

        where:
            std_t(err) is the std over time_dim
            N_t is the number of finite samples along time_dim.

        - order="2d" or "3d":
            Returns the SE map at each (lat, lon, ...).

        - order="1d":
            First compute the SE map as above, then:
              * if geo_weighted: area-weighted mean over lat/lon
              * else: plain mean over lat/lon.

            Returns a scalar per variable (and any non-spatial dims).
        """
        out: list[xr.Dataset] = []

        for d in self.data:
            err = self._err_field(self.truth, d)

            # number of valid time samples per grid cell
            n = err.count(dim=self.time_dim)

            # std over time
            sigma = err.std(dim=self.time_dim, skipna=True)

            # avoid division by zero where n == 0
            n_safe = xr.where(n > 0, n, np.nan)

            stderr_map = sigma / np.sqrt(n_safe)

            if order in ("2d", "3d"):
                out.append(stderr_map)
            elif order == "1d":
                if geo_weighted:
                    out.append(geo_avg(stderr_map, self.lat_dim, self.lon_dim))
                else:
                    out.append(stderr_map.mean(dim=(self.lat_dim, self.lon_dim), skipna=True))
            else:
                raise ValueError(f"Invalid order: {order}")

        return out

    def rmse (self, order: Literal["2d", "1d"] = "1d", geo_weighted: bool = True) -> List[xr.Dataset]:
        return self._generic_metric(
            self.truth, self.data,
            lambda truth, pred: (pred - truth)**2,
            final_metric_fn=np.sqrt,
            order=order,
            geo_weighted=geo_weighted
        )
    
    def mape (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Mean Absolute Percentage Error.

        Returns:
        - order="3d": pointwise APE (|err|/|truth| * 100)
        - order="2d": time-mean of APE (2D field)
        - order="1d": time-mean then spatial mean (scalar), geo-weighted if requested
        """
        def _mape_field (truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
            denom = xr.where(np.abs(truth) > eps, np.abs(truth), np.nan)
            return (np.abs(pred - truth) / denom) * 100.0

        return self._generic_metric(self.truth, self.data, _mape_field, order=order, geo_weighted=geo_weighted)

    def smape (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> list[xr.Dataset]:
        """
        Symmetric Mean Absolute Percentage Error
        """
        def _smape_field (truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
            denom = np.abs(truth) + np.abs(pred)
            denom = xr.where(denom > 0, denom, np.nan)
            return 100.0 * (2 * np.abs(pred - truth) / denom)

        return self._generic_metric(self.truth, self.data, _smape_field, order=order, geo_weighted=geo_weighted)

    # Normalized metrics, maps
    def _std_map (self, data, eps: float = 0.0) -> xr.Dataset:
        """Truth std map over time, with eps thresholding."""
        std_map = data.std(dim=self.time_dim, skipna=True)
        return xr.where(std_map > eps, std_map, np.nan)

    def nrmse_map (self, order: Literal["2d", "1d"] = "2d", geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        NRMSE normalized by truth std over time (map-wise).

        - order="2d": returns a 2D NRMSE map: RMSE_map / std_map
        - order="1d": returns a scalar per dataset by spatially averaging the 2D NRMSE map
                    (geo-weighted if geo_weighted=True)

        Notes:
        - std_map = std(truth over time_dim)
        - cells where std_map <= eps are set to NaN (ignored by skipna reductions)
        """

        # RMSE map per dataset (time-reduced)
        rmse_maps = self.rmse(order="2d", geo_weighted=geo_weighted) # geo_weighted irrelevant for 2d

        # Truth std map over time (same for all predictions)
        std_map_truth = self._std_map(self.truth, eps=eps)

        out: list[xr.Dataset] = []
        for rmse_map in rmse_maps:
            nrmse_map = rmse_map / std_map_truth
            if order == "2d":
                out.append(nrmse_map)
            elif order == "1d":
                if geo_weighted:
                    out.append(geo_avg(nrmse_map, self.lat_dim, self.lon_dim)) # already a scalar Dataset (since nrmse_map has no time dim)
                else:
                    out.append(nrmse_map.mean(dim=(self.lat_dim, self.lon_dim), skipna=True))
            else:
                raise ValueError(f"Invalid order: {order}")

        return out

    def nmae_map (
        self,
        order: Literal["2d", "1d"] = "2d",
        geo_weighted: bool = True,
        eps: float = 0.0,
    ) -> list[xr.Dataset]:
        """
        Normalized MAE by truth std over time (map-wise).

        - order="2d": returns 2D map: MAE_map / std_map
        - order="1d": spatial average of that 2D map (geo-weighted if requested)
        """
        # MAE map per dataset (time mean of abs error)
        mae_maps = self.mae(order="2d", geo_weighted=geo_weighted)  # geo_weighted irrelevant for 2d

        # Truth std map over time
        std_map_truth = self._std_map(self.truth, eps=eps)

        out: list[xr.Dataset] = []
        for mae_map in mae_maps:
            nmae_map = mae_map / std_map_truth
            if order == "2d":
                out.append(nmae_map)
            elif order == "1d":
                if geo_weighted:
                    out.append(geo_avg(nmae_map, self.lat_dim, self.lon_dim))
                else:
                    out.append(nmae_map.mean(dim=(self.lat_dim, self.lon_dim), skipna=True))
            else:
                raise ValueError(f"Invalid order: {order}")

        return out

    def nbias_map (
        self,
        order: Literal["2d", "1d"] = "2d",
        geo_weighted: bool = True,
        eps: float = 0.0,
    ) -> list[xr.Dataset]:
        """
        Normalized bias by truth std over time (map-wise).

        - bias map is mean(err) over time: (pred - truth).mean(time)
        - normalized bias: bias_map / std_map
        """
        bias_maps = self.err(order="2d", geo_weighted=geo_weighted)  # time-mean error (bias), 2D

        std_map_truth = self._std_map(self.truth, eps=eps)

        out: list[xr.Dataset] = []
        for bias_map in bias_maps:
            nbias_map = bias_map / std_map_truth

            if order == "2d":
                out.append(nbias_map)
            elif order == "1d":
                if geo_weighted:
                    out.append(geo_avg(nbias_map, self.lat_dim, self.lon_dim))
                else:
                    out.append(nbias_map.mean(dim=(self.lat_dim, self.lon_dim), skipna=True))
            else:
                raise ValueError(f"Invalid order: {order}")

        return out
    
    def nabsbias_map (
        self,
        order: Literal["2d", "1d"] = "2d",
        geo_weighted: bool = True,
        eps: float = 0.0,
    ) -> list[xr.Dataset]:
        """
        Normalized absolute bias magnitude by truth std over time (map-wise).

        - bias_map = mean_time(pred - truth)
        - abs bias magnitude map = abs(bias_map)
        - normalized abs bias = abs(bias_map) / std_map(truth)
        """
        abs_bias_maps = [abs(x) for x in self.err(order="2d", geo_weighted=geo_weighted)]  # time-mean abs error (abs bias), 2D

        std_map_truth = self._std_map(self.truth, eps=eps)

        out: list[xr.Dataset] = []
        for abs_bias_map in abs_bias_maps:
            abs_nbias_map = abs_bias_map / std_map_truth

            if order == "2d":
                out.append(abs_nbias_map)
            elif order == "1d":
                if geo_weighted:
                    out.append(geo_avg(abs_nbias_map, self.lat_dim, self.lon_dim))
                else:
                    out.append(abs_nbias_map.mean(dim=(self.lat_dim, self.lon_dim), skipna=True))
            else:
                raise ValueError(f"Invalid order: {order}")

        return out

    # Normalized metrics, global scalars
    def _std_scalar (self, data, geo_weighted: bool = True, eps: float = 0.0) -> xr.Dataset:
        # Denominator: global std of truth
        if geo_weighted:
            # spatially averaged time series, then std over time
            truth_bar = geo_avg(data, self.lat_dim, self.lon_dim)        # dims: time
            denom = truth_bar.std(dim=self.time_dim, skipna=True)
        else:
            denom = data.std(
                dim=(self.time_dim, self.lat_dim, self.lon_dim),
                skipna=True,
            )

        return xr.where(denom > eps, denom, np.nan)

    def nrmse_global (self, geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Global NRMSE normalized by global truth standard deviation.

        Returns one scalar per dataset.

        Definition:
        NRMSE = RMSE_global / std_global(truth)

        If geo_weighted=True:
        - spatial means are cosine(lat) weighted
        - time is unweighted
        """
        std_truth = self._std_scalar(self.truth, geo_weighted=geo_weighted, eps=eps)
        rmse_list = self.rmse(order="1d", geo_weighted=geo_weighted) # list of scalar Datasets

        return [rmse / std_truth for rmse in rmse_list]

    def nmae_global (self, geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Global NMAE = MAE_global / std_global(truth)
        """
        std_truth = self._std_scalar(self.truth, geo_weighted=geo_weighted, eps=eps)
        mae_list = self.mae(order="1d", geo_weighted=geo_weighted)

        return [mae / std_truth for mae in mae_list]

    def nbias_global (self, geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Global normalized bias = bias_global / std_global(truth)
        where bias_global = mean(pred - truth)
        """
        std_truth = self._std_scalar(self.truth, geo_weighted=geo_weighted, eps=eps)
        bias_list = self.err(order="1d", geo_weighted=geo_weighted)

        return [bias / std_truth for bias in bias_list]

    def nabsbias_global (self, geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Global normalized abs bias = abs_bias_global / std_global(truth)
        where abs_bias_global = abs(mean(pred - truth))
        """
        nbias = self.nbias_global(geo_weighted=geo_weighted, eps=eps)
        return [abs(x) for x in nbias]

    # Difference metrics between prediction datasets
    def _generic_diff_metric (
        self,
        truth: xr.Dataset,
        data: List[xr.Dataset], # must be len 2
        parent_metric_fn: MetricFn,
        final_metric_fn: Optional[FinalFn] = None,
        order: Literal["3d", "2d", "1d"] = "1d",
        geo_weighted: bool = True,
    ) -> list[xr.Dataset]:
        if len(data) == 2:
            data_a, data_b = data[0], data[1] # fc, pr
            metric = parent_metric_fn(data_a - truth, data_b - truth)

            if order == "1d":
                if geo_weighted:
                    val = geo_avg(metric, self.lat_dim, self.lon_dim).mean(dim=self.time_dim, skipna=True)
                else:
                    val = metric.mean(dim=(self.time_dim, self.lat_dim, self.lon_dim), skipna=True)
            elif order == "2d":
                val = metric.mean(dim=self.time_dim, skipna=True)
            elif order == "3d":
                val = metric
            else:
                raise ValueError(f"Invalid order: {order}")

            if final_metric_fn is not None:
                val = final_metric_fn(val)

            return [val]

        else:
            raise ValueError(f"Data len must be 2, not {len(self.data)}, to calcualte diff metrics")

    def diff_err (self, order: Literal["3d", "2d", "1d"] = "1d", geo_weighted: bool = True) -> List[xr.Dataset]:
        # 2d and 1d cases are bias
        return self._generic_diff_metric(self.truth, self.data, self._err_field, order=order, geo_weighted=geo_weighted)

    def nbias_diff_global(self, geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Global *normalized* bias of the difference between two prediction datasets.

        Definition used here (returns one Dataset per variable, wrapped in a list):
            nbias_diff = nbias_a - nbias_b

        where for a model M:
            nbias_M = mean_over_time_and_space( (truth - M) / std_global(truth) )

        Equivalently (algebraic simplification):
            nbias_diff = mean((M_b - M_a) / std_global(truth))

        Notes
        -----
        - Requires exactly two prediction datasets (self.data[0] = model_a, self.data[1] = model_b).
        - If geo_weighted=True, spatial averages use cosine(lat) weighting (same convention as other methods).
        - Cells where the denominator (std_global(truth)) <= eps are set to NaN and ignored by reductions.
        - Returns a list with a single xr.Dataset (to match other methods' return type).
        """
        if len(self.data) != 2:
            raise ValueError(f"Data length must be 2 to calculate diff metrics, got {len(self.data)}")

        # align inputs
        truth, data_a, data_b = xr.align(self.truth, self.data[0], self.data[1], join="inner")

        # denominator: global std of truth (same convention as nbias_global)
        if geo_weighted:
            truth_bar = geo_avg(truth, self.lat_dim, self.lon_dim)                      # time series
            denom = truth_bar.std(dim=self.time_dim, skipna=True)
        else:
            denom = truth.std(dim=(self.time_dim, self.lat_dim, self.lon_dim), skipna=True)

        denom = xr.where(denom > eps, denom, np.nan)

        # normalized difference field
        diff_field = (data_b - data_a) / denom

        # reduce to scalar
        if geo_weighted:
            diff_ts = geo_avg(diff_field, self.lat_dim, self.lon_dim)                     # time series
            nbias_diff = diff_ts.mean(dim=self.time_dim, skipna=True)
        else:
            nbias_diff = diff_field.mean(
                dim=(self.time_dim, self.lat_dim, self.lon_dim),
                skipna=True,
            )

        return [nbias_diff]

    def nbias_diff_map(self, eps: float = 0.0) -> list[xr.Dataset]:
        """
        Normalized bias *map* of the difference between two prediction datasets.

        Definition:
            nbias_diff_map = mean_time( data_b - data_a ) / std_map(truth)

        Returns:
            [xr.Dataset] : single-element list containing the 2D normalized difference map
                           (coords: lat, lon; data_vars: same as truth/pred).
        Notes:
        - Requires exactly two prediction datasets (self.data[0] = model_a, self.data[1] = model_b).
        - eps: cells with truth std <= eps are masked to NaN.
        """
        if len(self.data) != 2:
            raise ValueError(f"Data length must be 2 to calculate diff metrics, got {len(self.data)}")

        truth, data_a, data_b = xr.align(self.truth, self.data[0], self.data[1], join="inner")

        # std map over time
        std_map = truth.std(dim=self.time_dim, skipna=True)
        denom = xr.where(std_map > eps, std_map, np.nan)

        # time-mean difference (b - a)
        diff_map = (data_b - data_a).mean(dim=self.time_dim, skipna=True)

        nbias_diff_map = diff_map / denom

        return [nbias_diff_map]

    # R² metrics
    def r2_map (self) -> list[xr.Dataset]:
        # SST: sum over time of squared anomalies (2D field per var)
        anom = self.truth - self.truth.mean(dim=self.time_dim, skipna=True)
        sst = (anom ** 2).sum(dim=self.time_dim, skipna=True)

        out: list[xr.Dataset] = []
        for d in self.data:
            sse = ((d - self.truth) ** 2).sum(dim=self.time_dim, skipna=True)
            r2 = 1.0 - sse / xr.where(sst > 0, sst, np.nan)
            out.append(r2)

        return out

    def r2_global (self, geo_weighted: bool = True) -> list[xr.Dataset]:
        # anomalies over time
        anom = self.truth - self.truth.mean(dim=self.time_dim, skipna=True)
        sst_t = (anom ** 2).sum(dim=self.time_dim, skipna=True)  # 2D field

        out: list[xr.Dataset] = []
        for d in self.data:
            sse_t = ((d - self.truth) ** 2).sum(dim=self.time_dim, skipna=True)  # 2D field

            if geo_weighted:
                sse = geo_avg(sse_t, self.lat_dim, self.lon_dim)   # scalar or 1D depending on _geo_avg
                sst = geo_avg(sst_t, self.lat_dim, self.lon_dim)
            else:
                sse = sse_t.mean(dim=(self.lat_dim, self.lon_dim), skipna=True)
                sst = sst_t.mean(dim=(self.lat_dim, self.lon_dim), skipna=True)

            r2 = 1 - sse / xr.where(sst > 0, sst, np.nan)
            out.append(r2)

        return out

    # Correlation metrics
    @staticmethod
    def _ds_corr (
        a: xr.Dataset,
        b: xr.Dataset, dim: str,
        min_periods: int = 2,
        weights: xr.DataArray | None = None,
    ) -> xr.Dataset:
        """
        Pearson correlation per variable between two Datasets along `dim`.
        Returns a Dataset with same data_vars (intersection).
        """
        common = [v for v in a.data_vars if v in b.data_vars]
        out = {}

        for v in common:
            x = a[v]
            y = b[v]

            r = xr.corr(x, y, dim=dim, weights=weights)

            # enforce min_periods (pairwise finite samples)
            n = (np.isfinite(x) & np.isfinite(y)).sum(dim=dim)
            r = r.where(n >= min_periods)

            out[v] = r

        return xr.Dataset(out, attrs=a.attrs)

    def corr_map (self, min_periods: int = 2) -> list[xr.Dataset]:
        """
        Pearson correlation over time at each (lat, lon) cell, per variable.
        """
        out: list[xr.Dataset] = []
        for d in self.data:
            out.append(self._ds_corr(self.truth, d, dim=self.time_dim, min_periods=min_periods))
        return out

    def corr_global (self, geo_weighted: bool = True, min_periods: int = 2) -> list[xr.Dataset]:
        """
        Correlation of global-mean time series (truth vs pred), per variable.
        """
        if geo_weighted:
            t = geo_avg(self.truth, self.lat_dim, self.lon_dim)  # Dataset with dim: time
        else:
            t = self.truth.mean(dim=(self.lat_dim, self.lon_dim), skipna=True)

        out: list[xr.Dataset] = []
        for d in self.data:
            if geo_weighted:
                p = geo_avg(d, self.lat_dim, self.lon_dim)
            else:
                p = d.mean(dim=(self.lat_dim, self.lon_dim), skipna=True)

            out.append(self._ds_corr(t, p, dim=self.time_dim, min_periods=min_periods))

        return out

    def corr_flat (self, geo_weighted: bool = False, min_periods: int = 2) -> list[xr.Dataset]:
        """
        Correlation across all space-time points, per variable.
        If geo_weighted=True, uses cos(lat) weights (area weighting for regular lat/lon grids).
        """
        out: list[xr.Dataset] = []
        dims = (self.time_dim, self.lat_dim, self.lon_dim)

        # xr.corr requires 1D vectors
        truth_stacked = self.truth.stack(_allpoints=dims)

        w_stacked = None
        if geo_weighted:
            # 1D lat weights from the same logic as _geo_avg
            w_lat = geo_weights(self.truth, self.lat_dim)  # dims: (lat,)

            # Broadcast to match a representative variable (time, lat, lon), then stack
            template = next(iter(self.truth.data_vars.values()))
            w_3d = w_lat.broadcast_like(template)               # (time, lat, lon)
            w_stacked = w_3d.stack(_allpoints=dims)             # (_allpoints,)

        for d in self.data:
            pred_stacked = d.stack(_allpoints=dims)
            out.append(
                self._ds_corr(
                    truth_stacked,
                    pred_stacked,
                    dim="_allpoints",
                    min_periods=min_periods,
                    weights=w_stacked,  # for xr.corr
                )
            )

        return out

    # Ensemble metrics
    def rmse_ens (self, order: Literal["2d", "1d"] = "1d", geo_weighted: bool = True) -> list[xr.Dataset]:
        """
        RMSE of the ensemble mean prediction.

        Assumes self.data contains ensemble members for a single model.
        """
        # Check realization dimension exists
        if self.realization_dim is None:
            return []

        # Ensemble means
        truth_ens_mean = self.truth.mean(dim=self.realization_dim, skipna=True)
        data_ens_mean = [d.mean(dim=self.realization_dim, skipna=True) for d in self.data]

        # RMSE of ensemble mean
        return self._generic_metric(
            truth_ens_mean, data_ens_mean,
            lambda truth, pred: (pred - truth) ** 2,
            final_metric_fn=np.sqrt,
            order=order,
            geo_weighted=geo_weighted,
        )

    def mae_ens (self, order: Literal["2d", "1d"] = "1d", geo_weighted: bool = True) -> list[xr.Dataset]:
        """
        MAE of the ensemble mean prediction.

        Assumes self.data contains ensemble members for a single model.
        """
        # Check realization dimension exists
        if self.realization_dim is None:
            return []

        # Ensemble means
        truth_ens_mean = self.truth.mean(dim=self.realization_dim, skipna=True)
        data_ens_mean = [d.mean(dim=self.realization_dim, skipna=True) for d in self.data]

        # MAE of ensemble mean
        return self._generic_metric(
            truth_ens_mean, data_ens_mean,
            lambda truth, pred: abs(pred - truth),
            order=order,
            geo_weighted=geo_weighted,
        )

    def bias_ens (self, order: Literal["2d", "1d"] = "1d", geo_weighted: bool = True) -> list[xr.Dataset]:
        """
        Bias of the ensemble mean prediction.

        Assumes self.data contains ensemble members for a single model.
        """
        # Check realization dimension exists
        if self.realization_dim is None:
            return []

        # Ensemble means
        truth_ens_mean = self.truth.mean(dim=self.realization_dim, skipna=True)
        data_ens_mean = [d.mean(dim=self.realization_dim, skipna=True) for d in self.data]

        # Bias of ensemble mean
        return self._generic_metric(
            truth_ens_mean, data_ens_mean,
            self._err_field,
            order=order,
            geo_weighted=geo_weighted,
        )

    # Ensemble normalized metrics
    def nrmse_ens (self, geo_weighted: bool = True, eps: float = 0.0) -> list[xr.Dataset]:
        """
        NRMSE of the ensemble mean prediction.

        Assumes self.data contains ensemble members for a single model.
        """
        # Check realization dimension exists
        if self.realization_dim is None:
            return []

        # Ensemble means
        truth_ens_mean = self.truth.mean(dim=self.realization_dim, skipna=True)
        data_ens_mean = [d.mean(dim=self.realization_dim, skipna=True) for d in self.data]

        # NRMSE of ensemble mean
        return self._generic_metric(
            truth_ens_mean, data_ens_mean,
            lambda truth, pred: (pred - truth) ** 2,
            final_metric_fn=np.sqrt,
            order="1d",
            geo_weighted=geo_weighted,
        )

    def compute_all_metrics(self, *, geo_weighted: bool = True, eps: float = 0.0) -> dict[str, Any]:
        """
        Compute a standard suite of metrics and return:
          res["models"][model]["scalar"][metric] -> xr.Dataset (0D per var)
          res["models"][model]["map"][metric]    -> xr.Dataset (2D per var: lat/lon)

        Notes
        -----
        - Scalar metrics use geo-weighting if geo_weighted=True (where applicable).
        - 3D fields are intentionally not returned.
        """
        def by_model(lst: list[xr.Dataset]) -> dict[str, xr.Dataset]:
            return {name: lst[i] for i, name in enumerate(self.data_name)}

        # metric name -> callables producing list[xr.Dataset] aligned with self.data
        scalar_fns = {
            # Basic metrics, become member metrics if realization dim present
            "mae":               lambda: self.mae(order="1d", geo_weighted=geo_weighted),
            "bias":              lambda: self.err(order="1d", geo_weighted=geo_weighted),
            "abs_bias":          lambda: self.abs_bias(order="1d", geo_weighted=geo_weighted),
            "rmse":              lambda: self.rmse(order="1d", geo_weighted=geo_weighted),
            "stderr":            lambda: self.stderr(order="1d", geo_weighted=geo_weighted),
            "std_of_errs":       lambda: self.std_of_errs(order="1d", geo_weighted=geo_weighted),
            "mape":              lambda: self.mape(order="1d", geo_weighted=geo_weighted, eps=eps),
            "smape":             lambda: self.smape(order="1d", geo_weighted=geo_weighted),
            # Normalized metrics
            "nrmse_global":      lambda: self.nrmse_global(geo_weighted=geo_weighted, eps=eps),
            "nrmse_map_mean":    lambda: self.nrmse_map(order="1d", geo_weighted=geo_weighted, eps=eps),
            "nmae_global":       lambda: self.nmae_global(geo_weighted=geo_weighted, eps=eps),
            "nbias_global":      lambda: self.nbias_global(geo_weighted=geo_weighted, eps=eps),
            "abs_nbias_global":  lambda: self.nabsbias_global(geo_weighted=geo_weighted, eps=eps),
            "nmae_map_mean":     lambda: self.nmae_map(order="1d", geo_weighted=geo_weighted, eps=eps),
            "nbias_map_mean":    lambda: self.nbias_map(order="1d", geo_weighted=geo_weighted, eps=eps),
            # Skill metrics
            "r2_global":         lambda: self.r2_global(geo_weighted=geo_weighted),
            "corr_global":       lambda: self.corr_global(geo_weighted=geo_weighted, min_periods=2),
            "corr_flat":         lambda: self.corr_flat(geo_weighted=geo_weighted, min_periods=2),
            # Ensemble metrics
            "rmse_ens":          lambda: self.rmse_ens(order="1d", geo_weighted=geo_weighted),
            "mae_ens":           lambda: self.mae_ens(order="1d", geo_weighted=geo_weighted),
            "bias_ens":          lambda: self.bias_ens(order="1d", geo_weighted=geo_weighted),
        }

        map_fns = {
            "mae":               lambda: self.mae(order="2d", geo_weighted=geo_weighted),
            "bias":              lambda: self.err(order="2d", geo_weighted=geo_weighted),
            "abs_bias":          lambda: self.abs_bias(order="2d", geo_weighted=geo_weighted),
            "rmse":              lambda: self.rmse(order="2d", geo_weighted=geo_weighted),
            "stderr":            lambda: self.stderr(order="2d", geo_weighted=geo_weighted),
            "std_of_errs":       lambda: self.std_of_errs(order="2d", geo_weighted=geo_weighted),
            "mape":              lambda: self.mape(order="2d", geo_weighted=geo_weighted, eps=eps),
            "smape":             lambda: self.smape(order="2d", geo_weighted=geo_weighted),
            "nrmse_map":         lambda: self.nrmse_map(order="2d", geo_weighted=geo_weighted, eps=eps),
            "nmae_map":          lambda: self.nmae_map(order="2d", geo_weighted=geo_weighted, eps=eps),
            "nbias_map":         lambda: self.nbias_map(order="2d", geo_weighted=geo_weighted, eps=eps),
            "abs_nbias_map":     lambda: self.nabsbias_map(order="2d", geo_weighted=geo_weighted, eps=eps),
            "r2_map":            lambda: self.r2_map(),
            "corr_map":          lambda: self.corr_map(min_periods=2),
        }

        # init models
        models: dict[str, dict[str, dict[str, xr.Dataset]]] = {
            name: {"scalar": {}, "map": {}} for name in self.data_name
        }

        # fill scalars
        for metric, fn in scalar_fns.items():
            for model, ds in by_model(fn()).items():
                models[model]["scalar"][metric] = ds

        # fill maps
        for metric, fn in map_fns.items():
            for model, ds in by_model(fn()).items():
                models[model]["map"][metric] = ds

        # fill diff
        if len(self.data) == 2:
            diff_name = "diff"
            models[diff_name] = {"scalar": {}, "map": {}}
            models[diff_name]["scalar"]["bias_diff"] = self.diff_err(order="1d", geo_weighted=geo_weighted)[0]
            models[diff_name]["scalar"]["nbias_diff_global"] = self.nbias_diff_global(geo_weighted=geo_weighted, eps=eps)[0]
            models[diff_name]["map"]["nbias_diff_map"] = self.nbias_diff_map(eps=eps)[0]

        return {
            "meta": {
                "truth_name": self.truth_name,
                "data_names": list(self.data_name),
                "dims": {"time": self.time_dim, "lat": self.lat_dim, "lon": self.lon_dim},
                "geo_weighted": geo_weighted,
                "eps": eps,
            },
            "models": models,
        }

    # @staticmethod
    # def fss (a, b, dims: Tuple, threshold, eps):
    #     """Fraction Skill Score"""
    #     A, B = (a > threshold).astype(float), (b > threshold).astype(float)
    #     return 1 - ((A - B)**2).mean(dim=time_dim) / ((A**2 + B**2).mean(dim=time_dim) + eps)

    # @staticmethod
    # def sal (a, b, dims: Tuple, eps): #TODO better understand this metric
    #     """Std ratio, Avg (mean) ratio, weighted mean difference (L)"""
    #     return {
    #         "S": (b.std(dim=time_dim) - a.std(dim=time_dim)) / (a.std(dim=time_dim) + eps),
    #         "A": (b.mean(dim=time_dim) - a.mean(dim=time_dim)) / (a.mean(dim=time_dim) + eps),
    #         "L": (b.weighted(abs(b)).mean(dim=time_dim) - a.weighted(abs(a)).mean(dim=time_dim)) # TODO probably wrong, check weights
    #     }
    @staticmethod
    def spectral_metrics (a, b, time_dim, spatial_dims, eps):
        import numpy as np

        # Helper: FFT power with consistent axis order
        def fft_power(da, time_dim, spatial_dims):
            arr = da.transpose(time_dim, *spatial_dims).astype("float64").values
            fft = np.fft.rfftn(arr, axes=(-2, -1)) # 2D spatial FFT
            return fft

        # Get FFTs
        A = fft_power(a, time_dim, spatial_dims)
        B = fft_power(b, time_dim, spatial_dims)

        # Power spectra (averaged over time axis = 0)
        Pa = np.mean(np.abs(A) ** 2, axis=0)
        Pb = np.mean(np.abs(B) ** 2, axis=0)

        # Power ratio
        ratio = Pb / (Pa + eps)

        # Cross-spectrum
        cross = np.mean(A * np.conj(B), axis=0)

        # Coherence: |E[AB*]| / sqrt(E[|A|^2] E[|B|^2])
        # IMPORTANT: avoid overflow by not doing Pa*Pb directly
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            denom = np.sqrt(Pa) * np.sqrt(Pb) + eps
            coh = np.abs(cross) / denom
            # Optional: coherence should be in [0, 1]
            coh = np.clip(coh, 0, 1)

        return {
            "power_a": Pa,
            "power_b": Pb,
            "ratio": ratio,
            "coherence": coh
        }

class PowerSpectrum:
    def __init__(
        self,
        data: xr.Dataset,
        dx_deg: float | None = None,
        dy_deg: float | None = None,
        n_bins: int = 40,
    ):
        """
        Compute time-mean radial power spectra (as function of wavelength in km)
        for all variables in an xarray Dataset on a lat-lon grid.

        Parameters
        ----------
        data : xr.Dataset
            Input dataset with at least lat/lon coordinates.
        dx_deg, dy_deg : float or None
            Horizontal resolution in degrees. If None, inferred from coordinates.
        n_bins : int
            Number of radial bins in wavelength space.
        """
        self.data = data
        self.lat_name = self.data.cf['latitude'].name
        self.lon_name = self.data.cf['longitude'].name
        self.time_name = self.data.cf['time'].name
        self.n_bins = n_bins

        # --- Infer grid information (lat array, dx_deg, dy_deg) ---
        lat_1d, lon_1d, dx_deg, dy_deg = self._infer_grid_info(
            self.lat_name, self.lon_name, dx_deg, dy_deg
        )
        self.lat_1d = lat_1d
        self.lon_1d = lon_1d
        self.dx_deg = dx_deg
        self.dy_deg = dy_deg

        ps_vars = {}

        for v in self.data.data_vars:
            da = self.data[v]

            # Ensure it has the spatial dims
            if not (self.lat_name in da.dims and self.lon_name in da.dims):
                # Skip non-spatial variables
                continue

            # --- Time-average of the power spectrum ---
            if self.time_name in da.dims:
                time_dim = self.time_name
                nt = da.sizes[time_dim]

                lam_centers = None
                spectra = []

                for it in range(nt):
                    slice_da = da.isel({time_dim: it})
                    field2d = self._prepare_field2d(slice_da)

                    lam, ps = self.compute_radial_spectrum(
                        field2d,
                        self.lat_1d,
                        self.dx_deg,
                        self.dy_deg,
                        n_bins=self.n_bins,
                    )

                    if lam_centers is None:
                        lam_centers = lam
                    else:
                        if not np.allclose(lam_centers, lam):
                            raise ValueError(f"Inconsistent wavelength grid for variable {v}")

                    spectra.append(ps)

                spectra = np.stack(spectra, axis=0)  # (time, wavelength)
                ps_mean = np.nanmean(spectra, axis=0)  # time-mean spectrum
            else:
                # No time dimension: just one spectrum
                field2d = self._prepare_field2d(da)
                lam_centers, ps_mean = self.compute_radial_spectrum(
                #     field2d,
                #     self.lat_1d,
                #     self.dx_deg,
                #     self.dy_deg,
                #     n_bins=self.n_bins,
                # )
                    field2d,
                    lat_1d,
                    dx_deg,
                    dy_deg,
                    n_bins=40,
                    nperseg_y=1024,
                    nperseg_x=1024,
                    noverlap_y=64,
                    noverlap_x=64,
                    window="hann",
                )


            # Store as DataArray with coordinate "wavelength_km"
            ps_vars[v] = xr.DataArray(
                ps_mean,
                coords={"wavelength_km": lam_centers},
                dims=("wavelength_km",),
                name=v,
            )

        # Dataset of power spectra, with common wavelength axis
        self.ps = xr.Dataset(ps_vars)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_grid_info(
        self,
        lat_name: str,
        lon_name: str,
        dx_deg: float | None,
        dy_deg: float | None,
    ):
        """Infer 1D lat/lon arrays and degree spacing from the dataset."""
        if lat_name not in self.data.coords or lon_name not in self.data.coords:
            raise ValueError(
                f"Dataset must have '{lat_name}' and '{lon_name}' coordinates."
            )

        lat = self.data[lat_name]
        lon = self.data[lon_name]

        if lat.ndim != 1 or lon.ndim != 1:
            raise NotImplementedError(
                "Currently only 1D lat/lon coordinates are supported."
            )

        lat_1d = lat.values
        lon_1d = lon.values

        # Infer dy_deg and dx_deg if not provided
        if dy_deg is None:
            dy_deg = float(np.nanmean(np.diff(lat_1d)))
        if dx_deg is None:
            # If lon wraps (e.g. 0..360), we still assume uniform spacing
            diffs = np.diff(lon_1d)
            dx_deg = float(np.nanmean(diffs))

        return lat_1d, lon_1d, dx_deg, dy_deg

    def _prepare_field2d(self, da: xr.DataArray) -> np.ndarray:
        """
        Take a DataArray and reduce it to a 2D (lat, lon) field by:
        - averaging over any non-spatial dims
        - transposing to (lat, lon)
        """
        dims_to_avg = [d for d in da.dims if d not in (self.lat_name, self.lon_name)]
        if len(dims_to_avg) > 0:
            da2 = da.mean(dim=dims_to_avg)
        else:
            da2 = da

        # Reorder so that lat is first, lon is second
        da2 = da2.transpose(self.lat_name, self.lon_name)

        field2d = da2.values
        if field2d.ndim != 2:
            raise ValueError(f"Expected 2D field after reduction, got shape {field2d.shape}")
        if field2d.shape[0] != self.lat_1d.size:
            raise ValueError("Latitude size mismatch between field and lat_1d.")
        return field2d

    # ------------------------------------------------------------------
    # Core spectrum computation
    # ------------------------------------------------------------------
    # @staticmethod
    # def compute_radial_spectrum(
    #     field2d: np.ndarray,
    #     lat_1d: np.ndarray,
    #     dx_deg: float,
    #     dy_deg: float,
    #     n_bins: int = 40,
    # ):
    #     """
    #     Compute 1D radial average power spectrum from a 2D field, and express it
    #     as a function of wavelength (km).

    #     - field2d is on a regular lat-lon grid (in degrees).
    #     - FFT is done in index/degree space; then we convert to physical km
    #       using spherical geometry:
    #         1° lat  ≈ 111.32 km
    #         1° lon  ≈ 111.32 * cos(lat) km
    #     - This handles varying dx (km) with latitude.

    #     Parameters
    #     ----------
    #     field2d : 2D ndarray
    #         Field on (lat, lon) grid, shape (ny, nx).
    #     lat_1d : 1D ndarray
    #         Latitude values (degrees), length ny.
    #     dx_deg, dy_deg : float
    #         Grid resolution in degrees (assumed uniform).
    #     n_bins : int
    #         Number of radial bins in wavelength space.

    #     Returns
    #     -------
    #     lam_centers : 1D ndarray
    #         Bin-center wavelengths in km (log-spaced).
    #     radial_power : 1D ndarray
    #         Radially-averaged power for each wavelength bin.
    #     """
    #     field2d = np.asarray(field2d)
    #     if field2d.ndim != 2:
    #         raise ValueError(f"compute_radial_spectrum expects 2D data, got {field2d.shape}")

    #     ny, nx = field2d.shape
    #     if lat_1d.shape[0] != ny:
    #         raise ValueError("lat_1d length must match field2d.shape[0]")

    #     # 2D real FFT
    #     fft = np.fft.rfftn(field2d)
    #     power = np.abs(fft) ** 2
    #     ny_p, nx_r = power.shape
    #     if ny_p != ny:
    #         raise RuntimeError("Internal size mismatch after FFT.")

    #     # Frequencies in cycles per degree
    #     ky_deg = np.fft.fftfreq(ny, d=dy_deg)       # shape (ny,)
    #     kx_deg = np.fft.rfftfreq(nx, d=dx_deg)      # shape (nx_r,)

    #     # Convert to wavenumber in cycles per km (km^-1), accounting for latitude
    #     km_per_deg_lat = 111.32
    #     # latitude-dependent km per deg in longitude
    #     lat_rad = np.deg2rad(lat_1d)
    #     km_per_deg_lon = km_per_deg_lat * np.cos(lat_rad)  # shape (ny,)

    #     # ky: same for every latitude row
    #     ky_km = ky_deg / km_per_deg_lat                 # shape (ny,)
    #     ky_km_2d = ky_km[:, np.newaxis]                 # (ny, 1)

    #     # kx: depends on latitude row via km_per_deg_lon
    #     # kx_deg is (nx_r,), km_per_deg_lon is (ny,)
    #     kx_km_2d = kx_deg[np.newaxis, :] / km_per_deg_lon[:, np.newaxis]

    #     # 2D radial wavenumber magnitude (km^-1)
    #     k_mag = np.sqrt(kx_km_2d**2 + ky_km_2d**2)

    #     # Avoid division by zero at k=0
    #     mask_nonzero = k_mag > 0
    #     if not mask_nonzero.any():
    #         raise RuntimeError("All wavenumbers are zero; something is wrong with the grid.")

    #     # Wavelength in km
    #     wavelength = np.zeros_like(k_mag)
    #     wavelength[mask_nonzero] = 1.0 / k_mag[mask_nonzero]

    #     # Define radial bins in wavelength space (log-spaced)
    #     lam_min = np.nanmin(wavelength[mask_nonzero])
    #     lam_max = np.nanmax(wavelength[mask_nonzero])

    #     lam_bins = np.logspace(np.log10(lam_min), np.log10(lam_max), n_bins)
    #     lam_centers = 0.5 * (lam_bins[:-1] + lam_bins[1:])

    #     radial_power = np.zeros_like(lam_centers)

    #     for i in range(len(lam_centers)):
    #         binmask = (wavelength >= lam_bins[i]) & (wavelength < lam_bins[i + 1])
    #         if binmask.any():
    #             radial_power[i] = power[binmask].mean()
    #         else:
    #             radial_power[i] = np.nan

    #     return lam_centers, radial_power

    @staticmethod
    def compute_radial_spectrum(
        field2d: np.ndarray,
        lat_1d: np.ndarray,
        dx_deg: float,
        dy_deg: float,
        n_bins: int = 40,
        nperseg_y: int | None = None,
        nperseg_x: int | None = None,
        noverlap_y: int | None = None,
        noverlap_x: int | None = None,
        window: str = "hann",
    ):
        """
        Welch-style 2D radial power spectrum expressed as a function of wavelength (km).

        - field2d is on a regular lat–lon grid (in degrees).
        - We split the field into overlapping 2D segments, window each segment,
        take a 2D rFFT, square (power), and average over segments (Welch method).
        - Then we convert to physical wavenumber using spherical geometry:
            1° lat  ≈ 111.32 km
            1° lon  ≈ 111.32 * cos(lat) km
        and finally to wavelength λ = 1 / |k| (km).
        - dx varies with latitude through cos(lat); dy is constant.

        Parameters
        ----------
        field2d : 2D ndarray
            Field on (lat, lon) grid, shape (ny, nx).
        lat_1d : 1D ndarray
            Latitude values (degrees), length ny.
        dx_deg, dy_deg : float
            Grid resolution in degrees (assumed uniform in lat/lon index).
        n_bins : int
            Number of radial bins in wavelength space (edges); centers are n_bins-1.
        nperseg_y, nperseg_x : int or None
            Segment size in y and x. If None, chosen automatically.
        noverlap_y, noverlap_x : int or None
            Overlap between segments. If None, 50% overlap.
        window : str
            Window name passed to scipy.signal.get_window (e.g. 'hann').

        Returns
        -------
        lam_centers : 1D ndarray
            Bin-center wavelengths in km (log-spaced), length n_bins - 1.
        radial_power : 1D ndarray
            Radially-averaged power for each wavelength bin.
        """
        field2d = np.asarray(field2d)
        if field2d.ndim != 2:
            raise ValueError(f"compute_radial_spectrum expects 2D data, got {field2d.shape}")

        ny, nx = field2d.shape
        if lat_1d.shape[0] != ny:
            raise ValueError("lat_1d length must match field2d.shape[0].")

        # --- Choose segment sizes and overlaps (Welch parameters) ---
        if nperseg_y is None:
            nperseg_y = min(256, ny)  # or tune as you like
        if nperseg_x is None:
            nperseg_x = min(256, nx)

        nperseg_y = min(nperseg_y, ny)
        nperseg_x = min(nperseg_x, nx)

        if noverlap_y is None:
            noverlap_y = nperseg_y // 2
        if noverlap_x is None:
            noverlap_x = nperseg_x // 2

        step_y = max(nperseg_y - noverlap_y, 1)
        step_x = max(nperseg_x - noverlap_x, 1)

        y_starts = np.arange(0, ny - nperseg_y + 1, step_y)
        x_starts = np.arange(0, nx - nperseg_x + 1, step_x)

        if len(y_starts) == 0 or len(x_starts) == 0:
            # Fall back to a single segment
            y_starts = np.array([0])
            x_starts = np.array([0])
            nperseg_y = ny
            nperseg_x = nx

        # --- 1D frequency grids for a segment (in cycles per degree) ---
        ky_deg = np.fft.fftfreq(nperseg_y, d=dy_deg)     # (nperseg_y,)
        kx_deg = np.fft.rfftfreq(nperseg_x, d=dx_deg)    # (nperseg_x_r,)

        nseg_x_r = kx_deg.size

        # --- Window (Welch taper) ---
        wy = get_window(window, nperseg_y, fftbins=True)
        wx = get_window(window, nperseg_x, fftbins=True)
        window_2d = wy[:, None] * wx[None, :]  # outer product

        # We'll accumulate power in radial bins
        lam_bins = None
        lam_centers = None
        radial_sum = None
        radial_count = None

        km_per_deg_lat = 111.32  # approximate

        # --- Loop over segments (Welch) ---
        for y0 in y_starts:
            y1 = y0 + nperseg_y
            lat_seg = lat_1d[y0:y1]  # (nperseg_y,)
            lat_rad = np.deg2rad(lat_seg)
            km_per_deg_lon_seg = km_per_deg_lat * np.cos(lat_rad)  # (nperseg_y,)

            # Frequencies in cycles per km for this segment
            ky_km = ky_deg / km_per_deg_lat          # (nperseg_y,)
            ky_km_2d = ky_km[:, None]                # (nperseg_y, 1)

            # kx depends on latitude for each row via km_per_deg_lon_seg
            kx_km_2d = kx_deg[None, :] / km_per_deg_lon_seg[:, None]  # (nperseg_y, nperseg_x_r)

            # 2D radial wavenumber magnitude (km^-1)
            k_mag = np.sqrt(kx_km_2d**2 + ky_km_2d**2)
            mask_nonzero = k_mag > 0

            # Wavelength (km)
            wavelength = np.zeros_like(k_mag)
            wavelength[mask_nonzero] = 1.0 / k_mag[mask_nonzero]

            # Define wavelength bins once (from first segment)
            if lam_bins is None:
                lam_min = np.nanmin(wavelength[mask_nonzero])
                lam_max = np.nanmax(wavelength[mask_nonzero])

                lam_bins = np.logspace(np.log10(lam_min), np.log10(lam_max), n_bins)
                lam_centers = 0.5 * (lam_bins[:-1] + lam_bins[1:])
                radial_sum = np.zeros_like(lam_centers)
                radial_count = np.zeros_like(lam_centers, dtype=np.int64)

            for x0 in x_starts:
                x1 = x0 + nperseg_x

                # Extract segment and apply window
                seg = field2d[y0:y1, x0:x1]
                seg = seg - np.nanmean(seg)  # detrend (remove mean)
                seg_win = seg * window_2d

                # 2D rFFT on segment
                fft_seg = np.fft.rfftn(seg_win)
                power_seg = np.abs(fft_seg) ** 2  # (nperseg_y, nperseg_x_r)

                # Radial binning in wavelength space
                for i in range(lam_centers.size):
                    binmask = (wavelength >= lam_bins[i]) & (wavelength < lam_bins[i + 1])
                    if binmask.any():
                        radial_sum[i] += power_seg[binmask].sum()
                        radial_count[i] += int(binmask.sum())

        # --- Final average over all pixels in all segments ---
        radial_power = np.full_like(radial_sum, np.nan, dtype=float)
        valid = radial_count > 0
        radial_power[valid] = radial_sum[valid] / radial_count[valid]

        return lam_centers, radial_power


    # ------------------------------------------------------------------
    # Optional plotting helper
    # ------------------------------------------------------------------
    def plot_all(
        self,
        normalize: bool = True,
        logx: bool = True,
        logy: bool = True,
        savepath: str | None = None,
        show: bool = True,
    ):
        """
        Quick comparison plot of spectra for all variables in self.ps.

        Parameters
        ----------
        normalize : bool
            If True, divide each spectrum by its max (compare shapes).
        logx, logy : bool
            Use log scale for x and/or y.
        savepath : str or None
            If given, save figure to this path.
        show : bool
            If True, call plt.show().
        """
        import matplotlib.pyplot as plt
        import os

        plt.figure(figsize=(6, 4))

        for v in self.ps.data_vars:
            da = self.ps[v]
            lam = da["wavelength_km"].values
            p = da.values.astype(float)

            mask = np.isfinite(lam) & np.isfinite(p)
            if not mask.any():
                continue

            lam = lam[mask]
            p = p[mask]

            if normalize and p.max() > 0:
                p = p / p.max()

            if logx and logy:
                plt.loglog(lam, p, label=str(v))
            elif logx:
                plt.semilogx(lam, p, label=str(v))
            elif logy:
                plt.semilogy(lam, p, label=str(v))
            else:
                plt.plot(lam, p, label=str(v))

        plt.xlabel("Wavelength (km)")
        plt.ylabel("Power" + (" (normalized)" if normalize else ""))
        plt.legend()
        plt.tight_layout()

        if savepath is not None:
            os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
            plt.savefig(savepath, dpi=150)

        if show:
            plt.show()
        else:
            plt.close()
