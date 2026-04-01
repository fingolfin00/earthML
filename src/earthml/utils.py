from pathlib import Path
import joblib, warnings
from typing import Sequence, Optional, Literal, Callable, List, Any
from dataclasses import is_dataclass, fields as dc_fields
from rich import print
from rich.pretty import pprint
from rich.table import Table as RichTable
from rich.highlighter import ReprHighlighter
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import cf_xarray
import xarray as xr
import pandas as pd
from earthkit.data.sources.empty import EmptySource
import os, psutil, multiprocessing, tempfile, logging, re, time, shutil

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import lightning as L
# import xesmf as xe
from scipy.interpolate import griddata
# Local imports
from .base.dataclasses import DataSelection
from .sources.registry import build_source


#------------------------
# Module-level functions
#------------------------

# TODO unsed, remove?
def remove_unwanted_dims_and_coords(ds, allowed_dims):
    ds_out = ds.copy()

    # Remove unwanted dimensions safely (only if size == 1)
    dims_to_remove = [d for d in ds_out.dims if d not in allowed_dims]

    for dim in dims_to_remove:
        if ds_out.sizes[dim] == 1:
            ds_out = ds_out.squeeze(dim=dim, drop=True)
        else:
            warnings.warn(
                f"Cannot remove dimension '{dim}' because its size is "
                f"{ds_out.sizes[dim]} (>1). Dimension left unchanged."
            )

    # Drop coordinates that depend on unwanted dimensions
    coords_to_drop = [
        cname for cname, coord in ds_out.coords.items()
        if not set(coord.dims).issubset(allowed_dims)
    ]

    if coords_to_drop:
        ds_out = ds_out.drop_vars(coords_to_drop)

    return ds_out






def inpaint_nan_bilinear(ds: xr.Dataset) -> xr.Dataset:
    # Bilinear inpainting on the rectilinear grid using index positions
    return (
        ds
        .interpolate_na(dim=ds.earthml.guessed_dims.latitude, method="linear", use_coordinate=False)
        .interpolate_na(dim=ds.earthml.guessed_dims.longitude, method="linear", use_coordinate=False)
    )

def print_ds_info(
    ds: xr.Dataset,
    quantiles: Optional[Sequence[float]] = None,
    regimes: Optional[Sequence[float]] = None,
) -> None:
    for name, da in ds.data_vars.items():
        # Basic stats
        mean = float(da.mean().values)
        std = float(da.std().values)
        vmin = float(da.min().values)
        vmax = float(da.max().values)
        print(f"var {name}")
        print(f"  shape:    {da.shape}")
        print(f"  dtype:    {da.dtype}")
        print(f"  mean:     {mean:.4g}")
        print(f"  std:      {std:.4g}")
        print(f"  min/max:  {vmin:.4g} / {vmax:.4g}")
        print(f"  attrs:    {da.attrs}")
        print(f"  encoding: {da.encoding}")
        # Quantiles
        if quantiles is not None:
            default_qs = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
            qs = da.quantile(default_qs if len(quantiles) == 0 else quantiles)
            print("  quantiles:")
            for q, v in zip(qs["quantile"].values, qs.values):
                print(f"    q={float(q):.3f}: {float(v):.4g}")
        # Separate regimes # TODO test
        if regimes is not None and len(regimes) > 1:
            edges = np.sort(np.asarray(regimes))
            loc_regimes = []
            # lower tail
            loc_regimes.append(da.where(da <= edges[0]))
            # interior bins
            for low, high in zip(edges[:-1], edges[1:]):
                loc_regimes.append(da.where((da > low) & (da <= high)))
            # upper tail
            loc_regimes.append(da.where(da > edges[-1]))
            for i, r in enumerate(loc_regimes):
                r_mean = float(r.mean().values)
                r_std = float(r.std().values)
                print(f"  regime {i} mean/std: {r_mean:.4g} / {r_std:.4g}")



def make_exp_folder_path(
        exp_root: Path,
        exp_name: str,
        v_d: dict, # variable dict
        lt: str,
        lt_unit: str,
        train_period: str,
        test_period: str,
    ):
    exp_full_name = (
        f"exp_{exp_name}_{v_d['exp_var']['fc']}-{v_d['exp_var']['an']}"
        f"_{lt}{lt_unit}_{v_d['region']}"
        f"_{train_period}"
        f"_{test_period}"
        f"_{v_d['input_provider']}_{v_d['target_provider']}"
        f"{v_d['exp_suffix']}"
    )
    return Path(exp_root) / exp_full_name

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

def _is_int_leadtime(ds, coord="leadtime"):
    if coord not in ds.coords:
        return False
    return np.issubdtype(ds[coord].dtype, np.integer)

# def harmonize_leadtime_int(*datasets, coord="leadtime", fallback="range", force_single_leadtime=False):
#     """
#     Make all datasets share the same integer leadtime coordinate.
#     - If any dataset already has int leadtime, that coord is used as the target.
#     - Otherwise fallback:
#         * "range": set leadtime = 1..N based on the first dataset
#         * "days": convert timedelta leadtime to integer days (requires timedelta)
#     Returns: list of datasets in same order as input.
#     """
#     dsets = list(datasets)

#     # Pick target integer leadtime if present
#     int_sources = [ds for ds in dsets if _is_int_leadtime(ds, coord)]
#     if int_sources:
#         target = int_sources[0][coord].astype("int64")
#     else:
#         # No integer leadtime anywhere -> choose fallback
#         if coord not in dsets[0].coords:
#             raise ValueError(f"No coord {coord!r} found in the first dataset.")
#         n = 1 if force_single_leadtime else dsets[0].sizes.get(coord, dsets[0][coord].size)

#         if fallback == "range":
#             target = xr.DataArray(np.arange(1, n + 1), dims=coord, coords={coord: np.arange(1, n + 1)})
#         elif fallback == "days":
#             lt0 = dsets[0][coord]
#             if not np.issubdtype(lt0.dtype, np.timedelta64):
#                 raise ValueError("fallback='days' requires timedelta64 leadtime.")
#             days = (lt0 / np.timedelta64(1, "D")).astype("int64")
#             target = xr.DataArray(days.values, dims=coord, coords={coord: days.values})
#         else:
#             raise ValueError("fallback must be 'range' or 'days'.")

#     out = []
#     for ds in dsets:
#         if coord not in ds.coords:
#             raise ValueError(f"Dataset missing coord {coord!r}.")
#         if ds.sizes.get(coord, ds[coord].size) != target.size:
#             raise ValueError(f"Leadtime length mismatch: {ds.sizes.get(coord)} vs {target.size}")

#         ds2 = ds.assign_coords({coord: target})
#         out.append(ds2)

#     return out

def harmonize_leadtime_int(*datasets, coord="leadtime", method="int_source"):
    """
    Make all datasets share the same integer leadtime coordinate, shifting each
    leadtime value up to the nearest 30-day boundary.

    Rules
    -----
    - If method="int_source":
        * Use the first dataset that already has integer leadtime as the base
        * Then round that integer leadtime up to the nearest multiple of 30
        * If no dataset has integer leadtime, fall back to "days"
    - If method="range":
        * Set leadtime = 1..N based on the first dataset
        * Then round up to the nearest multiple of 30
    - If method="days":
        * Convert the first dataset's timedelta64 leadtime to integer days
        * Then round up to the nearest multiple of 30

    Returns
    -------
    list[xr.Dataset]
        Datasets in the same order as input, with harmonized leadtime coords.
    """
    dsets = list(datasets)
    if not dsets:
        return []

    def _round_up_to_30(values):
        values = np.asarray(values, dtype=np.int64)
        return ((values + 29) // 30) * 30

    def _make_target_from_values(values):
        values = np.asarray(values, dtype=np.int64)
        return xr.DataArray(values, dims=(coord,), coords={coord: values})

    target = None

    if method == "int_source":
        int_sources = [ds for ds in dsets if _is_int_leadtime(ds, coord)]
        if int_sources:
            raw = int_sources[0][coord].astype("int64").values
            target = _make_target_from_values(_round_up_to_30(raw))
        else:
            method = "days"

    if target is None and method == "range":
        if coord not in dsets[0].coords:
            raise ValueError(f"No coord {coord!r} found in the first dataset.")
        n = dsets[0].sizes.get(coord, dsets[0][coord].size)
        raw = np.arange(1, n + 1, dtype=np.int64)
        target = _make_target_from_values(_round_up_to_30(raw))

    if target is None and method == "days":
        if coord not in dsets[0].coords:
            raise ValueError(f"No coord {coord!r} found in the first dataset.")
        lt0 = dsets[0][coord]
        if not np.issubdtype(lt0.dtype, np.timedelta64):
            raise ValueError("method='days' requires timedelta64 leadtime in the first dataset.")
        raw = (lt0 / np.timedelta64(1, "D")).astype("int64").values
        target = _make_target_from_values(_round_up_to_30(raw))

    if target is None:
        raise ValueError("method must be 'int_source', 'range', or 'days'.")

    out = []
    for ds in dsets:
        if coord not in ds.coords:
            raise ValueError(f"Dataset missing coord {coord!r}.")
        if ds.sizes.get(coord, ds[coord].size) != target.size:
            raise ValueError(
                f"Leadtime length mismatch: {ds.sizes.get(coord, ds[coord].size)} vs {target.size}"
            )
        out.append(ds.assign_coords({coord: target}))

    return out

def rename_dim_and_coord(ds: xr.Dataset, src: str, target: str) -> xr.Dataset:
    """
    Rename a dimension and any coordinate/variable with the same name,
    and clean up leftover old-name coords/vars.

    Behavior:
      1. If `src` is a dimension, rename it to `target`.
      2. If a coord or data_var named `src` exists, attempt to rename it to `target`.
      3. After step 1-2, if an object named `src` still exists:
           - If `target` does NOT exist (in coords or data_vars), rename `src` -> `target`.
           - Else (target already exists) drop the leftover `src` to avoid duplicate names.

    Notes:
      - This function prefers dropping an old-named variable if the target name is already present
        (to avoid collisions / inconsistent shapes).
    """
    # Rename dimension first (xarray's rename_dims)
    if src in ds.dims:
        ds = ds.rename_dims({src: target})

    # If there's a variable or coordinate with the source name, try renaming it.
    # Use rename_vars which handles both coords and data_vars; rename_vars will fail if
    # target already exists, so we guard against that.
    if (src in ds.coords) or (src in ds.data_vars):
        if target not in ds.coords and target not in ds.data_vars:
            # safe to rename directly
            ds = ds.rename_vars({src: target})
        else:
            # target already exists: decide policy — drop the old src to avoid duplicates
            # But if shapes are identical and you want to keep src -> merge, modify here.
            # We'll prefer dropping the leftover src variable/coord.
            if src in ds.coords:
                ds = ds.drop_vars(src)
            if src in ds.data_vars:
                # drop_vars will also drop data_vars; coords already handled above.
                ds = ds.drop_vars(src)

    # Defensive pass: if something named `src` still exists (e.g., created by xarray internals),
    # attempt final resolution: if target free -> rename, else drop.
    if (src in ds.coords) or (src in ds.data_vars):
        if (target not in ds.coords) and (target not in ds.data_vars):
            ds = ds.rename_vars({src: target})
        else:
            ds = ds.drop_vars(src)

    return ds

def normalize_ds_dims_and_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize the input dataset by renaming guessed dimensions to standard names.
    """
    for guessed_dim in (
        (ds.earthml.guessed_dims.time, "time"),
        (ds.earthml.guessed_dims.latitude, "latitude"),
        (ds.earthml.guessed_dims.longitude, "longitude"),
        (ds.earthml.guessed_dims.realization, "realization"),
        (ds.earthml.guessed_dims.leadtime, "leadtime")
    ):
        src_dim, target_dim = guessed_dim
        if src_dim is not None and src_dim != target_dim:
            ds = rename_dim_and_coord(ds, src_dim, target_dim)

    return ds

def get_and_rename_dim(reference: xr.Dataset, data: List[xr.Dataset]):
    # Normalize reference first
    norm_r = normalize_ds_dims_and_coords(reference)

    print("Reference dataset after normalization:", norm_r.coords)

    # Decide final target names (match reference for time/lat/lon; standard for the others)
    target_time = norm_r.earthml.guessed_dims.time
    target_lat  = norm_r.earthml.guessed_dims.latitude
    target_lon  = norm_r.earthml.guessed_dims.longitude

    target_dims = (target_time, target_lat, target_lon, "realization", "leadtime")

    out: List[xr.Dataset] = []
    for d in data:
        ds = d

        # Guess source names on the current ds
        src_time = ds.earthml.guessed_dims.time
        src_lat  = ds.earthml.guessed_dims.latitude
        src_lon  = ds.earthml.guessed_dims.longitude
        src_real = ds.earthml.guessed_dims.realization
        src_lead = ds.earthml.guessed_dims.leadtime

        src_dims = (src_time, src_lat, src_lon, src_real, src_lead)

        for target, src in zip(target_dims, src_dims):
            if target is None or src is None or src == target:
                continue
            ds = rename_dim_and_coord(ds, src, target)

        out.append(ds)

    return norm_r, out, target_dims  # (time, lat, lon, realization, leadtime)

def load_exp(
    exp_cfg,
    type_data: str,
    load_train_preds: bool = False,
    load_models: Sequence[str] | None = None,
    only_sizes: bool = False,
    merge_compat: str = "override"
) -> dict:
    """
    Return dict keyed by train_period and number of valid samples.

    out[tp]["fc"] -> Dataset with dim 'leadtime'
    out[tp]["an"] -> Dataset with dim 'leadtime'
    out[tp]["pr"] -> Dataset with dim 'leadtime' (or None if type_data != 'test')
    """
    exp_root      = exp_cfg["root"]
    exp_name      = exp_cfg["name"]
    var_specs     = exp_cfg["vars"]
    leadtimes     = exp_cfg["leadtimes"]
    lt_unit       = exp_cfg["leadtime_unit"]
    train_periods = exp_cfg["train_periods"]
    test_period   = exp_cfg["test_period"]

    # print(exp_cfg)

    load_models = {"fc", "an", "pr"} if load_models is None else set(load_models)
    need_fc = "fc" in load_models
    need_an = "an" in load_models
    need_pr = "pr" in load_models

    out, n_valid_samples = {}, {}
    for tp in train_periods:
        n_valid_samples[tp] = {}
        fc_lt, an_lt, pr_lt, mask_lt = [], [], [], []
        for lt in leadtimes:
            pr_list, fc_list, an_list, mask_list = [], [], [], []
            n_valid_samples[tp][lt] = {}

            for v, v_d in var_specs.items():
                exp_folder_path = make_exp_folder_path(
                    exp_root, exp_name, v_d, lt, lt_unit, tp, test_period,
                )
                exp_path = exp_folder_path / "experiment.cfg"
                print(f"Load experiment: {exp_path}")

                experiment = joblib.load(exp_path)
                source = experiment[f"{type_data}_data"]  # e.g. "test_data"

                # Create mask data source
                mask_exp = experiment[f"{type_data}_mask_data"]
                mask_source = build_source(
                    name=mask_exp.datasource.source,
                    datasource=mask_exp.datasource,
                    **mask_exp.source_configs,
                )

                n_valid_samples[tp][lt][v] = {
                    "input": (len(source["input"].elements.samples)),
                    "target": (len(source["target"].elements.samples)),
                }
                if need_pr and (type_data == "test" or load_train_preds):
                    n_valid_samples[tp][lt][v]["prediction"] = (len(source["prediction"].elements.samples))

                if only_sizes:
                    continue

                if need_pr and (type_data == "test" or load_train_preds):
                    source["prediction"] = type(source["prediction"])(
                        source["prediction"].datasource,
                        exp_folder_path / f"{type_data}_preds.zarr"
                    )
                    # source["prediction"].__init__(source["prediction"].datasource, exp_folder_path / "test_preds.zarr")
                    pr = source["prediction"].reload() # reload? YES!
                    pr = normalize_ds_dims_and_coords(pr)
                    pr_list.append(pr.rename_vars({v_d["exp_var"]["fc"]: v}))

                fc = an = None
                if need_fc:
                    source["input"] = type(source["input"])(
                        source["input"].datasource,
                        exp_folder_path / f"{type_data}_input.zarr"
                    )
                    fc = source["input"].reload()

                if need_an:
                    source["target"] = type(source["target"])(
                        source["target"].datasource,
                        exp_folder_path / f"{type_data}_target.zarr"
                    )
                    an = source["target"].reload()

                mask = type(mask_source)(
                    mask_source.datasource,
                    exp_folder_path / f"mask/{type_data}_mask.zarr"
                ).reload()

                if fc is not None:
                    fc = normalize_ds_dims_and_coords(fc)
                if an is not None:
                    an = normalize_ds_dims_and_coords(an)
                mask = normalize_ds_dims_and_coords(mask)

                # print(f"After norm mean fc: {float(fc.to_array().mean().values)}")
                # print(f"After norm mean an: {float(an.to_array().mean().values)}")
                # print(f"After norm mean pr: {float(pr.to_array().mean().values)}")
                if fc is not None:
                    fc_list.append(fc.rename_vars({v_d["exp_var"]["fc"]: v}))
                if an is not None:
                    an_list.append(an.rename_vars({v_d["exp_var"]["an"]: v}))

                mask_list.append(mask.rename_vars({"common_mask": v}))

            if only_sizes:
                continue

            # for name, ds_list in zip(["fc", "an", "pr"], [fc_list, an_list, pr_list]):
            #     print(name, [id(ds) for ds in ds_list] if ds_list else None)

            ds_lt = []

            for ds_list in [fc_list, an_list, pr_list, mask_list]:
                if ds_list:
                    if "leadtime" in ds_list[0].dims:
                        ds_list_harm = harmonize_leadtime_int(
                            *ds_list, coord="leadtime", method="days"
                        )
                    else:
                        ds_list_harm = ds_list

                    ds = xr.merge(ds_list_harm, compat=merge_compat)
                else:
                    ds = None

                # if ds is not None:
                #     print(f"After leadtime harm mean ds: {float(ds.to_array().mean().values)}")
                # else:
                #     print("ds is None")

                ds_lt.append(ds)

            # Rename dims to standard names
            fc_single_lt, an_single_lt, pr_single_lt, mask_single_lt = ds_lt
            # print(f"load_exp, leadtime={lt}, merged vars ds's:")
            # print(f"   fc_single_lt.dims: {fc_single_lt.dims if fc_single_lt is not None else 'None'}, {fc_single_lt["leadtime"].values if fc_single_lt is not None and "leadtime" in fc_single_lt.dims else 'N/A'}")
            # print(f"   an_single_lt.dims: {an_single_lt.dims if an_single_lt is not None else 'None'}, {an_single_lt["leadtime"].values if an_single_lt is not None and "leadtime" in an_single_lt.dims else 'N/A'}")
            # print(f"   pr_single_lt.dims: {pr_single_lt.dims if pr_single_lt is not None else 'None'}, {pr_single_lt["leadtime"].values if pr_single_lt is not None and "leadtime" in pr_single_lt.dims else 'N/A'}")
            # print(f"   mask_single_lt.dims: {mask_single_lt.dims if mask_single_lt is not None else 'None'}, {mask_single_lt["leadtime"].values if mask_single_lt is not None and "leadtime" in mask_single_lt.dims else 'N/A'}")

            fc_lt.append(fc_single_lt)
            an_lt.append(an_single_lt)
            mask_lt.append(mask_single_lt)
            if need_pr and (type_data == "test" or load_train_preds):
                pr_lt.append(pr_single_lt)

        if only_sizes:
            out[tp] = {}
            continue

        ds_tp = []
        leadtime_fc_dim, leadtime_an_dim, leadtime_pr_dim, leadtime_mask_dim = (
            fc_lt[0].earthml.guessed_dims.leadtime,
            an_lt[0].earthml.guessed_dims.leadtime,
            pr_lt[0].earthml.guessed_dims.leadtime if pr_lt else None,
            mask_lt[0].earthml.guessed_dims.leadtime,
        )
        for leadtime_dim, ds_list in zip(
            [leadtime_fc_dim, leadtime_an_dim, leadtime_pr_dim, leadtime_mask_dim],
            [fc_lt, an_lt, pr_lt, mask_lt]
        ):
            if ds_list:
                if leadtime_dim is None:
                    leadtime_dim = "leadtime"
                ds = xr.concat(ds_list, dim=leadtime_dim).assign_coords({leadtime_dim: np.array(leadtimes)}).assign_attrs(leadtime_unit=lt_unit, train_period=tp)
                print(f"Concatenated {len(ds_list)} datasets along leadtime dimension '{leadtime_dim}' for train_period {tp}.")
                print(f"   Resulting shape: {ds.dims}, leadtime values: {ds[leadtime_dim].values}")
            else:
                ds = None
            # print(ds)
            ds_tp.append(ds)
        fc_tp, an_tp, pr_tp, mask_tp = ds_tp

        out[tp] = {"fc": fc_tp, "an": an_tp, "pr": pr_tp, "mask": mask_tp}

    return out, n_valid_samples

def add_ke_to_runs(runs, suffixes=("_mse",), dataset_keys=("fc", "an", "pr"),
                   u_name="u10", v_name="v10", ke_name="ke"):
    """
    Adds kinetic energy variables ke{suf} to each runs[tp][key] dataset (in place).
    Works with datasets that have leadtime dim (it will be preserved).
    """
    for tp, dd in runs.items():
        for key in dataset_keys:
            ds = dd.get(key)
            if ds is None:
                continue

            for suf in suffixes:
                uvar = f"{u_name}{suf}"
                vvar = f"{v_name}{suf}"
                kvar = f"{ke_name}{suf}"

                if uvar in ds.data_vars and vvar in ds.data_vars:
                    ke = 0.5 * (ds[uvar] ** 2 + ds[vvar] ** 2)
                    ds[kvar] = ke.rename(kvar)

    return runs

def calculate_climatology(
    ds: xr.Dataset,
    groupby: str = "month",
    time_dim: str | None = None,
    start: str | None = None,
    end: str | None = None,
    keep_attrs: bool = True,
) -> xr.Dataset:
    if time_dim is None:
        time_dim = ds.earthml.guessed_dims.time

    if start is not None or end is not None:
        ds = ds.sel({time_dim: slice(start, end)})

    time = ds[time_dim]
    if not hasattr(time, "dt"):
        raise TypeError(f"{time_dim!r} must be datetime-like")

    group = getattr(time.dt, groupby)
    ds = ds.assign_coords({groupby: (time_dim, group.data)})

    # Cast before grouped reduction is built to avoid mixed types
    cast_map = {
        v: np.float32
        for v in ds.data_vars
        if np.issubdtype(ds[v].dtype, np.floating)
    }
    if cast_map:
        ds = ds.astype(cast_map)

    return ds.groupby(groupby).mean(
        dim=time_dim,
        keep_attrs=keep_attrs,
        engine="numpy", # safest option
    )

def create_valid_mask_ds(ds: xr.Dataset) -> xr.DataArray:
    """
    Returns a boolean mask that is True where all variables in the dataset
    are non-null, after broadcasting across shared dims.
    """
    if not ds.data_vars:
        raise ValueError("Cannot build mask from an empty dataset.")

    masks = []
    for var_name, da in ds.data_vars.items():
        # True where this variable is valid
        masks.append(da.notnull())

    valid_mask = masks[0]
    for m in masks[1:]:
        valid_mask = valid_mask & m

    valid_mask.name = "valid_mask"
    return valid_mask
