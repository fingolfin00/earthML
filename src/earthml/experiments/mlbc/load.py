from typing import Sequence, List, Tuple

from pathlib import Path
import joblib

from rich import print

import numpy as np
import cf_xarray
import xarray as xr
from dateutil.relativedelta import relativedelta


from ...sources import build_source


def _is_int_leadtime(ds, coord="leadtime"):
    if coord not in ds.coords:
        return False
    return np.issubdtype(ds[coord].dtype, np.integer)

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
