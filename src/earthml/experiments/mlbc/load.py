from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, Sequence, List
from datetime import datetime, date

import os
from pathlib import Path
import joblib

from rich import print

import numpy as np
import cf_xarray
import xarray as xr
from dateutil.relativedelta import relativedelta

from ...sources import build_source, BaseSource

from .dataclasses import MLBCExperimentConfig, MLBCExperimentDatasetRole, MLBCExperimentDataset
from .train_test import MLBCExperiment


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


def _load_saved_experiment(
    exp_cfg: MLBCExperimentConfig | str | Path
) -> tuple[Path, dict]: # returns a dict as defined in joblib.dump in train_test.py
    if isinstance(exp_cfg, MLBCExperimentConfig):
        exp_path = Path(exp_cfg.work_path)
    else:
        exp_path = Path(exp_cfg)
        if exp_path.is_file():
            exp_path = exp_path.parent

    return exp_path, joblib.load(exp_path / "experiment.cfg")


def _source_with_root_path(
    source: BaseSource,
    root_path: str | Path,
) -> BaseSource:
    config = getattr(source, "config", None)
    root_path = Path(root_path)

    if config is not None and hasattr(config, "root_path"):
        if is_dataclass(config):
            config = replace(config, root_path=root_path)
        else:
            config.root_path = root_path
        return type(source)(source.datasource, config)

    return type(source)(source.datasource, root_path)


def _load_single_exp(
    exp_cfg: MLBCExperimentConfig | str | Path,
    type_data: str,
    load_train_preds: bool = False,
    load_models: Sequence[str] | None = None,
    only_sizes: bool = False,
) -> tuple[dict[str, xr.Dataset], dict]:
    exp_path, experiment = _load_saved_experiment(exp_cfg)

    load_models = {"fc", "an", "pr"} if load_models is None else set(load_models)
    need_fc = "fc" in load_models
    need_an = "an" in load_models
    need_pr = "pr" in load_models

    source: dict[MLBCExperimentDatasetRole, BaseSource] = experiment[f"{type_data}_data"]
    mask_exp: MLBCExperimentDataset = experiment[f"{type_data}_mask_data"]
    mask_source = build_source(
        name=mask_exp.datasource.source,
        params={
            "datasource": mask_exp.datasource,
            "config": mask_exp.source_configs,
        },
    )

    n_valid_samples = {
        "input": len(source["input"].elements.samples),
        "target": len(source["target"].elements.samples),
    }
    if need_pr and (type_data == "test" or load_train_preds) and "prediction" in source:
        n_valid_samples["prediction"] = len(source["prediction"].elements.samples)

    if only_sizes:
        return {}, n_valid_samples

    fc = an = pr = None
    if need_fc:
        source["input"] = _source_with_root_path(
            source["input"],
            exp_path / f"{type_data}_input.zarr",
        )
        fc: xr.Dataset = source["input"].reload().earthml.normalize_dims_and_coords()

    if need_an:
        source["target"] = _source_with_root_path(
            source["target"],
            exp_path / f"{type_data}_target.zarr",
        )
        an: xr.Dataset = source["target"].reload().earthml.normalize_dims_and_coords()

    if need_pr and (type_data == "test" or load_train_preds) and "prediction" in source:
        source["prediction"] = _source_with_root_path(
            source["prediction"],
            exp_path / f"{type_data}_preds.zarr",
        )
        pr: xr.Dataset = source["prediction"].reload().earthml.normalize_dims_and_coords()

    mask = _source_with_root_path(
        mask_source,
        exp_path / f"mask/{type_data}_mask.zarr",
    ).reload()
    mask: xr.Dataset = mask.earthml.normalize_dims_and_coords()

    out = {}
    for model_name, ds in (("fc", fc), ("an", an), ("pr", pr)):
        if ds is None:
            continue
        allowed_dims = ds.earthml.guessed_dims
        ds = ds.earthml.remove_dims_and_coords(allowed_dims, drop_non_dim_coords=True)
        out[model_name] = ds

    if mask is not None:
        out['mask'] = mask

    return out, n_valid_samples


def load_exp(
    exp_configs,
    type_data: str,
    load_train_preds: bool = False,
    load_models: Sequence[str] | None = None,
    only_sizes: bool = False,
) -> tuple[dict[str, xr.Dataset], dict]:
    if isinstance(exp_configs, (MLBCExperimentConfig, str, Path)):
        return _load_single_exp(
            exp_cfg=exp_configs,
            type_data=type_data,
            load_train_preds=load_train_preds,
            load_models=load_models,
            only_sizes=only_sizes,
        )

    # TODO verify this makes still sense
    if isinstance(exp_configs, Mapping):
        out, n_valid_samples = {}, {}
        for key, cfg in exp_configs.items():
            runs, sizes = _load_single_exp(
                exp_cfg=cfg,
                type_data=type_data,
                load_train_preds=load_train_preds,
                load_models=load_models,
                only_sizes=only_sizes,
            )
            out[key] = runs
            n_valid_samples[key] = sizes
        return out, n_valid_samples

    raise TypeError("exp_configs must be an MLBCExperimentConfig, a path, or a mapping of experiment configs")


def load_all_exp_from_folder(
    exp_root: str | Path,
    type_data: str,
    load_train_preds: bool = False,
    load_models: Sequence[str] | None = None,
    only_sizes: bool = False,
    cfg_name: str = "experiment.cfg",
) -> tuple[dict[str, xr.Dataset], dict]:
    """
    Fetch all experiments in a provided folder.

    Returns
    -------
    runs_out : dict[str, xr.Dataset]
        Dictionary keyed by model name ("fc", "an", "pr", "mask"), where each
        dataset contains all merged experiment groups/variables.

    sizes_out : dict[str, dict[tuple, dict]]
        Per-group map of experiment coordinates -> valid sample sizes.
    """
    exp_root = Path(exp_root)

    load_models = {"fc", "an", "pr"} if load_models is None else set(load_models)
    need_fc = "fc" in load_models
    need_an = "an" in load_models
    need_pr = "pr" in load_models and (type_data == "test" or load_train_preds)

    def _has_required_artifacts(exp_dir: Path) -> bool:
        required = [exp_dir / cfg_name]

        if need_fc:
            required.append(exp_dir / f"{type_data}_input.zarr")
        if need_an:
            required.append(exp_dir / f"{type_data}_target.zarr")
        if need_pr:
            required.append(exp_dir / f"{type_data}_preds.zarr")

        # mask is always loaded by _load_single_exp
        required.append(exp_dir / "mask" / f"{type_data}_mask.zarr")

        return all(path.exists() for path in required)

    stack = [exp_root]
    configs = []

    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                        continue

                    if not entry.is_file(follow_symlinks=False) or entry.name != cfg_name:
                        continue

                    current_path = Path(current)

                    if not _has_required_artifacts(current_path):
                        continue

                    experiment = joblib.load(entry.path)
                    config: MLBCExperimentConfig = experiment["config"]
                    configs.append(config)
        except FileNotFoundError:
            continue

    grouped_runs: dict[str, dict[str, list[xr.Dataset]]] = {}
    grouped_sizes: dict[str, dict[tuple, dict]] = {}

    for config in configs:
        train_datasets = config.train_dataset
        train_datasets = train_datasets if isinstance(train_datasets, Sequence) else [train_datasets]

        idx_input = None
        for i, td in enumerate(train_datasets):
            if td.role == "input":
                idx_input = i
                break
        if idx_input is None:
            continue

        train_ds_input = train_datasets[idx_input]
        datasource_input = train_ds_input.datasource
        datasource_input = datasource_input[0] if isinstance(datasource_input, Sequence) else datasource_input

        source_configs = train_ds_input.source_configs
        source_config = source_configs[0] if isinstance(source_configs, Sequence) else source_configs
        leadtime_rd = source_config.leadtime

        if leadtime_rd.hours >= 1:
            leadtime = leadtime_rd.hours
            leadtime_unit = "hours"
        elif leadtime_rd.months >= 1:
            leadtime = leadtime_rd.months
            leadtime_unit = "months"
        else:
            raise ValueError(f"Unsupported leadtime for experiment {config.name}: {leadtime_rd}")

        var_input = datasource_input.data_selection.variable.name

        train_periods = datasource_input.data_selection.period
        train_periods = train_periods if isinstance(train_periods, Sequence) else [train_periods]
        train_period = f"{train_periods[0].start:%Y%m%d}-{train_periods[-1].end:%Y%m%d}"

        region = datasource_input.data_selection.region.name
        loss = config.net.loss.lower()

        group_name = var_input

        run_dict, size = _load_single_exp(
            exp_cfg=config,
            type_data=type_data,
            load_train_preds=load_train_preds,
            load_models=load_models,
            only_sizes=only_sizes,
        )

        size_key = (leadtime, train_period, region, loss)
        grouped_sizes.setdefault(group_name, {})[size_key] = size

        if only_sizes:
            continue

        grouped_runs.setdefault(group_name, {})

        for model_name, ds in run_dict.items():
            if ds is None:
                continue

            # leadtime may be already present in dataset
            if "leadtime" in ds.dims:
                ds = ds.assign_coords(leadtime=[leadtime])
            else:
                ds = ds.expand_dims({"leadtime": [leadtime]})

            # introduce new train_period and loss dims
            ds = ds.expand_dims({
                "train_period": [train_period],
                "loss": [loss],
            })

            ds.coords["leadtime"].attrs["unit"] = leadtime_unit

            # region is metadata
            ds.attrs["region"] = region

            grouped_runs[group_name].setdefault(model_name, []).append(ds)

    if only_sizes:
        return {}, grouped_sizes

    combined_by_group_and_model: dict[str, dict[str, xr.Dataset]] = {}
    for group_name, model_map in grouped_runs.items():
        combined_by_group_and_model[group_name] = {}
        for model_name, runs in model_map.items():
            if not runs:
                continue

            combined_by_group_and_model[group_name][model_name] = xr.combine_by_coords(
                runs,
                combine_attrs="drop_conflicts",
            )

    runs_out: dict[str, xr.Dataset] = {}
    all_model_names = {
        model_name
        for model_map in combined_by_group_and_model.values()
        for model_name in model_map
    }

    for model_name in all_model_names:
        model_datasets = [
            model_map[model_name]
            for model_map in combined_by_group_and_model.values()
            if model_name in model_map
        ]
        if not model_datasets:
            continue

        runs_out[model_name] = xr.merge(
            model_datasets,
            join="outer",
            compat="no_conflicts",
            combine_attrs="drop_conflicts",
        )

    return runs_out, grouped_sizes


def add_ke_to_runs(
    runs: xr.Dataset,
    dataset_keys: Sequence[str] = ("fc", "an", "pr"),
    u_name: str = "u10",
    v_name: str = "v10",
    ke_name: str = "ke",
) -> xr.Dataset:
    """
    Adds kinetic energy variables ke{suf} to the runs dataset in place.
    If a model dimension is present, computation can be restricted to dataset_keys.
    """
    if u_name not in runs.data_vars or v_name not in runs.data_vars:
        return runs

    model_mask = None
    if "model" in runs.dims and dataset_keys is not None:
        selected_models = [key for key in dataset_keys if key in runs["model"].values]
        if selected_models and len(selected_models) != runs.sizes["model"]:
            model_mask = xr.DataArray(
                np.isin(runs["model"].values, selected_models),
                dims=("model",),
                coords={"model": runs["model"].values},
            )

    ke = 0.5 * (runs[u_name] ** 2 + runs[v_name] ** 2)
    if model_mask is not None and "model" in ke.dims:
        ke = ke.where(model_mask)
    runs[ke_name] = ke.rename(ke_name)

    return runs
