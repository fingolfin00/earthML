from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import is_dataclass, replace
from typing import Sequence

import joblib
import os
from pathlib import Path

import numpy as np
import cf_xarray
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
import xarray as xr

from ...logging import get_logger
from ...sources import build_source, BaseSource
from ...base import Leadtime

from .dataclasses import MLBCExperimentConfig, MLBCExperimentDatasetRole, MLBCExperimentDataset


logger = get_logger(__name__)


_CANONICAL_VARIABLE_NAMES = {
    "sosstsst": "sst",
    "thetao": "sst",
    "votemper": "sst",
    "sst": "sst",
    "sosaline": "sss",
    "vosaline": "sss",
    "sss": "sss",
    "sos": "sss",
    "so": "sss",
    "so14chgt": "t14d",
    "t14d": "t14d",
    "so17chgt": "t17d",
    "t17d": "t17d",
    "so20chgt": "t20d",
    "t20d": "t20d",
    "sossheig": "ssh",
    "zos": "ssh",
    "ssh": "ssh",
}


def _rehydrate_saved_experiment(exp_path: Path, experiment: dict) -> dict:
    """
    Rebind location-dependent fields after loading a saved experiment.

    Experiments may be moved after creation, so any serialized absolute path
    must be reconciled with the current folder that contains ``experiment.cfg``.
    """
    config = experiment.get("config")
    if config is not None and hasattr(config, "work_path"):
        config.work_path = exp_path
    return experiment


def _canonicalize_dataset_variable_names(ds: xr.Dataset, dataset_name: str | None = None) -> xr.Dataset:
    """
    Rename dataset variables to a canonical short-name namespace.
    Only performs low-risk renames and avoids collisions.
    """
    rename_map: dict[str, str] = {}

    for var in ds.data_vars:
        canonical = _CANONICAL_VARIABLE_NAMES.get(var)
        if canonical is None or canonical == var:
            continue

        if canonical in ds.data_vars:
            # Avoid overwriting an existing variable
            continue

        rename_map[var] = canonical

    if rename_map:
        label = f" for {dataset_name!r}" if dataset_name else ""
        logger.info("Canonicalizing variables%s: %s", label, rename_map)
        ds = ds.rename_vars(rename_map)

    return ds


def _drop_auxiliary_merge_metadata(ds: xr.Dataset, dataset_name: str | None = None) -> xr.Dataset:
    """
    Drop non-essential source metadata that can vary across otherwise mergeable runs.
    """
    drop_names = [name for name in ("reftime",) if name in ds.variables]
    if not drop_names:
        return ds

    label = f" for {dataset_name!r}" if dataset_name else ""
    logger.info("Dropping auxiliary merge metadata%s: %s", label, drop_names)
    return ds.drop_vars(drop_names, errors="ignore")


def _drop_auxiliary_merge_metadata_many(
    datasets: Sequence[xr.Dataset],
    dataset_name: str | None = None,
) -> list[xr.Dataset]:
    return [
        _drop_auxiliary_merge_metadata(ds, dataset_name=dataset_name)
        for ds in datasets
    ]


def _normalize_selection_values(values: object, *, keep_empty: bool = False) -> list[object]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return [
            value for value in values
            if value is not None and (keep_empty or str(value) != "")
        ]
    if not keep_empty and str(values) == "":
        return []
    return [values]


def _matches_selection(value: object, allowed: object, *, keep_empty: bool = False) -> bool:
    allowed_values = _normalize_selection_values(allowed, keep_empty=keep_empty)
    if not allowed_values:
        return True
    value_str = str(value)
    return any(str(candidate) == value_str for candidate in allowed_values)


def _subset_dataset_variables(ds: xr.Dataset, variables: Sequence[str] | None) -> xr.Dataset:
    if variables is None:
        return ds

    selected = []
    requested = {str(variable) for variable in variables}
    for variable in ds.data_vars:
        if str(variable) in requested:
            selected.append(variable)

    return ds[selected] if selected else xr.Dataset(coords=ds.coords, attrs=ds.attrs)


def _is_int_leadtime(ds, coord="leadtime"):
    if coord not in ds.coords:
        return False
    return np.issubdtype(ds[coord].dtype, np.integer)


def get_leadtime_value_and_unit(leadtime: Leadtime) -> tuple[int, str]:
    if not isinstance(leadtime, Leadtime):
        raise TypeError(f"Expected Leadtime, got {type(leadtime).__name__}")
    return leadtime.value, leadtime.unit


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

    experiment = joblib.load(exp_path / "experiment.cfg")
    experiment = _rehydrate_saved_experiment(exp_path, experiment)
    return exp_path, experiment


def _normalize_experiment_variant(value: object) -> str:
    if value is None:
        return ""
    return str(value).lstrip("_").strip()


def _resolve_run_name_suffix(config: MLBCExperimentConfig, exp_path: Path) -> str:
    explicit_suffix = getattr(config, "run_name_suffix", None)
    if explicit_suffix is not None and str(explicit_suffix) != "":
        return str(explicit_suffix)

    loss_name = getattr(getattr(config, "net", None), "loss", "") or ""
    loss_suffix = f"_{str(loss_name).lower()}" if loss_name else ""
    run_name = exp_path.name

    if loss_suffix:
        idx = run_name.rfind(loss_suffix)
        if idx != -1:
            return run_name[idx:]

    return loss_suffix


def _resolve_experiment_variant(config: MLBCExperimentConfig, exp_path: Path) -> str:
    full_suffix = _resolve_run_name_suffix(config, exp_path)
    loss_name = getattr(getattr(config, "net", None), "loss", "") or ""
    loss_suffix = f"_{str(loss_name).lower()}" if loss_name else ""

    if loss_suffix and full_suffix.startswith(loss_suffix):
        return _normalize_experiment_variant(full_suffix[len(loss_suffix):])

    return _normalize_experiment_variant(full_suffix)


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


def _snap_time_to_nearest_month_start(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.datetime64):
        return arr
    idx = pd.DatetimeIndex(pd.to_datetime(arr))
    month_start = idx.to_period("M").to_timestamp(how="start")
    next_month = (idx.to_period("M") + 1).to_timestamp(how="start")
    snapped = month_start.where((idx - month_start) <= (next_month - idx), next_month)
    return snapped.to_numpy()


def _prefer_valid_time_for_alignment(
    input_ds: xr.Dataset,
    target_ds: xr.Dataset,
) -> xr.Dataset:
    input_time_dim = input_ds.earthml.guessed_dims.time
    target_time_dim = target_ds.earthml.guessed_dims.time
    if input_time_dim is None or target_time_dim is None:
        return input_ds
    if input_time_dim not in input_ds.coords or target_time_dim not in target_ds.coords:
        return input_ds
    if "valid_time" not in input_ds.coords:
        return input_ds

    valid_time = input_ds["valid_time"]
    if valid_time.dims != (input_time_dim,):
        return input_ds
    if valid_time.sizes.get(input_time_dim, 0) != input_ds.sizes.get(input_time_dim, 0):
        return input_ds

    input_time = np.asarray(input_ds[input_time_dim].values)
    input_valid_time = np.asarray(valid_time.values)
    target_time = np.asarray(target_ds[target_time_dim].values)

    snapped_valid_time = _snap_time_to_nearest_month_start(input_valid_time)
    snapped_target_time = _snap_time_to_nearest_month_start(target_time)

    source_overlap = len(np.intersect1d(input_time, target_time))
    valid_overlap = len(np.intersect1d(snapped_valid_time, snapped_target_time))
    if valid_overlap <= source_overlap:
        return input_ds

    logger.info(
        "Promote forecast time to valid_time while loading saved experiment: overlap %s -> %s",
        source_overlap,
        valid_overlap,
    )
    return input_ds.assign_coords({input_time_dim: snapped_valid_time})


def _load_single_exp(
    exp_cfg: MLBCExperimentConfig | str | Path,
    type_data: str,
    load_train_preds: bool = False,
    load_models: Sequence[str] | None = None,
    variables: Sequence[str] | None = None,
    only_sizes: bool = False,
    show_dask_progress: bool = True,
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
        fc: xr.Dataset = source["input"].reload(
            show_dask_progress=show_dask_progress,
        ).earthml.normalize_dims_and_coords()

    if need_an:
        source["target"] = _source_with_root_path(
            source["target"],
            exp_path / f"{type_data}_target.zarr",
        )
        an: xr.Dataset = source["target"].reload(
            show_dask_progress=show_dask_progress,
        ).earthml.normalize_dims_and_coords()

    if need_pr and (type_data == "test" or load_train_preds) and "prediction" in source:
        source["prediction"] = _source_with_root_path(
            source["prediction"],
            exp_path / f"{type_data}_preds.zarr",
        )
        pr: xr.Dataset = source["prediction"].reload(
            show_dask_progress=show_dask_progress,
        ).earthml.normalize_dims_and_coords()

    if fc is not None and an is not None:
        fc = _prefer_valid_time_for_alignment(fc, an)

    mask = _source_with_root_path(
        mask_source,
        exp_path / f"mask/{type_data}_mask.zarr",
    ).reload(
        show_dask_progress=show_dask_progress,
    )
    mask: xr.Dataset = mask.earthml.normalize_dims_and_coords()

    out = {}
    for model_name, ds in (("fc", fc), ("an", an), ("pr", pr)):
        if ds is None:
            continue
        ds = _canonicalize_dataset_variable_names(ds, dataset_name=model_name)
        ds = _subset_dataset_variables(ds, variables)
        ds = _drop_auxiliary_merge_metadata(ds, dataset_name=model_name)
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
    variables: Sequence[str] | None = None,
    only_sizes: bool = False,
) -> tuple[dict[str, xr.Dataset], dict]:
    if isinstance(exp_configs, (MLBCExperimentConfig, str, Path)):
        return _load_single_exp(
            exp_cfg=exp_configs,
            type_data=type_data,
            load_train_preds=load_train_preds,
            load_models=load_models,
            variables=variables,
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
                variables=variables,
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
    variables: Sequence[str] | None = None,
    run_filters: Mapping[str, object] | None = None,
    only_sizes: bool = False,
    cfg_name: str = "experiment.cfg",
    show_progress: bool = True,
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
                    experiment = _rehydrate_saved_experiment(current_path, experiment)
                    config: MLBCExperimentConfig = experiment["config"]
                    config.run_name_suffix = _resolve_run_name_suffix(config, current_path)
                    configs.append(config)
        except FileNotFoundError:
            continue

    grouped_runs: dict[str, dict[str, list[xr.Dataset]]] = {}
    grouped_sizes: dict[str, dict[tuple, dict]] = {}

    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        if show_progress else nullcontext()
    )

    with progress_cm as prog:
        task = prog.add_task("Loading experiments", total=len(configs)) if show_progress else None

        for config in configs:
            train_datasets = config.train_dataset
            train_datasets = train_datasets if isinstance(train_datasets, Sequence) else [train_datasets]

            idx_input = None
            for i, td in enumerate(train_datasets):
                if td.role == "input":
                    idx_input = i
                    break
            if idx_input is None:
                if show_progress:
                    prog.advance(task)
                continue

            train_ds_input = train_datasets[idx_input]
            datasource_input = train_ds_input.datasource
            datasource_input = datasource_input[0] if isinstance(datasource_input, Sequence) else datasource_input

            source_configs = train_ds_input.source_configs
            source_config = source_configs[0] if isinstance(source_configs, Sequence) else source_configs
            leadtime_rd = source_config.leadtime

            try:
                leadtime, leadtime_unit = get_leadtime_value_and_unit(leadtime_rd)
            except ValueError as exc:
                raise ValueError(f"Unsupported leadtime for experiment {config.name}: {leadtime_rd}") from exc

            var_input = datasource_input.data_selection.variable.name

            train_periods = datasource_input.data_selection.period
            train_periods = train_periods if isinstance(train_periods, Sequence) else [train_periods]
            train_period = f"{train_periods[0].start:%Y%m%d}-{train_periods[-1].end:%Y%m%d}"

            region = datasource_input.data_selection.region.name
            loss = config.net.loss.lower()
            variant = _resolve_experiment_variant(config, Path(config.work_path))

            group_name = _CANONICAL_VARIABLE_NAMES.get(var_input, var_input)
            if show_progress:
                prog.update(
                    task,
                    description=(
                        f"Loading {group_name} | leadtime={leadtime} {leadtime_unit} | "
                        f"region={region} | loss={loss} | variant={variant or '<default>'}"
                    ),
                )

            if not _matches_selection(group_name, (run_filters or {}).get("variable", variables)):
                if show_progress:
                    prog.advance(task)
                continue
            if not _matches_selection(leadtime, (run_filters or {}).get("leadtime")):
                if show_progress:
                    prog.advance(task)
                continue
            if not _matches_selection(train_period, (run_filters or {}).get("train_period")):
                if show_progress:
                    prog.advance(task)
                continue
            if not _matches_selection(region, (run_filters or {}).get("region")):
                if show_progress:
                    prog.advance(task)
                continue
            if not _matches_selection(loss, (run_filters or {}).get("loss")):
                if show_progress:
                    prog.advance(task)
                continue
            variant_filters = (run_filters or {}).get("variant")
            if variant_filters is not None:
                variant_filters = [
                    _normalize_experiment_variant(value)
                    for value in _normalize_selection_values(variant_filters, keep_empty=True)
                ]
            if not _matches_selection(variant, variant_filters, keep_empty=True):
                if show_progress:
                    prog.advance(task)
                continue

            run_dict, size = _load_single_exp(
                exp_cfg=config,
                type_data=type_data,
                load_train_preds=load_train_preds,
                load_models=load_models,
                variables=variables,
                only_sizes=only_sizes,
                show_dask_progress=not show_progress,
            )

            size_key = (leadtime, train_period, region, loss, variant)
            grouped_sizes.setdefault(group_name, {})[size_key] = size

            if only_sizes:
                if show_progress:
                    prog.advance(task)
                continue

            grouped_runs.setdefault(group_name, {})

            for model_name, ds in run_dict.items():
                if ds is None:
                    continue
                if model_name in {"fc", "an", "pr"} and not ds.data_vars:
                    logger.info(
                        "Skipping empty %s dataset for group=%s leadtime=%s train_period=%s region=%s loss=%s",
                        model_name,
                        group_name,
                        leadtime,
                        train_period,
                        region,
                        loss,
                    )
                    continue

                for dim in ["train_period", "loss", "variant", "region"]:
                    if dim in ds.dims or dim in ds.coords:
                        raise ValueError(f"{dim} already exists in dataset")

                # leadtime may be already present in dataset
                if "leadtime" in ds.dims:
                    if ds.sizes["leadtime"] != 1:
                        raise ValueError(
                            f"Expected singleton leadtime dimension, got {ds.sizes['leadtime']}"
                        )
                    ds = ds.squeeze("leadtime", drop=True)

                if "leadtime" in ds.coords:
                    ds = ds.drop_vars("leadtime")

                ds = ds.expand_dims(leadtime=[leadtime])

                # set new train_period and loss dims
                ds = ds.expand_dims(
                    {
                        "train_period": [train_period],
                        "loss": [loss],
                        "variant": [variant],
                        "region": [region],
                    }
                )

                ds.coords["leadtime"].attrs["unit"] = leadtime_unit

                grouped_runs[group_name].setdefault(model_name, []).append(ds)

                # print(model_name, group_name, ds.dims, ds.coords)
                # for v in ds.data_vars:
                #     print("   ", v, ds[v].dims)

            if show_progress:
                prog.advance(task)

    if only_sizes:
        return {}, grouped_sizes

    combine_steps = sum(
        1
        for model_map in grouped_runs.values()
        for runs in model_map.values()
        if runs
    )
    combined_by_group_and_model: dict[str, dict[str, xr.Dataset]] = {}
    runs_out: dict[str, xr.Dataset] = {}
    all_model_names = {
        model_name
        for model_map in grouped_runs.values()
        for model_name in model_map
    }
    merge_steps = len(all_model_names)

    progress_cm = (
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        if show_progress else nullcontext()
    )

    with progress_cm as prog:
        combine_task = prog.add_task("Combining grouped runs", total=combine_steps) if show_progress else None
        merge_task = prog.add_task("Merging models", total=merge_steps) if show_progress else None

        for group_name, model_map in grouped_runs.items():
            combined_by_group_and_model[group_name] = {}
            for model_name, runs in model_map.items():
                if not runs:
                    continue

                runs = _drop_auxiliary_merge_metadata_many(runs, dataset_name=model_name)
                combined_by_group_and_model[group_name][model_name] = xr.combine_by_coords(
                    runs,
                    combine_attrs="drop_conflicts",
                )
                if show_progress:
                    prog.advance(combine_task)

        for model_name in all_model_names:
            model_datasets = [
                model_map[model_name]
                for model_map in combined_by_group_and_model.values()
                if model_name in model_map
            ]
            if not model_datasets:
                if show_progress:
                    prog.advance(merge_task)
                continue

            model_datasets = _drop_auxiliary_merge_metadata_many(
                model_datasets,
                dataset_name=model_name,
            )
            runs_out[model_name] = xr.merge(
                model_datasets,
                join="outer",
                compat="no_conflicts",
                combine_attrs="drop_conflicts",
            )
            if show_progress:
                prog.advance(merge_task)

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
