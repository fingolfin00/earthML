from typing import Sequence, Literal
from dataclasses import asdict

import os
import time, multiprocessing, joblib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from zarr.codecs import BloscCodec

import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from .dataclasses import MLBCExperimentDataset, MLBCExperimentConfig
from .registry import MLBCExperimentDatasetRole, MLBCExperimentMode

from ...sources import build_source, BaseSource, DataSource, XarrayLocalSourceConfig
from ...misc import Table
from ...logging import get_logger, log_renderable
from ...neural import XarrayDataset, Normalize, EpochRandomSplitDataModule
from ...neural.nets import build_net
from .plot import (
    run_stage_plot_bundle,
)

ENABLE_STAGE_PLOTTING = True


class MLBCExperiment:
    def __init__(
        self,
        config: MLBCExperimentConfig,
    ):
        self._configure_torch_env()
        self.config = config
        self.logger = get_logger(__name__)

        self.per_channel_masked_mean = False if self.config.output_realizations else True # return (1, C, 1, 1) in _masked_metrics (see neural/normalize.py) # TODO not sure it's super general

        # Get test variable list,
        self.test_var_list = self._get_var_list(self.config.test_dataset)
        self.train_var_list = self._get_var_list(self.config.train_dataset)

        # Setup paths and make dirs if necessary
        self._path_setup()

        # General torch and Lightning setup
        self.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_cpus = multiprocessing.cpu_count()
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.torch_workers = 0
        self.logger.info(
            "Torch workers in use: %s (CPUs=%s, CUDA devices=%s, accelerator=%s)",
            self.torch_workers,
            num_cpus,
            num_gpus,
            self.accelerator,
        )
        L.seed_everything(self.config.net.seed)
        # Tensorboard
        self.tl_logger = TensorBoardLogger(self.config.work_path, name="tensorboard_logs") # version=self.run_number

        # Init
        self.normalize_input = None
        self.normalize_target = None
        self.train_datamodule = None

        # Init climatologies
        self.input_clim_ts = None
        self.target_clim_ts = None

        self.consolidated_zarr = False

        # Init prediction experiment datasets and add to test and train lists
        self.test_preds_store = self.config.work_path / Path("test_preds").with_suffix(".zarr")
        test_preds_exp = self._init_experiment_dataset(self.test_preds_store, self.config.test_dataset, MLBCExperimentDatasetRole.PREDICTION)
        self.config.test_dataset.append(test_preds_exp)

        self.train_preds_store = self.config.work_path / Path("train_preds").with_suffix(".zarr")
        train_preds_exp = self._init_experiment_dataset(self.train_preds_store, self.config.train_dataset, MLBCExperimentDatasetRole.PREDICTION)
        self.config.train_dataset.append(train_preds_exp)

        # Init source data objects and save original datasets. Entry point for download and preprocessing in Source
        self.source_train_data = self._init_source_data(self.config.train_dataset, "train")
        self.source_test_data  = self._init_source_data(self.config.test_dataset,  "test")

        # Generate xarray datsets
        self.train_input_ds, self.train_target_ds = self._generate_xarray_ds(self.source_train_data, self.config.train_dataset, MLBCExperimentMode.TRAIN) # capital T just for logging
        self.test_input_ds, self.test_target_ds = self._generate_xarray_ds(self.source_test_data, self.config.test_dataset, MLBCExperimentMode.TEST)

        # Handle latitudes for optional spatial weighting
        self.latitudes = None
        loss_cfg = self.config.net.loss_params.get("loss", {})  # safe access
        net_cfg = self.config.net.loss_params.get("net", {})

        lat_cfg = loss_cfg.get("latitudes", False)
        if lat_cfg is True:
            # Compute from test data
            ds_in = self.source_test_data["input"].load()
            lat_np = self._get_latitudes(ds_in).astype(np.float32)
            self.latitudes = torch.from_numpy(lat_np)
        elif lat_cfg is False or lat_cfg is None:
            # Explicit opt-out or absent
            self.latitudes = None
        else:
            # User provided array-like or torch.Tensor in config
            if isinstance(lat_cfg, torch.Tensor):
                self.latitudes = lat_cfg.to(dtype=torch.float32).cpu()
            else:
                lat_arr = np.asarray(lat_cfg, dtype=np.float32)
                self.latitudes = torch.from_numpy(lat_arr)

        # TODO add latitudes validation
        # if self.latitudes is not None:
        #     expected_H = ...  # compute or fetch from a sample input shape
        #     if self.latitudes.numel() != expected_H:
        #         raise ValueError(f"latitudes length {self.latitudes.numel()} != H {expected_H}")

        # Remove climatology (monthly) if requested
        if self.config.anomaly:
            self._init_anomaly_climatologies()
            self.train_input_ds, self.train_target_ds = self._calculate_anomaly(self.train_input_ds, self.train_target_ds)
            self.test_input_ds, self.test_target_ds = self._calculate_anomaly(self.test_input_ds, self.test_target_ds)
            self._save_anomaly_datasets(self.train_input_ds, self.train_target_ds, "train")
            self._save_anomaly_datasets(self.test_input_ds, self.test_target_ds, "test")

        # Inpaint nan if requested
        if self.config.inpaint_nan:
            self.logger.info("Inpainting NaN values in datasets with bilinear interpolation...")
            self.train_input_ds, self.train_target_ds = (
                self.train_input_ds.earthml.inpaint_nan_lonlat_bilinear(),
                self.train_target_ds.earthml.inpaint_nan_lonlat_bilinear(),
            )
            self.test_input_ds, self.test_target_ds = (
                self.test_input_ds.earthml.inpaint_nan_lonlat_bilinear(),
                self.test_target_ds.earthml.inpaint_nan_lonlat_bilinear(),
            )

        # Hook preprocessing function
        if self.config.torch_preprocess_fn is not None:
            self.logger.info("Apply preprocessing custom function...")
            self.train_input_ds, self.train_target_ds = self.config.torch_preprocess_fn(self.train_input_ds, self.train_target_ds)
            self.test_input_ds, self.test_target_ds = self.config.torch_preprocess_fn(self.test_input_ds, self.test_target_ds)

        # Init mask dataset
        mask_filename = "mask"

        self.test_mask_store = self.mask_folder_path.joinpath(Path("test_"+mask_filename).with_suffix(".zarr"))
        self.test_mask_exp = self._init_experiment_dataset(self.test_mask_store, self.config.test_dataset, MLBCExperimentDatasetRole.PREDICTION)

        self.train_mask_store = self.mask_folder_path.joinpath(Path("train_"+mask_filename).with_suffix(".zarr"))
        self.train_mask_exp = self._init_experiment_dataset(self.train_mask_store, self.config.train_dataset, MLBCExperimentDatasetRole.PREDICTION)

        self.train_mask_ds = self._create_and_save_common_mask(self.train_input_ds, self.train_target_ds, MLBCExperimentMode.TRAIN)
        self.test_mask_ds = self._create_and_save_common_mask(self.test_input_ds, self.test_target_ds, MLBCExperimentMode.TEST)

        # Build merged loss_params
        base_loss_params = dict(self.config.net.loss_params)  # shallow copy
        nested_loss = dict(base_loss_params.get("loss", {}))
        if self.latitudes is not None:
            nested_loss["latitudes"] = self.latitudes
        base_loss_params["loss"] = nested_loss
        base_loss_params["net"] = net_cfg

        # Initialize model
        self.model = build_net(
            name=self.config.net.name,
            learning_rate=self.config.net.learning_rate,
            loss=self.config.net.loss,
            loss_params=base_loss_params,
            norm=self.config.net.norm_strategy,
            supervised=self.config.net.supervised,
            **self.config.net.extra_net_args,
        ).to(self.device)

        # Log model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info("Net %s trainable parameters: %s", type(self.model).__name__, f"{trainable_params:,}")
        log_renderable(Table(self.config, title="ExperimentConfig", twocols=True).table, logger=self.logger)

        # Save experiment
        with open(self.work_path.joinpath("experiment.cfg"), 'wb') as f:
            joblib.dump({
                'config': self.config,
                'device': self.device,
                'torch_workers': self.torch_workers,
                'tl_logger': self.tl_logger,
                # 'model': self.model,
                'test_data': self.source_test_data,
                'train_data': self.source_train_data,
                'test_mask_data': self.test_mask_exp,
                'train_mask_data': self.train_mask_exp,
            }, f)


    def _configure_torch_env(self):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def _configure_torch_runtime(self):
        torch.set_float32_matmul_precision('medium')  # or 'high'
        if torch.cuda.is_available():
            deterministic = False
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic

    def _path_setup(self):
        self.work_path = Path(self.config.work_path)
        # Diagnostic plots location
        self.plots_folder_path = self.work_path.joinpath("./plots")
        self.plots_folder_path.mkdir(parents=True, exist_ok=True)
        # Weights location
        self.weights_folder_path = self.work_path.joinpath("./weights")
        self.weights_folder_path.mkdir(parents=True, exist_ok=True)
        self.weights_filename = f"{self.config.name}_weights"
        self.weights_path = self.weights_folder_path.joinpath(self.weights_filename+'.ckpt')
        # Train dataset normalization data location
        norm_data_folder_path = self.work_path.joinpath("./normdata")
        normdata_input_filename = f"{self.config.name}_normdata_input.gz"
        normdata_target_filename = f"{self.config.name}_normdata_target.gz"
        norm_data_folder_path.mkdir(parents=True, exist_ok=True)
        self.normdata_input_path = norm_data_folder_path.joinpath(normdata_input_filename)
        self.normdata_target_path = norm_data_folder_path.joinpath(normdata_target_filename)
        # Mask location
        self.mask_folder_path = self.work_path.joinpath("./mask")
        self.mask_folder_path.mkdir(parents=True, exist_ok=True)
        # Anomaly location
        self.anomaly_folder_path = self.work_path.joinpath("./anomaly")
        if self.config.anomaly:
            self.anomaly_folder_path.mkdir(parents=True, exist_ok=True)
        # Lightning checkpoints location
        self.ckpt_filename = "checkpoint"
        self.ckpt_folder_path = self.work_path.joinpath("./checkpoints")
        self.ckpt_folder_path.mkdir(parents=True, exist_ok=True)
        ckpt_files = list(self.ckpt_folder_path.glob(f"{self.ckpt_filename}*.ckpt"))
        try:
            self.ckpt_path = max(ckpt_files, key=lambda p: p.stat().st_ctime) # get most recent
        except ValueError:  # empty sequence
            self.ckpt_path = self.ckpt_folder_path / f"{self.ckpt_filename}.ckpt"

    def _get_latitudes(self, ds: xr.Dataset) -> np.ndarray:
        self.logger.info("Getting latitudes from dataset for loss function...")
        return ds[ds.earthml.guessed_dims.latitude].values

    @staticmethod
    def _get_var_list(exp: MLBCExperimentDataset | Sequence[MLBCExperimentDataset]):
        # TODO I have doubts on this implementation cause we are picking the first test datasource and it may not be representative of the others,
        # but for now we assume all test datasources have the same variable and region selection
        if isinstance(exp, list):
            test_config_datasource = exp[0].datasource[0] if isinstance(exp[0].datasource, list) else exp[0].datasource
            test_config_datasel = test_config_datasource.data_selection
        else:
            test_config_datasel = exp.datasource.data_selection
        return test_config_datasel.variable if isinstance(test_config_datasel.variable, list) else [test_config_datasel.variable]


    def _init_experiment_dataset(
        self,
        exp_store: Path,
        exp_datasets: Sequence[MLBCExperimentDataset],
        role = MLBCExperimentDatasetRole,
    ):
        base_dataset = exp_datasets[0]
        datasource = base_dataset.datasource
        data_selection = (
            datasource[0].data_selection
            if isinstance(datasource, list)
            else datasource.data_selection
        )

        exp = MLBCExperimentDataset(
            role=role,
            datasource=DataSource("xarray-local", deepcopy(data_selection)),
            source_configs=XarrayLocalSourceConfig(
                root_path=exp_store,
                xarray_args={"consolidated": self.consolidated_zarr},
            ),
        )

        return exp


    def _create_xarray_local_source(self, save_path: str | Path, datasource_list: list[DataSource]):
        source_configs = XarrayLocalSourceConfig(
            root_path=save_path,
            xarray_args={
                "consolidated": self.consolidated_zarr,
                "decode_times": True,
            },
        )

        # DO not reuse objects, can create issues later
        combined = deepcopy(sum(datasource_list))
        combined = DataSource(
            source="xarray-local",
            data_selection=deepcopy(combined.data_selection),
        )

        src = build_source(
            "xarray-local",
            params={
                "datasource": combined,
                "config": source_configs,
            },
        )
        return source_configs, src

    def _generate_xarray_ds(
        self,
        source_data: dict[str, BaseSource],
        experiments: MLBCExperimentDataset | Sequence[MLBCExperimentDataset],
        mode: MLBCExperimentMode
    ) -> tuple[xr.Dataset, xr.Dataset]:
        if not isinstance(experiments, list):
            experiments = [experiments]
        exp_roles = [e.role for e in experiments if e.role != MLBCExperimentDatasetRole.PREDICTION]
        ds_d = {}
        s = time.time()

        for role in exp_roles:
            # print(role)
            ds = source_data[role].load()

            # Keep only allowed dims and coords
            allowed_dims = {
                ds.earthml.guessed_dims.time,
                ds.earthml.guessed_dims.latitude,
                ds.earthml.guessed_dims.longitude,
                ds.earthml.guessed_dims.realization,
                # ds.earthml.guessed_dims.leadtime, # leadtime dim must be removed (should be 1D at most)
                # "missed_time", # remove missed_times from the dataset to avoid interference with later torch code
            }
            ds = ds.earthml.remove_dims_and_coords(allowed_dims)

            self.logger.info(f"{mode.capitalize()} {role} dims after removal: {ds.sizes}")

            ds_d[role] = ds

        loading_time = time.time() - s
        self.logger.info("%s loading time: %.1fs", mode.capitalize(), loading_time)

        # For now only support input and target roles
        assert "input" in exp_roles and "target" in exp_roles
        input_ds, target_ds = ds_d[MLBCExperimentDatasetRole.INPUT], ds_d[MLBCExperimentDatasetRole.TARGET]

        # Align shared sample axes, but never silently inner-join spatial float coordinates.
        input_ds, target_ds = self._align_input_target_ds(input_ds, target_ds)

        self._plot_dataset_stage(
            input_ds=input_ds,
            target_ds=target_ds,
            mode=mode,
            stage="input_target",
        )

        return input_ds, target_ds

    def _align_input_target_ds(
        self,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        lat_in = input_ds.earthml.guessed_dims.latitude
        lon_in = input_ds.earthml.guessed_dims.longitude
        lat_tgt = target_ds.earthml.guessed_dims.latitude
        lon_tgt = target_ds.earthml.guessed_dims.longitude

        input_ds, target_ds = self._harmonize_spatial_coords(
            input_ds=input_ds,
            target_ds=target_ds,
            input_dim=lat_in,
            target_dim=lat_tgt,
            axis_name="latitude",
        )
        input_ds, target_ds = self._harmonize_spatial_coords(
            input_ds=input_ds,
            target_ds=target_ds,
            input_dim=lon_in,
            target_dim=lon_tgt,
            axis_name="longitude",
        )

        exclude_align_dims = tuple(
            dim
            for dim in {
                input_ds.earthml.guessed_dims.realization,
                target_ds.earthml.guessed_dims.realization,
                lat_in,
                lon_in,
                lat_tgt,
                lon_tgt,
            }
            if dim is not None
        )

        before_sizes = {"input": dict(input_ds.sizes), "target": dict(target_ds.sizes)}
        input_ds, target_ds = xr.align(
            input_ds,
            target_ds,
            join="inner",
            exclude=exclude_align_dims,
        )
        self.logger.info(
            "Aligned input/target with spatial dims excluded from inner join. "
            "Before sizes: %s | After sizes: %s",
            before_sizes,
            {"input": dict(input_ds.sizes), "target": dict(target_ds.sizes)},
        )
        return input_ds, target_ds

    def _harmonize_spatial_coords(
        self,
        *,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        input_dim: str | None,
        target_dim: str | None,
        axis_name: str,
        atol: float = 1e-5,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        if input_dim is None or target_dim is None:
            return input_ds, target_ds
        if input_dim not in input_ds.coords or target_dim not in target_ds.coords:
            return input_ds, target_ds

        coord_in = np.asarray(input_ds[input_dim].values)
        coord_tgt = np.asarray(target_ds[target_dim].values)

        def _coord_preview(values: np.ndarray) -> dict:
            flat = np.asarray(values, dtype=np.float64).ravel()
            finite = flat[np.isfinite(flat)]
            if finite.size == 0:
                return {"shape": tuple(values.shape), "first": None, "last": None, "min": None, "max": None}
            return {
                "shape": tuple(values.shape),
                "first": float(finite[0]),
                "last": float(finite[-1]),
                "min": float(finite.min()),
                "max": float(finite.max()),
            }

        self.logger.info(
            "Compare input/target %s coordinates before align: input=%s | target=%s",
            axis_name,
            _coord_preview(coord_in),
            _coord_preview(coord_tgt),
        )

        if coord_in.shape != coord_tgt.shape:
            raise ValueError(
                f"Input and target {axis_name} coordinates have different shapes: "
                f"{coord_in.shape} vs {coord_tgt.shape}. Refusing dangerous implicit alignment."
            )

        if np.array_equal(coord_in, coord_tgt):
            return input_ds, target_ds

        if np.allclose(coord_in, coord_tgt, atol=atol, equal_nan=True):
            max_abs_diff = float(np.nanmax(np.abs(coord_in - coord_tgt)))
            self.logger.warning(
                "Input and target %s coordinates differ only numerically "
                "(max abs diff=%s). Normalizing target coords to input coords before alignment.",
                axis_name,
                max_abs_diff,
            )
            target_ds = target_ds.assign_coords({target_dim: input_ds[input_dim].values})
            return input_ds, target_ds

        max_abs_diff = float(np.nanmax(np.abs(coord_in - coord_tgt)))
        raise ValueError(
            f"Input and target {axis_name} coordinates differ materially "
            f"(shape={coord_in.shape}, max_abs_diff={max_abs_diff}). "
            "Refusing dangerous inner alignment on spatial coordinates."
        )

    def _plot_dataset_stage(
        self,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        mode: MLBCExperimentMode,
        stage: str,
    ) -> None:
        """
        Hook for lightweight diagnostic plots during the experiment lifecycle.

        Intended use:
        - keep all plotting calls centralized in train_test.py
        - save artifacts under a train/test folder in the run work path
        - remain a no-op until plotting behavior is implemented
        """
        if not ENABLE_STAGE_PLOTTING:
            self.logger.info("Skip dataset-stage plotting because ENABLE_STAGE_PLOTTING=False")
            return

        plot_specs = [
            {"ds": input_ds, "label": "input", "mean_label": "input mean", "color": "tab:blue"},
            {"ds": target_ds, "label": "target", "mean_label": "target mean", "color": "tab:orange"},
        ]
        lag_steps = self._requested_lag_steps(mode, input_ds)
        run_stage_plot_bundle(
            logger=self.logger,
            plots_folder_path=self.plots_folder_path,
            plot_specs=plot_specs,
            left_ds=input_ds,
            right_ds=target_ds,
            data_type=mode,
            stage=stage,
            stage_kind="dataset",
            residual_label="input-target",
            residual_mean_label="input-target mean",
            lag_steps=lag_steps,
        )

    def _plot_prediction_stage(
        self,
        pred_ds: xr.Dataset,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        mode: MLBCExperimentMode,
        stage: str,
    ) -> None:
        if not ENABLE_STAGE_PLOTTING:
            self.logger.info("Skip prediction-stage plotting because ENABLE_STAGE_PLOTTING=False")
            return

        plot_specs = [
            {"ds": pred_ds, "label": "pred", "mean_label": "pred mean", "color": "tab:green"},
            {"ds": input_ds, "label": "input", "mean_label": "input mean", "color": "tab:blue"},
            {"ds": target_ds, "label": "target", "mean_label": "target mean", "color": "tab:orange"},
        ]
        lag_steps = self._requested_lag_steps(mode, input_ds)
        run_stage_plot_bundle(
            logger=self.logger,
            plots_folder_path=self.plots_folder_path,
            plot_specs=plot_specs,
            left_ds=pred_ds,
            right_ds=target_ds,
            data_type=mode,
            stage=stage,
            stage_kind="prediction",
            residual_label="pred-target",
            residual_mean_label="pred-target mean",
            lag_steps=lag_steps,
        )

    def _requested_lag_steps(
        self,
        mode: MLBCExperimentMode,
        ref_ds: xr.Dataset,
    ) -> int | None:
        datasets_cfg = self.config.train_dataset if mode == MLBCExperimentMode.TRAIN else self.config.test_dataset
        input_cfg = next((ds for ds in datasets_cfg if ds.role == MLBCExperimentDatasetRole.INPUT), None)
        if input_cfg is None:
            return None

        source_configs = input_cfg.source_configs
        if isinstance(source_configs, Sequence) and not isinstance(source_configs, (str, bytes)):
            source_config = source_configs[0]
        else:
            source_config = source_configs

        leadtime_rd = getattr(source_config, "leadtime", None)
        if leadtime_rd is None:
            return None

        time_dim = ref_ds.earthml.guessed_dims.time
        if time_dim is None or ref_ds.sizes.get(time_dim, 0) < 2:
            return None

        try:
            time_vals = np.asarray(ref_ds[time_dim].values).astype("datetime64[ns]")
            deltas = np.diff(time_vals).astype("timedelta64[ns]").astype(np.int64)
            if deltas.size == 0:
                return None
            dt_ns = int(np.median(deltas))
            lt_ns = int(pd.Timedelta(
                days=leadtime_rd.days + leadtime_rd.months * 30 + leadtime_rd.years * 365,
                hours=leadtime_rd.hours,
                minutes=leadtime_rd.minutes,
                seconds=leadtime_rd.seconds,
            ).to_timedelta64().astype("timedelta64[ns]").astype(np.int64))
            if dt_ns <= 0 or lt_ns <= 0:
                return None
            return max(1, int(round(lt_ns / dt_ns)))
        except Exception:
            return None

    def _create_and_save_common_mask(
        self,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        data_type: str,
    ):
        # Create common mask and save
        input_valid_mask = input_ds.earthml.mask()
        target_valid_mask = target_ds.earthml.mask()

        common_mask = (input_valid_mask & target_valid_mask).rename("common_mask")

        mask_ds = xr.Dataset(
            {
                "common_mask": common_mask.astype("uint8"),  # smaller than bool
            }
        )
        mask_ds["common_mask"].attrs.update(
            {
                "description": "Common validity mask for aligned input and target datasets",
                "values": "1=valid, 0=masked",
            }
        )

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = {"common_mask": {"compressors": compressor}}

        if data_type == "train":
            self.logger.info("Save mask to %s", self.train_mask_store)
            mask_ds.to_zarr(self.train_mask_store, encoding=encoding_zarr, mode="w", consolidated=self.consolidated_zarr)
        elif data_type == "test":
            self.logger.info("Save mask to %s", self.test_mask_store)
            mask_ds.to_zarr(self.test_mask_store, encoding=encoding_zarr, mode="w", consolidated=self.consolidated_zarr)
        else:
            raise ValueError(f"Unknown data_type {data_type} for mask saving")

        return mask_ds

    def _init_source_data(
        self,
        exp_ds: MLBCExperimentDataset | Sequence[MLBCExperimentDataset],
        source_type: str
    ) -> dict[MLBCExperimentDatasetRole, BaseSource]:
        """Returns dictionary of populated Source instances"""

        def _to_list(data):
            if isinstance(data, list):
                return data
            if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
                return list(data)
            return [data]

        # Normalize to list
        exp_ds = _to_list(exp_ds)

        sources: dict[MLBCExperimentDatasetRole, BaseSource] = {}
        missed = set()
        for e in exp_ds:
            # Guard for duplicate roles
            if e.role in sources:
                raise ValueError(f"Duplicate role encountered: {e.role}")

            datasource = e.datasource
            source_configs = e.source_configs

            # Normalize to lists
            datasource = _to_list(datasource)
            source_configs = _to_list(source_configs)
            if not datasource:
                raise ValueError(f"No datasource entries provided for role={e.role}")

            save_path = self.config.work_path.joinpath(Path(f"{source_type}_{e.role}")).with_suffix(".zarr")

            # Check if dataset already exists in Zarr format
            is_zarr_store = save_path.is_dir() and any(
                save_path.joinpath(marker).exists()
                for marker in ("zarr.json",) # only supports Zarr v3
                # for marker in ("zarr.json", ".zgroup", ".zmetadata") # supports also older Zarr versions
            )

            if is_zarr_store:
                xr_loc_source_configs, sources[e.role] = self._create_xarray_local_source(save_path, datasource)
                log_renderable(
                    Table({f"Source '{sources[e.role].datasource.source}' {source_type} {e.role} params": asdict(xr_loc_source_configs)}, twocols=True).table,
                    logger=self.logger,
                )
            else:
                if len(datasource) != len(source_configs):
                    raise ValueError(
                        f"datasource and source_configs length mismatch for role={e.role}: "
                        f"{len(datasource)} != {len(source_configs)}"
                    )

                sources_list: list[BaseSource] = []
                for i, (d, sc) in enumerate(zip(datasource, source_configs)):
                    sources_list.append(
                        build_source(
                            d.source,
                            params=dict(
                                datasource=d,
                                config=sc,
                            ),
                        )
                    )
                    log_renderable(
                        Table({f"Source '{d.source}' {source_type} {e.role} [{i}] params": asdict(sc)}, twocols=True).table,
                        logger=self.logger,
                    )
                source: BaseSource = sum(sources_list)

                # Union of missing samples
                if e.role != MLBCExperimentDatasetRole.PREDICTION: # preds exists as folder but cannot be materialized
                    ds = source.load() # this is the first load!
                    source_missed = getattr(source.elements, "missed", None)
                    if source_missed:
                        missed |= set(source_missed)
                        if "missed_time" in ds:
                            missed |= set(pd.to_datetime(ds["missed_time"].values).to_pydatetime())

                # Save datasets if requested
                if e.save:
                    source.save(save_path) # TODO examine, save may be not atomic
                    # Regenerate as xarray-local source type
                    xr_loc_source_configs, sources[e.role] = self._create_xarray_local_source(save_path, datasource)
                    log_renderable(
                        Table({f"Source '{sources[e.role].datasource.source}' {source_type} {e.role} params": asdict(xr_loc_source_configs)}, twocols=True).table,
                        logger=self.logger,
                    )
                else:
                    sources[e.role] = source

        self.logger.info("Combined missed dates (%s): %s", source_type, missed)

        # Reload to remove union of missed elements from all non prediction  datasets (for dimension consistency)
        if missed: # TODO this logic may be a bit weak
            for role, source in sources.items():
                if role != MLBCExperimentDatasetRole.PREDICTION:
                    sources[role].elements.missed = missed # TODO we shouldn't update the internal source state here, but for now it's ok
                    sources[role].reload()
                    # _ = ds.sizes # just to trigger reload

        return sources


    # Anomaly helpers
    def _init_anomaly_climatologies(self):
        self.logger.info("Calculate target and input climatologies from train dataset...")

        self.target_clim_ts = self.train_target_ds.earthml.climatology(groupby="month")
        self.input_clim_ts = self.train_input_ds.earthml.climatology(groupby="month")

    def _calculate_anomaly(
        self,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        if self.target_clim_ts is None or self.input_clim_ts is None:
            raise RuntimeError("Anomaly climatologies were not initialized from the Train dataset.")

        target_ds = target_ds.groupby(f"{target_ds.earthml.guessed_dims.time}.month") - self.target_clim_ts
        input_ds = input_ds.groupby(f"{input_ds.earthml.guessed_dims.time}.month") - self.input_clim_ts
        return input_ds, target_ds

    def _save_anomaly_datasets(
        self,
        input_anom_ds: xr.Dataset,
        target_anom_ds: xr.Dataset,
        data_type: str,
    ):
        for role, ds in (("input", input_anom_ds), ("target", target_anom_ds)):
            anomaly_store = self.anomaly_folder_path.joinpath(f"{data_type.lower()}_{role}.zarr")
            ds.to_zarr(anomaly_store, mode="w", consolidated=self.consolidated_zarr)


    def _init_callbacks(self):
        # Initialize trainer callbacks
        callbacks = []
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.config.net.earlystopping_patience,
            verbose=True,
            mode="min"
        )
        callbacks.append(early_stop_callback)
        # Checkpointing every N epochs
        periodic_checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_folder_path,
            every_n_epochs=1,
            # filename="checkpoint_{epoch:02d}-{val_loss:.2f}"
            filename=self.ckpt_filename
        )
        callbacks.append(periodic_checkpoint_callback)
        # Model best weights, MUST be the last ModelCheckpoint. Can append from here since ModelCheckpoint will always be the last callbacks
        best_weights_callback = ModelCheckpoint(
            dirpath=self.weights_folder_path,
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            filename=self.weights_filename
        )
        callbacks.append(best_weights_callback)
        return callbacks

    def _init_train_trainer(self):
        self._configure_torch_runtime()
        # Initialize Lightning trainer
        return L.Trainer(
            max_epochs=self.config.net.epochs,
            accelerator=self.accelerator,
            devices=1,
            precision="16-mixed",
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            log_every_n_steps=1,
            logger=self.tl_logger,
            accumulate_grad_batches=self.config.net.accumulate_grad_batches,
            callbacks=self._init_callbacks(),
            # deterministic=True
        )

    def _init_test_trainer(self):
        self._configure_torch_runtime()
        return L.Trainer(
            max_epochs=self.config.net.epochs,
            accelerator=self.accelerator,
            devices=1,
            precision="16-mixed",
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            accumulate_grad_batches=self.config.net.accumulate_grad_batches,
            # deterministic=True
        )


    def _generate_torch_dataset(
        self,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        source_data: dict[str, BaseSource],
        mode: MLBCExperimentMode
    ):
        # Create torch dataset
        torch_dataset = XarrayDataset(
            input_ds, target_ds,
            target_realization_avg=self.config.target_realization_avg,
            realization_as_channel=self.config.realization_as_channel,
            output_realizations=self.config.output_realizations,
        )

        # Compute stats
        x_stats = Normalize._masked_metrics(
            pred=torch_dataset.x,
            target=torch_dataset.x,
            pred_mask=torch_dataset.x_mask,
            target_mask=torch_dataset.x_mask,
            metric_mask=torch_dataset.x_mask,
            per_channel_mean=self.per_channel_masked_mean,
        )

        y_stats = Normalize._masked_metrics(
            pred=torch_dataset.y,
            target=torch_dataset.y,
            pred_mask=torch_dataset.y_mask,
            target_mask=torch_dataset.y_mask,
            metric_mask=torch_dataset.y_mask,
            per_channel_mean=self.per_channel_masked_mean,
        )

        x_mean, x_std = x_stats["target_mean"], x_stats["target_std"]
        y_mean, y_std = y_stats["target_mean"], y_stats["target_std"]
        x_mean_np, x_std_np = np.nanmean(torch_dataset.x_np), np.nanstd(torch_dataset.x_np)
        y_mean_np, y_std_np = np.nanmean(torch_dataset.y_np), np.nanstd(torch_dataset.y_np)

        # Count masked/invalid elements
        x_masked = (~torch_dataset.x_mask).sum().item()
        y_masked = (~torch_dataset.y_mask).sum().item()
        x_masked_np = (~torch_dataset.mask_x_np).sum()
        y_masked_np = (~torch_dataset.mask_y_np).sum()

        x_total = torch_dataset.x_mask.numel()
        y_total = torch_dataset.y_mask.numel()
        x_total_np = torch_dataset.mask_x_np.size
        y_total_np = torch_dataset.mask_y_np.size

        if x_total == 0 or y_total == 0:
            raise ValueError("Mask tensor is empty: check dataset construction")

        log_renderable(Table({f'{mode.capitalize()} dataset': {
            MLBCExperimentDatasetRole.INPUT: {
                'numpy shape': torch_dataset.x_np.shape,
                'numpy mean': x_mean_np,
                'numpy std': x_std_np,
                'numpy masked_elements': x_masked_np,
                'numpy masked_fraction': x_masked_np / x_total_np,
                'torch shape': torch_dataset.x.shape,
                'torch mean': x_mean,
                'torch std': x_std,
                'torch masked_elements': x_masked,
                'torch masked_fraction': x_masked / x_total,
                'source': source_data[MLBCExperimentDatasetRole.INPUT].datasource.source,
            },
            MLBCExperimentDatasetRole.TARGET: {
                'numpy shape': torch_dataset.y_np.shape,
                'numpy mean': y_mean_np,
                'numpy std': y_std_np,
                'numpy masked_elements': y_masked_np,
                'numpy masked_fraction': y_masked_np / y_total_np,
                'torch shape': torch_dataset.y.shape,
                'torch mean': y_mean,
                'torch std': y_std,
                'torch masked_elements': y_masked,
                'torch masked_fraction': y_masked / y_total,
                'source': source_data[MLBCExperimentDatasetRole.TARGET].datasource.source,
            },
        }}).table, logger=self.logger)

        return torch_dataset

    @staticmethod
    def _to_float(x):
        try:
            return float(x.item()) if hasattr(x, "item") else float(x)
        except Exception:
            return x

    def _make_test_info_table(
        self,
        test_dataset,
        source_test_data,
        preds_norm,
        preds_rescaled,
        test_var_list: Sequence,
    ):
        """
        Returns:
        meta: dict (run metadata)
        var_cols: dict keyed by variable name with shapes + masked metrics

        Notes:
        - Uses Normalize._masked_metrics(pred, target, pred_mask=..., target_mask=..., metric_mask=...)
        - For "stats" on x/y alone, calls with pred==target so mae/rmse/etc are defined (0) but you mainly read *_mean/*_std
        - Prediction metrics are computed vs y, using y_mask by default (or intersection if you pass both masks)
        """
        var_cols = {}

        def _pack(
                name: str,
            x: torch.Tensor, x_mask: torch.Tensor,
            y: torch.Tensor, y_mask: torch.Tensor,
            pn: torch.Tensor, pr: torch.Tensor
        ):
            # Masked stats (means/stds) for x and y (computed on their own masks)
            x_stats = Normalize._masked_metrics(
                pred=x, target=x,
                pred_mask=x_mask, target_mask=x_mask, metric_mask=x_mask,
                per_channel_mean=self.per_channel_masked_mean,
            )
            y_stats = Normalize._masked_metrics(
                pred=y, target=y,
                pred_mask=y_mask, target_mask=y_mask, metric_mask=y_mask,
                per_channel_mean=self.per_channel_masked_mean,
            )

            # Pred stats (means/stds) on x_mask (usually where the input is valid)
            pn_stats = Normalize._masked_metrics(
                pred=pn, target=pn,
                pred_mask=x_mask, target_mask=x_mask, metric_mask=x_mask,
                per_channel_mean=self.per_channel_masked_mean,
            )
            pr_stats = Normalize._masked_metrics(
                pred=pr, target=pr,
                pred_mask=x_mask, target_mask=x_mask, metric_mask=x_mask,
                per_channel_mean=self.per_channel_masked_mean,
            )

            # Pred vs target metrics on an "intelligent" mask:
            # use intersection of x/y validity if both exist; the helper will do this if metric_mask=None.
            pred_vs_y = Normalize._masked_metrics(
                pred=pr, target=y,
                pred_mask=x_mask, target_mask=y_mask, metric_mask=None,
                per_channel_mean=self.per_channel_masked_mean,
            )

            var_cols[name] = {
                "x shape": tuple(x.shape),
                "x mean(masked)": x_stats["target_mean"],
                "x std(masked)": x_stats["target_std"],

                "y shape": tuple(y.shape),
                "y mean(masked)": y_stats["target_mean"],
                "y std(masked)": y_stats["target_std"],

                "pred(norm) shape": tuple(pn.shape),
                "pred(norm) mean(masked)": pn_stats["pred_mean"],
                "pred(norm) std(masked)": pn_stats["pred_std"],

                "pred shape": tuple(pr.shape),
                "pred mean(masked)": pr_stats["pred_mean"],
                "pred std(masked)": pr_stats["pred_std"],

                # full masked prediction metrics
                "mae(y,pred)": pred_vs_y["mae"],
                "rmse(y,pred)": pred_vs_y["rmse"],
                "r2(y,pred)": pred_vs_y["r2"],
                "corr(y,pred)": pred_vs_y["corr"],
            }

        if self.config.realization_as_channel:
            # only 1 var support for R as C
            name = getattr(next(iter(test_var_list)), "name", str(next(iter(test_var_list))))
            _pack(
                name=name,
                x=test_dataset.x, x_mask=test_dataset.x_mask,
                y=test_dataset.y, y_mask=test_dataset.y_mask,
                pn=preds_norm, pr=preds_rescaled,
            )
        else:
            for i, var in enumerate(test_var_list):
                name = getattr(var, "name", str(var))
                _pack(
                    name=name,
                    x=test_dataset.x[:, i:i+1, :, :], x_mask=test_dataset.x_mask[:, i:i+1, :, :],
                    y=test_dataset.y[:, i:i+1, :, :], y_mask=test_dataset.y_mask[:, i:i+1, :, :],
                    pn=preds_norm[:, i:i+1, :, :], pr=preds_rescaled[:, i:i+1, :, :],
                )

        meta = {
            "device": str(self.device),
            "torch_workers": self.torch_workers,
            "input source": getattr(source_test_data["input"].datasource, "source", ""),
            "target source": getattr(source_test_data["target"].datasource, "source", ""),
            "ckpt_path": str(getattr(self, "ckpt_path", "")),
            "weights_path": str(getattr(self, "weights_path", "")),
        }

        return meta, var_cols

    def train(self):
        # Generate torch train dataset
        train_dataset = self._generate_torch_dataset(self.train_input_ds, self.train_target_ds, self.source_train_data, MLBCExperimentMode.TRAIN)

        # Normalize
        self.normalize_input  = Normalize().fit(train_dataset, filepath=self.normdata_input_path, dim='x')  # mean and std of train input (x) data
        self.normalize_target = Normalize().fit(train_dataset, filepath=self.normdata_target_path, dim='y') # mean and std of train target (y) data
        # print(f"Normalize type: {type(self.normalize.mean)} and shape: {self.normalize.mean.shape}")
        train_dataset.transform_x = self.normalize_input
        train_dataset.transform_y = self.normalize_target

        # Create train datamodule and split train datase\t into train and validation based on self.config.train_percent
        self.train_datamodule = EpochRandomSplitDataModule(
            train_dataset,
            train_fraction=self.config.net.train_percent,
            batch_size=self.config.net.batch_size,
            seed=self.config.net.seed,
            num_workers=self.torch_workers,
            per_epoch_resplit=self.config.per_epoch_resplit,
        )

        # Train
        trainer = self._init_train_trainer()
        ckpt_path = Path(self.ckpt_path) if Path(self.ckpt_path).exists() else None
        if ckpt_path is None:
            self.logger.info("Starting training from scratch")
        trainer.fit(
            self.model,
            datamodule=self.train_datamodule,
            ckpt_path=ckpt_path,
        )

    def _test(
        self,
        test_data_mode: MLBCExperimentMode,
        test_data: dict[str, BaseSource],
        test_input_ds: xr.Dataset,
        test_target_ds: xr.Dataset,
        preds_store: Path,
        var_list: Sequence,
        weights_filename: str | Path | None = None,
    ):
        dataset = self._generate_torch_dataset(test_input_ds, test_target_ds, test_data, test_data_mode)

        # Normalize input
        if not self.normalize_input:
            self.logger.info("Load normalization data from %s", self.normdata_input_path)
            self.normalize_input = Normalize().load(self.normdata_input_path)
        dataset.transform_x = self.normalize_input

        # Normalize target
        if not self.normalize_target:
            self.logger.info("Load normalization data from %s", self.normdata_target_path)
            self.normalize_target = Normalize().load(self.normdata_target_path)
        dataset.transform_y = self.normalize_target

        # Create test dataloader
        dataloader = DataLoader(dataset, batch_size=1, num_workers=self.torch_workers, shuffle=False)
        if not weights_filename:
            # Load checkpoint file
            self.logger.info("Load checkpoints from %s", self.ckpt_path)
            checkpoint = torch.load(self.ckpt_path, map_location=self.device)
            # print(f"Available keys in checkpoint file: {checkpoint.keys()}")
            # dict_keys(['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'MixedPrecision'])
            # state_dict = checkpoint["state_dict"]
            # print(f"State dict: {state_dict.keys()}")
            callbacks = checkpoint["callbacks"]
            # print(f"Callback: {dict(callbacks)}")
            last_key, last_callback = next(reversed(callbacks.items()))
            # print(f"Last callback: {last_key}")
            # print(f"Available keys in callback {last_key}: {last_callback.keys()}")
            # dict_keys(['monitor', 'best_model_score', 'best_model_path', 'current_score', 'dirpath', 'best_k_models', 'kth_best_model_path', 'kth_value', 'last_model_path'])
            weights_file = Path(last_callback["best_model_path"])
            if not weights_file.is_file(): # fallback to config weights filename
                weights_file = Path(self.weights_path)
        else:
            weights_file = Path(weights_filename)

        # Load weights
        self.logger.info("Load weights from file: %s", weights_file)
        weights = torch.load(weights_file, map_location=self.device)
        self.model.load_state_dict(weights['state_dict'])

        # Test
        self._init_test_trainer().test(self.model, dataloaders=dataloader)
        # print(f"Available attributes in model: {dir(self.model)}")

        # Rescale preds with target normalization
        preds_norm = self.model.test_preds.float()
        preds = self.normalize_target.inverse_tensor(preds_norm, self.normdata_target_path) # .squeeze()

        # Print info
        meta, var_cols = self._make_test_info_table(dataset, test_data, preds_norm, preds, var_list)
        log_renderable(Table({f"Test on {test_data_mode.capitalize()} dataset run info": meta}, twocols=True).table, logger=self.logger)
        log_renderable(Table({f"Test on {test_data_mode.capitalize()} dataset metrics (per variable)": var_cols}).table, logger=self.logger)

        pred_ds = self.save(preds, dataset, var_list, preds_store)
        self._plot_prediction_stage(
            pred_ds=pred_ds,
            input_ds=test_input_ds,
            target_ds=test_target_ds,
            mode=test_data_mode,
            stage="test_preds",
        )

        return dataloader

    def test(self, weights_filename=None):
        self.test_dataloader  = self._test(MLBCExperimentMode.TEST, self.source_test_data, self.test_input_ds, self.test_target_ds, self.test_preds_store, self.test_var_list, weights_filename)

    def test_on_train(self, weights_filename=None):
        self.test_dataloader  = self._test(MLBCExperimentMode.TRAIN, self.source_train_data, self.train_input_ds, self.train_target_ds, self.train_preds_store, self.train_var_list, weights_filename)

    def _infer_RT_from_source(self, dataset: XarrayDataset) -> tuple[int, int]:
        """
        Infer (R_out, T_out) from the original xarray datasets attached to the torch dataset.
        """
        input_ds = dataset.input_ds
        target_ds = dataset.target_ds

        tdim_tgt = target_ds.earthml.guessed_dims.time
        T_out = int(target_ds.sizes.get(tdim_tgt, 1))

        # realization sizes (safe even if no realization dim exists)
        rdim_in = input_ds.earthml.guessed_dims.realization
        rdim_tgt = target_ds.earthml.guessed_dims.realization
        R_in = int(input_ds.sizes.get(rdim_in, 1))
        R_tgt = int(target_ds.sizes.get(rdim_tgt, 1))

        R_out = R_in

        if R_out < 1:
            raise ValueError(f"Inferred invalid R_out={R_out}")
        if T_out < 1:
            raise ValueError(f"Inferred invalid T_out={T_out}")

        return R_out, T_out

    def _reconstruct_pred_tensor(
        self,
        data: torch.Tensor,     # (N,C,H,W)
        meta_ds: xr.Dataset,    # ONLY for slicing/copying coords/attrs
        R_out: int,
        T_out: int,
    ) -> tuple[torch.Tensor, xr.Dataset]:
        """
        Canonical output ALWAYS (C, R_out, T_out, H, W).
        If output is deterministic, ensures R_out==1 and adds singleton realization dim.
        """
        if data.ndim != 4:
            raise ValueError(f"Expected preds as (N,C,H,W), got {tuple(data.shape)}")

        # (C,N,H,W)
        x = data.permute(1, 0, 2, 3).contiguous()
        C_model, N, H, W = x.shape

        if self.config.realization_as_channel:
            # Here N should be time
            if N != T_out:
                raise ValueError(f"realization_as_channel=True expects N==T ({N} != {T_out})")

            if R_out == 1:
                x = x.unsqueeze(1)  # (C,1,T,H,W)
            else:
                # channels contain realizations: C_model == C * R_out
                if C_model % R_out != 0:
                    raise ValueError(f"C_model={C_model} not divisible by R_out={R_out}")
                C = C_model // R_out
                # (C*R, T, H, W) -> (R, C, T, H, W) -> (C, R, T, H, W)
                x = x.unflatten(0, (R_out, C)).permute(1, 0, 2, 3, 4).contiguous()

        else:
            # Here realizations are flattened into N if R_out>1, else N is time
            if R_out == 1:
                if N != T_out:
                    raise ValueError(f"deterministic expects N==T ({N} != {T_out})")
                x = x.unsqueeze(1)  # (C,1,T,H,W)
            else:
                if N != T_out * R_out:
                    raise ValueError(f"ensemble expects N==T*R ({N} != {T_out*R_out})")
                # time-major flatten (t slow, r fast): (C,N,H,W)->(C,T,R,H,W)->(C,R,T,H,W)
                x = x.unflatten(1, (T_out, R_out)).permute(0, 2, 1, 3, 4).contiguous()

        # meta_ds only sliced to match inferred sizes; never used to infer them
        tdim = meta_ds.earthml.guessed_dims.time
        rdim = meta_ds.earthml.guessed_dims.realization
        ydim = meta_ds.earthml.guessed_dims.latitude
        xdim = meta_ds.earthml.guessed_dims.longitude

        indexers: dict[str, slice] = {}
        if rdim in meta_ds.dims:
            indexers[rdim] = slice(0, R_out)
        if tdim in meta_ds.dims:
            indexers[tdim] = slice(0, T_out)
        if ydim in meta_ds.dims:
            indexers[ydim] = slice(0, H)
        if xdim in meta_ds.dims:
            indexers[xdim] = slice(0, W)
        if indexers:
            meta_ds = meta_ds.isel(indexers)

        return x, meta_ds

    def save(
        self,
        data: torch.Tensor,
        dataset: XarrayDataset,
        var_list: Sequence,
        preds_store: Path
    ):
        # Predictions live on the target grid, so keep target metadata/coords.
        meta_ds = dataset.target_ds

        tdim = meta_ds.earthml.guessed_dims.time
        rdim = meta_ds.earthml.guessed_dims.realization
        ydim = meta_ds.earthml.guessed_dims.latitude
        xdim = meta_ds.earthml.guessed_dims.longitude

        allowed_dims = {tdim, ydim, xdim, rdim, "missed_time"}
        meta_ds = meta_ds.earthml.remove_dims_and_coords(allowed_dims)
        base_order = [rdim, tdim, ydim, xdim]
        order = [d for d in base_order if d in meta_ds.dims]
        order += [d for d in meta_ds.dims if d not in order]
        meta_ds = meta_ds.transpose(*order, missing_dims="ignore")

        R_out, T_out = self._infer_RT_from_source(dataset)

        input_ds = dataset.input_ds
        input_rdim = input_ds.earthml.guessed_dims.realization
        input_has_matching_r = (
            input_rdim is not None
            and input_rdim in input_ds.dims
            and int(input_ds.sizes[input_rdim]) == R_out
        )

        # Ensemble predictions should follow the input realization convention
        # when the target metadata is deterministic/singleton.
        if (
            R_out > 1
            and rdim is not None
            and (
                rdim not in meta_ds.dims
                or int(meta_ds.sizes.get(rdim, 1)) != R_out
            )
        ):
            if not input_has_matching_r:
                raise ValueError(
                    "Cannot reconstruct ensemble prediction coordinates: "
                    f"target realization dim '{rdim}' has size "
                    f"{meta_ds.sizes.get(rdim, 'missing')} but expected {R_out}, "
                    f"and input realization dim '{input_rdim}' is unavailable."
                )

            if rdim in meta_ds.dims:
                meta_ds = meta_ds.drop_dims(rdim)

            input_rcoord = input_ds.coords.get(input_rdim)
            if input_rcoord is not None:
                if input_rdim != rdim:
                    input_rcoord = input_rcoord.rename({input_rdim: rdim})
                meta_ds = meta_ds.assign_coords({rdim: input_rcoord})
            else:
                meta_ds = meta_ds.assign_coords({rdim: np.arange(R_out, dtype=np.int64)})

            base_order = [rdim, tdim, ydim, xdim]
            order = [d for d in base_order if d in meta_ds.dims]
            order += [d for d in meta_ds.dims if d not in order]
            meta_ds = meta_ds.transpose(*order, missing_dims="ignore")

        data_vars, meta_ds = self._reconstruct_pred_tensor(data, meta_ds, R_out=R_out, T_out=T_out)

        # Use meta dim names if present; otherwise drop them safely.
        dims = []
        if rdim in meta_ds.dims:
            dims.append(rdim)
        dims += [d for d in (tdim, ydim, xdim) if d in meta_ds.dims]

        # If metadata lacks realization dim, write deterministic view
        if rdim not in meta_ds.dims:
            data_vars = data_vars.squeeze(1)  # (C,T,H,W)

        ds = xr.Dataset(
            {
                var.name: (dims, data_vars[i].cpu().numpy())
                for i, var in enumerate(var_list)
            },
            coords={c: meta_ds.coords[c] for c in meta_ds.coords},
            attrs=meta_ds.attrs,
        )

        # Restore with target climatology if trained on anomalies
        # print("PREDS vars:", ds.data_vars)
        if self.config.anomaly:
            # print("TARGET CLIM vars:", self.target_clim_ts.data_vars)
            # Rename clim vars
            assert len(self.target_clim_ts.data_vars) == len(ds.data_vars)
            clim_ds = self.target_clim_ts.rename({
                var_clim: var
                for var_clim, var in zip(self.target_clim_ts.data_vars, ds.data_vars)
            })
            ds = ds.groupby(f"{tdim}.month") + clim_ds

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = {v.name: {"compressors": compressor} for v in var_list}

        self.logger.info("Save preds to %s", preds_store)
        ds.to_zarr(preds_store, encoding=encoding_zarr, mode="w", consolidated=self.consolidated_zarr)
        return ds
