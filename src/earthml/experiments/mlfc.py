import time, multiprocessing, joblib
from pathlib import Path
from typing import List, Dict, Tuple
from rich import print
from rich.console import Console
import numpy as np
import pandas as pd
import xarray as xr
from zarr.codecs import BloscCodec
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from ..sources.registry import build_source
from ..sources.base import BaseSource
from ..dataclasses import ExperimentDataset, ExperimentConfig, DataSource
from ..utils import Table, print_ds_info, guess_time_dim, guess_lon_dim, guess_lat_dim, guess_realization_dim, guess_leadtime_dim, remove_unwanted_dims_and_coords
from ..lightning import XarrayDataset, Normalize, EpochRandomSplitDataModule
from ..nets.registry import build_net

class ExperimentMLFC:
    def __init__(
        self,
        config: ExperimentConfig,
    ):
        self._configure_torch_env()
        self.config = config
        self.rich_console = Console()

        self.per_channel_masked_mean = False if self.config.output_realizations else True # TODO not sure it's super general

        # Get test variable list,
        # TODO I have doubts on this implementation cause we are picking the first test datasource and it may not be representative of the others,
        # but for now we assume all test datasources have the same variable and region selection
        if isinstance(self.config.test, list):
            test_config_datasource = self.config.test[0].datasource[0] if isinstance(self.config.test[0].datasource, list) else self.config.test[0].datasource
            test_config_datasel = test_config_datasource.data_selection
        else:
            test_config_datasel = self.config.test.datasource.data_selection
        self.test_var_list = test_config_datasel.variable if isinstance(test_config_datasel.variable, list) else [test_config_datasel.variable]

        # Setup paths and make dirs if necessary
        self._path_setup()

        # General torch and Lightning setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_cpus, num_gpus = multiprocessing.cpu_count(), torch.cuda.device_count() if torch.cuda.is_available() else 1 # torch.backends.mps.is_available()
        self.torch_workers = max(1, num_cpus // num_gpus // 2) if not torch.backends.mps.is_available() else 1 # could be moved in datamodule definition, but in test() still using dataloader
        print(f"Torch workers in use: {self.torch_workers} ({num_cpus} CPUs // {num_gpus} GPU (CUDA or others) devices // 2)")
        L.seed_everything(self.config.seed)
        # Tensorboard
        self.tl_logger = TensorBoardLogger(self.config.work_path, name="tensorboard_logs") # version=self.run_number

        # Init
        self.normalize = None
        self.train_datamodule = None

        # Init predictions
        preds_filename = "test_preds"
        self.consolidated_zarr = False
        self.preds_store = self.config.work_path.joinpath(Path(preds_filename).with_suffix(".zarr"))
        preds_exp = ExperimentDataset(
            role='prediction',
            datasource=DataSource(
                "xarray-local",
                self.config.test[0].datasource[0].data_selection if isinstance(self.config.test[0].datasource, list) else self.config.test[0].datasource.data_selection,
            ),
            source_params={
                'root_path': self.preds_store,
                'xarray_args': {'consolidated': self.consolidated_zarr}
            }
        )
        self.config.test.append(preds_exp)

        # Init source data objects
        self.source_train_data = self._init_source_data(self.config.train, "train")
        self.source_test_data  = self._init_source_data(self.config.test,  "test")

        # Handle latitudes for optional spatial weighting
        self.latitudes = None
        loss_cfg = self.config.loss_params.get("loss", {})  # safe access
        net_cfg = self.config.loss_params.get("net", {})

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

        # Build merged loss_params
        base_loss_params = dict(self.config.loss_params)  # shallow copy
        nested_loss = dict(base_loss_params.get("loss", {}))
        if self.latitudes is not None:
            nested_loss["latitudes"] = self.latitudes
        base_loss_params["loss"] = nested_loss
        base_loss_params["net"] = net_cfg

        # Initialize model
        self.model = build_net(
            self.config.net,
            learning_rate=self.config.learning_rate,
            loss=self.config.loss,
            loss_params=base_loss_params,
            norm=self.config.norm_strategy,
            supervised=self.config.supervised,
            **self.config.extra_net_args,
        ).to(self.device)

        # Log model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Net {type(self.model).__name__} trainable parameters: {trainable_params:,}")

        self.rich_console.print(Table(self.config, title="ExperimentConfig", twocols=True).table)

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
            }, f)

    def _configure_torch_env (self):
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def _configure_torch_runtime (self):
        torch.set_float32_matmul_precision('medium')  # or 'high'
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

    def _path_setup (self):
        self.work_path = Path(self.config.work_path)
        # Weights location
        self.weights_folder_path = self.work_path.joinpath("./weights")
        self.weights_filename = f"{self.config.name}_weights"
        self.weights_path = self.weights_folder_path.joinpath(self.weights_filename+'.ckpt')
        # Train dataset normalization data location
        norm_data_folder_path = self.work_path.joinpath("./normdata")
        normdata_input_filename = f"{self.config.name}_normdata_input.gz"
        normdata_target_filename = f"{self.config.name}_normdata_target.gz"
        norm_data_folder_path.mkdir(parents=True, exist_ok=True)
        self.normdata_input_path = norm_data_folder_path.joinpath(normdata_input_filename)
        self.normdata_target_path = norm_data_folder_path.joinpath(normdata_target_filename)
        # Lightning checkpoints location
        self.ckpt_filename = "checkpoint"
        self.ckpt_folder_path = self.work_path.joinpath("./checkpoints")
        self.ckpt_folder_path.mkdir(parents=True, exist_ok=True)
        ckpt_files = self.ckpt_folder_path.glob(f"{self.ckpt_filename}*.ckpt")
        try:
            self.ckpt_path = max(ckpt_files, key=lambda item: item.stat().st_ctime) # get most recent
        except:
            self.ckpt_path = self.ckpt_folder_path.joinpath(f"{self.ckpt_filename}.ckpt")

    @staticmethod
    def _get_latitudes (ds: xr.Dataset) -> np.ndarray:
        print("Getting latitudes from dataset for loss function...")
        return ds[guess_lat_dim(ds)].values

    def _remove_corrupted_ts (self, ds: xr.Dataset) -> Tuple[xr.Dataset, set]:
        time_dim = guess_time_dim(ds)
        if time_dim is None:
            return ds, set()

        keep = xr.ones_like(ds[time_dim], dtype=bool)

        for var in [v for v in ds.data_vars if v != "_has_var"]:
            da = ds[var]
            if time_dim not in da.dims:
                continue

            reduce_dims = [d for d in da.dims if d != time_dim]

            # Select corrupted timestep, mean over non-time dims is NaN
            ts_mean = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
            keep = keep & ts_mean.notnull()

        # Compute only the 1D mask (cheaper)
        keep = keep.compute() if hasattr(keep.data, "compute") else keep

        # Return a python set of python datetimes
        missed_sel = ds[time_dim].where(~keep, drop=True)
        missed = set(pd.to_datetime(missed_sel.values).to_pydatetime()) if missed_sel.values.size else set()

        # Drop corrupted timesteps
        ds = ds.drop_sel({time_dim: missed_sel})

        # Add missed info to dataset
        if "missed_time" in ds.coords:
            prev_missed = set(ds["missed_time"].values.tolist())
        else:
            prev_missed = set()
        missed_np = np.array(sorted(missed | prev_missed), dtype="datetime64[ns]")
        ds = ds.assign_coords(missed_time=("missed_time", missed_np))
        ds["missed_time"].encoding.update({
            "units": "nanoseconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        })

        return ds, missed

    def _create_xarray_local_source (self, save_path: str | Path, datasource_list: list[DataSource]):
        source_params = dict(
            root_path=save_path,
            xarray_args={
                "consolidated": self.consolidated_zarr,
                "decode_times": True,
            },
        )

        datasource_sum = sum(datasource_list)
        datasource_sum.source = "xarray-local"

        src = build_source(
            "xarray-local",
            datasource=datasource_sum,
            **source_params,
        )
        return source_params, src

    def _init_source_data (self, exp_ds: ExperimentDataset | List[ExperimentDataset], source_type: str):
        """Returns populated Source instances"""

        # Normalize to list
        if not isinstance(exp_ds, list):
            exp_ds = [exp_ds]

        sources = {}
        for e in exp_ds:
            datasource = e.datasource
            source_params = e.source_params

            # Normalize to lists
            if not isinstance(datasource, list) and not isinstance(source_params, list):
                datasource = [datasource]
                source_params = [source_params]

            save_path = self.config.work_path.joinpath(Path(f"{source_type}_{e.role}")).with_suffix(".zarr")

            if save_path.exists(): # TODO weak check, make more robust, like check it is really a zarr store
                xr_loc_source_params, sources[e.role] = self._create_xarray_local_source(save_path, datasource)
                self.rich_console.print(Table({f"Source '{sources[e.role].datasource.source}' {source_type} {e.role} params": xr_loc_source_params}, twocols=True).table)
            else:
                sources_list: list[BaseSource] = []
                for i, (d, sp) in enumerate(zip(datasource, source_params)):
                    sources_list.append(
                        build_source(
                            d.source,
                            datasource=d,
                            **sp,
                        )
                    )
                    self.rich_console.print(Table({f"Source '{d.source}' {source_type} {e.role} [{i}] params": sp}, twocols=True).table)
                source = sum(sources_list)

                # Remove corrupted timesteps only for input and target
                if e.role != "prediction":
                    with source.load() as ds:
                        ds_sel, corrupted = self._remove_corrupted_ts(ds)
                        print(f"Corrupted timesteps: {corrupted}")
                        source.ds = ds_sel # drop corrupted timesteps
                        source.elements.missed |= corrupted

                sources[e.role] = source
                # Save datasets if requested
                if e.save: # and not save_path.exists():
                    sources[e.role].save(save_path)
                    # Regenerate as xarray-local source type
                    xr_loc_source_params, sources[e.role] = self._create_xarray_local_source(save_path, datasource)
                    self.rich_console.print(Table({f"Source '{sources[e.role].datasource.source}' {source_type} {e.role} params": xr_loc_source_params}, twocols=True).table)

        # Detect missing samples
        missed = set()
        for role, source in sources.items():
            if isinstance(source.elements.samples, dict):
                missed |= {p for p in source.elements.missed} # missed is a set
            if source.source_name == "xarray-local" and role != "prediction":
                with source.load() as ds:
                    if "missed_time" in ds:
                        missed |= set(pd.to_datetime(ds["missed_time"].values).to_pydatetime())
        print(f"Missed dates ({source_type}): {missed}")

        # Reload to remove union of missed elements from all datasets (for dimension consistency)
        if missed:
            for role, source in sources.items():
                if source.source_name == "xarray-local" and role != "prediction":
                    source.elements.missed = missed
                    with source.reload() as ds:
                        _ = ds.sizes # just to trigger reload
        # source_table_dict = {f"{source.source_name} [{role}]": source.load().sizes for role, source in sources.items() if role != "prediction"}
        # self.rich_console.print(Table({f"Reloaded source shapes": source_table_dict}).table)

        # missed = {p for source in sources.values() for p in source.elements.get('missed_samples', [])} # missed is a set
        # if missed:
        #     for role, source in sources.items():
        #         source.elements.samples = {k: v for k, v in source.elements.samples.items() if k not in missed}
        #         print(f"Removed missed data {missed} from {role}")
        #         print(f"Number of {source_type} {role}: {len(source.elements.samples)}")
        return sources

    def _init_callbacks (self):
        # Initialize trainer callbacks
        callbacks = []
        # Early stopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=self.config.earlystopping_patience,
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

    def _init_train_trainer (self):
        self._configure_torch_runtime()
        # Initialize Lightning trainer
        return L.Trainer(
            max_epochs=self.config.epochs,
            precision="16-mixed",
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            log_every_n_steps=1,
            logger=self.tl_logger,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            callbacks=self._init_callbacks(),
            # deterministic=True
        )

    def _init_test_trainer (self):
        self._configure_torch_runtime()
        return L.Trainer(
            max_epochs=self.config.epochs,
            precision="16-mixed",
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            # deterministic=True
        )

    def _generate_torch_dataset (self, source_data: Dict[str, BaseSource], experiments: ExperimentDataset | List[ExperimentDataset], data_type: str):
        if not isinstance(experiments, list):
            experiments = [experiments]
        exp_roles = [e.role for e in experiments if e.role != 'prediction']
        ds_d = {}
        s = time.time()
        for role in exp_roles:
            # print(role)
            ds = source_data[role].load()

            # Keep only allowed dims and coords
            allowed_dims = {
                guess_time_dim(ds),
                guess_lat_dim(ds),
                guess_lon_dim(ds),
                guess_realization_dim(ds),
                # guess_leadtime_dim(ds), # leadtime dim must be removed (should be 1D at most)
                "missed_time",
            }
            ds = remove_unwanted_dims_and_coords(ds, allowed_dims)

            # Remove missed_times from the dataset to avoid interference with later torch code
            ds = ds.drop_dims("missed_time")
            ds_d[role] = ds

        loading_time = time.time() - s
        print(f"{data_type} loading time: {loading_time:.1f}s")

        # For now only support input and target roles
        assert "input" in exp_roles and "target" in exp_roles
        input_ds, target_ds = ds_d['input'], ds_d['target']

        # Hook preprocessing function
        if self.config.torch_preprocess_fn is not None:
            input_ds, target_ds = self.config.torch_preprocess_fn(input_ds, target_ds)

        torch_dataset = XarrayDataset(
            input_ds, target_ds,
            target_realization_avg=self.config.target_realization_avg,
            realization_as_channel=self.config.realization_as_channel,
            output_realizations=self.config.output_realizations,
        )

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

        self.rich_console.print(Table({f'{data_type} dataset': {
            'input': {
                'shape': torch_dataset.x.shape,
                'mean': x_mean,
                'std': x_std,
                'source': source_data['input'].datasource.source
            },
            'target': {
                'shape': torch_dataset.y.shape,
                'mean': y_mean,
                'std': y_std,
                'source': source_data['target'].datasource.source
            },
            # 'loading_time': loading_time
        }}).table)

        return torch_dataset

    @staticmethod
    def _to_float (x):
        try:
            return float(x.item()) if hasattr(x, "item") else float(x)
        except Exception:
            return x

    def _make_test_info_table (self, test_dataset, preds_norm, preds_rescaled):
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

        def _pack (name: str,
                x: torch.Tensor, x_mask: torch.Tensor,
                y: torch.Tensor, y_mask: torch.Tensor,
                pn: torch.Tensor, pr: torch.Tensor):
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
            name = getattr(next(iter(self.test_var_list)), "name", str(next(iter(self.test_var_list))))
            _pack(
                name=name,
                x=test_dataset.x, x_mask=test_dataset.x_mask,
                y=test_dataset.y, y_mask=test_dataset.y_mask,
                pn=preds_norm, pr=preds_rescaled,
            )
        else:
            for i, var in enumerate(self.test_var_list):
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
            "input source": getattr(self.source_test_data["input"].datasource, "source", ""),
            "target source": getattr(self.source_test_data["target"].datasource, "source", ""),
            "ckpt_path": str(getattr(self, "ckpt_path", "")),
            "weights_path": str(getattr(self, "weights_path", "")),
        }

        return meta, var_cols

    def train (self):
        # Generate torch train dataset
        train_dataset = self._generate_torch_dataset(self.source_train_data, self.config.train, 'Train')

        # Normalize
        self.normalize_input  = Normalize().fit(train_dataset, filepath=self.normdata_input_path, dim='x')  # mean and std of train input (x) data
        self.normalize_target = Normalize().fit(train_dataset, filepath=self.normdata_target_path, dim='y') # mean and std of train target (y) data
        # print(f"Normalize type: {type(self.normalize.mean)} and shape: {self.normalize.mean.shape}")
        train_dataset.transform_x = self.normalize_input
        train_dataset.transform_y = self.normalize_target

        # Create train datamodule and split train datase\t into train and validation based on self.config.train_percent
        self.train_datamodule = EpochRandomSplitDataModule(
            train_dataset,
            train_fraction=self.config.train_percent,
            batch_size=self.config.batch_size,
            seed=self.config.seed,
            num_workers=self.torch_workers,
            per_epoch_replit=False
        )

        # Train
        trainer = self._init_train_trainer()
        ckpt_path = Path(self.ckpt_path) if Path(self.ckpt_path).exists() else None
        if ckpt_path is None:
            print("Starting training from scratch")
        trainer.fit(
            self.model,
            datamodule=self.train_datamodule,
            ckpt_path=ckpt_path,
        )

    def test (self, weights_filename=None):
        test_dataset = self._generate_torch_dataset(self.source_test_data, self.config.test, 'Test')

        # Normalize input
        if not self.normalize_input:
            print(f"Load normalization data from {self.normdata_input_path}")
            self.normalize_input = Normalize().load(self.normdata_input_path)
        test_dataset.transform_x = self.normalize_input

        # Normalize target
        if not self.normalize_target:
            print(f"Load normalization data from {self.normdata_target_path}")
            self.normalize_target = Normalize().load(self.normdata_target_path)
        test_dataset.transform_y = self.normalize_target

        # Create test dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=self.torch_workers, shuffle=False)
        if not weights_filename:
            # Load checkpoint file
            print(f"Load checkpoints from {self.ckpt_path}")
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

        # Load weights
        print(f"Load weights from file: {weights_file}")
        weights = torch.load(weights_file, map_location=self.device)
        self.model.load_state_dict(weights['state_dict'])

        # Test
        self._init_test_trainer().test(self.model, dataloaders=self.test_dataloader)
        # print(f"Available attributes in model: {dir(self.model)}")

        # Rescale preds with target normalization
        self.preds = self.normalize_target.inverse_tensor(self.model.test_preds, self.normdata_target_path) # .squeeze()

        # Print info
        meta, var_cols = self._make_test_info_table(test_dataset, self.model.test_preds, self.preds)
        self.rich_console.print(Table({"Test run info": meta}, twocols=True).table)
        self.rich_console.print(Table({"Test metrics (per variable)": var_cols}).table)

        self.save(self.preds, 'input')

    def _reconstruct_pred_tensor (
        self,
        data: torch.Tensor,     # expected (N, C, H, W)
        meta_ds: xr.Dataset,    # transposed already
    ) -> tuple[torch.Tensor, xr.Dataset]:
        """
        Returns:
        data_vars: torch.Tensor shaped either (C,T,H,W) or (C,R,T,H,W) depending on meta_ds dims
        meta_ds: possibly sliced so dims match (e.g., R->1 for deterministic preds)
        """
        if data.ndim != 4:
            raise ValueError(f"Expected preds as (N,C,H,W), got {data.shape}")

        meta_ds_ndims = [meta_ds[var.name].ndim for var in self.test_var_list]
        meta_ds_shapes = [meta_ds[var.name].shape for var in self.test_var_list]

        # (C,N,H,W)
        data_permuted = data.permute(1, 0, 2, 3).contiguous()

        tdim = guess_time_dim(meta_ds)
        rdim = guess_realization_dim(meta_ds)
        has_r = rdim in meta_ds.dims
        T_meta = meta_ds.dims.get(tdim, None)
        R_meta = meta_ds.dims.get(rdim, None) if has_r else None

        N = data_permuted.shape[1]

        if self.config.realization_as_channel:
            # In this mode N should be time
            if T_meta is not None and N != T_meta:
                raise ValueError(f"realization_as_channel=True but preds N={N} != T_meta={T_meta}")

            if has_r:
                # deterministic output: keep only one realization in metadata
                meta_ds = meta_ds.isel({rdim: slice(0, 1)})
                # add singleton realization dim to data: (C,1,T,H,W)
                data_permuted = data_permuted.unsqueeze(1)
            # else: meta has no R dim -> keep (C,T,H,W) as-is

            print(f"Input dataset shape: {len(meta_ds_ndims)},{meta_ds_shapes[0]}, permuted pred tensor shape: {data_permuted.shape}")
            return data_permuted, meta_ds

        else:
            # In this mode N is (T*R) (or sometimes R*T depending on your flattening; yours is T*R)
            if not has_r:
                raise ValueError("realization_as_channel=False but metadata has no realization dim to unflatten into.")

            if T_meta is None or R_meta is None:
                raise ValueError(f"Missing required dims in metadata: T={T_meta}, R={R_meta}")

            if N != (T_meta * R_meta):
                raise ValueError(f"Expected preds N=T*R={T_meta*R_meta}, got N={N}")

            # Our dataset uses flatten(start_dim=1,end_dim=2) with (C,T,R,H,W) -> (C,T*R,H,W)
            # so the flattened order is (time-major): t changes slowest, r fastest.
            # Unflatten dim=1 back to (T,R), then permute to (R,T) to match canonical meta order (R,T,H,W).
            data_permuted = data_permuted.unflatten(1, (T_meta, R_meta))      # (C,T,R,H,W)
            data_permuted = data_permuted.permute(0, 2, 1, 3, 4).contiguous()  # (C,R,T,H,W)

            print(f"Input dataset shape: {len(meta_ds_ndims)},{meta_ds_shapes[0]}, permuted pred tensor shape: {data_permuted.shape}")
            return data_permuted, meta_ds

    def save (self, data: torch.Tensor, metadata_source: str):
        """
        Convert torch.Tensor to xarray.Dataset using metadata_source
        to select ds metadata and save it to Zarr storage
        """
        meta_ds = self.source_test_data[metadata_source].load()

        # Keep only allowed dims and coords
        allowed_dims = {
            guess_time_dim(meta_ds),
            guess_lat_dim(meta_ds),
            guess_lon_dim(meta_ds),
            guess_realization_dim(meta_ds),
            # guess_leadtime_dim(meta_ds), # leadtime dim must be removed (should be 1D at most)
            "missed_time",
        }
        meta_ds = remove_unwanted_dims_and_coords(meta_ds, allowed_dims)

        # Canonical dim order
        base_order = [guess_realization_dim(meta_ds), guess_time_dim(meta_ds), guess_lat_dim(meta_ds), guess_lon_dim(meta_ds)]
        # Build the actual order based on what exists
        order = [d for d in base_order if d in meta_ds.dims]
        # If there are other unexpected dims, tack them on at the end
        order += [d for d in meta_ds.dims if d not in order]

        # Force dataset-wide dim order (dims missing on some vars are ignored)
        meta_ds = meta_ds.transpose(*order, missing_dims="ignore")

        data_vars, meta_ds = self._reconstruct_pred_tensor(data, meta_ds)

        # Now build xr.Dataset variables with meta_ds[var].dims
        ds = xr.Dataset(
            {
                var.name: (meta_ds[var.name].dims, data_vars[i].cpu().numpy())
                for i, var in enumerate(self.test_var_list)
            },
            coords={c: meta_ds.coords[c] for c in meta_ds.coords},
            attrs=meta_ds.attrs,
        )

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = (
            {v.name: {"compressors": compressor} for v in self.test_var_list}
        )
        print(f"Save to {self.preds_store}")
        ds.to_zarr(self.preds_store, encoding=encoding_zarr, mode='w', consolidated=self.consolidated_zarr)

        return ds
