import os, time, copy, multiprocessing, pickle, joblib
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# from datetime import datetime, timedelta
from pathlib import Path
# from copy import deepcopy
# from itertools import zip_longest
from typing import List
from rich import print
# from rich.pretty import pprint
from rich.console import Console
import numpy as np
import pandas as pd
import xarray as xr
from zarr.codecs import BloscCodec
# from zarr.storage import ZipStore
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
# Local imports
from .source import SourceRegistry
from .dataclasses import ExperimentDataset, ExperimentConfig, DataSource
from .utils import Table, print_ds_info
from .lightning import XarrayDataset, Normalize, EpochRandomSplitDataModule
from .nets.smaatunet import SmaAt_UNet

torch.set_float32_matmul_precision('medium')  # or 'high'
# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 

class ExperimentRegistry:
    def __init__ (self, source_name: str):
        self.class_registry = {
            "ML-forecast-correction": ExperimentMLFC
        }
        self.source_name = source_name

    def get_class (self):
        return self.class_registry.get(self.source_name)

class ExperimentMLFC:
    def __init__(
        self,
        config: ExperimentConfig
    ):
        self.config = config
        self.rich_console = Console()
        # Get test variable list, TODO I have doubts on this implementation
        test_config_datasel = self.config.test[0].datasource.data_selection if isinstance(self.config.test, list) else self.config.test.datasource.data_selection
        self.test_var_list = test_config_datasel.variable if isinstance(test_config_datasel.variable, list) else [test_config_datasel.variable]
        # Setup paths and make dirs if necessary
        self._path_setup()
        # General torch and Lightning setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_cpus, num_cuda_devices = multiprocessing.cpu_count(), torch.cuda.device_count() 
        self.torch_workers = max(1, num_cpus // num_cuda_devices // 2) # could be moved in datamodule definition, but in test() still using dataloader
        print(f"Torch workers in use: {self.torch_workers} ({num_cpus} CPUs // {num_cuda_devices} CUDA devices // 2)")
        L.seed_everything(self.config.seed)
        # Tensorboard
        self.tl_logger = TensorBoardLogger(self.config.work_path, name="tensorboard_logs") # version=self.run_number
        # Initialize model
        self.Net = globals()[self.config.net]
        self.model = self.Net(
            learning_rate=self.config.learning_rate,
            loss=self.config.loss,
            loss_params=self.config.loss_params,
            norm=self.config.norm_strategy,
            supervised=self.config.supervised,
            **self.config.extra_net_args,
        ).to(self.device)
        # Log model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Net {self.Net.__name__} trainable parameters: {trainable_params:,}")
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
                self.config.test[0].datasource.data_selection,
            ),
            source_params={
                'root_path': self.preds_store,
                'xarray_args': {'consolidated': self.consolidated_zarr}
            }
        )
        self.config.test.append(preds_exp)
        # Init source data objects
        self.source_test_data = self._init_source_data(self.config.test, 'test') #, deltas)
        self.source_train_data = self._init_source_data(self.config.train, 'train') # ), deltas)
        # Save experiment
        with open(self.work_path.joinpath("experiment.cfg"), 'wb') as f:
            joblib.dump({
                'config': self.config,
                'device': self.device,
                'torch_workers': self.torch_workers,
                'tl_logger': self.tl_logger,
                # 'model': self.model,
                'test_data': self.source_test_data,
                'train_data': self.source_train_data
            }, f)

    def _path_setup (self):
        self.work_path = Path(self.config.work_path)
        # Weights location
        self.weights_folder_path = self.work_path.joinpath("./weights")
        self.weights_filename = f"{self.config.name}_weights"
        self.weights_path = self.weights_folder_path.joinpath(self.weights_filename+'.ckpt')
        # Train dataset normalization data location
        norm_data_folder_path = self.work_path.joinpath("./normdata")
        normdata_filename = f"{self.config.name}_normdata.gz"
        norm_data_folder_path.mkdir(parents=True, exist_ok=True)
        self.normdata_path = norm_data_folder_path.joinpath(normdata_filename)
        # Lightning checkpoints location
        self.ckpt_filename = "checkpoint"
        self.ckpt_folder_path = self.work_path.joinpath("./checkpoints")
        self.ckpt_folder_path.mkdir(parents=True, exist_ok=True)
        ckpt_files = self.ckpt_folder_path.glob(f"{self.ckpt_filename}*.ckpt")
        try:
            self.ckpt_path = max(ckpt_files, key=lambda item: item.stat().st_ctime) # get most recent
        except:
            self.ckpt_path = self.ckpt_folder_path.joinpath(f"{self.ckpt_filename}.ckpt")

    def _init_source_data (self, exp_ds: ExperimentDataset | List[ExperimentDataset], source_type: str, deltas: List[timedelta] = None):
        """Returns populated selected Source instances"""
        if not isinstance(exp_ds, list):
            exp_ds = [exp_ds]
        sources = {}
        for i, e in enumerate(exp_ds):
            Source = SourceRegistry(e.datasource.source).get_class()
            # data_sel = e.datasource.data_selection
            # data_sel.period.start += deltas[i] if deltas else data_sel.period.start
            # data_sel.period.end += deltas[i] if deltas else data_sel.period.end
            self.rich_console.print(Table({f"Source {source_type} {e.role} params": e.source_params}, twocols=True).table) # TODO move tables to source.py
            sources[e.role] = Source(
                datasource=e.datasource,
                **e.source_params
            )
        # Clear missing samples if supported
        missed = set()
        for source in sources.values():
            if isinstance(source.elements.samples, dict):
                missed |= {p for p in source.elements.missed} # missed is a set
        print(f"Missed dates ({source_type}): {missed}")
        for source in sources.values():
            source.elements.missed = missed
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
        return L.Trainer(
            max_epochs=self.config.epochs,
            precision="16-mixed",
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            # deterministic=True
        )

    def train (self):
        # Generate torch train dataset
        s = time.time()
        train_dataset = XarrayDataset(
            self.source_train_data['input'].load(),
            self.source_train_data['target'].load()
        )
        loading_time = time.time() - s
        self.rich_console.print(Table({'Train dataset': {
            'input': {
                'shape': train_dataset.x.shape,
                'source': self.config.train[0].datasource.source
            },
            'target': {
                'shape': train_dataset.y.shape,
                'source': self.config.train[1].datasource.source
            },
            'loading_time': loading_time
        }}).table)
        # Normalize
        self.normalize = Normalize().fit(train_dataset, filepath=self.normdata_path, dim='x') # uses mean and std of train input (x) data
        # print(f"Normalize type: {type(self.normalize.mean)} and shape: {self.normalize.mean.shape}")
        train_dataset.transform = self.normalize
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
        self._init_train_trainer().fit(self.model, datamodule=self.train_datamodule)

    def test (self, weights_filename=None):
        # Generate torch test dataset
        s = time.time()
        test_dataset = XarrayDataset(
            self.source_test_data['input'].load(),
            self.source_test_data['target'].load()
        )
        loading_time = time.time() - s
        self.rich_console.print(Table({'Test dataset': {
            'input': {
                'shape': test_dataset.x.shape,
                'source': self.config.test[0].datasource.source
            },
            'target': {
                'shape': test_dataset.y.shape,
                'source': self.config.test[1].datasource.source
            },
            'loading_time': loading_time
        }}, params_name='data type').table)
        self.rich_console.print(Table({"Test dataset metrics": {
                'input mean': {var.name: test_dataset.x[:,i,:,:].mean().item() for i, var in enumerate(self.test_var_list)},
                'input std': {var.name: test_dataset.x[:,i,:,:].std().item() for i, var in enumerate(self.test_var_list)},
                'target mean': {var.name: test_dataset.y[:,i,:,:].mean().item() for i, var in enumerate(self.test_var_list)},
                'target std': {var.name: test_dataset.y[:,i,:,:].std().item() for i, var in enumerate(self.test_var_list)},
                'rmse target-input': {var.name: np.sqrt(((test_dataset.y[:,i,:,:] - test_dataset.x[:,i,:,:])**2).mean().item()) for i, var in enumerate(self.test_var_list)}
        }}, params_name='metric').table)
        # Normalize
        if not self.normalize:
            print(f"Load normalization data from {self.normdata_path}")
            self.normalize = Normalize().load(self.normdata_path)
        test_dataset.transform = self.normalize
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
        mean_norm_pred_d = {var.name: self.model.test_preds[:,i,:,:].mean().item() for i, var in enumerate(self.test_var_list)}
        std_norm_pred_d = {var.name: self.model.test_preds[:,i,:,:].std().item() for i, var in enumerate(self.test_var_list)}
        print(f"Normalized prediction shape: {self.model.test_preds.shape}, mean: {mean_norm_pred_d}, std: {std_norm_pred_d}")
        # Rescale
        self.preds = self.normalize.inverse_tensor(self.model.test_preds, self.normdata_path) # .squeeze()
        mean_pred_d = {var.name: self.preds[:,i,:,:].mean().item() for i, var in enumerate(self.test_var_list)}
        std_pred_d = {var.name: self.preds[:,i,:,:].std().item() for i, var in enumerate(self.test_var_list)}
        print(f"Rescaled prediction shape: {self.preds.shape}, mean: {mean_pred_d}, std: {std_pred_d}")
        rmse_pred_d = {var.name: np.sqrt(((test_dataset.y[:,i,:,:] - self.preds[:,i,:,:])**2).mean().item()) for i, var in enumerate(self.test_var_list)}
        print(f"RMSE target-prediction: {rmse_pred_d}")
        self.save(self.preds, 'input')

    def save (self, data: torch.Tensor, metadata_source: xr.Dataset):
        """Convert torch.Tensor to xarray.Dataset using metadata_source as ds metadata and save it to Zarr storage"""
        meta_ds = self.source_test_data[metadata_source].load()
        ds = xr.Dataset(
            {var.name: (meta_ds[var.name].dims, data[:,i,:,:].cpu().numpy()) for i, var in enumerate(self.test_var_list)},
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
