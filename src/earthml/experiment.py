import os, time, copy, multiprocessing
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datetime import datetime, timedelta
from pathlib import Path
from itertools import zip_longest
# from abc import ABC, abstractmethod
import numpy as np
import xarray as xr
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
# Local imports
from .source import SourceRegistry
from .dataclasses import ExperimentConfig
from .utils import EpochRandomSplitDataModule, XarrayDataset, Normalize
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
        config: ExperimentConfig,
        source_input_args: dict,
        source_target_args: dict
    ):
        self.config = config
        self.source_input_args = source_input_args
        self.source_target_args = source_target_args
        # Setup paths and make dirs if necessary
        self._path_setup()
        # General torch and Lightning setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_cpus, num_cuda_devices = multiprocessing.cpu_count(), torch.cuda.device_count() 
        self.torch_workers = max(1, num_cpus // num_cuda_devices // 2) # could be moved in datamodule definition, but in test() still using dataloader
        print(f"Torch workers in use: {self.torch_workers} ({num_cpus} CPUs // {num_cuda_devices} CUDA devices // 2)")
        L.seed_everything(self.config.seed)
        # Tensorboard
        self.tl_logger = TensorBoardLogger(self.config.work_path, name=self.config.name) # version=self.run_number
        # Initialize model
        self.Net = globals()[self.config.net]
        self.model = self.Net(
            learning_rate=self.config.learning_rate,
            loss=self.config.loss,
            norm=self.config.norm_strategy,
            supervised=self.config.supervised,
        ).to(self.device)
        # Log model info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Net {self.Net.__name__} trainable parameters: {trainable_params:,}")
        # Initialize data objects used and populated in train() and test()
        self._source_train_setup()
        self._source_test_setup()
        self.normalize = None
        self.train_datamodule = None
        # Initialize trainer for train and test
        self._init_trainer()

    def _path_setup (self):
        self.work_path = Path(self.config.work_path)
        # Weights location
        self.weights_folder_path = self.work_path.joinpath("./weights")
        self.weights_filename = f"{self.config.name}_weights"
        # self.weights_path = self.weights_folder_path.joinpath(self.weights_filename)
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

    def _source_train_setup (self):
        SourceTrain = SourceRegistry(self.config.train.source).get_class()
        input_data_sel = self.config.train.data_selection
        # print(f"Input data sel: {input_data_sel}")
        target_data_sel = copy.deepcopy(self.config.train.data_selection)
        target_data_sel.period.start += pd.to_timedelta(self.config.lead_time)
        target_data_sel.period.end += pd.to_timedelta(self.config.lead_time)
        # print(f"Input data sel: {target_data_sel}")
        source_train_params = {
            "input": {
                "data_selection": input_data_sel,
            } | self.source_input_args,
            "target": {
                "data_selection": target_data_sel,
            } | self.source_target_args
        }
        # Create source instances
        # print(f"Source train input params: {source_train_params["input"]}")
        # print(f"Source train target params: {source_train_params["target"]}")
        self.source_train_input = SourceTrain(**source_train_params["input"])
        self.source_train_target = SourceTrain(**source_train_params["target"])
        # Clear missing samples
        missed = set(self.source_train_input.file_paths['missed_samples']) | set(self.source_train_target.file_paths['missed_samples'])
        if missed:
            self.source_train_input.file_paths['samples'] = {k: v for k, v in self.source_train_input.file_paths['samples'].items() if k not in missed}
            self.source_train_target.file_paths['samples'] = {k: v for k, v in self.source_train_target.file_paths['samples'].items() if k not in missed}
            print(f"Remove missed data {missed} from inputs/targets")
            print(f"Number of test inputs: {len(self.source_train_input.file_paths['samples'])}, "
                f"targets: {len(self.source_train_target.file_paths['samples'])}")

    def _source_test_setup (self):
        SourceTest = SourceRegistry(self.config.test.source).get_class()
        input_data_sel = self.config.test.data_selection
        target_data_sel = copy.deepcopy(self.config.test.data_selection)
        target_data_sel.period.start += pd.to_timedelta(self.config.lead_time)
        target_data_sel.period.end += pd.to_timedelta(self.config.lead_time)
        source_test_params = {
            "input": {
                "data_selection": input_data_sel,
            } | self.source_input_args,
            "target": {
                "data_selection": target_data_sel,
            } | self.source_target_args
        }
        # Create source instances
        # print(f"Source test input params: {source_test_params["input"]}")
        # print(f"Source test target params: {source_test_params["target"]}")
        self.source_test_input = SourceTest(**source_test_params["input"])
        self.source_test_target = SourceTest(**source_test_params["target"])
        # Clear missing samples
        missed = set(self.source_test_input.file_paths['missed_samples']) | set(self.source_test_target.file_paths['missed_samples'])
        if missed:
            self.source_test_input.file_paths['samples'] = {k: v for k, v in self.source_test_input.file_paths['samples'].items() if k not in missed}
            self.source_test_target.file_paths['samples'] = {k: v for k, v in self.source_test_target.file_paths['samples'].items() if k not in missed}
            print(f"Remove missed data {missed} from inputs/targets")
            print(f"Number of test inputs: {len(self.source_test_input.file_paths['samples'])}, "
                f"targets: {len(self.source_test_target.file_paths['samples'])}")

    def _init_trainer (self):
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
        callbacks.append(best_weights_callback)# Initialize Lightning trainer
        self.train_trainer = L.Trainer(
            max_epochs=self.config.epochs,
            precision="16-mixed",
            # gradient_clip_val=1.0,           # Recommended starting value (e.g., 0.5, 1.0, 5.0)
            # gradient_clip_algorithm="norm",  # "norm" for clipping by norm, "value" for clipping by value
            log_every_n_steps=1,
            logger=self.tl_logger,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            callbacks=callbacks,
            # deterministic=True
        )
        self.test_trainer = L.Trainer(
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
            self.source_train_input.load(),
            self.source_train_target.load()
        )
        print(f"Train dataset, input {train_dataset.x.shape}, target {train_dataset.y.shape} (source {self.config.train.source}) loaded in {(time.time() - s):.3f}s")
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
        self.train_trainer.fit(self.model, datamodule=self.train_datamodule)

    def test (self, weights_filename=None):
        # Generate torch test dataset
        s = time.time()
        test_dataset = XarrayDataset(
            self.source_test_input.load(),
            self.source_test_target.load()
        )
        print(f"Test dataset, input {test_dataset.x.shape}, target {test_dataset.y.shape} (source {self.config.test.source}) loaded in {(time.time() - s):.3f}s")
        print(f"Test dataset input mean: {test_dataset.x.mean()}, std: {test_dataset.x.std()}")
        print(f"Test dataset target mean: {test_dataset.y.mean()}, std: {test_dataset.y.std()}")
        rmse_input = np.sqrt(((test_dataset.y - test_dataset.x)**2).mean())
        print(f"RMSE target-input: {rmse_input}")
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
            weights_filename = last_callback["best_model_path"]
        # Load weights
        print(f"Load weights from file: {weights_filename}")
        weights = torch.load(weights_filename, map_location=self.device)
        self.model.load_state_dict(weights['state_dict'])
        # Test
        self.test_trainer.test(self.model, dataloaders=self.test_dataloader)
        # print(f"Available attributes in model: {dir(self.model)}")
        print(f"Normalized prediction shape: {self.model.test_preds.shape}, mean: {self.model.test_preds.mean()}, std: {self.model.test_preds.std()}")
        # Rescale and squeeze
        preds = self.normalize.inverse_tensor(self.model.test_preds, self.normdata_path).squeeze()
        print(f"Rescaled prediction shape: {preds.shape}, mean: {preds.mean()}, std: {preds.std()}")
        rmse_preds = np.sqrt(((test_dataset.y - preds.unsqueeze(1))**2).mean())
        print(f"RMSE target-prediction: {rmse_preds}")
        # Convert to Xarray using input ds metadata
        test_var = self.config.test.data_selection.variable.name
        test_input_ds = self.source_test_input.load()
        self.test_pred_ds = xr.Dataset(
            {test_var: (test_input_ds[test_var].dims, preds.cpu().numpy())},
            coords={c: test_input_ds.coords[c] for c in test_input_ds.coords},
            attrs=test_input_ds.attrs,
        )
