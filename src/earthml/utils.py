from pathlib import Path
from rich import print
from rich.pretty import pprint
from rich.table import Table
import joblib
import cf_xarray
import os, psutil, multiprocessing, tempfile, logging
from dask.distributed import Client, LocalCluster
import numpy as np
import torch
from torch.utils.data import random_split, Dataset, DataLoader, SubsetRandomSampler
import lightning as L

class Normalize:
    def __init__ (self, mean=None, std=None):
        """
        mean, std: optional scalars or tensors for normalization.
                   If None, must call fit(dataset) before using.
        """
        self.mean = mean
        self.std = std

    def _norm (self, data, epsilon=1e-6):
        return (data - self.mean) / (self.std + epsilon)
    
    def _denorm (self, data):
        return self.std * data + self.mean

    def _check_filepath (self, filepath):
        if self.mean is None or self.std is None:
            try:
                self = self.load(filepath)
            except Exception as e:
                print(e)
                raise ValueError("Transform not fitted.")

    def save (self, filepath):
        joblib.dump(self, filepath)

    def load (self, filepath):
        return joblib.load(filepath)

    def fit (self, dataset, filepath, dim='x'):
        """
        Compute global mean and std from an XarrayDataset instance.
        This assumes dataset has two torch.Tensor of shape [N, C, H, W].
        dim: select the data to fit, input (x) or target (y)
        """
        data = getattr(dataset, dim)  # (N, C, H, W)
        self.mean = data.mean(dim=(0, 2, 3), keepdim=True).reshape(1, -1, 1, 1)  # per-channel mean
        # print(f"Mean type: {type(self.mean)} and shape: {self.mean.shape}")
        self.std = data.std(dim=(0, 2, 3), keepdim=True).reshape(1, -1, 1, 1)
        # print(f"Std type: {type(self.std)} and shape: {self.std.shape}")
        print(f"Fitted mean: {self.mean.flatten().tolist()}")
        print(f"Fitted std: {self.std.flatten().tolist()}")
        self.save(filepath)
        return self

    def inverse (self, dataset, filepath=None):
        """
        Return a new dataset with denormalized data.
        Does not modify the original dataset in-place.
        """
        self._check_filepath(filepath)

        new_dataset = dataset.__class__.__new__(dataset.__class__)
        new_dataset.__dict__ = dataset.__dict__.copy()

        if hasattr(dataset, "x") and dataset.x is not None:
            new_dataset.x = self._denorm(dataset.x)
        if hasattr(dataset, "y") and dataset.y is not None:
            new_dataset.y = self._denorm(dataset.y)
        
        return new_dataset

    def inverse_tensor (self, data, filepath=None):
        """
        Return denormalized data.
        """
        self._check_filepath(filepath)
        return self._denorm(data)


    def __call__ (self, x, y, epsilon=1e-6):
        if self.mean is None or self.std is None:
            raise ValueError("Transform not fitted")
        x = self._norm(x, epsilon=epsilon)
        x = x.squeeze(0) # remove time dimension becasue we call this in Dataset __getitem__
        # print(f"X after norm type: {type(x)} and shape: {x.shape}")
        y = self._norm(y, epsilon=epsilon)
        y = y.squeeze(0)
        # print(f"Y after norm type: {type(y)} and shape: {y.shape}")
        return x, y

class EpochRandomSplitDataModule (L.LightningDataModule):
    def __init__ (self, dataset, train_fraction=0.9, batch_size=32, seed=42, num_workers=0, per_epoch_replit=False):
        super().__init__()
        self.dataset = dataset
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.per_epoch_replit = per_epoch_replit

    def setup (self, stage=None):
        # initial split
        self._resplit()

    def _resplit (self):
        torch.manual_seed(self.seed)

        num_samples = len(self.dataset)
        indices = torch.randperm(num_samples)
        subset_size = int(num_samples * self.train_fraction)

        train_indices = indices[:subset_size]
        valid_indices = indices[subset_size:]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        self._train_dl = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            sampler=train_sampler,  
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self._val_dl = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            sampler=valid_sampler,  
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader (self):
        return self._train_dl

    def val_dataloader (self):
        return self._val_dl

    def on_train_epoch_start (self):
        if self.per_epoch_replit:
            # re-split at every epoch
            self._resplit()

# import xesmf as xe
class XarrayDataset (Dataset):
    def __init__(self, input_ds, target_ds, transform=None, transform_args=None):
        """
        input_ds: xarray.Dataset
        target_ds: xarray.Dataset
        transform: callable with signature (x, y, **kwargs) -> (x, y)
        transform_args: dict of keyword args to pass to transform
        """
        self.input_ds = input_ds
        self.target_ds = target_ds
        self.transform = transform
        self.transform_args = transform_args or {}

        # print("Dataset before regrid")
        # for ds in [input_ds, target_ds]:
        #     for v in ds.variables:
        #         print(v, ds[v].dims, ds[v].shape)
        # Ensure identical spatial grid
        self.target_ds = self.regrid(self.input_ds, self.target_ds)
        # for ds in [input_ds, target_ds]:
        #     for v in ds.variables:
        #         print(v, ds[v].dims, ds[v].shape)
        # print("Dataset after regrid")

        self.time_dim = self.input_ds.cf['time'].name
        self.time = self.input_ds[self.time_dim].values

        self.x = torch.tensor(self.input_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        self.y = torch.tensor(self.target_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        assert len(self.x) == len(self.y), f"Mismatched dataset length: x={len(self.x)}, y={len(self.y)}"

    @staticmethod
    def regrid(input_ds, target_ds):
        """Ensure target has same lat/lon dims & coords as input, regrid if needed."""
        # Get dimension names from CF
        lat_in = input_ds.cf['latitude'].name
        lon_in = input_ds.cf['longitude'].name
        lat_tg = target_ds.cf['latitude'].name
        lon_tg = target_ds.cf['longitude'].name

        same_dims = (lat_in == lat_tg) and (lon_in == lon_tg)
        same_shape = (input_ds[lat_in].shape == target_ds[lat_tg].shape) and \
                     (input_ds[lon_in].shape == target_ds[lon_tg].shape)
        same_coords = (
            np.array_equal(input_ds[lat_in].values, target_ds[lat_tg].values) and
            np.array_equal(input_ds[lon_in].values, target_ds[lon_tg].values)
        )
        if same_dims and same_shape and same_coords:
            return target_ds  # perfect match, nothing to do
        # Build regridder once
        print("Regridding...")
        # TODO need to compile esmf and esmpy from source https://github.com/esmf-org/esmf (not available in pip)
        # regridder = xe.Regridder(
        #     target_ds,
        #     input_ds,
        #     method="bilinear",
        #     periodic=False,
        #     reuse_weights=False,
        # )
        # return regridder(target_ds)
        # Fallback to interpolation
        return target_ds.interp(
            {
                lat_tg: input_ds[lat_in],
                lon_tg: input_ds[lon_in],
            },
            method="nearest"
        )
    def __len__ (self):
        return len(self.x)

    def __getitem__ (self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x, y = self.transform(x, y, **self.transform_args)
        return x, y

class Dask:
    def __init__ (self, base_port=8787, n_workers=None):
        """
        Start an optimized LocalCluster on HPC or JupyterHub environments.

        - Detects local scratch directory (/scratch, /tmp, $TMPDIR)
        - Balances n_workers by available CPU and memory (~4 GB per worker)
        - Registers cf_xarray on workers
        - Returns (client, cluster)
        """
        # Silence Tornado/Bokeh websocket noise (optional)
        logging.getLogger("tornado.application").setLevel(logging.ERROR)
        logging.getLogger("bokeh").setLevel(logging.ERROR)
        # Detect best local scratch directory
        candidates = [
            os.environ.get("TMPDIR"),
            f"/scratch/{os.environ.get('USER')}",
            "/scratch",
            tempfile.gettempdir(),
            os.path.expanduser("~/.dask-tmp")
        ]
        local_dir = next((p for p in candidates if p and os.path.exists(p)), tempfile.gettempdir())
        os.makedirs(local_dir, exist_ok=True)
        # Auto-tune number of workers
        n_cores = multiprocessing.cpu_count()
        total_mem_gb = psutil.virtual_memory().total / 1e9
        if n_workers is None:
            n_workers = min(n_cores, max(1, int(total_mem_gb // 4)))  # ~4 GB per worker
        # Start cluster
        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            timeout='600s',  # 10 minutes
            heartbeat_interval='10s',
            memory_limit="auto",
            local_directory=local_dir,
            dashboard_address=f":{base_port}",  # uses free port if base_port busy
        )
        self.client = Client(self.cluster)
        self.client.run(lambda: __import__("cf_xarray"))
        print(f"Cores: {n_cores}, Mem: {total_mem_gb} GB -> Dask workers: {n_workers}")
        print(f"Write Dask local files in {local_dir}")

from rich.table import Table as RichTable
from rich.highlighter import ReprHighlighter
class Table ():
    """Helper class to create rich Tables from multinested dicts"""
    def __init__ (self, data: dict, title: str = None, params_name:str = None, twocols: bool = False) -> RichTable:
        assert isinstance(data, dict)
        if len(data.keys()) == 1:
            assert isinstance(next(iter(data.values())), dict) # there must be data
            data_name = next(iter(data.keys()))
            data = data[data_name] # promote first inner dict to actual data
            title = data_name if title is None else title
        has_inner_dicts = self._has_inner_dicts(data)
        rich_params = {
            "title": title,
            "show_header": bool(has_inner_dicts and title and not twocols),
        }
        self.table = RichTable(**rich_params)
        rowheads = self._get_rowheads(data) # check only first inner level
        highligher = ReprHighlighter()
        params_name = 'params' if params_name is None else params_name
        if has_inner_dicts and rowheads and not twocols:
            self.table.add_column("params", style='magenta')
            for k in data.keys():
                self.table.add_column(str(k), style='cyan')
            row = {}
            for i,r in enumerate(rowheads):
                row[r] = []
                for v in data.values():
                    if isinstance(v, dict):
                        row[r].append(highligher(str(list(v.values())[i]))) if r in v.keys() else ""
            for r in rowheads:
                self.table.add_row(str(r), *row[r])
        else:
            self.table.add_column(title, style='magenta')
            self.table.add_column("", style='cyan')
            for k, v in data.items():
                self.table.add_row(k, highligher(str(v)))

    def _has_inner_dicts (self, d: dict) -> bool:
        for v in d.values():
            if isinstance(v, dict):
                return True or self.has_inner_dicts(v)
            elif isinstance(v, (list, tuple)):
                if any(isinstance(i, dict) and self.has_inner_dicts(i) for i in v):
                    return True
        return False

    def _get_rowheads (self, d: dict, recursive: bool = False) -> list:
        rowheads = []
        for v in d.values():
            if isinstance(v, dict):
                rowheads.extend(map(str, v.keys()))
                if recursive:
                    rowheads.extend(self._get_rowheads(v, recursive))
        return list(dict.fromkeys(rowheads))
