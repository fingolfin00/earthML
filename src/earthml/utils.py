from pathlib import Path
import joblib
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

        self.time_dim = self.input_ds.cf['time'].name
        self.time = self.input_ds[self.time_dim].values

        self.x = torch.tensor(self.input_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        self.y = torch.tensor(self.target_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)

    def __len__ (self):
        return len(self.x)

    def __getitem__ (self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x, y = self.transform(x, y, **self.transform_args)
        return x, y
