from typing import List, Any, Optional, Callable
import inspect
import importlib, joblib
import numpy as np
import xarray as xr
import torch
from torch import nn
from torch.utils.data import random_split, Dataset, DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import lightning as L
from torchmetrics import MeanAbsoluteError, MeanSquaredError, Metric
from torchmetrics.image import SpatialCorrelationCoefficient
# import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .utils import guess_realization_dim, guess_time_dim, guess_lon_dim, guess_lat_dim

# Module level functions

def call_loss (
        loss_fn, y_pred: torch.Tensor, y_true: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
    """
    Calls loss_fn with supported kwargs only.
    Supports losses with signatures:
      - loss(y_pred, y_true)
      - loss(y_pred, y_true, mask=...)
      - loss(y_pred, y_true, x_input, mask=...)  (your custom)
    """
    sig = inspect.signature(loss_fn.forward if hasattr(loss_fn, "forward") else loss_fn)
    params = sig.parameters

    kwargs: dict[str, Any] = {}

    # only pass mask if accepted
    if mask is not None and "mask" in params:
        kwargs["mask"] = mask

    # pass x_input if requested and accepted (you call it x_input in your custom)
    if x0 is not None:
        if "x_input" in params:
            kwargs["x_input"] = x0
            return loss_fn(y_pred, y_true, **kwargs)
        # some custom losses might take it positionally, but your code uses positional (mu,y,x0,...)
        # If forward has 3rd positional param and you want to use it, do it explicitly:
        positional = [p for p in params.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(positional) >= 3:
            return loss_fn(y_pred, y_true, x0, **kwargs)

    return loss_fn(y_pred, y_true, **kwargs)

# Classes

# Masked metrics
class MaskedMAE(Metric):
    is_differentiable = False
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_abs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        m = mask.to(dtype=preds.dtype, device=preds.device)
        self.sum_abs += (preds.sub(target).abs() * m).sum()
        self.count += m.sum()

    def compute(self):
        return self.sum_abs / self.count # NaN if count is zero


class MaskedRMSE(Metric):
    is_differentiable = False
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        m = mask.to(dtype=preds.dtype, device=preds.device)
        self.sum_sq += ((preds - target) ** 2 * m).sum()
        self.count += m.sum()

    def compute(self):
        return torch.sqrt(self.sum_sq / self.count) # NaN if count is zero

class MaskedSpatialCorr(Metric):
    is_differentiable = False
    higher_is_better = True

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.add_state("sum_corr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # preds/target: (N,C,H,W), mask broadcastable to that
        m = mask.to(dtype=preds.dtype, device=preds.device)
        N, C = preds.shape[:2]
        x = preds.reshape(N, C, -1)
        y = target.reshape(N, C, -1)
        mm = m.reshape(N, C, -1)

        count = mm.sum(dim=-1)  # (N,C)
        valid = count > 1

        # masked means
        denom = count.clamp_min(1.0)
        mx = (x * mm).sum(dim=-1) / denom
        my = (y * mm).sum(dim=-1) / denom

        xc = (x - mx.unsqueeze(-1)) * mm
        yc = (y - my.unsqueeze(-1)) * mm

        cov = (xc * yc).sum(dim=-1)
        vx = (xc * xc).sum(dim=-1)
        vy = (yc * yc).sum(dim=-1)

        corr = cov / (torch.sqrt(vx * vy).clamp_min(self.eps))
        corr = corr[valid]

        self.sum_corr += corr.sum()
        self.num += torch.tensor(corr.numel(), device=self.sum_corr.device, dtype=self.sum_corr.dtype)

    def compute(self):
        return self.sum_corr / self.num # NaN if count is zero

class EarthMLLightningModule (L.LightningModule):
    def __init__ (self, use_first_input=False):
        super().__init__()
        # self.extra_logger = extra_logger

        self.use_first_input = use_first_input

        # Metrics
        self.train_mae = MaskedMAE()
        self.train_rmse = MaskedRMSE()
        self.train_scc = MaskedSpatialCorr()
        # self.train_acc = AnomalyCorrelationCoefficient()

        self.val_mae = MaskedMAE()
        self.val_rmse = MaskedRMSE()
        self.val_scc = MaskedSpatialCorr() # Your custom SCC
        # self.val_acc = AnomalyCorrelationCoefficient() # Your custom ACC

        self.test_mae = MaskedMAE()
        self.test_rmse = MaskedRMSE()
        self.test_scc = MaskedSpatialCorr()
        # self.test_acc = AnomalyCorrelationCoefficient()

        # Metrics storage
        self.test_step_outputs = []
        self.pic_log_interval = 1 # set to 1 to log at every epoch
        self.last_val_pred = None
        self.last_val_target = None

    @staticmethod
    def resolve_loss (name, params):
        # simple name -> try torch.nn
        if "." not in name:
            if hasattr(nn, name):
                return getattr(nn, name)(**params)
            raise ValueError(f"Loss '{name}' not found in torch.nn")
        # dotted path -> dynamic import
        module_path, class_name = name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)(**params)

    @staticmethod
    def center_crop_to (x, target_h, target_w):
        _, _, h, w = x.shape
        off_y = max((h - target_h) // 2, 0)
        off_x = max((w - target_w) // 2, 0)
        return x[:, :, off_y:off_y + target_h, off_x:off_x + target_w]

    def match_spatial (self, x, H, W):
        """Match tensor x to ref's HxW by center-cropping or padding (replicate) without interpolation."""
        _, _, h, w = x.shape
        dy, dx = H - h, W - w
        if dy == 0 and dx == 0:
            return x

        # If x is larger -> crop; if smaller -> pad (replicate) to avoid introducing zeros
        if dy < 0 or dx < 0:
            x = self.center_crop_to(x, min(h, H), min(w, W))
            _, _, h, w = x.shape
            dy, dx = H - h, W - w

        # Now only non-negative pads remain
        if dy != 0 or dx != 0:
            pad_left   = dx // 2
            pad_right  = dx - pad_left
            pad_top    = dy // 2
            pad_bottom = dy - pad_top
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
        return x

    def _squeeze_and_add_log_img (self, img_tensor, name_tag, logger_instance, colormap=None, vmin=None, vmax=None):
        """
        Logs a single 2D image to the logger, with optional colormap.
        img_tensor: A 2D tensor (H, W) for grayscale, or 3D tensor (C, H, W) for RGB/RGBA.
                    For colormapping, it should typically be a 2D grayscale tensor.
        name_tag: The name to appear in the logger (e.g., "Validation/Prediction_Sample").
        logger_instance: The logger object (e.g., self.logger).
        colormap: String name of a matplotlib colormap (e.g., 'viridis', 'jet', 'gray').
                  If None, the image is logged as is (assuming it's already in the correct format).
        vmin, vmax: Optional min/max values for normalization before applying colormap.
                    If None, min/max of the tensor will be used.
        """
        if self.trainer.sanity_checking:
            return # Don't log images during sanity check

        # Detach from graph and move to CPU if not already
        img_tensor = img_tensor.detach().cpu()

        if colormap:
            # Apply colormap
            if img_tensor.dim() not in [2, 3]: # Should be 2D (H,W) or (1,H,W) for colormapping
                print(f"Colormap applied to unexpected tensor dim: {img_tensor.shape}. Expecting 2D or (1,H,W).")
                return

            # If it's (1, H, W), squeeze to (H, W) for colormapping
            if img_tensor.dim() == 3 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)
            
            # Convert to numpy array
            np_img = img_tensor.numpy()

            # Normalize values before colormapping if vmin/vmax are provided, or use auto-scaling
            if vmin is None:
                vmin = np_img.min()
            if vmax is None:
                vmax = np_img.max()

            # Get the colormap
            cmap = cm.get_cmap(colormap)
            
            # Apply colormap. cmap returns RGBA (H, W, 4) in float [0, 1]
            # Convert to RGB (H, W, 3) and then to uint8 (0-255)
            # TensorBoard expects uint8 for image summaries for better visualization range
            color_mapped_np = (cmap(np_img / (vmax - vmin) - vmin / (vmax - vmin))[:, :, :3] * 255).astype(np.uint8)

            # Convert back to torch.Tensor and permute to (C, H, W)
            log_tensor = torch.from_numpy(color_mapped_np).permute(2, 0, 1) # HWC -> CHW
            
        else:
            # No colormap, assume image is already in suitable format (e.g., grayscale 1xHxW or RGB 3xHxW)
            if img_tensor.dim() == 2:
                log_tensor = img_tensor.unsqueeze(0) # Grayscale (H, W) -> (1, H, W)
            elif img_tensor.dim() == 3:
                log_tensor = img_tensor # Already (C, H, W)
            else:
                print(f"Unexpected tensor dimension for image logging without colormap: {img_tensor.shape}. Skipping.")
                return

        logger_instance.experiment.add_image(name_tag, log_tensor, self.global_step)
        # print(f"Logged image '{name_tag}' at step {self.global_step}")


    def _log_prediction (self, pred_tensor, target_tensor, tag_prefix, logger_instance, colormap='bwr'):
        """
        Logs a sample prediction and its corresponding target.
        pred_tensor: The prediction tensor (e.g., from self.last_val_pred). Expected 4D (N, C, H, W).
        target_tensor: The target tensor (e.g., from self.last_val_target). Expected 4D (N, C, H, W).
        tag_prefix: Base string for the log name (e.g., "Val").
        logger_instance: The logger object (e.g., self.logger).
        colormap: The colormap to apply. Defaults to 'bwr'.
        """
        if self.trainer.sanity_checking:
            return

        if pred_tensor is None or target_tensor is None:
            print("Attempted to log image, but prediction or target tensor is None.")
            return
        
        if pred_tensor.dim() != 4 or pred_tensor.shape[0] == 0:
            print(f"Prediction tensor not 4D (N,C,H,W) or empty for logging: {pred_tensor.shape}. Skipping.")
            return
        if target_tensor.dim() != 4 or target_tensor.shape[0] == 0:
            print(f"Target tensor not 4D (N,C,H,W) or empty for logging: {target_tensor.shape}. Skipping.")
            return

        # Choose the first sample (index 0) and the first channel (index 0)
        sample_idx = 0 
        channel_idx = 0 

        # Extract the single image slice (H, W)
        # pred_img_slice = pred_tensor[sample_idx, channel_idx, :, :]
        # target_img_slice = target_tensor[sample_idx, channel_idx, :, :]
        pred_img_slice = pred_tensor.mean(axis=(0,1))
        target_img_slice = target_tensor.mean(axis=(0,1))

        # Determine vmin/vmax for consistent color mapping across prediction and target
        # This is important for comparing them visually
        all_values = target_img_slice.flatten()-pred_img_slice.flatten()
        vmin = all_values.min().item()
        vmax = all_values.max().item()

        self._squeeze_and_add_log_img(
            target_img_slice-pred_img_slice, 
            tag_prefix, 
            logger_instance, 
            colormap=colormap, 
            vmin=vmin, vmax=vmax
        )

    def training_step (self, batch, batch_idx):
        if self.supervised:
            x, y, mask = batch
        else:
            x, _, mask = batch
            y = x
        pred = self(x)
        mask = mask.to(self.device)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()

        # If pred has 2C channels, first half is mean
        if pred.shape[1] == 2 * y.shape[1]:
            mu = pred[:, : y.shape[1], ...]
        else:
            mu = pred

        if self.use_first_input:
            loss = call_loss(self.loss, mu, y, x0=x[0], mask=mask)
        else:
            loss = call_loss(self.loss, mu, y, mask=mask)

        mu = mu.contiguous()
        y  = y.contiguous()

        self.train_mae.update(mu, y, mask)
        self.train_rmse.update(mu, y, mask)
        self.train_scc.update(mu, y, mask)
        # self.train_acc.update(pred, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        try:
            comps = self.loss.loss_components
            for k,v in comps.items():
                self.log(f"train_{k}", v)
        except:
            pass
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True)
        self.log("train_scc", self.train_scc, on_step=False, on_epoch=True)
        # self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step (self, batch, batch_idx):
        if self.supervised:
            x, y, mask = batch
        else:
            x, _, mask = batch
            y = x

        pred = self(x)
        mask = mask.to(self.device)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()

        # If pred has 2C channels, first half is mean
        if pred.shape[1] == 2 * y.shape[1]:
            mu = pred[:, : y.shape[1], ...]
        else:
            mu = pred

        if self.use_first_input:
            loss = call_loss(self.loss, mu, y, x0=x[0], mask=mask)
        else:
            loss = call_loss(self.loss, mu, y, mask=mask)

        mu = mu.contiguous()
        y  = y.contiguous()

        self.val_mae.update(mu, y, mask)
        self.val_rmse.update(mu, y, mask)
        self.val_scc.update(mu, y, mask)
        # self.val_acc.update(mu, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        try:
            comps = self.loss.loss_components
            for k,v in comps.items():
                self.log(f"val_{k}", v)
        except:
            pass
        self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_scc", self.val_scc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.last_val_pred = pred.detach().cpu()
        self.last_val_target = y.detach().cpu()

    def test_step (self, batch, batch_idx):
        if self.supervised:
            x, y, mask = batch
        else:
            x, _, mask = batch
            y = x
        pred = self(x)
        mask = mask.to(self.device)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()

        # If pred has 2C channels, first half is mean
        if pred.shape[1] == 2 * y.shape[1]:
            mu = pred[:, : y.shape[1], ...]
        else:
            mu = pred

        if self.use_first_input:
            loss = call_loss(self.loss, mu, y, x0=x[0], mask=mask)
        else:
            loss = call_loss(self.loss, mu, y, mask=mask)

        mu = mu.contiguous()
        y  = y.contiguous()

        self.test_mae.update(mu, y, mask)
        self.test_rmse.update(mu, y, mask)
        self.test_scc.update(mu, y, mask)
        # self.test_acc.update(mu, y)

        # Log per-batch loss. Metrics will be logged at epoch end using their aggregated state.
        self.log("test_loss", loss, on_step=True, on_epoch=True, logger=True) # Corrected: use `loss`
        self.log("test_mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_scc", self.test_scc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.test_step_outputs.append({"preds": pred.detach().cpu(), "targets": y.detach().cpu()})

    def on_train_epoch_start (self):
        # Access the optimizer's learning rate
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]  # list of LRs, usually one
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True)
        # optimizer = self.optimizers()
        # current_lr = optimizer.param_groups[0]['lr']
        # self.log('lr', current_lr, on_step=False, on_epoch=True)
        print(f"Epoch {self.current_epoch}: learning Rate = {current_lr}")

    # def on_validation_epoch_end (self):
    #     """Process validation results, including logging a sample image"""
    #     if self.current_epoch % self.pic_log_interval == 0:
    #         self._log_prediction(self.last_val_pred, self.last_val_target, "Prediction Error", self.logger)
    #     # Clean up the stored tensors to prevent accidental reuse or memory issues
    #     self.last_val_pred = None
    #     self.last_val_target = None

    def on_test_epoch_end (self):
        """Process test results and store for external access"""
        # If you need to access the final aggregated metric values from the Test stage
        # *after* trainer.test() has completed, you can get them from the logger or from trainer.callback_metrics.
        final_test_mae = self.test_mae.compute()
        final_test_rmse = self.test_rmse.compute()
        final_test_scc = self.test_scc.compute()
        # final_test_acc = self.test_acc.compute()
        
        print(f"Test Results - "
            f"MAE: {final_test_mae:.4f}, "
            f"RMSE: {final_test_rmse:.4f}, "
            f"SCC: {final_test_scc:.4f}, "
            # f"ACC: {final_test_acc:.4f}"
        )

        # Store all test_preds/targets for post-test analysis
        if self.test_step_outputs: # Check if there were any batches
            self.test_preds = torch.cat([out["preds"] for out in self.test_step_outputs], dim=0)
            self.test_targets = torch.cat([out["targets"] for out in self.test_step_outputs], dim=0)
        self.test_step_outputs.clear()

    def configure_optimizers (self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

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

# Normalizer
class Normalize:
    def __init__ (self, mean=None, std=None):
        """
        mean/std expected shapes after fit: (1, C, 1, 1)
        """
        self.mean = mean
        self.std = std

    @staticmethod
    def _masked_stats (data: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12):
        """
        data: (N, C, H, W)
        mask: (N, C, H, W) bool/0-1
        returns mean/std: (1, C, 1, 1)
        """
        if data.ndim != 4 or mask.ndim != 4:
            raise ValueError(f"Expected (N,C,H,W), got data {data.shape}, mask {mask.shape}")

        mask = mask.bool()
        count = mask.sum(dim=(0, 2, 3), keepdim=True).clamp_min(1)  # (1,C,1,1)

        sum_vals = (data * mask).sum(dim=(0, 2, 3), keepdim=True)
        mean = sum_vals / count

        var = ((data - mean) ** 2 * mask).sum(dim=(0, 2, 3), keepdim=True) / count
        std = torch.sqrt(var + eps)
        return mean, std

    def fitted (self) -> bool:
        return self.mean is not None and self.std is not None

    def save (self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load (cls, filepath: str) -> "Normalize":
        return joblib.load(filepath)

    def fit (self, dataset, filepath: str | None = None, dim: str = "x", eps: float = 1e-12):
        """
        dataset.<dim>      : (N,C,H,W)
        dataset.<dim>_mask : (N,C,H,W)
        """
        data = getattr(dataset, dim)
        mask = getattr(dataset, f"{dim}_mask")

        self.mean, self.std = self._masked_stats(data, mask, eps=eps)

        if filepath is not None:
            self.save(filepath)
        return self

    def _check_filepath (self, filepath: str):
        if self.mean is None or self.std is None:
            try:
                self = self.load(filepath)
            except Exception as e:
                print(e)
                raise ValueError("Transform not fitted.")

    def _broadcast_params (self, t: torch.Tensor):
        """
        Return mean/std broadcastable to t without mutating state.
        """
        if not self.fitted():
            raise ValueError("Transform not fitted (mean/std are None)")

        if t.ndim == 4:      # (N,C,H,W)
            mean = self.mean
            std = self.std
        elif t.ndim == 3:    # (C,H,W)
            # expect stored (1,C,1,1) -> (C,1,1)
            mean = self.mean.squeeze(0)
            std = self.std.squeeze(0)
        else:
            raise ValueError(f"Unsupported tensor shape {t.shape}")

        if t.ndim == 3 and t.shape[0] != mean.shape[0]:
            raise ValueError(f"Channel mismatch: tensor has C={t.shape[0]}, normalizer has C={mean.shape[0]}")

        return mean.to(device=t.device, dtype=t.dtype), std.to(device=t.device, dtype=t.dtype)

    def __call__ (self, t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        mean, std = self._broadcast_params(t)
        return (t - mean) / (std + eps)

    def inverse_tensor (self, t: torch.Tensor, filepath: str) -> torch.Tensor:
        self._check_filepath(filepath)
        mean, std = self._broadcast_params(t)
        return t * std + mean

# Torch dataset
class XarrayDataset (Dataset):
    def __init__ (
        self,
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        target_realization_avg: bool = True,
        transform_x: Callable = None,
        transform_y: Callable = None,
        transform_x_args: dict = None,
        transform_y_args: dict = None,
        realization_as_channel: bool = False,
    ):
        """
        input_ds: xarray.Dataset
        target_ds: xarray.Dataset
        transform_x: callable with signature (x, **kwargs) -> x (for input)
        transform_y: callable with signature (y, **kwargs) -> y (for target)
        transform_x_args: dict of keyword args to pass to transform_x
        transform_y_args: dict of keyword args to pass to transform_y
        """
        self.target_ds = target_ds # .load(scheduler="synchronous")
        self.input_ds = input_ds
        assert isinstance(self.input_ds, xr.Dataset) and isinstance(self.target_ds, xr.Dataset), f"Input ds type: {self.input_ds}, target ds type: {self.target_ds}"

        self.transform_x, self.transform_y = transform_x, transform_y
        self.transform_x_args = transform_x_args or {}
        self.transform_y_args = transform_y_args or {}

        # NumPy arrays with NaNs for missing data
        x_np = self._transpose_dims_ds_to_da(self.input_ds).to_numpy()
        assert len(x_np.shape) > 3 and len(x_np.shape) < 6, f"Input has shape {x_np.shape}"
        # if x_np.ndim == 5 and x_np.shape[1] == 1:
        #     x_np = x_np.squeeze(1)          # C,T,H,W squeeze R dimension if present
        mask_x_np = np.isfinite(x_np)       # True where valid, False where masked
        x_np_filled = np.where(mask_x_np, x_np, 0.0)

        y_np = self._transpose_dims_ds_to_da(self.target_ds).to_numpy()
        assert len(y_np.shape) > 3 and len(y_np.shape) < 6, f"Target has shape {y_np.shape}"
        # if y_np.ndim == 5 and y_np.shape[1] == 1:
        #     y_np = y_np.squeeze(1)
        mask_y_np = np.isfinite(y_np)
        y_np_filled = np.where(mask_y_np, y_np, 0.0)

        # print(f"Dataset x mean: {np.nanmean(x_np)}, std: {np.nanstd(x_np)}")
        # print(f"Dataset y mean: {np.nanmean(y_np)}, std: {np.nanstd(y_np)}")
        # print("input vars:", list(input_ds.data_vars))
        # print("target vars:", list(target_ds.data_vars))
        # print(f"Input shape: {x_np.shape}, target shape: {y_np.shape}")

        if realization_as_channel:
            print("Realization as channel branch")
            # X: merge R into C
            if x_np_filled.ndim == 5:  # (C,T,R,H,W)
                C, T, R, H, W = x_np_filled.shape
                x_np_filled_ = x_np_filled.reshape(C * R, T, H, W)          # (C*R,T,H,W)
                mask_x_np_   = mask_x_np.reshape(C * R, T, H, W)
            else:  # (C,T,H,W)
                x_np_filled_ = x_np_filled                                  # (C,T,H,W)
                mask_x_np_   = mask_x_np

            self.x      = torch.tensor(x_np_filled_, dtype=torch.float32).permute(1, 0, 2, 3)  # (T,Cin,H,W)
            self.x_mask = torch.tensor(mask_x_np_,   dtype=torch.bool).permute(1, 0, 2, 3)     # (T,Cin,H,W)

            # Y: keep deterministic (always average if it has R, target_realization_avg ignored in this branch)
            if y_np_filled.ndim == 5:  # (C,T,R,H,W)
                # If deterministic target is stored with R=1, this just squeezes/averages safely.
                y_np_filled_ = np.nanmean(y_np_filled, axis=2)   # -> (C,T,H,W)
                mask_y_np_   = np.any(mask_y_np, axis=2)         # -> (C,T,H,W)
            else:  # (C,T,H,W)
                y_np_filled_ = y_np_filled
                mask_y_np_   = mask_y_np

            self.y      = torch.tensor(y_np_filled_, dtype=torch.float32).permute(1, 0, 2, 3)  # (T,Cout,H,W)
            self.y_mask = torch.tensor(mask_y_np_,   dtype=torch.bool).permute(1, 0, 2, 3)     # (T,Cout,H,W)

            # print("TENSORS self.x", self.x.shape, "self.y", self.y.shape, "self.y_mask", self.y_mask.shape)
            assert self.y.shape == self.y_mask.shape, (self.y.shape, self.y_mask.shape)

            # Basic spatial + sample checks (channels may differ)
            assert self.x.shape[0]  == self.y.shape[0], (f"Mismatched dataset shape: x={self.x.shape}, y={self.y.shape}")  # same T
            assert self.x.shape[2:] == self.y.shape[2:], (f"Mismatched dataset shape: x={self.x.shape}, y={self.y.shape}") # same H,W

        else:
            if target_realization_avg and len(y_np.shape) == 5:
                y_np_filled = np.nanmean(y_np_filled, axis=2)          # average over R
                mask_y_np = np.any(mask_y_np, axis=2)                  # valid if any realization is valid

            # Uniform realizations: only works if one of input and target has R>1 and the other R=1
            if len(x_np_filled.shape) == 5: # C,T,R,H,W
                self.x = torch.tensor(x_np_filled, dtype=torch.float32).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3) # (T*R),C,H,W
                self.x_mask = torch.tensor(mask_x_np, dtype=torch.bool).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
            else: # C,T,H,W
                if len(y_np_filled.shape) == 5: # C,T,R,H,W
                    R = y_np_filled.shape[2]
                    x = torch.tensor(x_np_filled, dtype=torch.float32)      # C,T,H,W
                    x = x.unsqueeze(2).repeat(1, 1, R, 1, 1)                # C,T,R,H,W
                    self.x = x.flatten(1, 2).permute(1, 0, 2, 3)            # (T*R),C,H,W
                    x_mask = torch.tensor(mask_x_np, dtype=torch.bool)      # C,T,H,W
                    x_mask = x_mask.unsqueeze(2).repeat(1, 1, R, 1, 1)      # C,T,R,H,W
                    self.x_mask = x_mask.flatten(1, 2).permute(1, 0, 2, 3)  # (T*R),C,H,W
                else:
                    self.x = torch.tensor(x_np_filled, dtype=torch.float32).permute(1, 0, 2, 3) # T,C,H,W
                    self.x_mask = torch.tensor(mask_x_np, dtype=torch.bool).permute(1, 0, 2, 3)
            if len(y_np_filled.shape) == 5:
                self.y = torch.tensor(y_np_filled, dtype=torch.float32).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
                self.y_mask = torch.tensor(mask_y_np, dtype=torch.bool).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
            else:
                if len(x_np_filled.shape) == 5: # C,T,R,H,W
                    R = x_np_filled.shape[2]
                    y = torch.tensor(y_np_filled, dtype=torch.float32)      # C,T,H,W
                    y = y.unsqueeze(2).repeat(1, 1, R, 1, 1)                # C,T,R,H,W
                    self.y = y.flatten(1, 2).permute(1, 0, 2, 3)            # (T*R),C,H,W
                    y_mask = torch.tensor(mask_y_np, dtype=torch.bool)
                    y_mask = y_mask.unsqueeze(2).repeat(1, 1, R, 1, 1)
                    self.y_mask = y_mask.flatten(1, 2).permute(1, 0, 2, 3)
                else:
                    self.y = torch.tensor(y_np_filled, dtype=torch.float32).permute(1, 0, 2, 3)
                    self.y_mask = torch.tensor(mask_y_np, dtype=torch.bool).permute(1, 0, 2, 3)

            # print("TENSORS self.x", self.x.shape, "self.y", self.y.shape, "self.y_mask", self.y_mask.shape)
            assert self.y.shape == self.y_mask.shape, (self.y.shape, self.y_mask.shape)

            # self.x = torch.tensor(self.input_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
            # self.y = torch.tensor(self.target_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)

            assert self.x.shape == self.y.shape, (f"Mismatched dataset shape: x={self.x.shape}, y={self.y.shape}")

    @staticmethod
    def _transpose_dims_ds_to_da (ds: xr.Dataset, excluded_vars: str | List[str] | None = "_has_var") -> xr.DataArray:
        excluded_vars = excluded_vars or []
        vars = [v for v in ds.data_vars if v not in excluded_vars]
        da = ds[vars].to_array()

        required_dims = {
            'time': guess_time_dim(ds),
            'y': guess_lat_dim(ds),
            'x': guess_lon_dim(ds),
        }
        realization_dim = guess_realization_dim(da)
        if set(required_dims.values()) - set(da.dims):
            raise ValueError(f"Unexpected dims: {da.dims}")

        if realization_dim in da.dims and da.ndim == 5:
            return da.transpose(
                "variable",
                required_dims['time'],
                realization_dim,
                required_dims['y'],
                required_dims['x'],
            )
        elif realization_dim not in da.dims and da.ndim == 4:
            return da.transpose(
                "variable",
                required_dims['time'],
                required_dims['y'],
                required_dims['x'],
            )
        else:
            return da

    def __len__ (self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]              # shapes: (C,H,W) or (C,T,H,W) if using time
        mx, my = self.x_mask[idx], self.y_mask[idx]

        if self.transform_x:
            x = self.transform_x(x, **self.transform_x_args)
        if self.transform_y:
            y = self.transform_y(y, **self.transform_y_args)

        mask = my           # use target mask
        # mask = mx & my    # combine input and target masks

        return x, y, mask
