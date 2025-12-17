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
from .utils import _guess_dim_name

class EarthMLLightningModule (L.LightningModule):
    def __init__ (self, use_first_input=False):
        super().__init__()
        # self.extra_logger = extra_logger

        self.use_first_input = use_first_input

        # Metrics
        self.train_mae = MeanAbsoluteError()
        self.train_rmse = MeanSquaredError(squared=False) # squared=False for RMSE
        self.train_scc = SpatialCorrelationCoefficient()
        # self.train_acc = AnomalyCorrelationCoefficient()

        self.val_mae = MeanAbsoluteError()
        self.val_rmse = MeanSquaredError(squared=False) # squared=False for RMSE
        self.val_scc = SpatialCorrelationCoefficient() # Your custom SCC
        # self.val_acc = AnomalyCorrelationCoefficient() # Your custom ACC

        self.test_mae = MeanAbsoluteError()
        self.test_rmse = MeanSquaredError(squared=False)
        self.test_scc = SpatialCorrelationCoefficient()
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
            x, y = batch
        else:
            x, _ = batch
            y = x
        pred = self(x)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()

        if self.use_first_input:
            loss = self.loss(pred, y, x[0])
        else:
            loss = self.loss(pred, y)

        self.train_mae.update(pred, y)
        self.train_rmse.update(pred, y)
        self.train_scc.update(pred, y)
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
            x, y = batch
        else:
            x, _ = batch
            y = x

        pred = self(x)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()

        if self.use_first_input:
            loss = self.loss(pred, y, x[0])
        else:
            loss = self.loss(pred, y)

        self.val_mae.update(pred, y)
        self.val_rmse.update(pred, y)
        self.val_scc.update(pred, y)
        # self.val_acc.update(pred, y)

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
            x, y = batch
        else:
            x, _ = batch
            y = x
        pred = self(x)
        # ensure contiguous
        pred = pred.contiguous()
        y = y.contiguous()
        loss = self.loss(pred, y) # Corrected: use `loss`

        self.test_mae.update(pred, y)
        self.test_rmse.update(pred, y)
        self.test_scc.update(pred, y)
        # self.test_acc.update(pred, y)

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

class HeteroBiasCorrectionLoss(nn.Module):
    """
    Loss for bias correction with:
    - variance-normalized MSE
    - identity-preserving term when true bias is small
    """
    def __init__(
        self,
        lambda_identity: float = 0.1,
        bias_scale: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.bias_scale = bias_scale
        self.eps = eps
        self.loss_components = None

    def forward(
        self,
        y_pred: torch.Tensor,        # corrected forecast (field)
        y_true: torch.Tensor,        # analysis
        x_input: torch.Tensor,       # raw forecast
        var_field: torch.Tensor | None = None,
    ):
        # --- variance-normalized MSE term ---
        sq_err = (y_pred - y_true) ** 2

        if var_field is not None:
            denom = var_field + self.eps
        else:
            # per-channel variance estimated from batch
            est_var = y_true.var(dim=(0, 2, 3), keepdim=True)
            denom = est_var + self.eps

        var_norm_mse = (sq_err / denom).mean()

        # --- identity-preserving term ---
        # true bias between raw input and target
        true_bias = y_true - x_input
        bias_mag = true_bias.abs()

        # weight strong where true bias is small
        w_id = torch.exp(-bias_mag / self.bias_scale)

        # penalize changing the input when its bias is small
        id_err = (y_pred - x_input) ** 2
        identity_loss = (w_id * id_err).mean()

        total_loss = var_norm_mse + self.lambda_identity * identity_loss

        self.loss_components = {
            "var_norm_mse": var_norm_mse.detach(),
            "identity_loss": identity_loss.detach(),
        }
        return total_loss

class Normalize:
    def __init__ (self, mean=None, std=None):
        """
        mean, std: optional scalars or tensors for normalization.
                   If None, must call fit(dataset) before using.
        """
        self.mean = mean
        self.std = std

    @staticmethod
    def _masked_stats(data, mask):
        """
        Compute per-channel mean/std using only masked-valid entries.

        data: [N, C, H, W] float tensor (possibly contains zeros)
        mask: [N, C, H, W] bool or 0/1 tensor
        """
        mask = mask.bool()
        # Number of valid entries per-channel
        count = mask.sum(dim=(0, 2, 3), keepdim=True)  # shape [1, C, 1, 1]

        # Avoid divide-by-zero for channels with no valid data
        count = count.clamp(min=1)
        print(f"Normalization, masked count: {count}")

        # Masked mean
        sum_vals = (data * mask).sum(dim=(0, 2, 3), keepdim=True)
        mean = sum_vals / count

        # Masked variance
        var = ((data - mean) ** 2 * mask).sum(dim=(0, 2, 3), keepdim=True) / count
        std = torch.sqrt(var + 1e-6)

        return mean, std

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

    def fit(self, dataset, filepath, dim="x"):
        """
        Fit mean/std only on valid pixels.
        dataset.<dim>      = tensor [N, C, H, W]
        dataset.<dim>_mask = tensor [N, C, H, W]
        dim: select the data to fit, input (x) or target (y)
        """
        data = getattr(dataset, dim)
        mask = getattr(dataset, f"{dim}_mask")

        self.mean, self.std = self._masked_stats(data, mask)

        print(f"Masked (mean, std) for normalization: ({self.mean.flatten().tolist()}, {self.std.flatten().tolist()})")

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

        # Remove time dimension because we call this in Dataset __getitem__
        x = self._norm(x, epsilon=epsilon).squeeze(0)
        # print(f"X normalized shape: {x.shape}, mean: {x.mean()}, std: {x.std()}")
        y = self._norm(y, epsilon=epsilon).squeeze(0)
        # print(f"Y normalized shape: {y.shape}, mean: {y.mean()}, std: {y.std()}")
        return x, y

class XarrayDataset (Dataset):
    def __init__(self, input_ds: xr.Dataset, target_ds: xr.Dataset, transform=None, transform_args=None):
        """
        input_ds: xarray.Dataset
        target_ds: xarray.Dataset
        transform: callable with signature (x, y, **kwargs) -> (x, y)
        transform_args: dict of keyword args to pass to transform
        """
        self.target_ds = target_ds # .load(scheduler="synchronous")
        self.input_ds = input_ds
        assert isinstance(self.input_ds, xr.Dataset) and isinstance(self.target_ds, xr.Dataset), f"Input ds type: {self.input_ds}, target ds type: {self.target_ds}"

        self.transform = transform
        self.transform_args = transform_args or {}

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

        # Realizations
        if len(x_np.shape) == 5: # C,T,R,H,W
            self.x = torch.tensor(x_np_filled, dtype=torch.float32).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3) # (T*R),C,H,W
            self.x_mask = torch.tensor(mask_x_np, dtype=torch.bool).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
        else: # C,T,H,W
            if len(y_np.shape) == 5: # C,T,R,H,W
                R = y_np.shape[2]
                x = torch.tensor(x_np_filled, dtype=torch.float32)      # C,T,H,W
                x = x.unsqueeze(2).repeat(1, 1, R, 1, 1)                # C,T,R,H,W
                self.x = x.flatten(1, 2).permute(1, 0, 2, 3)            # (T*R),C,H,W
                x_mask = torch.tensor(mask_x_np, dtype=torch.bool)      # C,T,H,W
                x_mask = x_mask.unsqueeze(2).repeat(1, 1, R, 1, 1)      # C,T,R,H,W
                self.x_mask = x_mask.flatten(1, 2).permute(1, 0, 2, 3)  # (T*R),C,H,W
            else:
                self.x = torch.tensor(x_np_filled, dtype=torch.float32).permute(1, 0, 2, 3) # T,C,H,W
                self.x_mask = torch.tensor(mask_x_np, dtype=torch.bool).permute(1, 0, 2, 3)
        if len(y_np.shape) == 5:
            self.y = torch.tensor(y_np_filled, dtype=torch.float32).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
            self.y_mask = torch.tensor(mask_y_np, dtype=torch.bool).flatten(start_dim=1, end_dim=2).permute(1, 0, 2, 3)
        else:
            if len(x_np.shape) == 5: # C,R,T,H,W
                R = x_np.shape[2]
                y = torch.tensor(y_np_filled, dtype=torch.float32)      # C,T,H,W
                y = y.unsqueeze(2).repeat(1, 1, R, 1, 1)                # C,T,R,H,W
                self.y = y.flatten(1, 2).permute(1, 0, 2, 3)            # (T*R),C,H,W
                y_mask = torch.tensor(mask_y_np, dtype=torch.bool)
                y_mask = y_mask.unsqueeze(2).repeat(1, 1, R, 1, 1)
                self.y_mask = y_mask.flatten(1, 2).permute(1, 0, 2, 3)
            else:
                self.y = torch.tensor(y_np_filled, dtype=torch.float32).permute(1, 0, 2, 3)
                self.y_mask = torch.tensor(mask_y_np, dtype=torch.bool).permute(1, 0, 2, 3)

        # self.x = torch.tensor(self.input_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        # self.y = torch.tensor(self.target_ds.to_array().values, dtype=torch.float32).permute(1, 0, 2, 3)
        assert len(self.x) == len(self.y), f"Mismatched dataset length: x={len(self.x)}, y={len(self.y)}"

    @staticmethod
    def _transpose_dims_ds_to_da (ds: xr.Dataset) -> xr.DataArray:
        da = ds.to_array()
        required_dims = {
            'time': _guess_dim_name(ds, "time", ['valid_time', 'time_counter']),
            'y': _guess_dim_name(ds, "y", ['lat', 'latitude', 'nav_lat']),
            'x': _guess_dim_name(ds, "x", ['lon', 'longitude', 'nav_lon']),
        }
        if set(required_dims.values()) - set(da.dims):
            raise ValueError(f"Unexpected dims: {da.dims}")

        if 'realization' in da.dims and da.ndim == 5:
            return da.transpose(
                "variable",
                required_dims['time'],
                "realization",
                required_dims['y'],
                required_dims['x'],
            )
        elif 'realization' not in da.dims and da.ndim == 4:
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

    def __getitem__ (self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x, y = self.transform(x, y, **self.transform_args)
        return x, y
