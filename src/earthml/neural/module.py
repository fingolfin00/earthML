from typing import List, Any, Optional, Callable, Literal, Dict
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler, Dataset

import lightning as L

import matplotlib.cm as cm

from ..logging import get_logger
from .metrics import MaskedMAE, MaskedRMSE, MaskedSpatialCorr
from .utils import call_loss


logger = get_logger(__name__)


class EarthMLLightningModule(L.LightningModule):
    def __init__(
        self,
        use_full_batch_input_as_baseline: bool = False
    ):
        super().__init__()
        # self.extra_logger = extra_logger

        self.use_full_batch_input_as_baseline = use_full_batch_input_as_baseline

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
    def center_crop_to(
        x,
        target_h,
        target_w
    ):
        _, _, h, w = x.shape
        off_y = max((h - target_h) // 2, 0)
        off_x = max((w - target_w) // 2, 0)
        return x[:, :, off_y:off_y + target_h, off_x:off_x + target_w]

    def match_spatial(
        self,
        x,
        H,
        W
    ):
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

    def _squeeze_and_add_log_img(
        self,
        img_tensor,
        name_tag,
        logger_instance,
        colormap=None,
        vmin=None,
        vmax=None
    ):
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
                logger.warning(
                    "Colormap applied to unexpected tensor dim: %s. Expecting 2D or (1,H,W).",
                    img_tensor.shape,
                )
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
                logger.warning("Unexpected tensor dimension for image logging without colormap: %s. Skipping.", img_tensor.shape)
                return

        logger_instance.experiment.add_image(name_tag, log_tensor, self.global_step)
        # print(f"Logged image '{name_tag}' at step {self.global_step}")


    def _log_prediction(
        self,
        pred_tensor,
        target_tensor,
        tag_prefix,
        logger_instance,
        colormap: str = 'bwr'
    ):
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
            logger.warning("Attempted to log image, but prediction or target tensor is None.")
            return
        
        if pred_tensor.dim() != 4 or pred_tensor.shape[0] == 0:
            logger.warning("Prediction tensor not 4D (N,C,H,W) or empty for logging: %s. Skipping.", pred_tensor.shape)
            return
        if target_tensor.dim() != 4 or target_tensor.shape[0] == 0:
            logger.warning("Target tensor not 4D (N,C,H,W) or empty for logging: %s. Skipping.", target_tensor.shape)
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

    def training_step(
        self,
        batch,
        batch_idx
    ):
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

        if self.use_full_batch_input_as_baseline:
            loss = call_loss(self.loss, mu, y, x0=x, mask=mask)
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

    def validation_step(
        self,
        batch,
        batch_idx
    ):
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
        # # TODO make it more robust: very weak condition, what if we have input R=2 and target R=1?
        if pred.shape[1] == 2 * y.shape[1]:
            mu = pred[:, : y.shape[1], ...]
        else:
            mu = pred

        if self.use_full_batch_input_as_baseline:
            loss = call_loss(self.loss, mu, y, x0=x, mask=mask)
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
        
        self.last_val_pred = mu.detach().cpu()
        self.last_val_target = y.detach().cpu()

    def test_step(
        self,
        batch,
        batch_idx
    ):
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

        if self.use_full_batch_input_as_baseline:
            loss = call_loss(self.loss, mu, y, x0=x, mask=mask)
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

        self.test_step_outputs.append({"preds": mu.detach().cpu(), "targets": y.detach().cpu()})

    def on_train_epoch_start(self):
        # Access the optimizer's learning rate
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]  # list of LRs, usually one
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=True)
        # optimizer = self.optimizers()
        # current_lr = optimizer.param_groups[0]['lr']
        # self.log('lr', current_lr, on_step=False, on_epoch=True)
        logger.info("Epoch %s: learning Rate = %s", self.current_epoch, current_lr)

    # def on_validation_epoch_end (self):
    #     """Process validation results, including logging a sample image"""
    #     if self.current_epoch % self.pic_log_interval == 0:
    #         self._log_prediction(self.last_val_pred, self.last_val_target, "Prediction Error", self.logger)
    #     # Clean up the stored tensors to prevent accidental reuse or memory issues
    #     self.last_val_pred = None
    #     self.last_val_target = None

    def on_test_epoch_end(self):
        """Process test results and store for external access"""
        # If you need to access the final aggregated metric values from the Test stage
        # *after* trainer.test() has completed, you can get them from the logger or from trainer.callback_metrics.
        final_test_mae = self.test_mae.compute()
        final_test_rmse = self.test_rmse.compute()
        final_test_scc = self.test_scc.compute()
        # final_test_acc = self.test_acc.compute()
        
        logger.info(
            "Test Results - MAE: %.4f, RMSE: %.4f, SCC: %.4f",
            final_test_mae,
            final_test_rmse,
            final_test_scc,
        )

        # Store all test_preds/targets for post-test analysis
        if self.test_step_outputs: # Check if there were any batches
            self.test_preds = torch.cat([out["preds"] for out in self.test_step_outputs], dim=0)
            self.test_targets = torch.cat([out["targets"] for out in self.test_step_outputs], dim=0)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class EpochRandomSplitDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        train_fraction: float = 0.9,
        batch_size: int = 32,
        seed: int = 42,
        num_workers: int = 0,
        per_epoch_resplit: bool = False
    ):
        super().__init__()
        self.dataset = dataset
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.per_epoch_resplit = per_epoch_resplit

    def setup(self, stage=None):
        # initial split
        self._resplit()

    def _resplit(self):
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

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        return self._val_dl

    def on_train_epoch_start(self):
        if self.per_epoch_resplit:
            torch.manual_seed(self.seed + self.trainer.current_epoch)
            # re-split at every epoch
            self._resplit()
