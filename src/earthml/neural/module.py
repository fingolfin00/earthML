from typing import Literal

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from ..logging import get_logger
from .metrics import MaskedMAE, MaskedRMSE, MaskedSpatialCorr


logger = get_logger(__name__)

SplitStrategy = Literal[
    "explicit",  # separately supplied train and validation datasets
    "time",  # chronological percentage split by initialization time
    "random",  # random percentage split by initialization time
]

Stage = Literal["train", "validation", "test"]


class EarthMLLightningModule(L.LightningModule):
    optimizer_lr: float
    weight_decay: float

    loss: nn.Module
    loss_name: str
    supervised: bool

    def __init__(
        self,
        optimizer_lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()

        self.optimizer_lr = optimizer_lr
        self.weight_decay = weight_decay

        self.train_mae = MaskedMAE()
        self.train_rmse = MaskedRMSE()

        self.val_mae = MaskedMAE()
        self.val_rmse = MaskedRMSE()

        self.test_mae = MaskedMAE()
        self.test_rmse = MaskedRMSE()
        self.test_scc = MaskedSpatialCorr()

        self.test_step_outputs: list[dict[str, torch.Tensor]] = []
        self.test_preds: torch.Tensor | None = None
        self.test_targets: torch.Tensor | None = None
        self.test_months: torch.Tensor | None = None
        self.test_masks: torch.Tensor | None = None

    def _log_loss_components(self, stage: Stage) -> None:
        components = getattr(self.loss, "loss_components", None)
        if not components:
            return

        for name, value in components.items():
            self.log(
                f"{stage}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

    def _shared_step(
        self,
        batch,
        stage: Stage,
    ) -> torch.Tensor:
        if self.supervised:
            x, y, mask, months = batch
        else:
            x, _, mask, months = batch
            y = x

        pred = self(x).contiguous()
        y = y.contiguous()
        mask = mask.to(self.device)

        loss = self.compute_loss(
            prediction=pred,
            target=y,
            mask=mask,
            model_input=x,
            months=months,
        )

        # Probabilistic losses may return distribution parameters. Metrics use
        # only the predictive mean.
        if self.loss_name == "GaussianNLLFromLogits":
            mu, _ = torch.chunk(pred, 2, dim=1)
        else:
            mu = pred

        mu = mu.contiguous()
        batch_size = x.shape[0]

        if stage == "train":
            self.train_mae.update(mu, y, mask)
            self.train_rmse.update(mu, y, mask)

            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "train_mae",
                self.train_mae,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "train_rmse",
                self.train_rmse,
                on_step=False,
                on_epoch=True,
            )

        elif stage == "validation":
            self.val_mae.update(mu, y, mask)
            self.val_rmse.update(mu, y, mask)

            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "val_mae",
                self.val_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "val_rmse",
                self.val_rmse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        else:
            self.test_mae.update(mu, y, mask)
            self.test_rmse.update(mu, y, mask)
            self.test_scc.update(mu, y, mask)

            self.log(
                "test_loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )
            self.log(
                "test_mae",
                self.test_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test_rmse",
                self.test_rmse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "test_scc",
                self.test_scc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            self.test_step_outputs.append(
                {
                    "preds": mu.detach().float().cpu(),
                    "targets": y.detach().float().cpu(),
                    "mask": mask.detach().cpu(),
                    "months": months.detach().cpu(),
                }
            )

        self._log_loss_components(stage)
        return loss

    @staticmethod
    def center_crop_to(
        x: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        _, _, height, width = x.shape
        offset_y = max((height - target_h) // 2, 0)
        offset_x = max((width - target_w) // 2, 0)

        return x[
            :,
            :,
            offset_y : offset_y + target_h,
            offset_x : offset_x + target_w,
        ]

    def match_spatial(
        self,
        x: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Center-crop or replicate-pad ``x`` to the requested size."""
        _, _, height, width = x.shape
        delta_h = target_h - height
        delta_w = target_w - width

        if delta_h == 0 and delta_w == 0:
            return x

        if delta_h < 0 or delta_w < 0:
            x = self.center_crop_to(
                x,
                min(height, target_h),
                min(width, target_w),
            )
            _, _, height, width = x.shape
            delta_h = target_h - height
            delta_w = target_w - width

        if delta_h != 0 or delta_w != 0:
            pad_left = delta_w // 2
            pad_right = delta_w - pad_left
            pad_top = delta_h // 2
            pad_bottom = delta_h - pad_top

            x = F.pad(
                x,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="replicate",
            )

        return x

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "validation")

    def test_step(self, batch, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def on_train_epoch_start(self) -> None:
        scheduler = self.lr_schedulers()
        current_lr = scheduler.get_last_lr()[0]

        self.log(
            "lr",
            current_lr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        logger.info(
            "Epoch %s: learning rate = %s",
            self.current_epoch,
            current_lr,
        )

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs.clear()
        self.test_preds = None
        self.test_targets = None
        self.test_masks = None
        self.test_months = None

    def on_test_epoch_end(self) -> None:
        final_test_mae = self.test_mae.compute()
        final_test_rmse = self.test_rmse.compute()
        final_test_scc = self.test_scc.compute()

        logger.info(
            "Test Results - MAE: %.4f, RMSE: %.4f, SCC: %.4f",
            final_test_mae,
            final_test_rmse,
            final_test_scc,
        )

        if not self.test_step_outputs:
            raise RuntimeError("Testing produced no prediction batches.")

        self.test_preds = torch.cat(
            [output["preds"] for output in self.test_step_outputs],
            dim=0,
        )

        self.test_targets = torch.cat(
            [output["targets"] for output in self.test_step_outputs],
            dim=0,
        )

        self.test_months = torch.cat(
            [output["months"] for output in self.test_step_outputs],
            dim=0,
        )

        self.test_masks = torch.cat(
            [output["mask"] for output in self.test_step_outputs],
            dim=0,
        )

        self.test_step_outputs.clear()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def compute_loss(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        model_input: torch.Tensor | None = None,
        var_field: torch.Tensor | None = None,
        months: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_name in {
            "MSELoss",
            "GeoMSELoss",
        }:
            return self.loss(prediction, target)

        if self.loss_name in {
            "MaskedMSELoss",
            "GeoMaskedMSELoss",
            "GaussianNLLFromLogits",
            "EmpiricalCRPSLoss",
        }:
            return self.loss(
                prediction,
                target,
                mask=mask,
            )

        if self.loss_name == "GeoMaskedMSELowFreqLoss":
            return self.loss(
                prediction,
                target,
                mask=mask,
                months=months,
            )

        if self.loss_name == "VarNormMaskMSELoss":
            return self.loss(
                y_pred=prediction,
                y_true=target,
                var_field=var_field,
                mask=mask,
            )

        if self.loss_name == "HeteroBiasCorrectionLoss":
            if model_input is None:
                raise ValueError(
                    "HeteroBiasCorrectionLoss requires model_input."
                )

            return self.loss(
                y_pred=prediction,
                y_true=target,
                x_input=model_input,
                var_field=var_field,
                mask=mask,
            )

        raise ValueError(
            f"No loss-call rule is defined for {self.loss_name!r}"
        )


class SplitDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        *,
        train_fraction: float = 0.9,
        batch_size: int = 32,
        seed: int = 42,
        num_workers: int = 0,
        split_strategy: SplitStrategy = "time",
        shuffle_train: bool = True,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        drop_last_train: bool = False,
    ) -> None:
        super().__init__()

        if not 0.0 < train_fraction < 1.0:
            raise ValueError(
                "train_fraction must be between 0 and 1, "
                f"got {train_fraction}"
            )

        if split_strategy == "explicit" and val_dataset is None:
            raise ValueError(
                "val_dataset is required when split_strategy='explicit'."
            )

        if split_strategy != "explicit" and val_dataset is not None:
            raise ValueError(
                "val_dataset must be None unless split_strategy='explicit'."
            )

        self.source_dataset = train_dataset
        self.explicit_val_dataset = val_dataset

        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.split_strategy = split_strategy
        self.shuffle_train = shuffle_train
        self.drop_last_train = drop_last_train

        self.pin_memory = (
            torch.cuda.is_available()
            if pin_memory is None
            else pin_memory
        )
        self.persistent_workers = (
            num_workers > 0
            if persistent_workers is None
            else persistent_workers
        )

        if self.persistent_workers and self.num_workers == 0:
            raise ValueError(
                "persistent_workers=True requires num_workers > 0."
            )

        self.train_dataset: Dataset
        self.val_dataset: Dataset
        self.train_indices: list[int] | None = None
        self.val_indices: list[int] | None = None

    def _samples_per_initialization(self) -> tuple[int, int]:
        """
        Return the number of initialization times and samples per time.

        Samples are assumed to be flattened in time-major order.
        """
        dataset = self.source_dataset

        if not hasattr(dataset, "input_ds"):
            raise TypeError(
                f"split_strategy={self.split_strategy!r} requires a dataset "
                "with an input_ds time dimension."
            )

        input_ds = dataset.input_ds
        time_dim = input_ds.earthml.guessed_dims.time

        if time_dim is None or time_dim not in input_ds.dims:
            raise ValueError(
                "Could not determine the initialization-time dimension."
            )

        n_times = int(input_ds.sizes[time_dim])
        n_samples = len(dataset)

        if n_samples % n_times != 0:
            raise ValueError(
                "Cannot group samples by initialization time: "
                f"n_samples={n_samples}, n_times={n_times}."
            )

        return n_times, n_samples // n_times

    @staticmethod
    def _expand_time_indices(
        time_indices: list[int],
        samples_per_time: int,
    ) -> list[int]:
        sample_indices: list[int] = []

        for time_idx in time_indices:
            start = time_idx * samples_per_time
            sample_indices.extend(
                range(start, start + samples_per_time)
            )

        return sample_indices

    def _get_indices(self) -> tuple[list[int], list[int]]:
        if self.split_strategy == "explicit":
            raise RuntimeError(
                "_get_indices() is not used for "
                "split_strategy='explicit'."
            )

        n_times, samples_per_time = self._samples_per_initialization()
        n_train_times = int(n_times * self.train_fraction)

        if not 0 < n_train_times < n_times:
            raise ValueError(
                "The requested split produces an empty partition: "
                f"n_times={n_times}, "
                f"train_fraction={self.train_fraction}, "
                f"n_train_times={n_train_times}."
            )

        if self.split_strategy == "time":
            train_time_indices = list(range(n_train_times))
            val_time_indices = list(range(n_train_times, n_times))

        elif self.split_strategy == "random":
            generator = torch.Generator().manual_seed(self.seed)
            shuffled_times = torch.randperm(
                n_times,
                generator=generator,
            ).tolist()

            train_time_indices = sorted(
                shuffled_times[:n_train_times]
            )
            val_time_indices = sorted(
                shuffled_times[n_train_times:]
            )

        else:
            raise ValueError(
                f"Unknown split_strategy={self.split_strategy!r}"
            )

        return (
            self._expand_time_indices(
                train_time_indices,
                samples_per_time,
            ),
            self._expand_time_indices(
                val_time_indices,
                samples_per_time,
            ),
        )

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "train_dataset") and hasattr(self, "val_dataset"):
            return

        if self.split_strategy == "explicit":
            self.train_dataset = self.source_dataset
            assert self.explicit_val_dataset is not None
            self.val_dataset = self.explicit_val_dataset
            self.train_indices = None
            self.val_indices = None
            return

        self.train_indices, self.val_indices = self._get_indices()
        self.train_dataset = Subset(
            self.source_dataset,
            self.train_indices,
        )
        self.val_dataset = Subset(
            self.source_dataset,
            self.val_indices,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )
