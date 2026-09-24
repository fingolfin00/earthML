from typing import Literal
from collections.abc import Iterator

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, BatchSampler

import lightning as L
from lightning.pytorch.callbacks import Callback, EarlyStopping, RichProgressBar
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from torchmetrics import MetricCollection

from ..logging import get_logger
from .dataset import XarrayDataset, XarraySubset
from .metrics import (
    MaskedBias,
    MaskedGeoBias,
    MaskedGeoMAE,
    MaskedGeoRMSE,
    MaskedGeoSpatialCorr,
    MaskedGeoStdRatio,
    MaskedGeoTemporalCorr,
    MaskedMAE,
    MaskedRMSE,
    MaskedSpatialCorr,
    MaskedStdRatio,
    MaskedTemporalCorr,
)


logger = get_logger(__name__)


SplitStrategy = Literal[
    "explicit",  # separately supplied train and validation datasets
    "time",      # chronological percentage split by initialization time
    "random",    # random percentage split by initialization time
]

Stage = Literal[
    "train",
    "validation",
    "test",
]

# ------------------------------------------------------
# Callbacks
# ------------------------------------------------------

def rich_print(
    trainer: L.Trainer,
    message: str,
) -> None:
    progress_bar = trainer.progress_bar_callback

    if isinstance(progress_bar, RichProgressBar):
        progress = getattr(
            progress_bar,
            "progress",
            None,
        )

        if progress is not None:
            progress.console.print(message)

            logger.print(
                message,
                console=False,
            )
            return

    logger.print(message)


class RichEarlyStopping(EarlyStopping):
    @staticmethod
    def _log_info(
        trainer: L.Trainer,
        message: str,
        log_rank_zero_only: bool,
    ) -> None:
        if (
            log_rank_zero_only
            and trainer.global_rank != 0
        ):
            return

        rich_print(
            trainer,
            message,
        )

class LearningRateChangePrinter(Callback):
    def __init__(
        self,
        *,
        rtol: float = 1e-12,
        atol: float = 0.0,
    ) -> None:
        super().__init__()

        self.rtol = rtol
        self.atol = atol

        self.previous_lr: float | None = None

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        optimizer = trainer.optimizers[0]

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        if self.previous_lr is None:
            self.previous_lr = current_lr
            return

        if not np.isclose(
            current_lr,
            self.previous_lr,
            rtol=self.rtol,
            atol=self.atol,
        ):
            rich_print(
                trainer,
                (
                    f"[cyan]Epoch {trainer.current_epoch}: learning rate changed "
                    f"{self.previous_lr} → "
                    f"{current_lr}[/cyan]"
                ),
            )

        self.previous_lr = current_lr

# ------------------------------------------------------
# Module
# ------------------------------------------------------

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
        latitudes: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.optimizer_lr = optimizer_lr
        self.weight_decay = weight_decay

        # ------------------------------------------------------
        # Metric configuration
        # ------------------------------------------------------

        if latitudes is not None:
            latitudes = torch.as_tensor(
                latitudes,
                dtype=torch.float32,
            )

        self.register_buffer(
            "metric_latitudes",
            latitudes,
            persistent=True,
        )

        self._configure_metrics(latitudes)

        # ------------------------------------------------------
        # Test outputs
        # ------------------------------------------------------

        self.test_step_outputs: list[
            dict[str, torch.Tensor]
        ] = []

        self.test_preds: torch.Tensor | None = None
        self.test_targets: torch.Tensor | None = None
        self.test_months: torch.Tensor | None = None
        self.test_masks: torch.Tensor | None = None

    def _configure_metrics(
        self,
        latitudes: torch.Tensor | None,
    ) -> None:
        """
        Configure train, validation and test metrics.

        If latitude coordinates are available, use cosine-latitude
        weighted geographic metrics.

        Otherwise preserve the previous equal-grid-cell weighting.
        """
        if latitudes is None:
            base_metrics = MetricCollection(
                {
                    "bias": MaskedBias(),
                    "mae": MaskedMAE(),
                    "rmse": MaskedRMSE(),
                    "std_ratio": MaskedStdRatio(),
                    "tcc": MaskedTemporalCorr(),
                }
            )

            test_metrics = MetricCollection(
                {
                    "bias": MaskedBias(),
                    "mae": MaskedMAE(),
                    "rmse": MaskedRMSE(),
                    "std_ratio": MaskedStdRatio(),
                    "tcc": MaskedTemporalCorr(),
                    "scc": MaskedSpatialCorr(),
                },
                prefix="test_",
            )

        else:
            base_metrics = MetricCollection(
                {
                    "bias": MaskedGeoBias(
                        latitudes,
                    ),
                    "mae": MaskedGeoMAE(
                        latitudes,
                    ),
                    "rmse": MaskedGeoRMSE(
                        latitudes,
                    ),
                    "std_ratio": MaskedGeoStdRatio(
                        latitudes,
                    ),
                    "tcc": MaskedGeoTemporalCorr(
                        latitudes,
                    ),
                }
            )

            test_metrics = MetricCollection(
                {
                    "bias": MaskedGeoBias(
                        latitudes,
                    ),
                    "mae": MaskedGeoMAE(
                        latitudes,
                    ),
                    "rmse": MaskedGeoRMSE(
                        latitudes,
                    ),
                    "std_ratio": MaskedGeoStdRatio(
                        latitudes,
                    ),
                    "tcc": MaskedGeoTemporalCorr(
                        latitudes,
                    ),
                    "scc": MaskedGeoSpatialCorr(
                        latitudes,
                    ),
                },
                prefix="test_",
            )

        self.train_metrics = base_metrics.clone(
            prefix="train_",
        )

        self.val_metrics = base_metrics.clone(
            prefix="val_",
        )

        self.test_metrics = test_metrics

    def _log_loss_components(
        self,
        stage: Stage,
        batch_size: int,
    ) -> None:
        components = getattr(
            self.loss,
            "loss_components",
            None,
        )

        if not components:
            return

        for name, value in components.items():
            self.log(
                f"{stage}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
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

        # Probabilistic losses may return distribution parameters.
        # Metrics use only the predictive mean.
        if self.loss_name == "GaussianNLLFromLogits":
            mu, _ = torch.chunk(
                pred,
                2,
                dim=1,
            )
        else:
            mu = pred

        mu = mu.contiguous()

        batch_size = x.shape[0]

        if stage == "train":
            metrics = self.train_metrics

            self.log(
                "train_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )

        elif stage == "validation":
            metrics = self.val_metrics

            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )

        else:
            metrics = self.test_metrics

            self.log(
                "test_loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

        # ------------------------------------------------------
        # Metrics
        # ------------------------------------------------------

        metrics.update(
            mu,
            y,
            mask,
        )

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        # ------------------------------------------------------
        # Loss-specific components
        # ------------------------------------------------------

        self._log_loss_components(
            stage,
            batch_size=batch_size,
        )

        # ------------------------------------------------------
        # Store test predictions for post-test diagnostics/output
        # ------------------------------------------------------

        if stage == "test":
            self.test_step_outputs.append(
                {
                    "preds": mu.detach().cpu(),
                    "targets": y.detach().cpu(),
                    "months": months.detach().cpu(),
                    "mask": mask.detach().cpu(),
                }
            )

        return loss

    # ==========================================================
    # Spatial utilities
    # ==========================================================

    @staticmethod
    def center_crop_to(
        x: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        _, _, height, width = x.shape

        offset_y = max(
            (height - target_h) // 2,
            0,
        )

        offset_x = max(
            (width - target_w) // 2,
            0,
        )

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
        """
        Center-crop or replicate-pad x to the requested size.
        """
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
                (
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                ),
                mode="replicate",
            )

        return x

    # ==========================================================
    # Lightning steps
    # ==========================================================

    def training_step(
        self,
        batch,
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(
            batch,
            "train",
        )

    def validation_step(
        self,
        batch,
        batch_idx: int,
    ) -> None:
        self._shared_step(
            batch,
            "validation",
        )

    def test_step(
        self,
        batch,
        batch_idx: int,
    ) -> None:
        self._shared_step(
            batch,
            "test",
        )

    # ==========================================================
    # Epoch hooks
    # ==========================================================

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

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:
            return

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs.clear()

        self.test_preds = None
        self.test_targets = None
        self.test_months = None
        self.test_masks = None

    def on_test_epoch_end(self) -> None:
        test_metrics = self.test_metrics.compute()

        logger.info(
            (
                "Test Results - "
                "Bias: %.4f, "
                "MAE: %.4f, "
                "RMSE: %.4f, "
                "Std ratio: %.4f, "
                "TCC: %.4f, "
                "SCC: %.4f"
            ),
            test_metrics["test_bias"],
            test_metrics["test_mae"],
            test_metrics["test_rmse"],
            test_metrics["test_std_ratio"],
            test_metrics["test_tcc"],
            test_metrics["test_scc"],
        )

        if not self.test_step_outputs:
            raise RuntimeError(
                "Testing produced no prediction batches."
            )

        self.test_preds = torch.cat(
            [
                output["preds"]
                for output in self.test_step_outputs
            ],
            dim=0,
        )

        self.test_targets = torch.cat(
            [
                output["targets"]
                for output in self.test_step_outputs
            ],
            dim=0,
        )

        self.test_months = torch.cat(
            [
                output["months"]
                for output in self.test_step_outputs
            ],
            dim=0,
        )

        self.test_masks = torch.cat(
            [
                output["mask"]
                for output in self.test_step_outputs
            ],
            dim=0,
        )

        self.test_step_outputs.clear()

    # ==========================================================
    # Optimizer
    # ==========================================================

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        # TODO: allow passing optimizer and lr scheduler settings
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.optimizer_lr,
            weight_decay=self.weight_decay,
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                # mode="max", # in original NOAA implemenetation: probably a bug, definetely a bug
                factor=0.1,
                patience=4,
                min_lr=1e-8,
            )
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

    # ==========================================================
    # Loss dispatch
    # ==========================================================

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
            "HuberLoss",
            "GeoMSELoss",
        }:
            return self.loss(
                prediction,
                target,
            )

        if self.loss_name in {
            "MaskedMSELoss",
            "GeoMaskedMSELoss",
            "GaussianNLLFromLogits",
            "EmpiricalCRPSLoss",
            "SpatialCVaRMSELoss",
            "SpatialDegradationMSELoss",
        }:
            return self.loss(
                prediction,
                target,
                mask=mask,
            )

        if self.loss_name == "GeoMaskedMSEMultiScaleLoss":
            return self.loss(
                y_pred=prediction,
                y_true=target,
                x_input=model_input,
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
            f"No loss-call rule is defined for "
            f"{self.loss_name!r}"
        )


class SplitDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: XarrayDataset,
        val_dataset: XarrayDataset | None = None,
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
        group_batches_by_month: bool = False,
        train_subsamples: int | None = None,
        val_subsamples: int | None = None,
    ) -> None:
        super().__init__()

        if split_strategy != "explicit" and not 0.0 < train_fraction < 1.0:
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

        self._train_dataset: XarrayDataset | XarraySubset | None = None
        self._val_dataset: XarrayDataset | XarraySubset | None = None

        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers
        self.split_strategy = split_strategy
        self.shuffle_train = shuffle_train
        self.drop_last_train = drop_last_train
        self.group_batches_by_month = group_batches_by_month

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

        self.train_indices: list[int] | None = None
        self.val_indices: list[int] | None = None

        self.train_subsamples = train_subsamples
        self.val_subsamples = val_subsamples

        for name, value in (
            ("train_subsamples", self.train_subsamples),
            ("val_subsamples", self.val_subsamples),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0.")


    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        """Create the train/validation datasets.

        ``stage`` is ignored because the split is stage-independent.
        """

        if self._train_dataset is not None and self._val_dataset is not None:
            return

        if self.split_strategy == "explicit":
            assert self.explicit_val_dataset is not None

            self._train_dataset, self.train_indices = (
                self._subsample_explicit_dataset(
                    self.source_dataset,
                    self.train_subsamples,
                    partition="train",
                )
            )

            self._val_dataset, self.val_indices = (
                self._subsample_explicit_dataset(
                    self.explicit_val_dataset,
                    self.val_subsamples,
                    partition="val",
                )
            )

            return

        (
            self.train_indices,
            self.val_indices,
            train_time_indices,
            val_time_indices,
        ) = self._get_indices()

        self._train_dataset = XarraySubset(
            self.source_dataset,
            sample_indices=self.train_indices,
            time_indices=train_time_indices,
        )

        self._val_dataset = XarraySubset(
            self.source_dataset,
            sample_indices=self.val_indices,
            time_indices=val_time_indices,
        )


    def _samples_per_initialization(
        self,
        dataset: XarrayDataset,
    ) -> tuple[int, int]:
        return dataset.n_init_times, dataset.samples_per_init

    def _subsample_time_indices(
        self,
        time_indices: list[int],
        num_subsamples: int | None,
        *,
        partition: str,
    ) -> list[int]:
        if num_subsamples is None:
            return time_indices

        if num_subsamples > len(time_indices):
            raise ValueError(
                f"{partition}_subsamples cannot exceed the available "
                "initialization times in the partition: "
                f"requested={num_subsamples}, "
                f"available={len(time_indices)}."
            )

        generator = torch.Generator().manual_seed(self.seed)
        selected = torch.randperm(
            len(time_indices),
            generator=generator,
        )[:num_subsamples].tolist()

        return sorted(time_indices[i] for i in selected)

    def _subsample_explicit_dataset(
        self,
        dataset: XarrayDataset,
        num_subsamples: int | None,
        *,
        partition: str,
    ) -> tuple[XarrayDataset | XarraySubset, list[int] | None]:
        if num_subsamples is None:
            return dataset, None

        n_times, samples_per_time = self._samples_per_initialization(dataset)

        selected_times = self._subsample_time_indices(
            list(range(n_times)),
            num_subsamples,
            partition=partition,
        )

        sample_indices = self._expand_time_indices(
            selected_times,
            samples_per_time,
        )

        return (
            XarraySubset(
                dataset,
                sample_indices=sample_indices,
                time_indices=selected_times,
            ),
            sample_indices,
        )

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

    def _get_indices(
        self,
    ) -> tuple[
        list[int], # train_sample_indices
        list[int], # val_sample_indices
        list[int], # train_time_indices
        list[int], # val_time_indices
    ]:
        if self.split_strategy == "explicit":
            raise RuntimeError(
                "_get_indices() is not used for "
                "split_strategy='explicit'."
            )

        n_times, samples_per_time = self._samples_per_initialization(
            self.source_dataset
        )

        time_indices = list(range(n_times))

        n_selected_times = len(time_indices)
        n_train_times = int(n_selected_times * self.train_fraction)

        if not 0 < n_train_times < n_selected_times:
            raise ValueError(
                "The requested split produces an empty partition: "
                f"n_times={n_selected_times}, "
                f"train_fraction={self.train_fraction}, "
                f"n_train_times={n_train_times}."
            )

        if self.split_strategy == "time":
            train_time_indices = time_indices[:n_train_times]
            val_time_indices = time_indices[n_train_times:]

        elif self.split_strategy == "random":
            generator = torch.Generator().manual_seed(self.seed)

            shuffled = torch.randperm(
                n_selected_times,
                generator=generator,
            ).tolist()

            train_time_indices = sorted(
                time_indices[i]
                for i in shuffled[:n_train_times]
            )
            val_time_indices = sorted(
                time_indices[i]
                for i in shuffled[n_train_times:]
            )

        else:
            raise ValueError(
                f"Unknown split_strategy={self.split_strategy!r}"
            )

        train_time_indices = self._subsample_time_indices(
            train_time_indices,
            self.train_subsamples,
            partition="train",
        )
        val_time_indices = self._subsample_time_indices(
            val_time_indices,
            self.val_subsamples,
            partition="val",
        )

        train_sample_indices = self._expand_time_indices(
            train_time_indices,
            samples_per_time,
        )

        val_sample_indices = self._expand_time_indices(
            val_time_indices,
            samples_per_time,
        )

        return (
            train_sample_indices,
            val_sample_indices,
            train_time_indices,
            val_time_indices,
        )

    def train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset

        if self.group_batches_by_month:
            batch_sampler = SingleMonthBatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                drop_last=self.drop_last_train,
                seed=self.seed,
            )

            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last_train,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self.val_dataset

        if self.group_batches_by_month:
            batch_sampler = SingleMonthBatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                seed=self.seed,
            )

            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @property
    def train_dataset(self) -> XarrayDataset | XarraySubset:
        if self._train_dataset is None:
            raise RuntimeError("setup() must be called before accessing train_dataset.")
        return self._train_dataset

    @property
    def val_dataset(self) -> XarrayDataset | XarraySubset:
        if self._val_dataset is None:
            raise RuntimeError("setup() must be called before accessing val_dataset.")
        return self._val_dataset


class SingleMonthBatchSampler(BatchSampler):
    """Yield batches containing samples from exactly one calendar month."""

    def __init__(
        self,
        dataset: XarrayDataset | XarraySubset,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}"
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        months = self._extract_months(dataset)

        if len(months) != len(dataset):
            raise ValueError(
                f"Month count {len(months)} does not match "
                f"dataset length {len(dataset)}"
            )

        self.month_to_indices: dict[int, list[int]] = {}

        for index, month in enumerate(months):
            month = int(month)

            if not 1 <= month <= 12:
                raise ValueError(
                    f"Invalid month {month} at sample {index}"
                )

            self.month_to_indices.setdefault(month, []).append(index)

        if not self.month_to_indices:
            raise ValueError("Dataset contains no samples")

    @staticmethod
    def _extract_months(
        dataset: XarrayDataset | XarraySubset,
    ) -> torch.Tensor:
        months = torch.as_tensor(
            dataset.months,
            dtype=torch.long,
        )

        if months.ndim != 1:
            raise ValueError(
                f"Expected months shape (N,), got {tuple(months.shape)}"
            )

        return months

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        batches: list[list[int]] = []

        for month in sorted(self.month_to_indices):
            indices = torch.as_tensor(
                self.month_to_indices[month],
                dtype=torch.long,
            )

            if self.shuffle:
                permutation = torch.randperm(
                    len(indices),
                    generator=generator,
                )
                indices = indices[permutation]

            for start in range(0, len(indices), self.batch_size):
                batch = indices[
                    start : start + self.batch_size
                ].tolist()

                if len(batch) < self.batch_size and self.drop_last:
                    continue

                batches.append(batch)

        # Also mix the order of months/batches. Batch contents remain
        # month-homogeneous.
        if self.shuffle and batches:
            order = torch.randperm(
                len(batches),
                generator=generator,
            ).tolist()

            batches = [batches[index] for index in order]

        # Produce a different deterministic order next epoch even when
        # Lightning does not explicitly call set_epoch().
        self.epoch += 1

        yield from batches

    def __len__(self) -> int:
        total = 0

        for indices in self.month_to_indices.values():
            count = len(indices)

            if self.drop_last:
                total += count // self.batch_size
            else:
                total += (
                    count + self.batch_size - 1
                ) // self.batch_size

        return total
