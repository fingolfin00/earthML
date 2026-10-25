from pathlib import Path
from typing import Literal, Sequence

import joblib, torch

from ..logging import get_logger
from .dataset import XarrayDataset, XarraySubset


logger = get_logger(__name__)


NormalizationMode = float | int | torch.Tensor | Literal["global", "channel", "gridpoint", "sample"]


class Normalize:
    """Independently configure mean and standard deviation.

    Numeric/tensor values are fixed. Global/channel/gridpoint statistics are
    fitted on training data. Sample statistics are calculated dynamically.
    Data has shape (N,C,H,W), or (C,H,W) for a single sample.
    """

    def __init__(
        self,
        mean: NormalizationMode = "channel",
        std: NormalizationMode = "channel",
        *,
        eps: float = 1e-6,
        exclude_channels: Sequence[int] | None = None,
    ) -> None:
        self.mean_spec = mean
        self.std_spec = std
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.eps = eps
        self.exclude_channels = tuple(exclude_channels or ())
        self.n_channels: int | None = None
        self.included_channels: tuple[int, ...] | None = None
        self.excluded_channels: tuple[int, ...] | None = None
        for spec in (mean, std):
            if isinstance(spec, str) and spec not in {
                "global", "channel", "gridpoint", "sample"
            }:
                raise ValueError(f"Unknown statistic specification: {spec!r}")


    def fitted(self) -> bool:
        return (
            self.n_channels is not None
            and (self.mean is not None or self._is_sample(self.mean_spec))
            and (self.std is not None or self._is_sample(self.std_spec))
        )


    @staticmethod
    def _is_sample(spec: NormalizationMode) -> bool:
        return isinstance(spec, str) and spec == "sample"

    @property
    def uses_sample_stats(self) -> bool:
        return (
            self._is_sample(self.mean_spec)
            or self._is_sample(self.std_spec)
        )


    @staticmethod
    def _reduce_dims(spec: str) -> tuple[int, ...]:
        return {
            "global": (0, 1, 2, 3),
            "channel": (0, 2, 3),
            "gridpoint": (0,),
        }[spec]


    @staticmethod
    def _prepare_mask(
        data: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if data.ndim not in (3, 4):
            raise ValueError(
                "Expected data shape (N,C,H,W) or (C,H,W), "
                f"got {tuple(data.shape)}"
            )

        finite_mask = torch.isfinite(data)

        if mask is None:
            return finite_mask

        if mask.shape != data.shape:
            raise ValueError(
                f"Mask shape mismatch: expected {tuple(data.shape)}, "
                f"got {tuple(mask.shape)}"
            )

        return (
            mask.to(device=data.device, dtype=torch.bool)
            & finite_mask
        )


    @staticmethod
    def _masked_mean_std(
        data: torch.Tensor,
        mask: torch.Tensor,
        *,
        reduce_dims: tuple[int, ...],
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calc_dtype = (
            torch.float32
            if data.dtype in (torch.float16, torch.bfloat16)
            else data.dtype
        )

        data = data.to(dtype=calc_dtype)
        mask = mask.to(device=data.device, dtype=torch.bool)

        count = mask.sum(
            dim=reduce_dims,
            keepdim=True,
        )

        if torch.any(count == 0):
            empty_groups = torch.nonzero(
                count == 0,
                as_tuple=False,
            )
            raise ValueError(
                "Cannot calculate normalization statistics because "
                f"some groups contain no valid values: {empty_groups.tolist()}"
            )

        count = count.to(dtype=calc_dtype)

        safe_data = torch.where(
            mask,
            data,
            torch.zeros(
                (),
                device=data.device,
                dtype=calc_dtype,
            ),
        )

        mean = safe_data.sum(
            dim=reduce_dims,
            keepdim=True,
        ) / count

        centered = torch.where(
            mask,
            data - mean,
            torch.zeros(
                (),
                device=data.device,
                dtype=calc_dtype,
            ),
        )

        variance = centered.square().sum(
            dim=reduce_dims,
            keepdim=True,
        ) / count

        std = torch.sqrt(variance).clamp_min(eps)

        return mean, std


    def _resolve_channels(
        self,
        n_channels: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        resolved_excluded: set[int] = set()

        for channel in self.exclude_channels:
            resolved = channel if channel >= 0 else n_channels + channel

            if not 0 <= resolved < n_channels:
                raise ValueError(
                    f"Excluded channel index {channel} is invalid for "
                    f"{n_channels} channels."
                )

            resolved_excluded.add(resolved)

        excluded = tuple(sorted(resolved_excluded))
        included = tuple(
            channel
            for channel in range(n_channels)
            if channel not in resolved_excluded
        )

        if not included:
            raise ValueError(
                "All input channels were excluded from normalization."
            )

        return included, excluded


    @staticmethod
    def _validate_statistic(
        name: Literal["mean", "std"],
        value: torch.Tensor,
        data_shape: tuple[int, ...],
    ) -> None:
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")

        if name == "std" and not torch.all(value > 0):
            raise ValueError("std must be positive")

        try:
            result_shape = torch.broadcast_shapes(data_shape, value.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} cannot broadcast "
                f"to data shape {data_shape}"
            ) from exc

        if result_shape != data_shape:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} would expand "
                f"data shape {data_shape} to {result_shape}"
            )


    def _set_channel_configuration(
        self,
        n_channels: int,
    ) -> None:
        included, excluded = self._resolve_channels(n_channels)

        self.n_channels = n_channels
        self.included_channels = included
        self.excluded_channels = excluded


    def _check_channel_configuration(
        self,
        tensor: torch.Tensor,
    ) -> None:
        if self.n_channels is None or self.included_channels is None:
            raise ValueError("Normalizer channel configuration is not initialized")

        if tensor.ndim not in (3, 4):
            raise ValueError(
                "Expected tensor shape (N,C,H,W) or (C,H,W), "
                f"got {tuple(tensor.shape)}"
            )

        channel_dim = 1 if tensor.ndim == 4 else 0
        actual_channels = tensor.shape[channel_dim]

        if actual_channels != self.n_channels:
            raise ValueError(
                f"Normalizer was fitted with {self.n_channels} channels, "
                f"but received a tensor with {actual_channels} channels"
            )


    def _select_included_channels(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._check_channel_configuration(tensor)

        assert self.included_channels is not None

        indices = torch.as_tensor(
            self.included_channels,
            device=tensor.device,
            dtype=torch.long,
        )

        channel_dim = 1 if tensor.ndim == 4 else 0
        return tensor.index_select(channel_dim, indices)


    def _restore_included_channels(
        self,
        original: torch.Tensor,
        transformed: torch.Tensor,
    ) -> torch.Tensor:
        self._check_channel_configuration(original)

        assert self.included_channels is not None

        indices = torch.as_tensor(
            self.included_channels,
            device=original.device,
            dtype=torch.long,
        )

        result = original.clone()
        channel_dim = 1 if original.ndim == 4 else 0

        result.index_copy_(
            channel_dim,
            indices,
            transformed,
        )

        return result


    def fit(
        self,
        dataset: XarrayDataset | XarraySubset,
        *,
        dim: Literal["x", "y"] = "x",
        filepath: str | Path | None = None,
    ) -> "Normalize":
        data = getattr(dataset, dim)
        mask = getattr(dataset, f"{dim}_mask", None)

        if data.ndim != 4:
            raise ValueError(f"Expected (N,C,H,W), got {tuple(data.shape)}")

        self._set_channel_configuration(data.shape[1])

        included = self._select_included_channels(data)

        included_mask = self._select_included_channels(mask) if mask is not None else None
        valid_mask = self._prepare_mask(included, included_mask)

        fitted_stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for name, spec in (("mean", self.mean_spec), ("std", self.std_spec)):
            if self._is_sample(spec):
                value = None

            elif isinstance(spec, str):
                if spec not in fitted_stats:
                    fitted_stats[spec] = self._masked_mean_std(
                        data=included,
                        mask=valid_mask,
                        reduce_dims=self._reduce_dims(spec),
                        eps=self.eps,
                    )
                value = fitted_stats[spec][0 if name == "mean" else 1]

            else:
                value = torch.as_tensor(spec, dtype=data.dtype)

                if value.ndim == 1:
                    value = value.reshape(1, -1, 1, 1)
                elif value.ndim == 3:
                    value = value.unsqueeze(0)

            if value is not None:
                value = value.detach().cpu()

                self._validate_statistic(
                    name,
                    value,
                    (1, *included.shape[1:]),
                )

            setattr(self, name, value)

        if filepath is not None:
            self.save(filepath)

        return self


    def _resolve_mean_or_std(
        self,
        name: Literal["mean", "std"],
        included: torch.Tensor,
        provided: torch.Tensor | None,
        sample_value: torch.Tensor | None,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        spec = self.mean_spec if name == "mean" else self.std_spec

        if self._is_sample(spec):
            value = provided if provided is not None else sample_value

            if value is None:
                raise ValueError(
                    f"Inverse normalization needs the original sample {name}"
                    if inverse else f"Missing sample {name}"
                )

        else:
            if provided is not None:
                raise ValueError(f"Cannot override non-sample {name}")

            value = getattr(self, name)

        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)

        if included.ndim == 3 and value.ndim == 4 and value.shape[0] == 1:
            value = value.squeeze(0)

        value = value.to(device=included.device, dtype=included.dtype)

        if self._is_sample(spec):
            expected_shape = (
                (*included.shape[:2], 1, 1)
                if included.ndim == 4
                else (included.shape[0], 1, 1)
            )

            if value.shape != expected_shape:
                raise ValueError(
                    f"Sample {name} must have shape {expected_shape}, "
                    f"got {tuple(value.shape)}"
                )

        self._validate_statistic(
            name,
            value,
            tuple(included.shape),
        )

        return value


    def resolve_sample_mean_or_std(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked spatial mean/std for each sample and included channel"""

        included = self._select_included_channels(tensor)

        reduce_dims = (2, 3) if included.ndim == 4 else (1, 2)

        included_mask = self._select_included_channels(mask) if mask is not None else None
        valid_mask = self._prepare_mask(included, included_mask)

        return self._masked_mean_std(
            data=included,
            mask=valid_mask,
            reduce_dims=reduce_dims,
            eps=self.eps,
        )


    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fitted():
            raise ValueError("Normalizer is not fitted")

        included = self._select_included_channels(tensor)

        sample_mean = sample_std = None

        if self.uses_sample_stats and (
            (self._is_sample(self.mean_spec) and mean is None)
            or (self._is_sample(self.std_spec) and std is None)
        ):
            sample_mean, sample_std = self.resolve_sample_mean_or_std(tensor, mask=mask)

        mu = self._resolve_mean_or_std("mean", included, mean, sample_mean, inverse=False)
        sigma = self._resolve_mean_or_std("std", included, std, sample_std, inverse=False)

        normalized = (included - mu) / sigma

        return self._restore_included_channels(tensor, normalized)


    def inverse_tensor(
        self,
        tensor: torch.Tensor,
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not self.fitted():
            raise ValueError("Normalizer is not fitted")

        included = self._select_included_channels(tensor)

        mu = self._resolve_mean_or_std("mean", included, mean, None, inverse=True)
        sigma = self._resolve_mean_or_std("std", included, std, None, inverse=True)

        return self._restore_included_channels(tensor, included * sigma + mu)


    def save(
        self,
        filepath: str | Path,
    ) -> None:
        if not self.fitted():
            raise ValueError("Cannot save an unfitted normalizer")

        joblib.dump(self, filepath)


    @classmethod
    def load(
        cls,
        filepath: str | Path,
    ) -> "Normalize":
        loaded = joblib.load(filepath)

        if not isinstance(loaded, cls) or not loaded.fitted():
            raise ValueError("Loaded object is not a fitted Normalize")

        return loaded


class MonthlyNormalize(Normalize):
    """
    Monthly configurable normalizer.

    Resulting stored shapes:

    global:
        (12, 1, 1, 1)

    channel:
        (12, C, 1, 1)

    gridpoint:
        (12, C, H, W)
    """

    def fit(
        self,
        dataset,
        *,
        dim: str = "x",
        filepath: str | Path | None = None,
    ) -> "MonthlyNormalize":
        data = getattr(dataset, dim)
        mask = getattr(dataset, f"{dim}_mask", None)

        if data.ndim != 4:
            raise ValueError(
                "Expected dataset data shape (N,C,H,W), "
                f"got {tuple(data.shape)}"
            )

        self._set_channel_configuration(data.shape[1])

        data_for_fit = self._select_included_channels(data)

        if mask is not None:
            mask_for_fit = self._select_included_channels(mask)
        else:
            mask_for_fit = None

        months = torch.as_tensor(
            dataset.months,
            device=data.device,
            dtype=torch.long,
        )

        if months.ndim != 1:
            raise ValueError(
                f"Expected months shape (N,), got {tuple(months.shape)}"
            )

        if months.shape[0] != data.shape[0]:
            raise ValueError(
                f"Month count {months.shape[0]} does not match "
                f"sample count {data.shape[0]}"
            )

        if torch.any((months < 1) | (months > 12)):
            raise ValueError("Month values must be between 1 and 12")

        valid_mask = self._prepare_mask(
            data_for_fit,
            mask_for_fit,
        )

        means: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []

        reduce_dims = self._reduce_dims()

        for month in range(1, 13):
            selection = months == month

            if not torch.any(selection):
                raise ValueError(
                    f"No samples available for month {month}"
                )

            mean, std = self._masked_mean_std(
                data_for_fit[selection],
                valid_mask[selection],
                reduce_dims=reduce_dims,
                eps=self.eps,
            )

            # Remove the singleton sample/month axis.
            means.append(mean.squeeze(0))
            stds.append(std.squeeze(0))

        self.mean = torch.stack(means).detach().cpu()
        self.std = torch.stack(stds).detach().cpu()

        if filepath is not None:
            self.save(filepath)

        return self


    def _monthly_params_for(
        self,
        tensor: torch.Tensor,
        months: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fitted():
            raise ValueError("Normalizer is not fitted")

        months = torch.as_tensor(
            months,
            device=tensor.device,
            dtype=torch.long,
        )

        if torch.any((months < 1) | (months > 12)):
            raise ValueError("Month values must be between 1 and 12")

        all_mean = self.mean.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
        all_std = self.std.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )

        if tensor.ndim == 3:
            if months.numel() != 1:
                raise ValueError(
                    "A (C,H,W) tensor requires exactly one month"
                )

            month_idx = int(months.item()) - 1
            mean = all_mean[month_idx]
            std = all_std[month_idx]

        elif tensor.ndim == 4:
            if months.ndim != 1:
                raise ValueError(
                    "For batched data, months must have shape (N,)"
                )

            if months.shape[0] != tensor.shape[0]:
                raise ValueError(
                    f"Month count {months.shape[0]} does not match "
                    f"batch size {tensor.shape[0]}"
                )

            indices = months - 1
            mean = all_mean.index_select(0, indices)
            std = all_std.index_select(0, indices)

        else:
            raise ValueError(
                "Expected tensor shape (N,C,H,W) or (C,H,W), "
                f"got {tuple(tensor.shape)}"
            )

        try:
            torch.broadcast_shapes(
                tensor.shape,
                mean.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                f"Monthly parameter shape {tuple(mean.shape)} cannot "
                f"broadcast to tensor shape {tuple(tensor.shape)}"
            ) from exc

        return mean, std


    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        months: torch.Tensor | int,
        **kwargs,
    ) -> torch.Tensor:
        included = self._select_included_channels(tensor)

        mean, std = self._monthly_params_for(
            included,
            months,
        )

        normalized = (included - mean) / std

        return self._restore_included_channels(
            tensor,
            normalized,
        )


    def inverse_tensor(
        self,
        tensor: torch.Tensor,
        *,
        months: torch.Tensor | int,
        **kwargs,
    ) -> torch.Tensor:
        included = self._select_included_channels(tensor)

        mean, std = self._monthly_params_for(
            included,
            months,
        )

        restored = included * std + mean

        return self._restore_included_channels(
            tensor,
            restored,
        )
