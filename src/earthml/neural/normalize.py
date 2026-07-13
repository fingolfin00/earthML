from pathlib import Path
from typing import Literal

import joblib, torch

from ..logging import get_logger


logger = get_logger(__name__)


NormalizationMode = Literal[
    "global",
    "channel",
    "gridpoint",
]


class Normalize:
    """
    Configurable tensor normalizer.

    Supported modes
    ---------------
    global:
        One mean/std over samples, channels and spatial dimensions.
        Stored shape: (1, 1, 1, 1)

    channel:
        One mean/std per channel over samples and spatial dimensions.
        Stored shape: (1, C, 1, 1)

    gridpoint:
        One mean/std per channel and grid point over samples.
        Stored shape: (1, C, H, W)
    """

    def __init__(
        self,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        *,
        mode: NormalizationMode = "channel",
        eps: float = 1e-6,
    ) -> None:
        self.mean = mean
        self.std = std
        self.mode = mode
        self.eps = eps

    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def _reduce_dims(self) -> tuple[int, ...]:
        match self.mode:
            case "global":
                return (0, 1, 2, 3)
            case "channel":
                return (0, 2, 3)
            case "gridpoint":
                return (0,)
            case _:
                raise ValueError(
                    f"Unsupported normalization mode: {self.mode!r}"
                )

    @staticmethod
    def _prepare_mask(
        data: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if data.ndim != 4:
            raise ValueError(
                "Expected data shape (N,C,H,W), "
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

    def fit(
        self,
        dataset,
        *,
        dim: str = "x",
        filepath: str | Path | None = None,
    ) -> "Normalize":
        data = getattr(dataset, dim)
        mask = getattr(dataset, f"{dim}_mask", None)

        valid_mask = self._prepare_mask(data, mask)

        mean, std = self._masked_mean_std(
            data,
            valid_mask,
            reduce_dims=self._reduce_dims(),
            eps=self.eps,
        )

        self.mean = mean.detach().cpu()
        self.std = std.detach().cpu()

        if filepath is not None:
            self.save(filepath)

        return self

    def _params_for(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.fitted():
            raise ValueError("Normalizer is not fitted")

        if tensor.ndim == 4:
            mean = self.mean
            std = self.std
        elif tensor.ndim == 3:
            mean = self.mean.squeeze(0)
            std = self.std.squeeze(0)
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
                f"Stored parameter shape {tuple(mean.shape)} cannot be "
                f"broadcast to tensor shape {tuple(tensor.shape)}"
            ) from exc

        return (
            mean.to(
                device=tensor.device,
                dtype=tensor.dtype,
            ),
            std.to(
                device=tensor.device,
                dtype=tensor.dtype,
            ),
        )

    def __call__(
        self,
        tensor: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        mean, std = self._params_for(tensor)
        return (tensor - mean) / std

    def inverse_tensor(
        self,
        tensor: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        mean, std = self._params_for(tensor)
        return tensor * std + mean

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

        if not isinstance(loaded, cls):
            raise TypeError(
                f"Expected {cls.__name__}, "
                f"got {type(loaded).__name__}"
            )

        if not loaded.fitted():
            raise ValueError("Loaded normalizer is not fitted")

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

        valid_mask = self._prepare_mask(data, mask)

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
                data[selection],
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
        mean, std = self._monthly_params_for(
            tensor,
            months,
        )
        return (tensor - mean) / std

    def inverse_tensor(
        self,
        tensor: torch.Tensor,
        *,
        months: torch.Tensor | int,
        **kwargs,
    ) -> torch.Tensor:
        mean, std = self._monthly_params_for(
            tensor,
            months,
        )
        return tensor * std + mean
