from dataclasses import dataclass
from typing import Sequence

import numpy as np
import xarray as xr

from ..logging import get_logger


logger = get_logger(__name__)


@dataclass
class MetricDims:
    time: str
    latitude: str
    longitude: str
    realization: str | None
    leadtime: str | None


@dataclass
class MetricData:
    """
    Container for truth, model, and auxiliary datasets used in metric evaluation.

    This class standardizes and validates input datasets by:
    - Ensuring consistent list-based structure for model inputs
    - Aligning all datasets along shared coordinates (inner join)
    - Applying a common validity mask (either user-provided or computed)
    - Renaming dimensions to a standardized convention via `get_and_rename_dim`

    Notes
    -----
    - All datasets are aligned using `xr.align(..., join="inner")`, so only
      overlapping coordinates are retained.
    - Masking is applied uniformly across truth and model datasets:
        * If `mask_data` is provided, it defines valid points.
        * Otherwise, only points with finite values across ALL variables and
          ALL datasets are retained.
    - The class assumes that all datasets are compatible in terms of variables
      and dimensions after alignment.
    - `clim_data` is stored but NOT processed (no alignment, masking, or renaming).
      It is the caller's responsibility to ensure its consistency if used.

    Parameters
    ----------
    truth_data : xr.Dataset
        Reference dataset against which models are evaluated.

    model_data : list[xr.Dataset]
        List of model datasets to evaluate. Must be non-empty and aligned
        in structure with `truth_data`.

    clim_data : list[xr.Dataset | None]
        Optional climatology datasets. Expected to have length
        `len(model_data) + 1` (first entry typically corresponds to truth).
        These datasets are NOT modified or validated internally.

    mask_data : xr.Dataset or None
        Optional dataset defining valid grid points. Must be alignable with
        truth and model datasets. All variables must evaluate to True at a
        point for it to be considered valid.

    truth_name : str
        Name identifier for the truth dataset.

    model_names : list[str]
        Names corresponding to each dataset in `model_data`.

    dims : MetricDims
        Standardized dimension names after renaming.
    """
    truth_data: xr.Dataset
    model_data: list[xr.Dataset]
    clim_data: list[xr.Dataset | None]
    mask_data: xr.Dataset | None
    truth_name: str
    model_names: list[str]
    dims: MetricDims

    @classmethod
    def from_inputs(
        cls,
        truth_data: xr.Dataset,
        model_data: xr.Dataset | Sequence[xr.Dataset],
        truth_name: str,
        model_names: str | Sequence[str],
        clim_data: xr.Dataset | Sequence[xr.Dataset] | None = None,
        mask_data: xr.Dataset | None = None,
    ) -> "MetricData":
        """
        Construct a `MetricData` instance from raw input datasets.

        This method performs input normalization, validation, alignment, masking,
        and dimension standardization.

        Steps performed
        ---------------
        1. Normalize `model_data` and `model_names` to lists.
        2. Validate matching lengths and non-empty inputs.
        3. Optionally validate `clim_data` length (must equal len(model_data) + 1).
        4. Remove bookkeeping variables (currently only "_has_var").
        5. Align all datasets using an inner join over shared coordinates.
        6. Compute or apply a validity mask:
            - If `mask_data` is provided, use it directly.
            - Otherwise, require all values across all datasets and variables to be finite.
        7. Apply the mask to all datasets.
        8. Standardize dimension names via `get_and_rename_dim`.

        Parameters
        ----------
        truth_data : xr.Dataset
            Reference dataset.

        model_data : xr.Dataset or sequence of xr.Dataset
            One or more model datasets.

        truth_name : str
            Name of the truth dataset.

        model_names : str or sequence of str
            Name(s) corresponding to each model dataset.

        clim_data : xr.Dataset or sequence of xr.Dataset or None, optional
            Optional climatology datasets. If provided, must have length
            `len(model_data) + 1`. These datasets are NOT processed internally.

        mask_data : xr.Dataset or None, optional
            Optional dataset specifying valid points.

        Returns
        -------
        MetricData
            A fully validated and aligned `MetricData` instance.

        Raises
        ------
        ValueError
            If:
            - `model_data` is empty
            - `model_names` length does not match `model_data`
            - `clim_data` has incorrect length
        """
        model_data = [model_data] if isinstance(model_data, xr.Dataset) else list(model_data)
        if not model_data:
            raise ValueError("model_data must contain at least one dataset")

        model_names = [model_names] if isinstance(model_names, str) else list(model_names)
        if len(model_names) != len(model_data):
            raise ValueError(
                f"model_data and model_names must have the same length "
                f"({len(model_data)} != {len(model_names)})"
            )

        if clim_data is None:
            clim_data = [None] * (len(model_data) + 1)
        else:
            clim_data = [clim_data] if isinstance(clim_data, xr.Dataset) else list(clim_data)
            if len(clim_data) != len(model_data) + 1:
                raise ValueError(
                    f"clim_data must have length len(model_data) + 1 "
                    f"({len(clim_data)} != {len(model_data) + 1})"
                )

        # Drop bookkeeping vars
        truth_data = truth_data[[v for v in truth_data.data_vars if v != "_has_var"]]
        model_data = [
            ds[[v for v in ds.data_vars if v != "_has_var"]]
            for ds in model_data
        ]

        truth_data = truth_data.earthml.normalize_dims_and_coords()
        model_data = [ds.earthml.normalize_dims_and_coords() for ds in model_data]

        guessed_dims = model_data[0].earthml.guessed_dims
        time_dim = guessed_dims.time
        lat_dim = guessed_dims.latitude
        lon_dim = guessed_dims.longitude
        realization_dim = guessed_dims.realization
        leadtime_dim = guessed_dims.leadtime

        def _broadcast_singleton_realization(
            ds: xr.Dataset,
            target_values: np.ndarray | None,
        ) -> xr.Dataset:
            if realization_dim is None or target_values is None or target_values.size <= 1:
                return ds

            if realization_dim in ds.dims:
                if ds.sizes.get(realization_dim, 0) == 1:
                    ds = ds.squeeze(realization_dim, drop=True)
                    ds = ds.expand_dims({realization_dim: target_values})
                    ds = ds.assign_coords({realization_dim: target_values})
                return ds

            if realization_dim in ds.coords and ds[realization_dim].ndim == 0:
                ds = ds.drop_vars(realization_dim, errors="ignore")

            return ds.expand_dims({realization_dim: target_values}).assign_coords({realization_dim: target_values})

        target_realization_values = None
        for ds in model_data:
            if (
                realization_dim is not None
                and realization_dim in ds.dims
                and realization_dim in ds.coords
                and ds.sizes.get(realization_dim, 0) > 1
            ):
                target_realization_values = ds[realization_dim].values
                break

        if mask_data is not None:
            mask_data = mask_data.earthml.normalize_dims_and_coords()

        if target_realization_values is not None:
            truth_data = _broadcast_singleton_realization(truth_data, target_realization_values)
            model_data = [
                _broadcast_singleton_realization(ds, target_realization_values)
                for ds in model_data
            ]
            if mask_data is not None:
                mask_data = _broadcast_singleton_realization(mask_data, target_realization_values)
            clim_data = [
                _broadcast_singleton_realization(ds, target_realization_values) if ds is not None else None
                for ds in clim_data
            ]

        # Align data and mask
        datasets = [truth_data] + model_data
        if mask_data is not None:
            aligned = xr.align(*datasets, mask_data, join="inner")
            *core_aligned, mask_aligned = aligned
            valid = mask_aligned.to_array().all("variable")
        else:
            core_aligned = xr.align(*datasets, join="inner")
            masks = [np.isfinite(ds.to_array()).all("variable") for ds in core_aligned]
            valid = masks[0]
            for mask in masks[1:]:
                valid = valid & mask

        # Cast to avoid mixed types
        core_aligned = [
            ds.astype({
                v: np.float32
                for v in ds.data_vars
                if np.issubdtype(ds[v].dtype, np.floating)
            })
            for ds in core_aligned
        ]

        masked = [ds.where(valid) for ds in core_aligned]

        truth_data = masked[0]
        model_data = masked[1:]

        return cls(
            truth_data=truth_data,
            model_data=model_data,
            clim_data=clim_data,
            mask_data=mask_data,
            truth_name=truth_name,
            model_names=model_names,
            dims=MetricDims(
                time=time_dim,
                latitude=lat_dim,
                longitude=lon_dim,
                realization=realization_dim,
                leadtime=leadtime_dim,
            ),
        )

    def summary(self) -> None:
        """
        Print a summary of dataset dimensions for truth and model inputs.
        """
        logger.info("Metrics initialized with truth and model shapes:")
        logger.info("  %s: %s", self.truth_name, dict(self.truth_data.sizes))
        for ds, name in zip(self.model_data, self.model_names):
            logger.info("  %s: %s", name, dict(ds.sizes))
