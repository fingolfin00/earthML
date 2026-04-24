from abc import ABC, abstractmethod
from dataclasses import field
from typing import Sequence
import math

from copy import deepcopy
from pathlib import Path

import time
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

from dask.distributed import wait, progress

from ..base.dataclasses import TimeRange
from ..logging import get_logger

from .utils import save_zarr
from .dataclasses import DataSource, Sample


logger = get_logger(__name__)


class BaseSource(ABC):
    def __init__(
        self,
        datasource: DataSource,
    ):
        self.datasource = deepcopy(datasource) # deepcopy on entry to avoid unwanted mutations
        self.data_selection = self.datasource.data_selection
        self.source_name = datasource.source
        self.date_range = self.generate_date_range(self.data_selection.period)
        if len(self.date_range) == 0:
            raise ValueError(
                f"Generated empty date range for source={self.source_name} "
                f"from period={self.data_selection.period}"
            )
        # self.data_selection.period.start = self.date_range[0] # this may change the periods generated in the launcher script!
        # self.data_selection.period.end = self.date_range[-1]
        logger.info("Requested date range: %s to %s", self.data_selection.period.start, self.data_selection.period.end)
        logger.info("Effective date range: %s to %s, length %s", self.date_range[0], self.date_range[-1], len(self.date_range))
        # print(self.date_range)

        # Init
        self.elements = Sample()
        self.ds = None
        variables = self.data_selection.variable
        self.var_name_list = [v.name for v in variables] if isinstance(variables, Sequence) else [variables.name]
        self.select_area_after_request = False
        self.regrid_resolution = None
        self.convert_unit = None
        self.corruption_check_mode = "light"

    def __add__(self, other: "BaseSource") -> "BaseSource":
        if not isinstance(other, BaseSource):
            return NotImplemented
        from .combinators import SumSource
        return SumSource(self, other)

    def __radd__(self, other: "BaseSource") -> "BaseSource":
        # so sum([s1, s2, s3]) works
        if other == 0:
            return self
        return self.__add__(other)


    @staticmethod
    def generate_date_range(period: TimeRange) -> pd.DatetimeIndex | xr.CFTimeIndex | list[pd.Timestamp]:
        freq = period.freq
        start = period.start
        end = period.end
        shifted = period.shifted
        # Try to interpret freq as a Timedelta (works for H, D, etc., but not M/Y)
        try:
            freq_td = pd.to_timedelta(freq)
        except (TypeError, ValueError):
            freq_td = None
        # Only shift `end` forward for sub-daily frequencies
        if freq_td is not None and freq_td < pd.to_timedelta('24h'):
            end = end + freq_td
        dr = xr.date_range(
            start=start,
            end=end,
            freq=freq, # original string
            inclusive='both'
        )
        if shifted:
            dr = [d + relativedelta(**shifted) for d in dr]
        return dr


    @abstractmethod
    def _get_data(self) -> xr.Dataset:
        """
        Get data for the given data selection. Implement in subclasses.
        """
        pass


    def _remove_corrupted_timesteps(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        time_dim = ds.earthml.guessed_dims.time
        if time_dim is None:
            return ds

        mode = getattr(self, "corruption_check_mode", "full")
        if mode not in {"bookkeeping", "light", "full"}:
            raise ValueError(
                f"Unsupported corruption_check_mode={mode!r}. "
                "Expected one of {'bookkeeping', 'light', 'full'}."
            )

        status_keep = None
        if "_has_var" in ds and time_dim in ds["_has_var"].dims:
            status_da = ds["_has_var"]
            reduce_dims = [d for d in status_da.dims if d != time_dim]
            status_keep = status_da.any(dim=reduce_dims) if reduce_dims else status_da.astype(bool)

        if mode == "bookkeeping":
            if status_keep is None:
                logger.warning(
                    "%s: corruption_check_mode='bookkeeping' requested but _has_var is unavailable; "
                    "falling back to full scan",
                    self.source_name,
                )
                mode = "full"
            else:
                keep = status_keep
                logger.info("%s: using _has_var bookkeeping to determine valid timesteps", self.source_name)

        elif mode == "light":
            if status_keep is not None:
                keep = status_keep
                logger.info(
                    "%s: using light corruption check (_has_var plus per-variable non-null scan)",
                    self.source_name,
                )
            else:
                keep = xr.ones_like(ds[time_dim], dtype=bool)
                logger.info(
                    "%s: _has_var unavailable, using light corruption check without bookkeeping",
                    self.source_name,
                )

            for var in [v for v in ds.data_vars if v != "_has_var"]:
                da = ds[var]
                if time_dim not in da.dims:
                    continue

                reduce_dims = [d for d in da.dims if d != time_dim]
                valid = da.notnull().any(dim=reduce_dims) if reduce_dims else da.notnull()
                keep = keep & valid

        elif mode == "full":
            keep = xr.ones_like(ds[time_dim], dtype=bool)
            logger.info("%s: using full corruption check", self.source_name)

            for var in [v for v in ds.data_vars if v != "_has_var"]:
                da = ds[var]
                if time_dim not in da.dims:
                    continue

                reduce_dims = [d for d in da.dims if d != time_dim]

                # Select corrupted timestep, mean over non-time dims is NaN
                ts_mean = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
                keep = keep & ts_mean.notnull()

        # Compute only the 1D mask (cheaper)
        keep = keep.compute() if hasattr(keep.data, "compute") else keep

        n_keep = int(np.asarray(keep.values, dtype=bool).sum())
        logger.info("%s: non-null timesteps=%s/%s", self.source_name, n_keep, keep.size)

        # Return a python set of python datetimes
        corrupted_sel = ds[time_dim].where(~keep, drop=True)
        corrupted = (
            set(pd.to_datetime(corrupted_sel.values).to_pydatetime())
            if corrupted_sel.values.size else set()
        )
        logger.info("Corrupted %s samples: %s", self.source_name, len(corrupted))

        # Drop corrupted timesteps
        ds = ds.drop_sel({time_dim: corrupted_sel})

        # Update source class missed
        self.elements.missed = corrupted | self.elements.missed

        # Assign dataset missed_time coord
        missed_np = np.array(sorted(self.elements.missed), dtype="datetime64[ns]")
        ds = ds.assign_coords(missed_time=("missed_time", missed_np))
        ds["missed_time"].encoding.update({
            "units": "nanoseconds since 1970-01-01 00:00:00",
            "calendar": "proleptic_gregorian",
        })

        return ds

    def _finalize(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Final operations after _get_data
        """

        # Select area if necessary, juno-local subsets in _preprocess.py, xarray-local doesn't support it
        if self.select_area_after_request:
            logger.info("Select region lat: %s, lon: %s", self.data_selection.region.lat, self.data_selection.region.lon)
            ds = ds.earthml.subset(self.data_selection)

        # Grid resolution
        lat_res, lon_res = ds.earthml.resolution()
        logger.info("Native resolutions: lat %.2f, lon %.2f", lat_res, lon_res)

        # Regrid if required # TODO save weights for efficiency
        if self.regrid_resolution is not None:
            logger.info("Regridding %s to rectilinear grid with resolution %s", self.regrid_vars, self.regrid_resolution)
            ds = ds.earthml.regrid_to_rectilinear(
                region=self.data_selection.region,
                resolution=self.regrid_resolution,
                vars_to_regrid=self.regrid_vars,
            )
            lat_res_regrid, lon_res_regrid = ds.earthml.resolution()
            logger.info("Target rectilinear resolutions: lat %.2f, lon %.2f", lat_res_regrid, lon_res_regrid)

        # Convert unit of variables if necessary (e.g. from C to K for SST)
        if self.convert_unit is not None:
            logger.info("Converting variable units according to: %s", self.convert_unit)
            ds = ds.earthml.convert_unit(self.convert_unit)

        # Normalize dims
        logger.debug("Normalize dims and coords")
        ds = ds.earthml.normalize_dims_and_coords()

        # Remove corrupted timesteps and update missed (in class and in ds)
        ds = self._remove_corrupted_timesteps(ds)

        # self._debug_check_per_var(ds, "FINAL CHECK")

        time_like_dims = [d for d in ("time", "valid_time", "source_time", "time_counter", "t") if d in ds.dims]
        if len(time_like_dims) > 1:
            logger.warning("Multiple time-like dims present: %s", time_like_dims)

        return ds


    def load(
        self,
        show_dask_progress: bool = True,
    ) -> xr.Dataset:
        """Get data only if it hasn't loaded yet"""
        if self.ds is None:
            logger.info("Load data from %s...", self.source_name)
            t0 = time.time()

            logger.info("  1/3 building dataset...")
            ds = self._get_data()

            logger.info("  2/3 finalizing dataset...")
            ds = self._finalize(ds)

            # Persist here to materialise the dataset on the cluster
            # and shrink the graph that lives on the client.
            try:
                is_dask = (ds.chunks is not None)
            except ValueError:
                # inconsistent chunk layout -> it's dask-backed; unify before proceeding
                is_dask = True
                ds = ds.unify_chunks()

            if is_dask:
                logger.info("  3/3 materializing on cluster...")
                ds = ds.chunk() # ensure it's dask-backed (no-op if already)
                ds = ds.persist()
                if show_dask_progress:
                    progress(ds) # show progress bar
                wait(ds) # block load() until materialised

            self.ds = ds
            logger.info(" -> loading time: %.2fs", time.time() - t0)
            logger.info(" -> dataset shape: %s", self.ds.sizes)

        return self.ds

    def reload(
        self,
        show_dask_progress: bool = True,
    ) -> xr.Dataset:
        """Force data reload"""
        self.ds = None
        return self.load(show_dask_progress=show_dask_progress)


    def save(
        self,
        filepath: str | Path,
        consolidated: bool = False
    ):
        """Save dataset in Zarr format in filepath"""
        if self.ds is None:
            self.ds = self.load()

        save_zarr(self.ds, filepath, consolidated)

    def _debug_check_per_var(self, ds, print_str: str):
        for var in [v for v in ds.data_vars if v != "_has_var"]:
            da = ds[var]
            time_dim = ds.earthml.guessed_dims.time
            reduce_dims = [d for d in da.dims if d != time_dim]
            ts_mean = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
            n_ok = int(ts_mean.notnull().sum().compute() if hasattr(ts_mean.data, "compute") else ts_mean.notnull().sum())
            logger.info("%s: %s: var=%s, non-null timesteps=%s/%s", print_str, self.source_name, var, n_ok, ts_mean.size)
