from abc import ABC, abstractmethod
from dataclasses import field
from typing import Callable

from copy import deepcopy
from pathlib import Path

import time
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr

from dask.distributed import wait, progress
from zarr.codecs import BloscCodec

from rich import print

from ..base.dataclasses import TimeRange
from .dataclasses import DataSource, Sample


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
        print(f"Requested date range: {self.data_selection.period.start} to {self.data_selection.period.end}")
        print(f"Effective date range: {self.date_range[0]} to {self.date_range[-1]}, length {len(self.date_range)}")
        # print(self.date_range)

        # Init
        self.elements = Sample()
        self.ds = None
        self.var_name_list = [v.name for v in self.data_selection.variable] if isinstance(self.data_selection.variable, list) else [self.data_selection.variable.name]
        self.select_area_after_request = False
        self.regrid_resolution = None
        self.convert_unit = None

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
    def generate_date_range(period: TimeRange):
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

        keep = xr.ones_like(ds[time_dim], dtype=bool)

        for var in [v for v in ds.data_vars if v != "_has_var"]:
            da = ds[var]
            if time_dim not in da.dims:
                continue

            reduce_dims = [d for d in da.dims if d != time_dim]

            # Select corrupted timestep, mean over non-time dims is NaN
            ts_mean = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
            keep = keep & ts_mean.notnull()

            print(f"{self.source_name}: var={var}, non-null timesteps={int(ts_mean.notnull().sum())}/{ts_mean.size}")

        # Compute only the 1D mask (cheaper)
        keep = keep.compute() if hasattr(keep.data, "compute") else keep

        # Return a python set of python datetimes
        corrupted_sel = ds[time_dim].where(~keep, drop=True)
        corrupted = (
            set(pd.to_datetime(corrupted_sel.values).to_pydatetime())
            if corrupted_sel.values.size else set()
        )
        print(f"Corrupted {self.source_name} samples: {len(corrupted)}")

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
            print(f"Select region lat: {self.data_selection.region.lat}, lon: {self.data_selection.region.lon}")
            ds = ds.earthml.subset(self.data_selection)

        # Grid resolution
        lat_res, lon_res = ds.earthml.resolution()
        print(f"Native resolutions: lat {lat_res:.2f}, lon {lon_res:.2f}")

        # Regrid if required # TODO save weights for efficiency
        if self.regrid_resolution is not None:
            print(f"Regridding {self.regrid_vars} to rectilinear grid with resolution {self.regrid_resolution}")
            ds = ds.earthml.regrid_to_rectilinear(
                region=self.data_selection.region,
                resolution=self.regrid_resolution,
                vars_to_regrid=self.regrid_vars,
            )
            lat_res_regrid, lon_res_regrid = ds.earthml.resolution()
            print(f"Target rectilinear resolutions: lat {lat_res_regrid:.2f}, lon {lon_res_regrid:.2f}")

        # Convert unit of variables if necessary (e.g. from C to K for SST)
        if self.convert_unit is not None:
            print(f"Converting variable units according to: {self.convert_unit}")
            ds = ds.earthml.convert_unit(self.convert_unit)

        # Normalize dims
        ds = ds.earthml.normalize_dims_and_coords()

        # Remove corrupted timesteps and update missed (in class and in ds)
        ds = self._remove_corrupted_timesteps(ds)

        # self._debug_check_per_var(ds, "FINAL CHECK")

        time_like_dims = [d for d in ("time", "valid_time", "source_time", "time_counter", "t") if d in ds.dims]
        if len(time_like_dims) > 1:
            print(f"Warning: multiple time-like dims present: {time_like_dims}")

        return ds


    def load(self) -> xr.Dataset:
        """Get data only if it hasn't loaded yet"""
        if self.ds is None:
            print(f"Load data from {self.source_name}...")
            t0 = time.time()

            print("  1/3 building dataset...")
            ds = self._get_data()

            print("  2/3 finalizing dataset...")
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
                print("  3/3 materializing on cluster...")
                ds = ds.chunk() # ensure it's dask-backed (no-op if already)
                ds = ds.persist()
                progress(ds) # show progress bar
                wait(ds) # block load() until materialised

            self.ds = ds
            print(f" → loading time: {time.time() - t0:.2f}s")
            print(f" → dataset shape: {self.ds.sizes}")

        return self.ds


    def reload(self) -> xr.Dataset:
        """Force data reload"""
        self.ds = None
        return self.load()


    def save(
        self,
        filepath: str | Path,
        consolidated: bool = False
    ):
        """Save dataset in Zarr format in filepath"""
        if self.ds is None:
            self.ds = self.load()
        store = Path(filepath)
        print(f"Saving dataset to {store}")
        # print(f"Saving sizes: {self.ds.sizes}")
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = ({
            v: {"compressors": compressor} for v in self.ds.variables
        })
        self.ds.to_zarr(store, encoding=encoding_zarr, mode='w', consolidated=consolidated)


    def _debug_check_per_var(self, ds, print_str: str):
        for var in [v for v in ds.data_vars if v != "_has_var"]:
            da = ds[var]
            time_dim = ds.earthml.guessed_dims.time
            reduce_dims = [d for d in da.dims if d != time_dim]
            ts_mean = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
            n_ok = int(ts_mean.notnull().sum().compute() if hasattr(ts_mean.data, "compute") else ts_mean.notnull().sum())
            print(f"{print_str}: {self.source_name}: var={var}, non-null timesteps={n_ok}/{ts_mean.size}")
