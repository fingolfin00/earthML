from abc import ABC, abstractmethod
import time
from dateutil.relativedelta import relativedelta
from pathlib import Path
import pandas as pd
import xarray as xr
from dask.distributed import wait
from zarr.codecs import BloscCodec
from rich import print

from ..base.dataclasses import TimeRange
from .dataclasses import DataSource, Sample

class BaseSource(ABC):
    def __init__(
        self,
        datasource: DataSource,
    ):
        self.datasource = datasource
        self.data_selection = datasource.data_selection
        self.source_name = datasource.source
        self.date_range = self.generate_date_range(self.data_selection.period)
        self.data_selection.period.start = self.date_range[0] # this may change the periods generated in the launcher script!
        # self.data_selection.period.end = self.date_range[-1]
        print(f"Date range: length {len(self.date_range)}, {self.date_range[0]} to {self.date_range[-1]}")
        # print(self.date_range)
        self.elements = Sample()
        self.ds = None

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

    def load(self) -> xr.Dataset:
        """Get data only if it hasn't loaded yet"""
        if self.ds is None:
            print(f"Load data from {self.source_name}...")
            t0 = time.time()
            ds = self._get_data()

            # Persist here to materialise the dataset on the cluster
            # and shrink the graph that lives on the client.
            try:
                is_dask = (ds.chunks is not None)
            except ValueError:
                # inconsistent chunk layout -> it's dask-backed; unify before proceeding
                is_dask = True
                ds = ds.unify_chunks()

            if is_dask:
                ds = ds.chunk()   # ensure it's dask-backed (no-op if already)
                ds = ds.persist()
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
        if not self.ds:
            self.ds = self.load()
        store = Path(filepath)
        print(f"Saving dataset to {store}")
        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding_zarr = ({
            v: {"compressors": compressor} for v in self.ds.variables
        })
        self.ds.to_zarr(store, encoding=encoding_zarr, mode='w', consolidated=consolidated)
