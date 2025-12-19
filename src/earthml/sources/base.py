from abc import ABC, abstractmethod
import time
from pathlib import Path
import xarray as xr
from dask.distributed import wait
from zarr.codecs import BloscCodec
from rich import print

from ..dataclasses import DataSource, DataSelection, Sample
from ..utils import generate_date_range

class BaseSource (ABC):
    def __init__ (
        self,
        datasource: DataSource
    ):
        self.datasource = datasource
        self.data_selection = datasource.data_selection
        self.source_name = datasource.source
        self.date_range = generate_date_range(self.data_selection.period)
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

    @abstractmethod
    def _get_data (self) -> xr.Dataset:
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
            if hasattr(ds, "chunk"):  # i.e. Dask-backed
                ds = ds.chunk()   # ensure it’s dask-backed (no-op if already)
                ds = ds.persist()
                wait(ds) # block load() until materialised

            self.ds = ds
            print(f" → loading time: {time.time() - t0:.2f}s")
            print(f" → dataset shape: {self.ds.sizes}")
        return self.ds

    def reload (self) -> xr.Dataset:
        """Force data reload"""
        self.ds = self._get_data()
        return self.ds

    def save (self, filepath: str | Path, consolidated: bool = False):
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