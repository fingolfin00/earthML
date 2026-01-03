from abc import abstractmethod
from pathlib import Path
import numpy as np
import xarray as xr
from rich import print

from ..utils import _guess_dim_name
from ..dataclasses import DataSource, Sample
from .base import BaseSource

class XarrayLocalSource (BaseSource):
    def __init__ (
        self,
        datasource: DataSource,
        root_path: str | Path,
        xarray_args: dict = None,
    ):
        super().__init__ (datasource)
        self.path = Path(root_path)
        self.elements.samples = self.date_range
        self.xarray_args = {} if xarray_args is None else xarray_args

    def _get_data (self) -> xr.Dataset:
        self.ds = xr.open_dataset(self.path, **self.xarray_args)

        # Select only non-missed samples
        if self.elements.missed:
            time_dim = _guess_dim_name(self.ds, "time", ["valid_time", "time_counter"])
            missed = xr.DataArray(list(self.elements.missed), dims="missed_time", name="missed_time")
            keep_mask = ~self.ds[time_dim].isin(missed)
            self.ds = self.ds.sel({time_dim: keep_mask})

            # Update missed_time coord if present
            if "missed_time" in self.ds.coords:
                self.ds = self.ds.assign_coords(missed_time=missed)

        return self.ds

class MFXarrayLocalSource (BaseSource):
    def __init__ (
        self,
        datasource: DataSource,
        root_path: str,
        concat_dim: str = None,
    ):
        super().__init__ (datasource)
        self.path = Path(root_path)
        self.concat_dim = concat_dim

    @abstractmethod
    def _get_data_filenames (self) -> Sample:
        """
        Get the local data filenames for the given data selection. Implement in subclasses.
        """
        pass
