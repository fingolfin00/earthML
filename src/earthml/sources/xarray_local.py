from abc import abstractmethod
from pathlib import Path
import xarray as xr
from rich import print

from ..utils import _guess_coord_name
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
        if self.elements.missed:
            time_coord = _guess_coord_name(self.ds, "time", ["valid_time", "time_counter"])
            for var in self.ds:
                print(self.ds[var].shape)
            # self.ds[time_coord] = [d for d in self.date_range if d not in self.elements.missed]
            self.ds[time_coord] = self.date_range
            print(self.ds[time_coord])
            self.ds = self.ds.drop_sel({time_coord: list(self.elements.missed)})
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
