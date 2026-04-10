from dataclasses import dataclass, field
from abc import abstractmethod
from typing import Protocol

from pathlib import Path
import xarray as xr

from .dataclasses import DataSource, Sample
from .base import BaseSource


@dataclass
class XarrayLocalSourceConfig:
    root_path   : str | Path
    xarray_args : dict = field(default_factory=dict)

class HasRootPath(Protocol):
    root_path   : str | Path


class XarrayLocalSource(BaseSource):
    def __init__(
        self,
        datasource: DataSource,
        config: XarrayLocalSourceConfig | None = None,
    ):
        super().__init__(datasource)

        self.config = config

        self.path = Path(config.root_path)
        self.elements.samples = self.date_range
        self.xarray_args = {} if config.xarray_args is None else config.xarray_args

    def _get_data(self) -> xr.Dataset:
        self.ds = xr.open_dataset(self.path, **self.xarray_args)

        # Select only non-missed samples # TODO maybe can be removed
        if self.elements.missed:
            time_dim = self.ds.earthml.guessed_dims.time
            missed = xr.DataArray(list(self.elements.missed), dims="missed_time", name="missed_time")
            keep_mask = ~self.ds[time_dim].isin(missed)
            # print(missed)
            # print(self.ds[time_dim])
            # self.ds = self.ds.sel({time_dim: keep_mask})
            self.ds = self.ds.where(keep_mask, drop=True)

            # Update missed_time coord if present
            if "missed_time" in self.ds.coords:
                self.ds = self.ds.assign_coords(missed_time=missed)

        return self.ds


class MFXarrayLocalSource(BaseSource):
    def __init__(
        self,
        datasource: DataSource,
        config: HasRootPath,
    ):
        super().__init__(datasource)
        self.path = Path(config.root_path)

    @abstractmethod
    def _get_data_filenames(self) -> Sample:
        """
        Get the local data filenames for the given data selection. Implement in subclasses.
        """
        pass
