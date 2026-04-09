from typing import Callable
from dataclasses import dataclass, field

import os

from dateutil.relativedelta import relativedelta

import xarray as xr
import copernicusmarine

from rich import print

from .dataclasses import DataSource, RegridConfig
from .base import BaseSource


@dataclass
class CopernicusmarineSourceConfig:
    leadtime                    : relativedelta
    username                    : str
    password                    : str
    dataset                     : str
    regrid_config               : RegridConfig
    select_area_after_request   : bool = False
    convert_unit                : dict[str, tuple[Callable, str]] = field(default_factory=dict)
    # cache_dir                   : Path = Path("/tmp/copernicusmarine-cache/")


class CopernicusmarineSource(BaseSource):
    """
    Collect data using copernicusmarine library.
    """
    def __init__(
        self,
        datasource  : DataSource,
        config      : CopernicusmarineSourceConfig,
    ):
        super().__init__(datasource)
        self.config = config

        self.select_area_after_request = self.config.select_area_after_request
        self.convert_unit = self.config.convert_unit

        self.regrid_resolution = self.config.regrid_config.regrid_resolution
        self.regrid_vars = config.regrid_config.regrid_vars if config.regrid_config.regrid_vars is not None else self.var_name_list


    def _get_data(self) -> xr.Dataset:
        n_missed = len(self.elements.missed)
        samples = [s for s in self.elements.samples if s not in self.elements.missed] # TODO refactor to BaseSource?
        print(f"Samples: {len(samples)}, missed: {n_missed}")

        start = self.data_selection.period.start - self.config.leadtime
        end = self.data_selection.period.end - self.config.leadtime

        min_lat, max_lat, min_lon, max_lon = None, None, None, None
        if not self.select_area_after_request:
            min_lon = self.datasource.data_selection.region.lon[0]
            max_lon = self.datasource.data_selection.region.lon[1]
            min_lat = self.datasource.data_selection.region.lat[1]
            max_lat = self.datasource.data_selection.region.lat[0]
            if min_lon < 0:
                min_lon = (min_lon + 360.0) % 360.0
                max_lon = (max_lon + 360.0) % 360.0
            elif max_lon > 180.0 or min_lon < -180.0:
                min_lon = ((min_lon + 180.0) % 360.0) - 180.0
                max_lon = ((max_lon + 180.0) % 360.0) - 180.0

        depth, depth_unit = None, ""
        if self.data_selection.variable.levhpa is not None:
            depth = self.data_selection.variable.levhpa
            depth_unit = "hPa"
        if self.data_selection.variable.levm is not None:
            depth = self.data_selection.variable.levm
            depth_unit = "m"

        print(f"Copernicusmarine:")
        print(f"  - requested period: {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")
        print(f"  - requested lon: {min_lon}-{max_lon}, lat: {min_lat}-{max_lat}, depth: {depth} {depth_unit} (actual depth seleceted with 'nearest' method)")

        ds = copernicusmarine.open_dataset(
            dataset_id=self.config.dataset,
            variables=[self.datasource.data_selection.variable.name],
            start_datetime=start.strftime('%Y-%m-%d'),
            end_datetime=end.strftime('%Y-%m-%d'),
            username=self.config.username,
            password=self.config.password,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
        )

        print(ds)

        return ds.sel(depth=depth, method="nearest")
