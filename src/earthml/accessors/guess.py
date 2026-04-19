from typing import Sequence, Literal

import cf_xarray
import xarray as xr

from ..base.dataclasses import Dims


# Constants
DIM_NAMES = {
    "time":         ["time", "valid_time", "time_counter", "source_time", "t"],
    "latitude":     ["lat", "y", "nav_lat"],
    "longitude":    ["lon", "x", "nav_lon"],
    "level":        ["level", "lev", "z", "height", "depth", "deptht", "depthu", "depthv", "depthw"],
    "realization":  ["realization", "number", "ens"],
    "leadtime":     ["lead_time", "leadtime", "step", "forecastMonth"],
}


class EarthMLGuess:
    def _guess_dim_or_coord_or_datavar_name(
        self,
        ds: xr.Dataset,
        cf_name: str,
        name_type: Literal["coord", "dim", "data_var"],
        fallback_names: Sequence[str] | None = None,
    ) -> str | None:
        # Try cf_xarray key directly
        try:
            if name_type == 'dim':
                return ds.cf.indexes[cf_name].name
            elif name_type == 'coord':
                return ds.cf.coords[cf_name].name
            else:
                return ds.cf.data_vars[cf_name].name
        except (KeyError, AttributeError, ValueError):
            # cf_xarray can fail either because the mapping is missing or because
            # multiple coords satisfy the same CF role (for example `time`,
            # `valid_time`, and `source_time` coexisting in Earthkit datasets).
            pass
        # Try explicit fallback dimension names
        fallback_names = (fallback_names or []) + [cf_name]
        for name in fallback_names:
            # print(name_type, "Fallback:", name)
            # print(ds.coords.keys())
            # print(ds.data_vars.keys())
            if name_type == 'dim':
                name_ds = ds.dims
            elif name_type == 'coord':
                name_ds = ds.coords
            else:
                name_ds = ds.data_vars
            # print(name_ds)
            if name in name_ds:
                # print(name_type, "return", name)
                return name
        # Nothing found
        return None


    def guess_dim(
        self,
        ds: xr.Dataset,
        cf_name: str,
        fallback_names: list[str] | None = None,
    ):
        return self._guess_dim_or_coord_or_datavar_name(ds, cf_name, 'dim', fallback_names)

    def guess_coord(
        self,
        ds: xr.Dataset,
        cf_name: str,
        fallback_names: list[str] | None = None,
    ):
        coord = self._guess_dim_or_coord_or_datavar_name(ds, cf_name, 'coord', fallback_names)
        if coord is None: coord = self.guess_datavar(ds, cf_name, fallback_names)
        return coord

    def guess_datavar(
        self,
        ds: xr.Dataset,
        cf_name: str,
        fallback_names: list[str] | None = None,
    ):
        return self._guess_dim_or_coord_or_datavar_name(ds, cf_name, 'data_var', fallback_names)


    # Dims
    @property
    def guess_time_dim(self):
        return self.guess_dim(self._obj, "time", DIM_NAMES["time"])

    @property
    def guess_lat_dim(self):
        return self.guess_dim(self._obj, 'latitude', DIM_NAMES["latitude"])

    @property
    def guess_lon_dim(self):
        return self.guess_dim(self._obj, 'longitude', DIM_NAMES["longitude"])

    @property
    def guess_lev_dim(self):
        return self.guess_dim(self._obj, 'level', DIM_NAMES["level"])

    @property
    def guess_realization_dim(self):
        return self.guess_dim(self._obj, "realization", DIM_NAMES["realization"])

    @property
    def guess_leadtime_dim(self):
        return self.guess_dim(self._obj, "leadtime", DIM_NAMES["leadtime"])


    # Coords
    @property
    def guess_time_coord(self):
        return self.guess_coord(self._obj, "time", DIM_NAMES["time"])

    @property
    def guess_lat_coord(self):
        return self.guess_coord(self._obj, 'latitude', DIM_NAMES["latitude"])

    @property
    def guess_lon_coord(self):
        return self.guess_coord(self._obj, 'longitude', DIM_NAMES["longitude"])

    @property
    def guess_lev_coord(self):
        return self.guess_coord(self._obj, 'level', DIM_NAMES["level"])

    @property
    def guess_realization_coord(self):
        return self.guess_coord(self._obj, "realization", DIM_NAMES["realization"])

    @property
    def guess_leadtime_coord(self):
        return self.guess_coord(self._obj, "leadtime", DIM_NAMES["leadtime"])

    @property
    def guessed_dims(self):
        return Dims(
            time=self.guess_time_dim,
            latitude=self.guess_lat_dim,
            longitude=self.guess_lon_dim,
            level=self.guess_lev_dim,
            realization=self.guess_realization_dim,
            leadtime=self.guess_leadtime_dim,
        )

    @property
    def guessed_coords(self):
        return Dims(
            time=self.guess_time_coord,
            latitude=self.guess_lat_coord,
            longitude=self.guess_lon_coord,
            level=self.guess_lev_coord,
            realization=self.guess_realization_coord,
            leadtime=self.guess_leadtime_coord,
        )
