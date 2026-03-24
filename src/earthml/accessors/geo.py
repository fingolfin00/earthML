import numpy as np
import xarray as xr
from typing import Sequence

from ..utils import guess_lat_dim


@xr.register_dataset_accessor("earthml")
@xr.register_dataarray_accessor("earthml")
class EarthAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def lat_dim(self):
        return guess_lat_dim(self._obj)

    def geo_weights(self) -> xr.DataArray:
        """
        Cosine-of-latitude weights aligned to a 1D latitude dimension/coordinate.
        Appropriate as area-proxy weights for regular lat-lon grids.
        """
        data = self._obj
        lat_dim = self.lat_dim

        if lat_dim not in data.dims:
            raise ValueError(f"Latitude dimension {lat_dim!r} not found")
        if lat_dim not in data.coords:
            raise ValueError(f"Latitude coordinate {lat_dim!r} not found")

        lat = data.coords[lat_dim]
        if lat.ndim != 1:
            raise ValueError(f"Latitude coordinate {lat_dim!r} must be 1D")

        invalid = ((lat < -90) | (lat > 90)).fillna(False) # ignores nans
        if invalid.any().values:
            raise ValueError(f"Latitude coordinate {lat_dim!r} must be in [-90, 90]")

        return xr.DataArray(
            np.cos(np.deg2rad(lat)),
            coords={lat_dim: lat},
            dims=(lat_dim,),
        )

    def geo_mean(self, dim: str | Sequence[str]) -> xr.Dataset | xr.DataArray:
        """
        Geographically weighted mean over dim using cos(lat).
        Returns regular mean averaging over non-lat dim.
        """
        data = self._obj
        dim = (dim,) if isinstance(dim, str) else tuple(dim)

        if self.lat_dim not in dim:
            return data.mean(dim=dim, skipna=True)

        w = self.geo_weights()

        return data.weighted(w).mean(dim=dim, skipna=True)

    def geo_std(self, dim: str | Sequence[str]) -> xr.Dataset | xr.DataArray:
        """
        Geographically weighted population std over dim using cos(lat)
        Returns regular std averaging over non-lat dim.
        """
        data = self._obj
        dim = (dim,) if isinstance(dim, str) else tuple(dim)

        if self.lat_dim not in dim:
            return data.std(dim=dim, skipna=True)

        w = self.geo_weights()

        return data.weighted(w).std(dim=dim, skipna=True)

    def geo_cov(
        self,
        other: xr.DataArray,
        dim: str | Sequence[str],
    ) -> xr.DataArray:
        """
        Geographically weighted population covariance over dim using cos(lat).
        Returns regular covariance averaging over non-lat dims.
        """
        if not isinstance(other, xr.DataArray):
            raise TypeError("other must be an xarray.DataArray")

        data = self._obj
        dim = (dim,) if isinstance(dim, str) else tuple(dim)

        if self.lat_dim not in dim:
            x_mean = data.mean(dim=dim, skipna=True)
            y_mean = other.mean(dim=dim, skipna=True)
            return ((data - x_mean) * (other - y_mean)).mean(dim=dim, skipna=True)

        w = self.geo_weights()
        x_mean = data.weighted(w).mean(dim=dim, skipna=True)
        y_mean = other.weighted(w).mean(dim=dim, skipna=True)
        return ((data - x_mean) * (other - y_mean)).weighted(w).mean(dim=dim, skipna=True)
