from rich import print

import numpy as np
import cf_xarray
import xarray as xr
from scipy.interpolate import griddata

from ..base.dataclasses import Region


# TODO refactor regrid in one single optimized helper, possibly adding support for xesmf or other fast lib
class EarthMLRegrid:
    @staticmethod
    def _build_target_rect_grid(
        region: Region,
        resolution: float | tuple[float, float],
    ):
        """
        region: object with .lat (2 values: [north, south]) and .lon (2 values: [west, east])
        resolution: float or (lat_res, lon_res) in degrees (positive)
        Returns (lat_target, lon_target) as 1D numpy arrays.
        """
        # Resolution
        if isinstance(resolution, (tuple, list)):
            lat_res, lon_res = float(resolution[0]), float(resolution[1])
        else:
            lat_res = lon_res = float(resolution)

        lat0, lat1 = region.lat  # typically [north, south]
        lon0, lon1 = region.lon  # typically [west, east]

        # Orientation: ECMWF "area" is [north, west, south, east]
        # So lat usually decreases, lon usually increases.
        eps = 1e-6

        # latitude
        if lat0 > lat1:
            # descending: north -> south
            lat_vals = np.arange(lat0, lat1 - eps, -lat_res)
        else:
            # ascending
            lat_vals = np.arange(lat0, lat1 + eps, lat_res)

        # longitude
        if lon0 <= lon1:
            lon_vals = np.arange(lon0, lon1 + eps, lon_res)
        else:
            # if someone gives east < west, go descending
            lon_vals = np.arange(lon0, lon1 - eps, -lon_res)

        return lat_vals.astype("float32"), lon_vals.astype("float32")

    @staticmethod
    def _to_180(lon):
        return ((lon + 180) % 360) - 180


    def regrid_to_rectilinear(
        self,
        region: Region,
        resolution: float | tuple[float, float], # TODO why do we accept tuple?
        vars_to_regrid=None,
    ) -> xr.Dataset:
        ds = self._obj

        lon_name = ds.earthml.guessed_coords.longitude
        lat_name = ds.earthml.guessed_coords.latitude

        lat_da = ds[lat_name]
        lon_da = ds[lon_name]

        if isinstance(resolution, (tuple, list)):
            lat_res, lon_res = float(resolution[0]), float(resolution[1])
        else:
            lat_res = lon_res = float(resolution)

        lat0, lat1 = map(float, region.lat)   # [north, south]
        lon0, lon1 = map(float, region.lon)   # [west, east]

        eps = 1e-6

        # Build latitude target
        if lat0 > lat1:
            lat_target = np.arange(lat0, lat1 - eps, -lat_res)
        else:
            lat_target = np.arange(lat0, lat1 + eps, lat_res)

        rectilinear_src = (lat_da.ndim == 1 and lon_da.ndim == 1)

        print(f"Regrid: available data vars {list(ds.data_vars.keys())}, requested vars {vars_to_regrid}")

        # Normalize requested longitude
        lon0_t = self._to_180(lon0)
        lon1_t = self._to_180(lon1)

        # Build geographic target interval in [-180, 180)
        if lon0_t <= lon1_t:
            lon_target = np.arange(lon0_t, lon1_t + eps, lon_res)
        else:
            # true dateline-crossing request, e.g. 170 -> -170
            lon_target = np.concatenate([
                np.arange(lon0_t, 180, lon_res),
                np.arange(-180, lon1_t + eps, lon_res),
            ])

        # ===== Case 1: rectilinear source (1D lat/lon) =====
        if rectilinear_src:
            print("Regrid: rectilinear (1D) source -> rectilinear (1D) target via xarray.interp.")

            # Normalize source longitude to [-180, 180) and sort
            ds = ds.assign_coords({
                lon_name: self._to_180(ds[lon_name])
            }).sortby(lon_name)

            # Some source grids include duplicated endpoints after longitude
            # normalization (for example both 0 and 360 -> 0). interp()
            # requires unique coordinate indexes.
            _, lon_index = np.unique(ds[lon_name].values, return_index=True)
            if len(lon_index) != ds.sizes[lon_name]:
                ds = ds.isel({lon_name: np.sort(lon_index)})

            _, lat_index = np.unique(ds[lat_name].values, return_index=True)
            if len(lat_index) != ds.sizes[lat_name]:
                ds = ds.isel({lat_name: np.sort(lat_index)})

            lat_tgt_da = xr.DataArray(lat_target, dims=(lat_name,), name=lat_name)
            lon_tgt_da = xr.DataArray(lon_target, dims=(lon_name,), name=lon_name)

            regridded = ds.interp(
                {lat_name: lat_tgt_da, lon_name: lon_tgt_da},
                method="linear",
            )

            regridded = regridded.assign_coords(
                {lat_name: lat_tgt_da, lon_name: lon_tgt_da}
            )
            return regridded

        # ===== Case 2: curvilinear (2D) source (lat/lon 2D) -> rectilinear (1D) target =====
        print("Regrid: curvilinear (2D) source -> rectilinear (1D) target via scipy.griddata.")

        if lat_da.ndim != 2 or lon_da.ndim != 2:
            raise ValueError(
                "Curvilinear regrid assumes lat/lon either (1D,1D) or (2D,2D). "
                f"Got lat.ndim={lat_da.ndim}, lon.ndim={lon_da.ndim}."
            )

        y_dim_src, x_dim_src = lat_da.dims  # e.g. ("y", "x")

        # Source coords in same lon convention as lon_target
        lat_src_2d = lat_da.values
        lon_src_2d = ((lon_da.values + 180) % 360) - 180

        # Target grid shape
        Ny = lat_target.size
        Nx = lon_target.size

        # Flatten source coords
        lat_flat = lat_src_2d.ravel()
        lon_flat = lon_src_2d.ravel()
        points = np.column_stack([lon_flat, lat_flat])

        # Target grid as flat xi
        lon_out_2d, lat_out_2d = np.meshgrid(lon_target, lat_target)  # [Ny, Nx]
        xi = np.column_stack([lon_out_2d.ravel(), lat_out_2d.ravel()])  # [Ntgt, 2]

        data_vars_out = {}

        for name, da in ds.data_vars.items():
            if name not in vars_to_regrid:
                data_vars_out[name] = da
                continue

            if y_dim_src not in da.dims or x_dim_src not in da.dims:
                data_vars_out[name] = da
                continue

            print(f"  Regridding variable '{name}' with griddata...")

            # Move (y, x) to the end
            da_spatial = da.transpose(
                *[d for d in da.dims if d not in (y_dim_src, x_dim_src)],
                y_dim_src, x_dim_src,
            )
            data_np = da_spatial.values  # [..., Ny_src, Nx_src]

            leading_dims = da_spatial.dims[:-2]
            leading_shape = data_np.shape[:-2]
            ny_src, nx_src = data_np.shape[-2:]

            arr = data_np.reshape(-1, ny_src * nx_src)  # [Nlead, Nsrc]

            out_slices = []
            for i in range(arr.shape[0]):
                zi = arr[i, :]  # [Nsrc]

                valid = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(zi)

                if not np.any(valid):
                    zi_interp = np.full(xi.shape[0], np.nan)
                else:
                    zi_interp = griddata(
                        points[valid],
                        zi[valid],
                        xi,
                        method="linear",
                    )

                zi_interp_2d = zi_interp.reshape(Ny, Nx)
                out_slices.append(zi_interp_2d)

            out = np.stack(out_slices, axis=0).reshape(
                *leading_shape,
                Ny,
                Nx,
            )

            out_dims = leading_dims + (lat_name, lon_name)
            out_coords = {
                **{d: da_spatial.coords[d] for d in leading_dims if d in da_spatial.coords},
                lat_name: xr.DataArray(lat_target, dims=(lat_name,)),
                lon_name: xr.DataArray(lon_target, dims=(lon_name,)),
            }

            data_vars_out[name] = xr.DataArray(
                out,
                dims=out_dims,
                coords=out_coords,
                attrs=da.attrs,
            )

        # Dataset coords: keep non-lat/lon coords from src, override lat/lon
        coord_out = {
            k: v for k, v in ds.coords.items()
            if k not in (lat_name, lon_name)
        }
        coord_out[lat_name] = xr.DataArray(lat_target, dims=(lat_name,))
        coord_out[lon_name] = xr.DataArray(lon_target, dims=(lon_name,))

        out_ds = xr.Dataset(data_vars_out, coords=coord_out, attrs=ds.attrs)
        return out_ds
