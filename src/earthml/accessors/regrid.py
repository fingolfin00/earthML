from rich import print

import numpy as np
import cf_xarray
import xarray as xr
from scipy.interpolate import griddata

from ..base.dataclasses import Region


# TODO refactor regrid in one single optimized helper, possibly adding support for xesmf or other fast lib
class EarthMLRegrid:
    def regrid_to_rectilinear(
        self,
        region: Region,
        resolution: float | tuple[float, float],
        vars_to_regrid=None,
    ) -> xr.Dataset:
        """
        Regrid dataset variables onto a rectilinear latitude/longitude target grid.

        This accessor supports two source-grid layouts:

        1. Rectilinear source grid
        - latitude and longitude are both 1D coordinates
        - regridding is performed with ``xarray.Dataset.interp``

        2. Curvilinear source grid
        - latitude and longitude are both 2D coordinates
        - regridding is performed variable-by-variable with
            ``scipy.interpolate.griddata`` onto a 1D rectilinear target grid

        Parameters
        ----------
        region : Region
            Geographic target region. Expected shape:
            - ``region.lat = (north, south)`` or ``(south, north)``
            - ``region.lon = (west, east)``
            Longitudes may be given in any continuous degree convention
            (for example ``(-200, -120)`` is accepted).
        resolution : float or tuple[float, float]
            Target grid resolution in degrees.
            - If float: same resolution is used for latitude and longitude.
            - If tuple/list: interpreted as ``(lat_res, lon_res)``.
        vars_to_regrid : iterable of str or None, optional
            Names of data variables to regrid. Variables not listed are copied
            through unchanged. If None, all data variables are considered.

        Returns
        -------
        xr.Dataset
            Dataset with requested variables regridded onto 1D rectilinear target
            coordinates named using the dataset's guessed latitude/longitude
            coordinate names.

        Notes
        -----
        - Longitude interpolation is done in a continuous longitude frame chosen
        from the requested region, rather than by forcing everything into
        ``[-180, 180)`` first. This avoids artificial seams and inconsistent
        target spacing for dateline-crossing or extended-longitude requests such
        as ``(-200, -120)``.
        - For rectilinear sources, source longitudes are shifted by multiples of
        360 degrees to lie near the target longitude interval before sorting and
        interpolating.
        - For curvilinear sources, 2D source longitudes are shifted similarly
        before calling ``griddata``.
        """
        ds = self._obj

        lon_name = ds.earthml.guessed_coords.longitude
        lat_name = ds.earthml.guessed_coords.latitude

        lat_da = ds[lat_name]
        lon_da = ds[lon_name]

        if isinstance(resolution, (tuple, list)):
            lat_res, lon_res = float(resolution[0]), float(resolution[1])
        else:
            lat_res = lon_res = float(resolution)

        if lat_res <= 0 or lon_res <= 0:
            raise ValueError(
                f"Resolution must be positive. Got lat_res={lat_res}, lon_res={lon_res}."
            )

        if vars_to_regrid is None:
            vars_to_regrid = list(ds.data_vars)
        else:
            vars_to_regrid = list(vars_to_regrid)

        print(f"Regrid: available data vars {list(ds.data_vars.keys())}, requested vars {vars_to_regrid}")

        lat0, lat1 = map(float, region.lat)
        lon0, lon1 = map(float, region.lon)

        eps = 1e-6

        # ---- Target latitude: preserve requested orientation ----
        if lat0 > lat1:
            lat_target = np.arange(lat0, lat1 - eps, -lat_res, dtype="float32")
        else:
            lat_target = np.arange(lat0, lat1 + eps, lat_res, dtype="float32")

        # ---- Target longitude: keep a continuous interval, do NOT wrap to [-180, 180) ----
        # Example: (-200, -120) should remain continuous, not become [160..179, -180..-120]
        lon0_u, lon1_u = float(lon0), float(lon1)
        if lon_res > 0:
            while lon1_u < lon0_u:
                lon1_u += 360.0
            lon_target = np.arange(lon0_u, lon1_u + eps, lon_res, dtype="float32")
        else:
            while lon1_u > lon0_u:
                lon1_u -= 360.0
            lon_target = np.arange(lon0_u, lon1_u - eps, lon_res, dtype="float32")

        lon_center = 0.5 * (lon0_u + lon1_u)

        def _shift_longitudes_near_reference(lon_values: np.ndarray, reference: float) -> np.ndarray:
            """
            Shift longitudes by multiples of 360 so they lie as close as possible
            to a given reference longitude. Preserves local spacing while choosing
            a continuous longitude branch compatible with the target interval.
            """
            lon_values = np.asarray(lon_values, dtype=np.float64)
            return lon_values + 360.0 * np.round((reference - lon_values) / 360.0)

        def _deduplicate_sorted_1d_axis(ds_in: xr.Dataset, coord_name: str) -> xr.Dataset:
            """
            Sort dataset by a 1D coordinate and drop exact duplicate coordinate values,
            keeping first occurrence. interp() requires unique monotonic indexes.
            """
            ds_in = ds_in.sortby(coord_name)
            coord_vals = np.asarray(ds_in[coord_name].values)

            _, unique_idx = np.unique(coord_vals, return_index=True)
            if len(unique_idx) != ds_in.sizes[coord_name]:
                ds_in = ds_in.isel({coord_name: np.sort(unique_idx)})

            return ds_in

        rectilinear_src = (lat_da.ndim == 1 and lon_da.ndim == 1)

        # ======================================================================
        # Case 1: rectilinear (1D) source -> rectilinear (1D) target
        # ======================================================================
        if rectilinear_src:
            print("Regrid: rectilinear (1D) source -> rectilinear (1D) target via xarray.interp.")

            lat_src = np.asarray(ds[lat_name].values, dtype=np.float64)
            lon_src = _shift_longitudes_near_reference(ds[lon_name].values, lon_center)

            ds_interp = ds.assign_coords({
                lat_name: xr.DataArray(lat_src, dims=ds[lat_name].dims),
                lon_name: xr.DataArray(lon_src, dims=ds[lon_name].dims),
            })

            ds_interp = _deduplicate_sorted_1d_axis(ds_interp, lon_name)
            ds_interp = _deduplicate_sorted_1d_axis(ds_interp, lat_name)

            lat_tgt_da = xr.DataArray(lat_target, dims=(lat_name,), name=lat_name)
            lon_tgt_da = xr.DataArray(lon_target, dims=(lon_name,), name=lon_name)

            regridded = ds_interp.interp(
                {lat_name: lat_tgt_da, lon_name: lon_tgt_da},
                method="linear",
            )

            regridded = regridded.assign_coords(
                {lat_name: lat_tgt_da, lon_name: lon_tgt_da}
            )
            return regridded

        # ======================================================================
        # Case 2: curvilinear (2D) source -> rectilinear (1D) target
        # ======================================================================
        print("Regrid: curvilinear (2D) source -> rectilinear (1D) target via scipy.griddata.")

        if lat_da.ndim != 2 or lon_da.ndim != 2:
            raise ValueError(
                "Curvilinear regrid assumes lat/lon either (1D,1D) or (2D,2D). "
                f"Got lat.ndim={lat_da.ndim}, lon.ndim={lon_da.ndim}."
            )

        if lat_da.dims != lon_da.dims:
            raise ValueError(
                f"Curvilinear lat/lon must share the same dimensions. "
                f"Got lat.dims={lat_da.dims}, lon.dims={lon_da.dims}."
            )

        y_dim_src, x_dim_src = lat_da.dims

        lat_src_2d = np.asarray(lat_da.values, dtype=np.float64)
        lon_src_2d = _shift_longitudes_near_reference(lon_da.values, lon_center)

        Ny = lat_target.size
        Nx = lon_target.size

        lat_flat = lat_src_2d.ravel()
        lon_flat = lon_src_2d.ravel()

        coord_valid = np.isfinite(lat_flat) & np.isfinite(lon_flat)
        points = np.column_stack([lon_flat[coord_valid], lat_flat[coord_valid]])

        lon_out_2d, lat_out_2d = np.meshgrid(lon_target, lat_target)
        xi = np.column_stack([lon_out_2d.ravel(), lat_out_2d.ravel()])

        data_vars_out = {}

        for name, da in ds.data_vars.items():
            if name not in vars_to_regrid:
                data_vars_out[name] = da
                continue

            if y_dim_src not in da.dims or x_dim_src not in da.dims:
                data_vars_out[name] = da
                continue

            print(f"  Regridding variable '{name}' with griddata...")

            da_spatial = da.transpose(
                *[d for d in da.dims if d not in (y_dim_src, x_dim_src)],
                y_dim_src,
                x_dim_src,
            )
            data_np = da_spatial.values

            leading_dims = da_spatial.dims[:-2]
            leading_shape = data_np.shape[:-2]
            ny_src, nx_src = data_np.shape[-2:]

            if (ny_src, nx_src) != lat_src_2d.shape:
                raise ValueError(
                    f"Variable '{name}' spatial shape {(ny_src, nx_src)} does not match "
                    f"lat/lon shape {lat_src_2d.shape}."
                )

            arr = data_np.reshape(-1, ny_src * nx_src)

            out_slices = []
            for i in range(arr.shape[0]):
                zi = arr[i, :]

                valid = coord_valid & np.isfinite(zi)

                if valid.sum() == 0:
                    zi_interp = np.full(xi.shape[0], np.nan, dtype=np.float64)
                elif valid.sum() < 3:
                    # Not enough points for linear triangulation
                    zi_interp = np.full(xi.shape[0], np.nan, dtype=np.float64)
                else:
                    zi_interp = griddata(
                        points[valid[coord_valid]],
                        zi[valid],
                        xi,
                        method="linear",
                    )

                zi_interp_2d = zi_interp.reshape(Ny, Nx)
                out_slices.append(zi_interp_2d)

            out = np.stack(out_slices, axis=0).reshape(*leading_shape, Ny, Nx)

            out_dims = leading_dims + (lat_name, lon_name)
            out_coords = {
                **{
                    d: da_spatial.coords[d]
                    for d in leading_dims
                    if d in da_spatial.coords
                },
                lat_name: xr.DataArray(lat_target, dims=(lat_name,)),
                lon_name: xr.DataArray(lon_target, dims=(lon_name,)),
            }

            data_vars_out[name] = xr.DataArray(
                out,
                dims=out_dims,
                coords=out_coords,
                attrs=da.attrs,
            )

        coord_out = {
            k: v for k, v in ds.coords.items()
            if k not in (lat_name, lon_name)
        }
        coord_out[lat_name] = xr.DataArray(lat_target, dims=(lat_name,))
        coord_out[lon_name] = xr.DataArray(lon_target, dims=(lon_name,))

        out_ds = xr.Dataset(data_vars_out, coords=coord_out, attrs=ds.attrs)
        return out_ds
