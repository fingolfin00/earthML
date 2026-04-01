from rich import print # TODO thin the module, there shoulnd't be print here

import numpy as np
import cf_xarray
import xarray as xr
import pandas as pd

from ..base.dataclasses import DataSelection


class EarthMLSubset:
    """
    Xarray accessor to get a spatial subset of the same dataset
    """
    @staticmethod
    def _wrap_longitudes(lon, center):
        """
        Wrap longitudes into a continuous interval centered at `center`,
        i.e. (center - 180, center + 180].
        """
        return ((lon - center + 180.0) % 360.0) - 180.0 + center

    @staticmethod
    def _get_cutting_lon(lon_da: xr.DataArray, lon_dim: str, req_lon: tuple) -> tuple:
        if lon_da.ndim == 1: # rectilinear
            lon1d = lon_da.values
        elif lon_da.ndim == 2: # regular curvilinear
            non_lon_dims = [d for d in lon_da.dims if d != lon_dim]
            lon1d = lon_da.mean(dim=non_lon_dims).values
        else:
            raise ValueError(f"Not supporting rolling if DataArray dimension > 2: {lon_da.ndim}")
        idx = int(np.abs(lon1d - req_lon[0]).argmin())
        cutting_lon_idx = idx - 1
        cutting_lon = lon1d[cutting_lon_idx]
        return cutting_lon_idx, cutting_lon

    def roll_ds(self, ds: xr.Dataset, req_lon: tuple) -> xr.Dataset:
        if req_lon[0] < req_lon[1]: # nothing to do
            return ds

        lon_coord, lat_coord = ds.earthml.guessed_coords.longitude, ds.earthml.guessed_coords.latitude
        lon_dim = ds.earthml.guessed_dims.longitude
        lon_da, lat_da = ds[lon_coord], ds[lat_coord]

        cutting_lon_idx, cutting_lon = self._get_cutting_lon(lon_da, lon_dim, req_lon)
        # print(f"roll_ds: req_lon {req_lon}, cutting_lon_idx: {cutting_lon_idx}, cutting_lon: {cutting_lon}")
        # print(f"roll_ds: lon_da.sizes {lon_da.sizes}, lat_da.sizes: {lat_da.sizes}")

        ds = ds.assign_coords(**{lon_coord: self._wrap_longitudes(ds[lon_coord], cutting_lon)})
        return ds.roll(**{lon_dim: cutting_lon_idx}, roll_coords=True)

    @staticmethod
    def _dim_selection(
        dim_name: str, 
        values: int | float | tuple | None = None,
    ) -> dict:
        """Small helper to create the dim selection later"""
        sel_d: dict = {}

        if dim_name is not None and values is not None:
            if isinstance(values, tuple):
                sel_d[dim_name] = slice(*values)
            else:
                sel_d[dim_name] = values

        return sel_d

    @staticmethod
    def _normalize_lon_bounds(lon_min, lon_max, grid_min):
        """
        Normalize requested lon bounds to match dataset convention:
        - If grid is [0, 360], convert [-180, 180] requests to [0, 360].
        - If grid is [-180, 180], convert 0–360-style requests to [-180, 180].
        """
        lon_min = float(lon_min)
        lon_max = float(lon_max)

        if grid_min >= 0 and lon_min < 0:
            lon_min = (lon_min + 360.0) % 360.0
            lon_max = (lon_max + 360.0) % 360.0
        elif (grid_min < 0 and lon_max > 180.0) or (grid_min < 0 and lon_min < -180.0):
            lon_min = ((lon_min + 180.0) % 360.0) - 180.0
            lon_max = ((lon_max + 180.0) % 360.0) - 180.0
        # else:
        #     print("Unmanaged lon bound normalization")

        return lon_min, lon_max

    @staticmethod
    def _get_grid_extremes(ds: xr.Dataset, lon_coord, lat_coord) -> tuple:
        lon_da = ds[lon_coord]
        lat_da = ds[lat_coord]

        lon_vals = lon_da.values
        lat_vals = lat_da.values

        # print(f"Fields shape lon: {lon_da.shape}, lat: {lat_da.shape}")

        lon_grid_min = float(np.nanmin(lon_vals))
        lon_grid_max = float(np.nanmax(lon_vals))
        lat_grid_min = float(np.nanmin(lat_vals))
        lat_grid_max = float(np.nanmax(lat_vals))

        return (lon_grid_min, lon_grid_max), (lat_grid_min, lat_grid_max)


    def subset(
        self,
        data_selection: DataSelection,
    ) -> xr.Dataset:
        """
        Apply CF-based:
        - lon/lat convention normalization
        - region selection
        - vertical level selection (if levhpa / levm provided)
        - time selection (if leadtime provided in data_selection; currently commented)

        This runs once on the full combined dataset, not per file.
        """
        ds = self._obj

        # Leadtime selection
        leadtime = data_selection.variable.leadtime
        leadtime_sel_d = {}
        if leadtime is not None: # TODO refactor, it's used also in _preprocess
            # Build target timedelta from value + unit (e.g. "3 days", "12 hours")
            td = pd.to_timedelta(f"{leadtime.value} {leadtime.unit}")
            # Cast to same dtype as coord (usually timedelta64[ns])
            coord_dtype = ds[leadtime.name].dtype
            target = td.to_numpy().astype(coord_dtype)
            leadtime_sel_d = self._dim_selection(ds.earthml.guessed_dims.leadtime, target)

        # If only one leadtime, remove selection
        if leadtime_sel_d:
            leadtime_dim = next(iter(leadtime_sel_d))
            # print("Leadtime dim name:", leadtime_dim, "ndim:", ds[leadtime_dim].ndim, "shape:", ds[leadtime_dim].shape)
            if ds[leadtime_dim].ndim == 1 and ds[leadtime_dim].shape[0] == 1:
                leadtime_sel_d = {}

        # Vertical level selection
        levhpa = data_selection.variable.levhpa
        levm = data_selection.variable.levm
        level_value = next((lv for lv in (levhpa, levm) if lv is not None), None)
        vertical_dim = ds.earthml.guess_dim(ds, "vertical", ["level", "z"])
        level_sel_d = self._dim_selection(vertical_dim, level_value)

        lon_req = np.array(data_selection.region.lon, dtype=float)
        lat_req = np.array(data_selection.region.lat, dtype=float)

        lon_min_req, lon_max_req = float(lon_req.min()), float(lon_req.max())
        lat_min_req, lat_max_req = float(lat_req.min()), float(lat_req.max())

        # Lon/lat coordinate discovery
        lon_coord = ds.earthml.guessed_coords.longitude
        lat_coord = ds.earthml.guessed_coords.latitude
        if lon_coord is None or lat_coord is None:
            raise ValueError("Could not determine longitude/latitude coordinate names")
        lat_da, lon_da = ds[lat_coord], ds[lon_coord]
        # Get coordinate extremes
        (lon_grid_min, lon_grid_max), (lat_grid_min, lat_grid_max) = self._get_grid_extremes(ds, lon_coord, lat_coord)
        # Normalize request to ds grid
        lon_min_req, lon_max_req = self._normalize_lon_bounds(lon_min_req, lon_max_req, lon_grid_min)

        # Rectilinear vs curvilinear detection
        is_rectilinear = (
            lon_da.ndim == 1
            and lat_da.ndim == 1
            and lon_coord in ds.coords
            and lat_coord in ds.coords
        )
        grid_type = "rectilinear" if is_rectilinear else "curvilinear"
        print("Grid type:", grid_type)

        # Roll if request crosses longitude cut-line (like dateline or Greenwich)
        ds = self.roll_ds(ds, (lon_min_req, lon_max_req))
        
        # print(
        #     f"{grid_type} grid, limits: "
        #     f"lon ({lon_grid_min}, {lon_grid_max}), "
        #     f"lat ({lat_grid_min}, {lat_grid_max})"
        # )
        # print(
        #     f"lon requested: ({lon_min_req}, {lon_max_req}), "
        #     f"lat requested: ({lat_min_req}, {lat_max_req})"
        # )
        # print(ds)

        # if 'sosaline' in ds.data_vars:
        #     quickplot(ds, 'sosaline', "/data/cmcc/jd19424/ML/experiments_earthML/", 'subset_ds_after_roll.png')
        (lon_grid_min, lon_grid_max), (lat_grid_min, lat_grid_max) = self._get_grid_extremes(ds, lon_coord, lat_coord)
        lon_dim = ds.earthml.guessed_dims.longitude
        _, cutting_lon = self._get_cutting_lon(lon_da, lon_dim, (lon_min_req, lon_max_req))
        # print(f"subset_ds, cutting lon: {cutting_lon}")
        lon_min_req, lon_max_req = self._wrap_longitudes((lon_min_req, lon_max_req), cutting_lon)
        # print(ds)
        # print(
        #     f"{grid_type} grid, limits (after rolling): "
        #     f"lon ({lon_grid_min}, {lon_grid_max}), "
        #     f"lat ({lat_grid_min}, {lat_grid_max})"
        # )
        # print(
        #     f"lon requested (after rolling): ({lon_min_req}, {lon_max_req}), "
        #     f"lat requested (after rolling): ({lat_min_req}, {lat_max_req})"
        # )
        lat_da, lon_da = ds[lat_coord], ds[lon_coord]

        # For final logging
        sel_lon_min = None
        sel_lon_max = None
        sel_lat_min = None
        sel_lat_max = None

        if is_rectilinear:
            # Rectilinear horizontal selection via masks
            lon_vals = ds[lon_coord]
            lat_vals = ds[lat_coord]

            # latitude mask
            lat_lo = min(lat_min_req, lat_max_req)
            lat_hi = max(lat_min_req, lat_max_req)
            lat_mask = (lat_vals >= lat_lo) & (lat_vals <= lat_hi)

            # longitude mask including wrap-around
            if lon_min_req <= lon_max_req:
                lon_mask = (lon_vals >= lon_min_req) & (lon_vals <= lon_max_req)
            else:
                lon_mask = (lon_vals >= lon_min_req) | (lon_vals <= lon_max_req)

            # select coordinate values that satisfy mask
            sel_lon = lon_vals.where(lon_mask, drop=True)
            sel_lat = lat_vals.where(lat_mask, drop=True)

            sel_dict = {lon_coord: sel_lon, lat_coord: sel_lat}
            ds = ds.sel({lon_coord: sel_lon, lat_coord: sel_lat})

            # apply non-horizontal selections normally
            selection_d = {} | level_sel_d | leadtime_sel_d
            if selection_d:
                ds = ds.sel(**selection_d)

            # print(
            #     f"lat requested after reorientation {lon_slice}, "
            #     f"lat requested after reorientation {lat_slice})"
            # )

            # Compute selected extents from the rectilinear coords
            sel_lon_min = float(np.nanmin(ds[lon_coord].values))
            sel_lon_max = float(np.nanmax(ds[lon_coord].values))
            sel_lat_min = float(np.nanmin(ds[lat_coord].values))
            sel_lat_max = float(np.nanmax(ds[lat_coord].values))

        # Curvilinear grid
        else:
            # Longitude mask (with possible cutline crossing)
            if lon_min_req <= lon_max_req:
                lon_mask = (lon_da >= lon_min_req) & (lon_da < lon_max_req)
            else:
                # print("Subset: cutline crossing")
                lon_mask = (lon_da >= lon_min_req) | (lon_da < lon_max_req)

            # Latitude mask
            lat_mask = (lat_da >= lat_min_req) & (lat_da < lat_max_req)

            mask = (lon_mask & lat_mask).load()  # Dask-safe
            # print(mask)

            # For logging: compute extents from the masked coordinates
            masked_lon = lon_da.where(mask)
            masked_lat = lat_da.where(mask)
            # quickplot(mask, '', "/data/cmcc/jd19424/ML/experiments_earthML/", 'mask.png')

            if np.all(np.isnan(masked_lon)):
                raise ValueError("No longitude points in requested region.")
            if np.all(np.isnan(masked_lat)):
                raise ValueError("No latitude points in requested region.")

            sel_lon_min = float(np.nanmin(masked_lon.values))
            sel_lon_max = float(np.nanmax(masked_lon.values))
            sel_lat_min = float(np.nanmin(masked_lat.values))
            sel_lat_max = float(np.nanmax(masked_lat.values))

            # Apply horizontal mask to the dataset
            ds = ds.where(mask, drop=True)

            # Then apply vertical and leadtime selection
            selection_d = {} | level_sel_d | leadtime_sel_d
            if selection_d:
                ds = ds.sel(**selection_d)

        # Final sanity check
        if ds[lon_coord].size == 0 or ds[lat_coord].size == 0:
            raise ValueError(
                "Subset resulted in an empty dataset. "
                "Requested region may not overlap the dataset."
            )

        # print(
        #     "Selected limits "
        #     f"lon: ({sel_lon_min}, {sel_lon_max}), "
        #     f"lat: ({sel_lat_min}, {sel_lat_max})"
        # )

        print("Size after subsetting", ds.sizes)
        return ds
