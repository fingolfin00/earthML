import numpy as np
import cf_xarray
import xarray as xr
import pandas as pd

from ..base.dataclasses import DataSelection
from ..logging import get_logger


logger = get_logger(__name__)


class EarthMLSubset:
    """
    Xarray accessor to extract a spatial subset from a dataset.

    The accessor supports both rectilinear and curvilinear horizontal grids.

    Workflow
    --------
    1. Discover horizontal coordinates and dimensions.
    2. Normalize requested longitude bounds to the dataset convention.
    3. If needed, roll the dataset along longitude so the requested longitude
       interval becomes continuous on the native grid.
    4. Apply horizontal selection:
       - rectilinear: coordinate-based selection
       - curvilinear: boolean mask on 2D lon/lat coordinates
    5. Apply optional vertical-level and leadtime selections.

    Notes
    -----
    - Longitude handling here is designed for native-grid subsetting, not for
      interpolation. It may roll the dataset when the requested longitude
      interval crosses the dataset cutline.
    - This accessor intentionally keeps behavior close to the current version.
    """

    @staticmethod
    def _wrap_longitudes(lon, center):
        """
        Wrap longitudes into a continuous interval centered at `center`,
        i.e. approximately (center - 180, center + 180].
        """
        return ((lon - center + 180.0) % 360.0) - 180.0 + center

    @staticmethod
    def _shift_longitudes_near_reference(lon_values, reference):
        """
        Shift longitudes by multiples of 360 degrees so they lie near
        `reference`. Useful when selecting or comparing longitudes in a
        continuous frame.
        """
        lon_values = np.asarray(lon_values, dtype=np.float64)
        return lon_values + 360.0 * np.round((reference - lon_values) / 360.0)

    @staticmethod
    def _get_cutting_lon(lon_da: xr.DataArray, lon_dim: str, req_lon: tuple) -> tuple:
        """
        Return the rolling index and cutting longitude to make a crossing
        longitude interval continuous on the native grid.
        """
        if lon_da.ndim == 1:
            lon1d = lon_da.values
        elif lon_da.ndim == 2:
            non_lon_dims = [d for d in lon_da.dims if d != lon_dim]
            lon1d = lon_da.mean(dim=non_lon_dims).values
        else:
            raise ValueError(
                f"Not supporting rolling if DataArray dimension > 2: {lon_da.ndim}"
            )

        idx = int(np.abs(lon1d - req_lon[0]).argmin())
        cutting_lon_idx = idx - 1
        cutting_lon = lon1d[cutting_lon_idx]
        return cutting_lon_idx, cutting_lon

    def roll_ds(self, ds: xr.Dataset, req_lon: tuple) -> xr.Dataset:
        """
        Roll dataset along the longitude dimension when the requested longitude
        interval crosses the native cutline.

        If req_lon[0] < req_lon[1], the interval is already continuous in the
        current longitude frame and the dataset is returned unchanged.
        """
        if req_lon[0] < req_lon[1]:
            return ds

        lon_coord = ds.earthml.guessed_coords.longitude
        lon_dim = ds.earthml.guessed_dims.longitude

        lon_da = ds[lon_coord]
        cutting_lon_idx, cutting_lon = self._get_cutting_lon(lon_da, lon_dim, req_lon)

        ds = ds.assign_coords(
            **{lon_coord: self._wrap_longitudes(ds[lon_coord], cutting_lon)}
        )
        return ds.roll(**{lon_dim: cutting_lon_idx}, roll_coords=True)

    @staticmethod
    def _dim_selection(
        dim_name: str,
        values: int | float | tuple | None = None,
    ) -> dict:
        """Build a selection dictionary for one dimension."""
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
        Normalize requested lon bounds to match the broad dataset convention.

        - If grid is [0, 360], convert negative longitudes into that range.
        - If grid is roughly [-180, 180], convert 0-360 style requests into
          that range.

        This does not solve continuity across cutlines; it only aligns the
        request with the dataset's base longitude convention.
        """
        lon_min = float(lon_min)
        lon_max = float(lon_max)

        if grid_min >= 0 and lon_min < 0:
            lon_min = (lon_min + 360.0) % 360.0
            lon_max = (lon_max + 360.0) % 360.0
        elif grid_min < 0 and (lon_max > 180.0 or lon_min < -180.0):
            lon_min = ((lon_min + 180.0) % 360.0) - 180.0
            lon_max = ((lon_max + 180.0) % 360.0) - 180.0

        return lon_min, lon_max

    @staticmethod
    def _get_grid_extremes(ds: xr.Dataset, lon_coord, lat_coord) -> tuple:
        """Return min/max longitude and latitude values from dataset coordinates."""
        lon_da = ds[lon_coord]
        lat_da = ds[lat_coord]

        lon_grid_min = float(np.nanmin(lon_da.values))
        lon_grid_max = float(np.nanmax(lon_da.values))
        lat_grid_min = float(np.nanmin(lat_da.values))
        lat_grid_max = float(np.nanmax(lat_da.values))

        return (lon_grid_min, lon_grid_max), (lat_grid_min, lat_grid_max)

    @staticmethod
    def _is_rectilinear_grid(ds: xr.Dataset, lon_coord: str, lat_coord: str) -> bool:
        """Return True when longitude and latitude are both 1D coordinates."""
        lon_da = ds[lon_coord]
        lat_da = ds[lat_coord]
        return (
            lon_da.ndim == 1
            and lat_da.ndim == 1
            and lon_coord in ds.coords
            and lat_coord in ds.coords
        )

    def _build_extra_selection(self, ds: xr.Dataset, data_selection) -> dict:
        """
        Build non-horizontal selections: vertical level and leadtime.
        """
        selection_d = {}

        # Leadtime
        leadtime = data_selection.variable.leadtime
        if leadtime is not None:
            leadtime_coord = None
            if leadtime.name in ds.coords:
                leadtime_coord = leadtime.name
            else:
                leadtime_coord = ds.earthml.guessed_coords.leadtime

            leadtime_dim = ds.earthml.guessed_dims.leadtime

            if leadtime_coord is None or leadtime_dim is None:
                logger.info(
                    "Skipping leadtime selection for %s: dataset has no leadtime coord/dim",
                    leadtime.name,
                )
            else:
                td = pd.to_timedelta(f"{leadtime.value} {leadtime.unit}")
                coord_dtype = ds[leadtime_coord].dtype
                target = td.to_numpy().astype(coord_dtype)
                leadtime_sel_d = self._dim_selection(leadtime_dim, target)

                if leadtime_sel_d and not (
                    ds[leadtime_dim].ndim == 1
                    and ds[leadtime_dim].shape[0] == 1
                ):
                    selection_d |= leadtime_sel_d

        # Vertical level
        levhpa = data_selection.variable.levhpa
        levm = data_selection.variable.levm
        level_value = next((lv for lv in (levhpa, levm) if lv is not None), None)
        vertical_dim = ds.earthml.guess_dim(ds, "vertical", ["level", "z"])
        selection_d |= self._dim_selection(vertical_dim, level_value)

        return selection_d

    def subset(
        self,
        data_selection: DataSelection,
    ) -> xr.Dataset:
        """
        Subset dataset by region and optional non-horizontal selections.

        Applies:
        - longitude convention normalization
        - longitude rolling when the request crosses the native cutline
        - horizontal subset on rectilinear or curvilinear grids
        - optional vertical level selection
        - optional leadtime selection

        Parameters
        ----------
        data_selection : DataSelection
            Selection object containing at least:
            - region.lat / region.lon
            - optional variable.levhpa / variable.levm
            - optional variable.leadtime

        Returns
        -------
        xr.Dataset
            Spatially subsetted dataset.

        Raises
        ------
        ValueError
            If longitude/latitude coordinates cannot be determined or if the
            subset is empty.
        """
        ds = self._obj

        selection_d = self._build_extra_selection(ds, data_selection)

        lon_req = np.asarray(data_selection.region.lon, dtype=float)
        lat_req = np.asarray(data_selection.region.lat, dtype=float)

        lon_min_req = float(lon_req.min())
        lon_max_req = float(lon_req.max())
        lat_min_req = float(lat_req.min())
        lat_max_req = float(lat_req.max())

        lon_coord = ds.earthml.guessed_coords.longitude
        lat_coord = ds.earthml.guessed_coords.latitude
        if lon_coord is None or lat_coord is None:
            raise ValueError("Could not determine longitude/latitude coordinate names")

        lon_dim = ds.earthml.guessed_dims.longitude

        (lon_grid_min, lon_grid_max), (lat_grid_min, lat_grid_max) = self._get_grid_extremes(
            ds, lon_coord, lat_coord
        )

        lon_min_req, lon_max_req = self._normalize_lon_bounds(
            lon_min_req, lon_max_req, lon_grid_min
        )

        is_rectilinear = self._is_rectilinear_grid(ds, lon_coord, lat_coord)
        grid_type = "rectilinear" if is_rectilinear else "curvilinear"
        logger.info("Grid type: %s", grid_type)

        # Roll dataset if requested interval crosses the native cutline
        ds = self.roll_ds(ds, (lon_min_req, lon_max_req))

        # Refresh coords after rolling
        lat_da = ds[lat_coord]
        lon_da = ds[lon_coord]

        # Recompute cutting longitude from the rolled dataset, then wrap the
        # request into the same continuous longitude frame.
        _, cutting_lon = self._get_cutting_lon(lon_da, lon_dim, (lon_min_req, lon_max_req))
        lon_min_req, lon_max_req = self._wrap_longitudes(
            (lon_min_req, lon_max_req), cutting_lon
        )

        sel_lon_min = None
        sel_lon_max = None
        sel_lat_min = None
        sel_lat_max = None

        if is_rectilinear:
            lon_vals = ds[lon_coord]
            lat_vals = ds[lat_coord]

            lat_lo = min(lat_min_req, lat_max_req)
            lat_hi = max(lat_min_req, lat_max_req)
            lat_mask = (lat_vals >= lat_lo) & (lat_vals <= lat_hi)

            if lon_min_req <= lon_max_req:
                lon_mask = (lon_vals >= lon_min_req) & (lon_vals <= lon_max_req)
            else:
                lon_mask = (lon_vals >= lon_min_req) | (lon_vals <= lon_max_req)

            sel_lon = lon_vals.where(lon_mask, drop=True)
            sel_lat = lat_vals.where(lat_mask, drop=True)

            ds = ds.sel({lon_coord: sel_lon, lat_coord: sel_lat})
            # ds = ds.sel({lon_coord: sel_lon.values, lat_coord: sel_lat.values})

            if selection_d:
                ds = ds.sel(**selection_d)

            sel_lon_min = float(np.nanmin(ds[lon_coord].values))
            sel_lon_max = float(np.nanmax(ds[lon_coord].values))
            sel_lat_min = float(np.nanmin(ds[lat_coord].values))
            sel_lat_max = float(np.nanmax(ds[lat_coord].values))

        else:
            if lon_min_req <= lon_max_req:
                lon_mask = (lon_da >= lon_min_req) & (lon_da < lon_max_req)
            else:
                lon_mask = (lon_da >= lon_min_req) | (lon_da < lon_max_req)

            lat_lo = min(lat_min_req, lat_max_req)
            lat_hi = max(lat_min_req, lat_max_req)
            lat_mask = (lat_da >= lat_lo) & (lat_da < lat_hi)

            mask = (lon_mask & lat_mask).load()

            masked_lon = lon_da.where(mask)
            masked_lat = lat_da.where(mask)

            if np.all(np.isnan(masked_lon)):
                raise ValueError("No longitude points in requested region.")
            if np.all(np.isnan(masked_lat)):
                raise ValueError("No latitude points in requested region.")

            sel_lon_min = float(np.nanmin(masked_lon.values))
            sel_lon_max = float(np.nanmax(masked_lon.values))
            sel_lat_min = float(np.nanmin(masked_lat.values))
            sel_lat_max = float(np.nanmax(masked_lat.values))

            ds = ds.where(mask, drop=True)

            if selection_d:
                ds = ds.sel(**selection_d)

        if ds[lon_coord].size == 0 or ds[lat_coord].size == 0:
            raise ValueError(
                "Subset resulted in an empty dataset. "
                "Requested region may not overlap the dataset."
            )

        logger.info("Size after subsetting %s", ds.sizes)
        return ds
