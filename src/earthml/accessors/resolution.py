import numpy as np
import cf_xarray
import xarray as xr

from ..logging import get_logger


logger = get_logger(__name__)


class EarthMLResolution:
    @staticmethod
    def _coord_resolution_1d(values, name):
        vals = np.asarray(values)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            raise ValueError(f"Not enough points to infer resolution from {name}")
        diffs = np.diff(vals)
        diffs = np.abs(diffs[np.isfinite(diffs)])
        if diffs.size == 0:
            raise ValueError(f"No valid diffs to infer resolution from {name}")
        return float(diffs.mean())

    @staticmethod
    def _is_rectilinear_disguised(lat2d, lon2d, tol=1e-6):
        """Check if lat varies only along axis 0 and lon only along axis 1."""
        lat = np.asarray(lat2d)
        lon = np.asarray(lon2d)
        if lat.ndim != 2 or lon.ndim != 2:
            return False

        dlat_dx = np.diff(lat, axis=1)  # should be ~0 if lat=f(i)
        dlon_dy = np.diff(lon, axis=0)  # should be ~0 if lon=g(j)

        max_abs_dlat_dx = np.nanmax(np.abs(dlat_dx))
        max_abs_dlon_dy = np.nanmax(np.abs(dlon_dy))

        is_lat_only_y = not np.isnan(max_abs_dlat_dx) and max_abs_dlat_dx < tol
        is_lon_only_x = not np.isnan(max_abs_dlon_dy) and max_abs_dlon_dy < tol

        return is_lat_only_y and is_lon_only_x

    @staticmethod
    def _spacing_stats_line(line_deg, is_lon=False):
        """
        Return (mean_spacing, std_spacing, cv) along a 1D line,
        handling NaNs and longitude wrap if needed.
        """
        arr = np.asarray(line_deg)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return np.nan, np.nan, np.nan

        if is_lon:
            # unwrap for dateline crossings
            arr_rad = np.deg2rad(arr)
            arr_unwrapped = np.rad2deg(np.unwrap(arr_rad))
        else:
            arr_unwrapped = arr

        diffs = np.diff(arr_unwrapped)
        diffs = np.abs(diffs[np.isfinite(diffs)])
        if diffs.size == 0:
            return np.nan, np.nan, np.nan

        mean = float(diffs.mean())
        std = float(diffs.std())
        cv = std / mean if mean != 0 else np.inf
        return mean, std, cv

    def _is_regular_curvilinear(self, lat2d, lon2d, rel_tol=1e-2, min_points=10):
        """
        Check if spacing in a curvilinear grid is 'regular enough':
        cv (std/mean) < rel_tol along central lat and lon lines.
        """
        lat = np.asarray(lat2d)
        lon = np.asarray(lon2d)
        if lat.ndim != 2 or lon.ndim != 2:
            return False, (np.nan, np.nan)

        ny, nx = lat.shape
        j0 = nx // 2  # central column (meridional line)
        i0 = ny // 2  # central row (zonal line)

        lat_line = lat[:, j0]
        lon_line = lon[i0, :]

        lat_mean, lat_std, lat_cv = self._spacing_stats_line(lat_line, is_lon=False)
        lon_mean, lon_std, lon_cv = self._spacing_stats_line(lon_line, is_lon=True)

        n_lat = np.isfinite(lat_line).sum()
        n_lon = np.isfinite(lon_line).sum()

        if (
            np.isnan(lat_mean) or np.isnan(lon_mean)
            or n_lat < min_points or n_lon < min_points
        ):
            return False, (lat_mean, lon_mean)

        is_regular = (lat_cv < rel_tol) and (lon_cv < rel_tol)
        return is_regular, (abs(lat_mean), abs(lon_mean))

    @staticmethod
    def _fallback_curvilinear_res(lat2d, lon2d):
        """
        General heuristic for non-regular curvilinear:
        use median of absolute neighbor diffs in both directions.
        """
        lat = np.asarray(lat2d)
        lon = np.asarray(lon2d)

        # lat diffs
        lat_diffs = []
        for axis in (0, 1):
            d = np.diff(lat, axis=axis)
            d = np.abs(d[np.isfinite(d)])
            if d.size > 0:
                lat_diffs.append(d.ravel())
        if not lat_diffs:
            raise ValueError("Cannot infer latitude resolution from curvilinear grid")
        lat_all = np.concatenate(lat_diffs)
        lat_res = float(np.median(lat_all))

        # lon diffs with wrap handling
        lon_diffs = []
        for axis in (0, 1):
            d = np.diff(lon, axis=axis)
            # map to [-180, 180) to handle wrapping
            d = (d + 180.0) % 360.0 - 180.0
            d = np.abs(d[np.isfinite(d)])
            if d.size > 0:
                lon_diffs.append(d.ravel())
        if not lon_diffs:
            raise ValueError("Cannot infer longitude resolution from curvilinear grid")
        lon_all = np.concatenate(lon_diffs)
        lon_res = float(np.median(lon_all))

        return lat_res, lon_res

    def resolution(self):
        """
        Estimate latitude/longitude resolution (in degrees) for rectilinear or
        curvilinear grids.

        Logic:
        - If both lat & lon are 1D: treat as rectilinear.
        - If both are 2D:
            * check if they are rectilinear in disguise (lat=f(i), lon=g(j))
            * else check if spacing is roughly uniform (regular curvilinear)
            * else fall back to a more general heuristic on neighbor diffs.
        - Mixed 1D/2D dims are treated individually.
        """
        ds = self._obj

        lon_coord, lat_coord = ds.earthml.guessed_coords.longitude, ds.earthml.guessed_coords.latitude
        # print(f"Coords name lon: {lon_coord}, lat: {lat_coord}")

        lat_da = ds[lat_coord]
        lon_da = ds[lon_coord]

        # ---------- main logic ----------

        # Case 1: both 1D -> simple rectilinear
        if lat_da.ndim == 1 and lon_da.ndim == 1:
            lat_res = self._coord_resolution_1d(lat_da.values, lat_coord)
            lon_res = self._coord_resolution_1d(lon_da.values, lon_coord)
            return lat_res, lon_res

        # Case 2: both 2D -> combined detection logic
        if lat_da.ndim == 2 and lon_da.ndim == 2:
            lat2d = lat_da.values
            lon2d = lon_da.values

            # 2a. Rectilinear in disguise?
            if self._is_rectilinear_disguised(lat2d, lon2d):
                logger.info("Detected rectilinear grid stored as 2D.")
                lat_line = lat2d[:, 0]
                lon_line = lon2d[0, :]
                lat_res = self._coord_resolution_1d(lat_line, lat_coord)
                lon_res = self._coord_resolution_1d(lon_line, lon_coord)
                return lat_res, lon_res

            # 2b. Regular curvilinear?
            is_reg, (lat_reg, lon_reg) = self._is_regular_curvilinear(lat2d, lon2d, rel_tol=0.5)
            if is_reg:
                logger.info("Detected regular curvilinear grid.")
                return lat_reg, lon_reg

            # 2c. Fallback heuristic
            logger.info("Grid appears non-regular; using fallback curvilinear heuristic.")
            lat_res, lon_res = self._fallback_curvilinear_res(lat2d, lon2d)
            return lat_res, lon_res

        # Case 3: mixed dims (rare) – treat each independently
        logger.info("Mixed lat/lon dimensions (one 1D, one 2D); treating separately.")

        # Latitude
        if lat_da.ndim == 1:
            lat_res = self._coord_resolution_1d(lat_da.values, lat_coord)
        elif lat_da.ndim == 2:
            # use a simple central-column estimate
            lat_mean, _, _ = self._spacing_stats_line(lat_da.values[:, lat_da.values.shape[1] // 2], is_lon=False)
            if np.isnan(lat_mean):
                raise ValueError("Cannot infer latitude resolution from 2D latitude.")
            lat_res = abs(lat_mean)
        else:
            raise ValueError("Latitude coordinate must be 1D or 2D")

        # Longitude
        if lon_da.ndim == 1:
            lon_res = self._coord_resolution_1d(lon_da.values, lon_coord)
        elif lon_da.ndim == 2:
            lon_mean, _, _ = self._spacing_stats_line(
                lon_da.values[lon_da.values.shape[0] // 2, :],
                is_lon=True,
            )
            if np.isnan(lon_mean):
                raise ValueError("Cannot infer longitude resolution from 2D longitude.")
            lon_res = abs(lon_mean)
        else:
            raise ValueError("Longitude coordinate must be 1D or 2D")

        return lat_res, lon_res
