from collections import OrderedDict
import hashlib
import os
from pathlib import Path

import numpy as np
import cf_xarray
import xarray as xr

from scipy.interpolate import griddata
from scipy.spatial import Delaunay, QhullError

from ..base.dataclasses import Region
from ..logging import get_logger


logger = get_logger(__name__)


_DEFAULT_REGRID_CACHE_DIR = ".earthml-cache/regrid"
_DEFAULT_REGRID_XESMF_CACHE_DIR = ".earthml-cache/regrid-xesmf"
_REGRID_GEOMETRY_CACHE = OrderedDict()
_REGRID_GEOMETRY_CACHE_MAX_ITEMS = max(int(os.getenv("EARTHML_REGRID_CACHE_MAX_ITEMS", "16")), 1)
_REGRID_INTERP_MAP_CACHE = OrderedDict()
_REGRID_INTERP_MAP_CACHE_MAX_ITEMS = max(int(os.getenv("EARTHML_REGRID_INTERP_CACHE_MAX_ITEMS", "64")), 1)


def _import_xesmf():
    # Import lazily so plain dataset loading does not pull xESMF/ESMF native
    # libraries into the process before h5netcdf/h5py get initialized.
    import xesmf as xe

    return xe


def _invalid_curvilinear_coord_mask(
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
) -> np.ndarray:
    lat_2d = np.asarray(lat_2d, dtype=np.float64)
    lon_2d = np.asarray(lon_2d, dtype=np.float64)
    finite = np.isfinite(lat_2d) & np.isfinite(lon_2d)
    sentinel = np.isclose(lat_2d, -1.0, atol=1e-8) & np.isclose(lon_2d, -1.0, atol=1e-8)
    return (~finite) | sentinel


def _crop_curvilinear_valid_bbox(
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
    data_mask_2d: np.ndarray,
    da_spatial: xr.DataArray,
    y_dim_src: str,
    x_dim_src: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, xr.DataArray]:
    coord_invalid_2d = _invalid_curvilinear_coord_mask(lat_2d, lon_2d)
    valid_2d = np.asarray(data_mask_2d, dtype=bool) & (~coord_invalid_2d)
    row_has_valid = np.any(valid_2d, axis=1)
    col_has_valid = np.any(valid_2d, axis=0)
    if not row_has_valid.any() or not col_has_valid.any():
        raise ValueError("Curvilinear source grid has no valid support after masking invalid coordinates.")

    y_idx = np.flatnonzero(row_has_valid)
    x_idx = np.flatnonzero(col_has_valid)
    y0, y1 = int(y_idx[0]), int(y_idx[-1]) + 1
    x0, x1 = int(x_idx[0]), int(x_idx[-1]) + 1

    lat_crop = np.asarray(lat_2d[y0:y1, x0:x1], dtype=np.float64)
    lon_crop = np.asarray(lon_2d[y0:y1, x0:x1], dtype=np.float64)
    mask_crop = np.asarray(valid_2d[y0:y1, x0:x1], dtype=bool)
    da_crop = da_spatial.isel({y_dim_src: slice(y0, y1), x_dim_src: slice(x0, x1)})
    return lat_crop, lon_crop, mask_crop, da_crop


def _build_xesmf_weights_on_disk(
    ds_in_grid: xr.Dataset,
    ds_out_grid: xr.Dataset,
    weights_path: Path,
    *,
    method: str = "bilinear",
    ignore_degenerate: bool | None = None,
) -> None:
    import xesmf.backend as xb
    import xesmf.frontend as xf

    if weights_path.exists():
        weights_path.unlink()

    grid_in, _, _ = xf.ds_to_ESMFgrid(ds_in_grid, need_bounds=False, periodic=False)
    grid_out, _, _ = xf.ds_to_ESMFgrid(ds_out_grid, need_bounds=False, periodic=False)
    regrid = xb.esmf_regrid_build(
        grid_in,
        grid_out,
        method,
        filename=str(weights_path),
        ignore_degenerate=ignore_degenerate,
    )
    xb.esmf_regrid_finalize(regrid)


def _cache_base_dir() -> Path:
    cache_dir = os.getenv("EARTHML_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser()
    return Path.home() / ".earthml-cache"


def _cache_dir() -> Path:
    cache_dir = os.getenv("EARTHML_REGRID_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser()
    return _cache_base_dir() / Path(_DEFAULT_REGRID_CACHE_DIR).name


def _xesmf_cache_dir() -> Path:
    cache_dir = os.getenv("EARTHML_REGRID_XESMF_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir).expanduser()
    return _cache_base_dir() / Path(_DEFAULT_REGRID_XESMF_CACHE_DIR).name


def _hash_array(arr: np.ndarray) -> str:
    arr_c = np.ascontiguousarray(arr)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(arr_c.dtype).encode("ascii"))
    hasher.update(np.asarray(arr_c.shape, dtype=np.int64).tobytes())
    hasher.update(arr_c.view(np.uint8))
    return hasher.hexdigest()


def _build_curvilinear_cache_key(
    lat_src_2d: np.ndarray,
    lon_src_2d: np.ndarray,
    lat_target: np.ndarray,
    lon_target: np.ndarray,
) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for arr in (lat_src_2d, lon_src_2d, lat_target, lon_target):
        hasher.update(_hash_array(arr).encode("ascii"))
    return hasher.hexdigest()


def _remember_geometry_cache_entry(cache_key: str, entry: dict[str, np.ndarray]) -> None:
    _REGRID_GEOMETRY_CACHE[cache_key] = entry
    _REGRID_GEOMETRY_CACHE.move_to_end(cache_key)
    while len(_REGRID_GEOMETRY_CACHE) > _REGRID_GEOMETRY_CACHE_MAX_ITEMS:
        _REGRID_GEOMETRY_CACHE.popitem(last=False)


def _build_interp_map_cache_key(
    geometry_cache_key: str,
    data_valid_mask: np.ndarray,
) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(geometry_cache_key.encode("ascii"))
    hasher.update(_hash_array(np.asarray(data_valid_mask, dtype=bool)).encode("ascii"))
    return hasher.hexdigest()


def _remember_interp_map_cache_entry(cache_key: str, entry: dict[str, np.ndarray | bool | str]) -> None:
    _REGRID_INTERP_MAP_CACHE[cache_key] = entry
    _REGRID_INTERP_MAP_CACHE.move_to_end(cache_key)
    while len(_REGRID_INTERP_MAP_CACHE) > _REGRID_INTERP_MAP_CACHE_MAX_ITEMS:
        _REGRID_INTERP_MAP_CACHE.popitem(last=False)


def _load_or_build_curvilinear_geometry(
    lat_src_2d: np.ndarray,
    lon_src_2d: np.ndarray,
    lat_target: np.ndarray,
    lon_target: np.ndarray,
    *,
    silent: bool = False,
) -> tuple[str, dict[str, np.ndarray]]:
    def _log_info(msg: str, *args) -> None:
        if not silent:
            logger.info(msg, *args)

    cache_key = _build_curvilinear_cache_key(
        lat_src_2d=lat_src_2d,
        lon_src_2d=lon_src_2d,
        lat_target=lat_target,
        lon_target=lon_target,
    )

    entry = _REGRID_GEOMETRY_CACHE.get(cache_key)
    if entry is not None:
        _REGRID_GEOMETRY_CACHE.move_to_end(cache_key)
        _log_info("Regrid geometry cache hit (memory): %s", cache_key)
        return cache_key, entry

    cache_path = _cache_dir() / f"{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            entry = {
                "coord_valid": cached["coord_valid"],
                "points": cached["points"],
                "xi": cached["xi"],
            }
        _remember_geometry_cache_entry(cache_key, entry)
        _log_info("Regrid geometry cache hit (disk): %s", cache_path)
        return cache_key, entry

    lat_flat = lat_src_2d.ravel()
    lon_flat = lon_src_2d.ravel()
    coord_valid = np.isfinite(lat_flat) & np.isfinite(lon_flat)

    points = np.column_stack([lon_flat[coord_valid], lat_flat[coord_valid]])
    lon_out_2d, lat_out_2d = np.meshgrid(lon_target, lat_target)
    xi = np.column_stack([lon_out_2d.ravel(), lat_out_2d.ravel()])

    entry = {
        "coord_valid": coord_valid,
        "points": points,
        "xi": xi,
    }
    _remember_geometry_cache_entry(cache_key, entry)

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            coord_valid=coord_valid,
            points=points,
            xi=xi,
        )
        _log_info("Regrid geometry cache stored: %s", cache_path)
    except OSError as exc:
        logger.warning("Failed to persist regrid geometry cache %s: %s", cache_path, exc)

    _log_info("Regrid geometry cache miss: %s", cache_key)
    return cache_key, entry


def _load_or_build_curvilinear_interp_map(
    *,
    geometry_cache_key: str,
    points: np.ndarray,
    xi: np.ndarray,
    data_valid_mask: np.ndarray,
    silent: bool = False,
) -> dict[str, np.ndarray | bool | str]:
    def _log_info(msg: str, *args) -> None:
        if not silent:
            logger.info(msg, *args)

    cache_key = _build_interp_map_cache_key(
        geometry_cache_key=geometry_cache_key,
        data_valid_mask=data_valid_mask,
    )

    entry = _REGRID_INTERP_MAP_CACHE.get(cache_key)
    if entry is not None:
        _REGRID_INTERP_MAP_CACHE.move_to_end(cache_key)
        _log_info("Regrid interpolation map cache hit: %s", cache_key)
        return entry

    valid_idx = np.flatnonzero(data_valid_mask)
    if valid_idx.size < 3:
        entry = {"usable": False, "reason": "too_few_points"}
        _remember_interp_map_cache_entry(cache_key, entry)
        return entry

    points_valid = points[valid_idx]

    try:
        tri = Delaunay(points_valid)
        simplex = tri.find_simplex(xi)
        inside_mask = simplex >= 0
        inside_idx = np.flatnonzero(inside_mask)

        if inside_idx.size == 0:
            entry = {
                "usable": True,
                "inside_idx": inside_idx.astype(np.int64, copy=False),
                "vertices": np.empty((0, 3), dtype=np.int64),
                "weights": np.empty((0, 3), dtype=np.float64),
                "value_indices": valid_idx.astype(np.int64, copy=False),
            }
            _remember_interp_map_cache_entry(cache_key, entry)
            _log_info("Regrid interpolation map cache miss: %s", cache_key)
            return entry

        simplex_inside = simplex[inside_mask]
        transform = tri.transform[simplex_inside]
        delta = xi[inside_mask] - transform[:, 2, :]
        bary = np.einsum("ijk,ik->ij", transform[:, :2, :], delta)
        weights = np.concatenate(
            [bary, 1.0 - bary.sum(axis=1, keepdims=True)],
            axis=1,
        )
        vertices = tri.simplices[simplex_inside]

        entry = {
            "usable": True,
            "inside_idx": inside_idx.astype(np.int64, copy=False),
            "vertices": vertices.astype(np.int64, copy=False),
            "weights": weights.astype(np.float64, copy=False),
            "value_indices": valid_idx.astype(np.int64, copy=False),
        }
    except QhullError as exc:
        _log_info("Regrid interpolation map build failed (%s): %s", cache_key, exc)
        entry = {"usable": False, "reason": "qhull_error"}

    _remember_interp_map_cache_entry(cache_key, entry)
    _log_info("Regrid interpolation map cache miss: %s", cache_key)
    return entry


def _build_xesmf_grid(
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
    mask_2d: np.ndarray | None = None,
) -> xr.Dataset:
    ds_grid = xr.Dataset(
        coords={
            "lat": (("y", "x"), lat_2d),
            "lon": (("y", "x"), lon_2d),
        }
    )
    if mask_2d is not None:
        ds_grid["mask"] = xr.DataArray(mask_2d.astype(np.int32), dims=("y", "x"))
    return ds_grid


class EarthMLRegrid:
    def regrid_to_rectilinear(
        self,
        region: Region,
        resolution: float | tuple[float, float],
        vars_to_regrid=None,
        *,
        backend: str = "xesmf",
        silent: bool = False,
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

        def _log_info(msg: str, *args) -> None:
            if not silent:
                logger.info(msg, *args)

        _log_info("Regrid: available data vars %s, requested vars %s", list(ds.data_vars.keys()), vars_to_regrid)

        lat0, lat1 = map(float, region.lat)
        lon0, lon1 = map(float, region.lon)

        def _build_axis(start: float, stop: float, step: float) -> np.ndarray:
            step = abs(float(step))
            span = abs(stop - start)
            n = int(round(span / step))

            # Interpret requested region bounds as outer cell edges and build
            # target coordinates at cell centers. This keeps a single common
            # grid for all datasets while avoiding half-cell out-of-support
            # interpolation at the domain border.
            if start > stop:
                first = start - 0.5 * step
                return (first - np.arange(n, dtype=np.float32) * step).astype("float32")
            else:
                first = start + 0.5 * step
                return (first + np.arange(n, dtype=np.float32) * step).astype("float32")

        lat_target = _build_axis(lat0, lat1, lat_res)

        lon0_u, lon1_u = float(lon0), float(lon1)
        while lon1_u < lon0_u:
            lon1_u += 360.0
        lon_target = _build_axis(lon0_u, lon1_u, lon_res)

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
            _log_info("Regrid: rectilinear (1D) source -> rectilinear (1D) target via xarray.interp.")

            lat_src = np.asarray(ds[lat_name].values, dtype=np.float64)
            lon_src = _shift_longitudes_near_reference(ds[lon_name].values, lon_center)

            ds_interp = ds.assign_coords({
                lat_name: xr.DataArray(lat_src, dims=ds[lat_name].dims),
                lon_name: xr.DataArray(lon_src, dims=ds[lon_name].dims),
            })

            ds_interp = _deduplicate_sorted_1d_axis(ds_interp, lon_name)
            ds_interp = _deduplicate_sorted_1d_axis(ds_interp, lat_name)

            src_lat = np.asarray(ds_interp[lat_name].values, dtype=np.float64)
            src_lon = np.asarray(ds_interp[lon_name].values, dtype=np.float64)

            src_lat_res = float(np.abs(np.diff(src_lat)).mean()) if src_lat.size > 1 else np.nan
            src_lon_res = float(np.abs(np.diff(src_lon)).mean()) if src_lon.size > 1 else np.nan

            # logger.info(
            #     "Regrid rectilinear source/target preview: "
            #     "src_lat[%s..%s], tgt_lat[%s..%s], src_lon[%s..%s], tgt_lon[%s..%s]",
            #     src_lat[:3].tolist(),
            #     src_lat[-3:].tolist(),
            #     lat_target[:3].tolist(),
            #     lat_target[-3:].tolist(),
            #     src_lon[:3].tolist(),
            #     src_lon[-3:].tolist(),
            #     lon_target[:3].tolist(),
            #     lon_target[-3:].tolist(),
            # )

            # safest no-op: same spacing, keep existing grid exactly
            same_lat = src_lat.shape == lat_target.shape and np.allclose(src_lat, lat_target, atol=1e-6)
            same_lon = src_lon.shape == lon_target.shape and np.allclose(src_lon, lon_target, atol=1e-6)

            if same_lat and same_lon:
                _log_info("  no-op: keeping existing rectilinear grid")
                return ds_interp

            lat_tgt_da = xr.DataArray(lat_target, dims=(lat_name,), name=lat_name)
            lon_tgt_da = xr.DataArray(lon_target, dims=(lon_name,), name=lon_name)

            coord_out = {k: v for k, v in ds_interp.coords.items() if k not in (lat_name, lon_name)}
            coord_out[lat_name] = lat_tgt_da
            coord_out[lon_name] = lon_tgt_da

            data_vars_out = {}
            for name, da in ds_interp.data_vars.items():
                if name in vars_to_regrid and lat_name in da.dims and lon_name in da.dims:
                    data_vars_out[name] = da.interp(
                        {lat_name: lat_tgt_da, lon_name: lon_tgt_da},
                        method="linear",
                    )
                else:
                    # keep untouched vars only if they do not depend on lat/lon
                    if lat_name in da.dims or lon_name in da.dims:
                        data_vars_out[name] = da.interp(
                            {lat_name: lat_tgt_da, lon_name: lon_tgt_da},
                            method="nearest",
                        )
                    else:
                        data_vars_out[name] = da

            return xr.Dataset(data_vars_out, coords=coord_out, attrs=ds_interp.attrs)

        # ======================================================================
        # Case 2: curvilinear (2D) source -> rectilinear (1D) target
        # ======================================================================
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

        if backend == "xesmf":
            _log_info("Regrid: curvilinear (2D) source -> rectilinear (1D) target via xESMF.")

            lon_out_2d, lat_out_2d = np.meshgrid(lon_target, lat_target)
            ds_out_grid = _build_xesmf_grid(lat_out_2d, lon_out_2d)

            data_vars_out = {}
            for name, da in ds.data_vars.items():
                if name not in vars_to_regrid:
                    data_vars_out[name] = da
                    continue
                if y_dim_src not in da.dims or x_dim_src not in da.dims:
                    data_vars_out[name] = da
                    continue

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

                # static mask from first slice, or from external/source mask if you have one
                sample = da_spatial.isel({d: 0 for d in da_spatial.dims[:-2]}) if da_spatial.dims[:-2] else da_spatial
                data_mask_2d = np.isfinite(sample.values)
                lat_src_use, lon_src_use, mask_2d, da_spatial = _crop_curvilinear_valid_bbox(
                    lat_2d=lat_src_2d,
                    lon_2d=lon_src_2d,
                    data_mask_2d=data_mask_2d,
                    da_spatial=da_spatial,
                    y_dim_src=y_dim_src,
                    x_dim_src=x_dim_src,
                )
                data_np = da_spatial.values
                leading_dims = da_spatial.dims[:-2]
                leading_shape = data_np.shape[:-2]
                ny_src, nx_src = data_np.shape[-2:]

                ds_in_grid = _build_xesmf_grid(lat_src_use, lon_src_use, mask_2d=mask_2d)

                weights_key = hashlib.blake2b(digest_size=16)
                for arr in (lat_src_use, lon_src_use, lat_target, lon_target):
                    weights_key.update(_hash_array(np.asarray(arr)).encode("ascii"))
                weights_key.update(b"xesmf")
                weights_key.update(b"bilinear")
                weights_key.update(_hash_array(mask_2d.astype(np.int8)).encode("ascii"))

                weights_path = _xesmf_cache_dir() / f"{weights_key.hexdigest()}.nc"
                weights_path.parent.mkdir(parents=True, exist_ok=True)

                xe = _import_xesmf()
                reuse_weights = weights_path.exists()
                try:
                    regridder = xe.Regridder(
                        ds_in_grid,
                        ds_out_grid,
                        "bilinear",
                        unmapped_to_nan=True,
                        reuse_weights=reuse_weights,
                        filename=str(weights_path),
                    )
                except RuntimeError as exc:
                    if (
                        not reuse_weights
                        and "in-memory factors only supported with GNU (gfortran)" in str(exc)
                    ):
                        logger.debug(
                            "xESMF in-memory weight generation is unsupported on this ESMF build; "
                            "building weights on disk via ESMPy backend: %s",
                            weights_path,
                        )
                        _build_xesmf_weights_on_disk(
                            ds_in_grid=ds_in_grid,
                            ds_out_grid=ds_out_grid,
                            weights_path=weights_path,
                            method="bilinear",
                        )
                        regridder = xe.Regridder(
                            ds_in_grid,
                            ds_out_grid,
                            "bilinear",
                            unmapped_to_nan=True,
                            reuse_weights=True,
                            filename=str(weights_path),
                        )
                    else:
                        raise

                out = regridder(da_spatial, keep_attrs=True)
                # out = np.stack(out_slices, axis=0).reshape(*leading_shape, Ny, Nx)

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

        # scipy.griddata, inpaint NaNs if masked data
        else:
            _log_info("Regrid: curvilinear (2D) source -> rectilinear (1D) target via scipy.griddata.")

            Ny = lat_target.size
            Nx = lon_target.size

            geometry_cache_key, geometry = _load_or_build_curvilinear_geometry(
                lat_src_2d=lat_src_2d,
                lon_src_2d=lon_src_2d,
                lat_target=lat_target,
                lon_target=lon_target,
                silent=silent,
            )
            coord_valid = geometry["coord_valid"]
            points = geometry["points"]
            xi = geometry["xi"]

            data_vars_out = {}
            for name, da in ds.data_vars.items():
                if name not in vars_to_regrid:
                    data_vars_out[name] = da
                    continue
                if y_dim_src not in da.dims or x_dim_src not in da.dims:
                    data_vars_out[name] = da
                    continue
                
                _log_info("  Regridding variable '%s' with griddata...", name)

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
                    zi_coord_valid = zi[coord_valid]
                    data_valid = np.isfinite(zi_coord_valid)
                    valid_count = int(data_valid.sum())

                    if valid_count == 0:
                        zi_interp = np.full(xi.shape[0], np.nan, dtype=np.float64)
                    elif valid_count < 3:
                        # Not enough points for linear triangulation
                        zi_interp = np.full(xi.shape[0], np.nan, dtype=np.float64)
                    else:
                        interp_map = _load_or_build_curvilinear_interp_map(
                            geometry_cache_key=geometry_cache_key,
                            points=points,
                            xi=xi,
                            data_valid_mask=data_valid,
                            silent=silent,
                        )

                        if bool(interp_map["usable"]):
                            zi_interp = np.full(xi.shape[0], np.nan, dtype=np.float64)
                            inside_idx = interp_map["inside_idx"]
                            vertices = interp_map["vertices"]
                            weights = interp_map["weights"]
                            value_indices = interp_map["value_indices"]

                            if inside_idx.size:
                                zi_values = zi_coord_valid[value_indices]
                                zi_interp[inside_idx] = np.einsum(
                                    "ij,ij->i",
                                    zi_values[vertices],
                                    weights,
                                )
                        else:
                            zi_interp = griddata(
                                points[data_valid],
                                zi_coord_valid[data_valid],
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
