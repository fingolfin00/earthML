from collections import OrderedDict
from collections.abc import Iterable
from numbers import Real
from pathlib import Path
from typing import Any, Literal, TypeAlias

import hashlib
import os
import tempfile
import threading

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, QhullError

from ..base import Region
from ..logging import get_logger


logger = get_logger(__name__)

ResolutionLike: TypeAlias = float | tuple[float, float] | list[float] | Literal["source"] | None
Backend: TypeAlias = Literal["xesmf", "scipy", "griddata"]
NormalizedBackend: TypeAlias = Literal["xesmf", "scipy"]
GeometryEntry: TypeAlias = dict[str, np.ndarray]
InterpEntry: TypeAlias = dict[str, Any]

_REGRID_CACHE_SUBDIR = "regrid"
_XESMF_CACHE_SUBDIR = "regrid-xesmf"

_REGRID_GEOMETRY_CACHE: OrderedDict[str, GeometryEntry] = OrderedDict()
_REGRID_GEOMETRY_CACHE_MAX_ITEMS = max(
    int(os.getenv("EARTHML_REGRID_CACHE_MAX_ITEMS", "16")),
    1,
)

_REGRID_INTERP_MAP_CACHE: OrderedDict[str, InterpEntry] = OrderedDict()
_REGRID_INTERP_MAP_CACHE_MAX_ITEMS = max(
    int(os.getenv("EARTHML_REGRID_INTERP_CACHE_MAX_ITEMS", "64")),
    1,
)

_XESMF_WEIGHTS_LOCKS: dict[str, threading.Lock] = {}
_XESMF_WEIGHTS_LOCKS_GUARD = threading.Lock()


def _normalize_backend(backend: Backend | str) -> NormalizedBackend:
    if backend == "xesmf":
        return "xesmf"
    if backend in ("scipy", "griddata"):
        return "scipy"
    raise ValueError(
        "backend must be one of {'xesmf', 'scipy', 'griddata'}. "
        f"Got {backend!r}."
    )


def _normalize_resolution(
    ds: xr.Dataset,
    resolution: ResolutionLike,
) -> tuple[float, float]:
    def source_resolution() -> tuple[float, float]:
        earthml = getattr(ds, "earthml", None)
        if earthml is None or not hasattr(earthml, "resolution"):
            raise ValueError(
                "resolution='source' requires the earthml resolution accessor."
            )
        lat, lon = earthml.resolution()
        return float(lat), float(lon)

    if resolution is None:
        lat_res, lon_res = source_resolution()
    elif isinstance(resolution, str):
        if resolution != "source":
            raise ValueError(
                "String resolution values must be 'source'. "
                f"Got {resolution!r}."
            )
        lat_res, lon_res = source_resolution()
    elif isinstance(resolution, Real) and not isinstance(resolution, bool):
        lat_res = lon_res = float(resolution)
    elif isinstance(resolution, (tuple, list)):
        if len(resolution) != 2:
            raise ValueError(
                "Resolution sequence must have exactly two values: "
                "(lat_res, lon_res)."
            )
        lat_res = float(resolution[0])
        lon_res = float(resolution[1])
    else:
        raise TypeError(
            "resolution must be a float, a (lat_res, lon_res) tuple/list, "
            "'source', or None."
        )

    if not np.isfinite(lat_res) or not np.isfinite(lon_res):
        raise ValueError(
            f"Resolution must be finite. Got lat_res={lat_res}, lon_res={lon_res}."
        )
    if lat_res <= 0 or lon_res <= 0:
        raise ValueError(
            f"Resolution must be positive. Got lat_res={lat_res}, lon_res={lon_res}."
        )

    return float(lat_res), float(lon_res)


def _build_axis(start: float, stop: float, step: float) -> np.ndarray:
    step = abs(float(step))
    span = abs(float(stop) - float(start))
    n = int(round(span / step))

    if n < 1:
        raise ValueError(
            f"Cannot build target axis: start={start}, stop={stop}, step={step}."
        )

    # Region bounds are interpreted as outer cell edges; coordinates are centers.
    if start > stop:
        first = start - 0.5 * step
        values = first - np.arange(n, dtype=np.float64) * step
    else:
        first = start + 0.5 * step
        values = first + np.arange(n, dtype=np.float64) * step

    return values.astype("float32")


def _shift_longitudes_near_reference(
    lon_values: np.ndarray,
    reference: float,
) -> np.ndarray:
    lon_values = np.asarray(lon_values, dtype=np.float64)
    return lon_values + 360.0 * np.round((reference - lon_values) / 360.0)


def _prepare_rectilinear_source(
    ds: xr.Dataset,
    *,
    lat_name: str,
    lon_name: str,
    lon_center: float,
) -> xr.Dataset:
    lat_da = ds[lat_name]
    lon_da = ds[lon_name]

    if lat_da.ndim != 1 or lon_da.ndim != 1:
        raise ValueError("Rectilinear source preparation requires 1D lat/lon.")

    lat_dim = lat_da.dims[0]
    lon_dim = lon_da.dims[0]
    if lat_dim == lon_dim:
        raise ValueError(
            "1D latitude and longitude share the same dimension. "
            "This looks like an unstructured grid, not a rectilinear grid."
        )

    ds_out = ds.assign_coords(
        {
            lat_name: xr.DataArray(
                np.asarray(lat_da.values, dtype=np.float64),
                dims=lat_da.dims,
                attrs=lat_da.attrs,
                name=lat_name,
            ),
            lon_name: xr.DataArray(
                _shift_longitudes_near_reference(lon_da.values, lon_center),
                dims=lon_da.dims,
                attrs=lon_da.attrs,
                name=lon_name,
            ),
        }
    )

    for coord_name in (lon_name, lat_name):
        coord = ds_out[coord_name]
        dim = coord.dims[0]

        ds_out = ds_out.sortby(coord_name)
        coord_values = np.asarray(ds_out[coord_name].values)
        _, unique_idx = np.unique(coord_values, return_index=True)

        if unique_idx.size != ds_out.sizes[dim]:
            ds_out = ds_out.isel({dim: np.sort(unique_idx)})

    swap_dims: dict[str, str] = {}
    if lat_dim != lat_name:
        swap_dims[lat_dim] = lat_name
    if lon_dim != lon_name:
        swap_dims[lon_dim] = lon_name
    if swap_dims:
        ds_out = ds_out.swap_dims(swap_dims)

    return ds_out


def _cache_dir(env_var: str, default_subdir: str) -> Path:
    explicit = os.getenv(env_var)
    if explicit:
        return Path(explicit).expanduser()

    base = Path(os.getenv("EARTHML_CACHE_DIR", ".earthml-cache")).expanduser()
    return base / default_subdir


def _import_xesmf():
    # Import lazily so dataset loading does not eagerly load ESMF native libs.
    import xesmf as xe

    return xe


def _get_xesmf_weights_lock(weights_path: Path) -> threading.Lock:
    key = str(weights_path.resolve())
    with _XESMF_WEIGHTS_LOCKS_GUARD:
        lock = _XESMF_WEIGHTS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _XESMF_WEIGHTS_LOCKS[key] = lock
        return lock


def _invalid_curvilinear_coord_mask(
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
) -> np.ndarray:
    lat_2d = np.asarray(lat_2d, dtype=np.float64)
    lon_2d = np.asarray(lon_2d, dtype=np.float64)

    if lat_2d.shape != lon_2d.shape:
        raise ValueError(
            f"Latitude and longitude shapes differ: {lat_2d.shape} vs {lon_2d.shape}."
        )

    finite = np.isfinite(lat_2d) & np.isfinite(lon_2d)

    # Source-specific sentinel. Compute this before longitude branch shifting.
    sentinel = np.isclose(lat_2d, -1.0, atol=1e-8) & np.isclose(
        lon_2d,
        -1.0,
        atol=1e-8,
    )

    return (~finite) | sentinel


def _repair_curvilinear_coords(
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
    *,
    invalid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_fixed = np.asarray(lat_2d, dtype=np.float64).copy()
    lon_fixed = np.asarray(lon_2d, dtype=np.float64).copy()
    original_invalid = np.asarray(invalid_mask, dtype=bool).copy()

    if lat_fixed.shape != lon_fixed.shape or lat_fixed.shape != original_invalid.shape:
        raise ValueError(
            "lat_2d, lon_2d, and invalid_mask must have the same shape. "
            f"Got {lat_fixed.shape}, {lon_fixed.shape}, {original_invalid.shape}."
        )

    remaining_invalid = original_invalid.copy()
    ny, nx = lat_fixed.shape
    x_all = np.arange(nx, dtype=np.float64)

    # Repair partially invalid rows by interpolation/extension along x.
    for iy in range(ny):
        row_bad = remaining_invalid[iy]
        if not row_bad.any():
            continue

        valid_idx = np.flatnonzero(~row_bad)
        if valid_idx.size == 0:
            continue

        first = int(valid_idx[0])
        last = int(valid_idx[-1])

        if valid_idx.size >= 2:
            xp = valid_idx.astype(np.float64)
            lat_interp = np.interp(x_all, xp, lat_fixed[iy, valid_idx])
            lon_interp = np.interp(x_all, xp, lon_fixed[iy, valid_idx])

            gap_mask = row_bad & (x_all > first) & (x_all < last)
            lat_fixed[iy, gap_mask] = lat_interp[gap_mask]
            lon_fixed[iy, gap_mask] = lon_interp[gap_mask]

        if first > 0:
            if valid_idx.size >= 2:
                step_lat = lat_fixed[iy, valid_idx[1]] - lat_fixed[iy, first]
                step_lon = lon_fixed[iy, valid_idx[1]] - lon_fixed[iy, first]
            else:
                step_lat = step_lon = 0.0

            for ix in range(first - 1, -1, -1):
                lat_fixed[iy, ix] = lat_fixed[iy, ix + 1] - step_lat
                lon_fixed[iy, ix] = lon_fixed[iy, ix + 1] - step_lon

        if last < nx - 1:
            if valid_idx.size >= 2:
                step_lat = lat_fixed[iy, last] - lat_fixed[iy, valid_idx[-2]]
                step_lon = lon_fixed[iy, last] - lon_fixed[iy, valid_idx[-2]]
            else:
                step_lat = step_lon = 0.0

            for ix in range(last + 1, nx):
                lat_fixed[iy, ix] = lat_fixed[iy, ix - 1] + step_lat
                lon_fixed[iy, ix] = lon_fixed[iy, ix - 1] + step_lon

        remaining_invalid[iy, row_bad] = False

    # Fully invalid rows borrow from the nearest repaired row.
    row_all_bad = np.all(remaining_invalid, axis=1)
    good_rows = np.flatnonzero(~row_all_bad)
    if good_rows.size:
        for iy in np.flatnonzero(row_all_bad):
            nearest = int(good_rows[np.argmin(np.abs(good_rows - iy))])
            lat_fixed[iy, :] = lat_fixed[nearest, :]
            lon_fixed[iy, :] = lon_fixed[nearest, :]
            remaining_invalid[iy, :] = False

    # Last resort: column-wise interpolation for any remaining invalid points.
    if remaining_invalid.any():
        y_all = np.arange(ny, dtype=np.float64)
        for ix in range(nx):
            col_bad = remaining_invalid[:, ix]
            if not col_bad.any():
                continue

            valid_idx = np.flatnonzero(~col_bad)
            if valid_idx.size == 0:
                continue

            if valid_idx.size == 1:
                lat_fixed[col_bad, ix] = lat_fixed[valid_idx[0], ix]
                lon_fixed[col_bad, ix] = lon_fixed[valid_idx[0], ix]
            else:
                yp = valid_idx.astype(np.float64)
                lat_interp = np.interp(y_all, yp, lat_fixed[valid_idx, ix])
                lon_interp = np.interp(y_all, yp, lon_fixed[valid_idx, ix])
                lat_fixed[col_bad, ix] = lat_interp[col_bad]
                lon_fixed[col_bad, ix] = lon_interp[col_bad]

            remaining_invalid[col_bad, ix] = False

    return lat_fixed, lon_fixed, original_invalid


def _crop_curvilinear_valid_bbox(
    lat_2d: np.ndarray,
    lon_2d: np.ndarray,
    coord_invalid_2d: np.ndarray,
    data_mask_2d: np.ndarray,
    da_spatial: xr.DataArray,
    y_dim_src: str,
    x_dim_src: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, xr.DataArray]:
    lat_2d, lon_2d, coord_invalid_2d = _repair_curvilinear_coords(
        lat_2d,
        lon_2d,
        invalid_mask=coord_invalid_2d,
    )

    if data_mask_2d.shape != lat_2d.shape:
        raise ValueError(
            f"Data mask shape {data_mask_2d.shape} does not match grid shape {lat_2d.shape}."
        )

    valid_2d = np.asarray(data_mask_2d, dtype=bool) & (~coord_invalid_2d)
    row_has_valid = np.any(valid_2d, axis=1)
    col_has_valid = np.any(valid_2d, axis=0)

    if not row_has_valid.any() or not col_has_valid.any():
        raise ValueError(
            "Curvilinear source grid has no valid support after masking invalid coordinates."
        )

    y_idx = np.flatnonzero(row_has_valid)
    x_idx = np.flatnonzero(col_has_valid)
    y0, y1 = int(y_idx[0]), int(y_idx[-1]) + 1
    x0, x1 = int(x_idx[0]), int(x_idx[-1]) + 1

    return (
        np.asarray(lat_2d[y0:y1, x0:x1], dtype=np.float64),
        np.asarray(lon_2d[y0:y1, x0:x1], dtype=np.float64),
        np.asarray(valid_2d[y0:y1, x0:x1], dtype=bool),
        da_spatial.isel({y_dim_src: slice(y0, y1), x_dim_src: slice(x0, x1)}),
    )


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

    tmp_path = weights_path.with_name(
        f"{weights_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    if tmp_path.exists():
        tmp_path.unlink()

    grid_in, _, _ = xf.ds_to_ESMFgrid(ds_in_grid, need_bounds=False, periodic=False)
    grid_out, _, _ = xf.ds_to_ESMFgrid(ds_out_grid, need_bounds=False, periodic=False)

    regrid = xb.esmf_regrid_build(
        grid_in,
        grid_out,
        method,
        filename=str(tmp_path),
        ignore_degenerate=ignore_degenerate,
    )
    xb.esmf_regrid_finalize(regrid)

    if not tmp_path.exists():
        raise RuntimeError(f"ESMF failed to generate temporary weights file: {tmp_path}")

    tmp_path.replace(weights_path)


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
    coord_invalid_2d: np.ndarray,
    lat_target: np.ndarray,
    lon_target: np.ndarray,
) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"earthml-curvilinear-geometry-v2")

    for arr in (lat_src_2d, lon_src_2d, lat_target, lon_target):
        hasher.update(_hash_array(np.asarray(arr)).encode("ascii"))

    hasher.update(_hash_array(np.asarray(coord_invalid_2d, dtype=np.int8)).encode("ascii"))
    return hasher.hexdigest()


def _remember_lru(
    cache: OrderedDict[str, Any],
    key: str,
    value: Any,
    max_items: int,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        cache.popitem(last=False)


def _build_interp_map_cache_key(
    geometry_cache_key: str,
    data_valid_mask: np.ndarray,
) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(b"earthml-curvilinear-interp-v2")
    hasher.update(geometry_cache_key.encode("ascii"))
    hasher.update(_hash_array(np.asarray(data_valid_mask, dtype=bool)).encode("ascii"))
    return hasher.hexdigest()


def _load_or_build_curvilinear_geometry(
    lat_src_2d: np.ndarray,
    lon_src_2d: np.ndarray,
    coord_invalid_2d: np.ndarray,
    lat_target: np.ndarray,
    lon_target: np.ndarray,
    *,
    silent: bool = False,
) -> tuple[str, GeometryEntry]:
    def log_info(msg: str, *args: object) -> None:
        if not silent:
            logger.info(msg, *args)

    cache_key = _build_curvilinear_cache_key(
        lat_src_2d=lat_src_2d,
        lon_src_2d=lon_src_2d,
        coord_invalid_2d=coord_invalid_2d,
        lat_target=lat_target,
        lon_target=lon_target,
    )

    entry = _REGRID_GEOMETRY_CACHE.get(cache_key)
    if entry is not None:
        _REGRID_GEOMETRY_CACHE.move_to_end(cache_key)
        log_info("Regrid geometry cache hit (memory): %s", cache_key)
        return cache_key, entry

    cache_path = _cache_dir("EARTHML_REGRID_CACHE_DIR", _REGRID_CACHE_SUBDIR) / f"{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            entry = {
                "coord_valid": cached["coord_valid"],
                "points": cached["points"],
                "xi": cached["xi"],
            }
        _remember_lru(
            _REGRID_GEOMETRY_CACHE,
            cache_key,
            entry,
            _REGRID_GEOMETRY_CACHE_MAX_ITEMS,
        )
        log_info("Regrid geometry cache hit (disk): %s", cache_path)
        return cache_key, entry

    lat_flat = np.asarray(lat_src_2d, dtype=np.float64).ravel()
    lon_flat = np.asarray(lon_src_2d, dtype=np.float64).ravel()
    invalid_flat = np.asarray(coord_invalid_2d, dtype=bool).ravel()

    coord_valid = np.isfinite(lat_flat) & np.isfinite(lon_flat) & (~invalid_flat)
    points = np.column_stack([lon_flat[coord_valid], lat_flat[coord_valid]])

    lon_out_2d, lat_out_2d = np.meshgrid(lon_target, lat_target)
    xi = np.column_stack([lon_out_2d.ravel(), lat_out_2d.ravel()])

    entry = {
        "coord_valid": coord_valid,
        "points": points,
        "xi": xi,
    }
    _remember_lru(
        _REGRID_GEOMETRY_CACHE,
        cache_key,
        entry,
        _REGRID_GEOMETRY_CACHE_MAX_ITEMS,
    )

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            coord_valid=coord_valid,
            points=points,
            xi=xi,
        )
        log_info("Regrid geometry cache stored: %s", cache_path)
    except OSError as exc:
        logger.warning("Failed to persist regrid geometry cache %s: %s", cache_path, exc)

    log_info("Regrid geometry cache miss: %s", cache_key)
    return cache_key, entry


def _load_or_build_curvilinear_interp_map(
    *,
    geometry_cache_key: str,
    points: np.ndarray,
    xi: np.ndarray,
    data_valid_mask: np.ndarray,
    silent: bool = False,
) -> InterpEntry:
    def log_info(msg: str, *args: object) -> None:
        if not silent:
            logger.info(msg, *args)

    cache_key = _build_interp_map_cache_key(
        geometry_cache_key=geometry_cache_key,
        data_valid_mask=data_valid_mask,
    )

    entry = _REGRID_INTERP_MAP_CACHE.get(cache_key)
    if entry is not None:
        _REGRID_INTERP_MAP_CACHE.move_to_end(cache_key)
        log_info("Regrid interpolation map cache hit: %s", cache_key)
        return entry

    valid_idx = np.flatnonzero(data_valid_mask)
    if valid_idx.size < 3:
        entry = {"usable": False, "reason": "too_few_points"}
        _remember_lru(
            _REGRID_INTERP_MAP_CACHE,
            cache_key,
            entry,
            _REGRID_INTERP_MAP_CACHE_MAX_ITEMS,
        )
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
        else:
            simplex_inside = simplex[inside_mask]
            transform = tri.transform[simplex_inside]
            delta = xi[inside_mask] - transform[:, 2, :]
            bary = np.einsum("ijk,ik->ij", transform[:, :2, :], delta)
            weights = np.concatenate(
                [bary, 1.0 - bary.sum(axis=1, keepdims=True)],
                axis=1,
            )

            entry = {
                "usable": True,
                "inside_idx": inside_idx.astype(np.int64, copy=False),
                "vertices": tri.simplices[simplex_inside].astype(np.int64, copy=False),
                "weights": weights.astype(np.float64, copy=False),
                "value_indices": valid_idx.astype(np.int64, copy=False),
            }
    except QhullError as exc:
        log_info("Regrid interpolation map build failed (%s): %s", cache_key, exc)
        entry = {"usable": False, "reason": "qhull_error"}

    _remember_lru(
        _REGRID_INTERP_MAP_CACHE,
        cache_key,
        entry,
        _REGRID_INTERP_MAP_CACHE_MAX_ITEMS,
    )
    log_info("Regrid interpolation map cache miss: %s", cache_key)
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
    """
    Regridding accessor for EarthML datasets.

    This accessor provides utilities to transform datasets from their native
    spatial grid onto a regular latitude/longitude grid suitable for analysis,
    machine learning workflows, visualization, and intercomparison between
    datasets.

    Supported source grids
    ----------------------
    Rectilinear grids
        Latitude and longitude are stored as 1D coordinates.

        Example::

            latitude(lat)
            longitude(lon)

        Regridding is performed with ``xarray.Dataset.interp()``, which is
        typically fast and memory-efficient for regular grids.

    Curvilinear grids
        Latitude and longitude are stored as 2D coordinates.

        Example::

            latitude(y, x)
            longitude(y, x)

        Two backends are available:

        - ``backend="xesmf"``
          Uses xESMF/ESMF bilinear interpolation. This is the preferred
          backend for production workloads and large datasets.

        - ``backend="scipy"`` (or ``"griddata"``)
          Uses SciPy triangulation-based interpolation. This backend requires
          no ESMF installation and is useful as a fallback for irregular
          curvilinear grids.

    Longitude handling
    ------------------
    Longitudes are internally shifted to a continuous branch near the target
    region before interpolation. This avoids artificial discontinuities around
    the dateline and allows regions such as::

        (-200, -120)
        (150, 220)
        (170, -170)

    to be regridded consistently without manually normalizing longitudes.

    Resolution handling
    -------------------
    Target resolution can be specified explicitly::

        resolution=0.25
        resolution=(0.25, 0.25)

    or inferred automatically from the source dataset::

        resolution="source"

    which uses ``ds.earthml.resolution()`` to estimate the native latitude and
    longitude spacing.

    Large datasets
    --------------
    Regridding can dramatically increase dataset size. For example, converting
    a global 1° dataset to 0.25° resolution increases the number of spatial
    cells by approximately 16×.

    To avoid exhausting memory, the accessor can automatically process large
    outputs in batches. Batching is performed over non-spatial dimensions
    (typically time, ensemble member, or forecast lead time), writing
    intermediate results to temporary storage before recombining them into a
    single output dataset.

    Curvilinear coordinate repair
    -----------------------------
    Some upstream datasets contain invalid curvilinear coordinates represented
    by missing values or sentinel values.

    Before interpolation, the accessor can:

    - identify invalid coordinate points
    - repair isolated gaps through interpolation
    - extend partially missing rows
    - recover fully missing rows from neighboring valid rows
    - crop the source grid to the smallest valid bounding box

    These steps improve interpolation robustness while preserving the original
    data values.

    Caching
    -------
    Curvilinear interpolation can require expensive geometric operations
    (triangulation, interpolation map construction, xESMF weight generation).

    The accessor maintains both in-memory and on-disk caches for:

    - interpolation geometries
    - triangulation mappings
    - xESMF weight files

    Repeated regridding of identical grids therefore becomes substantially
    faster.

    Notes
    -----
    - Returned coordinates are always rectilinear latitude/longitude axes.
    - Regridding modifies only variables that depend on the spatial
      coordinates.
    - Non-spatial variables are copied unchanged.
    - Variables excluded from regridding are preserved whenever possible.
    - xESMF uses a static source-support mask derived from the first spatial
      slice. This is an intentional performance optimization to avoid
      rebuilding interpolation weights for every timestep or ensemble member.
    """
    _obj: xr.Dataset

    def regrid_to_rectilinear(
        self,
        region: Region,
        resolution: ResolutionLike = "source",
        vars_to_regrid: Iterable[str] | None = None,
        *,
        backend: Backend | str = "xesmf",
        silent: bool = False,
        max_output_gb: float | None = 24.0,
        auto_batch: bool = True,
        batch_output_gb: float = 2.0,
        batch_dim: str | None = None,
        tmp_dir: str | Path = "/tmp",
    ) -> xr.Dataset:
        """
        Regrid selected variables onto a regular latitude/longitude grid.

        Parameters
        ----------
        region : Region
            Target geographic region.

            Latitude bounds may be provided in either order::

                Region(lat=(90, -90), ...)
                Region(lat=(-90, 90), ...)

            Longitudes are interpreted as a continuous interval and may extend
            beyond the conventional [-180, 180] range.

        resolution : float, (float, float), "source", or None, default="source"
            Target grid spacing in degrees.

            - float:
            Use the same spacing for latitude and longitude.
            - (lat_res, lon_res):
            Use different spacings for latitude and longitude.
            - "source" or None:
            Use the native resolution estimated from
            ``ds.earthml.resolution()``.

        vars_to_regrid : iterable of str, optional
            Variables to interpolate.

            If None, all data variables are considered.

            Variables not selected are copied unchanged whenever possible.

        backend : {"xesmf", "scipy", "griddata"}, default="xesmf"
            Regridding backend used for curvilinear source grids.

            Rectilinear source grids always use xarray interpolation.

            - "xesmf":
            Bilinear interpolation using xESMF/ESMF.
            - "scipy" or "griddata":
            Triangulation-based interpolation using SciPy.

        silent : bool, default=False
            Suppress informational log messages.

        Returns
        -------
        xr.Dataset
            Dataset on a rectilinear latitude/longitude grid covering the
            requested region and resolution.

            The returned dataset contains 1D latitude and longitude coordinates,
            regardless of the source grid topology.

        Notes
        -----
        Rectilinear source grids are interpolated using
        ``xarray.Dataset.interp()``.

        Curvilinear source grids are interpolated using either xESMF or
        SciPy depending on the selected backend.

        Longitude coordinates are internally shifted to a continuous branch
        near the target region to avoid dateline discontinuities.

        Large outputs may be processed in batches to reduce peak memory
        usage.
        """
        ds = self._obj
        backend_normalized = _normalize_backend(backend)

        lon_name = ds.earthml.guessed_coords.longitude
        lat_name = ds.earthml.guessed_coords.latitude

        lat_da = ds[lat_name]
        lon_da = ds[lon_name]

        lat_res, lon_res = _normalize_resolution(ds, resolution)
        vars_to_regrid_list = (
            list(ds.data_vars) if vars_to_regrid is None else list(vars_to_regrid)
        )

        def log_info(msg: str, *args: object) -> None:
            if not silent:
                logger.info(msg, *args)

        log_info(
            "Regrid: available data vars %s, requested vars %s",
            list(ds.data_vars),
            vars_to_regrid_list,
        )
        log_info("Regrid: target resolution lat=%s, lon=%s", lat_res, lon_res)

        lat0, lat1 = map(float, region.lat)
        lon0, lon1 = map(float, region.lon)

        lat_target = _build_axis(lat0, lat1, lat_res)

        if max_output_gb is not None:
            max_output_gb = float(max_output_gb)
            if not np.isfinite(max_output_gb) or max_output_gb <= 0:
                raise ValueError(
                    f"max_output_gb must be positive or None. Got {max_output_gb!r}."
                )

        lon0_u, lon1_u = float(lon0), float(lon1)
        while lon1_u < lon0_u:
            lon1_u += 360.0

        lon_target = _build_axis(lon0_u, lon1_u, lon_res)
        lon_center = 0.5 * (lon0_u + lon1_u)

        if lat_da.ndim == 1 and lon_da.ndim == 1:
            log_info(
                "Regrid: rectilinear (1D) source -> rectilinear (1D) target via xarray.interp."
            )

            ds_interp = _prepare_rectilinear_source(
                ds,
                lat_name=lat_name,
                lon_name=lon_name,
                lon_center=lon_center,
            )

            src_lat = np.asarray(ds_interp[lat_name].values, dtype=np.float64)
            src_lon = np.asarray(ds_interp[lon_name].values, dtype=np.float64)

            same_lat = src_lat.shape == lat_target.shape and np.allclose(
                src_lat,
                lat_target,
                atol=1e-6,
            )
            same_lon = src_lon.shape == lon_target.shape and np.allclose(
                src_lon,
                lon_target,
                atol=1e-6,
            )

            if same_lat and same_lon:
                log_info("Regrid: no-op; keeping existing rectilinear grid.")
                return ds_interp

            lat_tgt_da = xr.DataArray(
                lat_target,
                dims=(lat_name,),
                name=lat_name,
                attrs=lat_da.attrs,
            )
            lon_tgt_da = xr.DataArray(
                lon_target,
                dims=(lon_name,),
                name=lon_name,
                attrs=lon_da.attrs,
            )

            target_spatial_dims = {lat_name, lon_name}
            coord_out = {
                name: coord
                for name, coord in ds_interp.coords.items()
                if name not in target_spatial_dims
                and not (target_spatial_dims & set(coord.dims))
            }
            coord_out[lat_name] = lat_tgt_da
            coord_out[lon_name] = lon_tgt_da

            if max_output_gb is not None and (not np.isfinite(batch_output_gb) or batch_output_gb <= 0):
                raise ValueError(
                    f"batch_output_gb must be positive. Got {batch_output_gb!r}."
                )

            def output_shape_and_gb(
                da_in: xr.DataArray,
                interp_dims: set[str],
                *,
                itemsize: int | None = None,
            ) -> tuple[tuple[int, ...], float]:
                shape: list[int] = []
                for dim, size in da_in.sizes.items():
                    if dim == lat_name and dim in interp_dims:
                        shape.append(int(lat_target.size))
                    elif dim == lon_name and dim in interp_dims:
                        shape.append(int(lon_target.size))
                    else:
                        shape.append(int(size))

                # xarray/scipy interpolation commonly promotes to float64.
                bytes_per_value = itemsize or max(np.dtype(da_in.dtype).itemsize, 8)
                n_values = int(np.prod(shape, dtype=np.int64))
                return tuple(shape), n_values * bytes_per_value / 1024**3

            def interp_rectilinear_da(
                var_name: str,
                da_in: xr.DataArray,
                interp_coords: dict[str, xr.DataArray],
                interp_dims: set[str],
                *,
                method: Literal["linear", "nearest"],
            ) -> xr.DataArray:
                shape, estimated_gb = output_shape_and_gb(da_in, interp_dims)
                too_large = (
                    max_output_gb is not None
                    and estimated_gb > float(max_output_gb)
                    and bool(interp_dims)
                )

                if not too_large:
                    return da_in.interp(
                        interp_coords,
                        method=method,
                        assume_sorted=True,
                    )

                if not auto_batch:
                    raise MemoryError(
                        f"Refusing to regrid variable {var_name!r}: estimated output "
                        f"array is {estimated_gb:.1f} GiB, above max_output_gb="
                        f"{float(max_output_gb):.1f}. Target spatial shape is "
                        f"({lat_target.size}, {lon_target.size}); full target shape "
                        f"would be {shape}."
                    )

                candidate_dims = [
                    dim
                    for dim in da_in.dims
                    if dim not in (lat_name, lon_name) and da_in.sizes[dim] > 1
                ]
                if batch_dim is not None:
                    if batch_dim not in candidate_dims:
                        raise ValueError(
                            f"batch_dim={batch_dim!r} is not a splittable dimension "
                            f"of variable {var_name!r}. Candidate dims: {candidate_dims}."
                        )
                    split_dim = batch_dim
                else:
                    if not candidate_dims:
                        raise MemoryError(
                            f"Variable {var_name!r} is too large to interpolate at once "
                            f"({estimated_gb:.1f} GiB), but it has no non-spatial "
                            "dimension to split over."
                        )
                    split_dim = max(candidate_dims, key=lambda dim: da_in.sizes[dim])

                split_axis = da_in.get_axis_num(split_dim)
                split_size = int(da_in.sizes[split_dim])
                bytes_per_index = estimated_gb * 1024**3 / split_size
                budget_bytes = float(batch_output_gb) * 1024**3
                chunk_len = max(1, min(split_size, int(budget_bytes // bytes_per_index)))

                log_info(
                    "Regrid: variable %r is %.1f GiB; batching over %r in chunks of %s.",
                    var_name,
                    estimated_gb,
                    split_dim,
                    chunk_len,
                )

                mm: np.memmap | None = None
                out_coords = {
                    dim: (
                        lat_tgt_da
                        if dim == lat_name
                        else lon_tgt_da
                        if dim == lon_name
                        else da_in.coords[dim]
                    )
                    for dim in da_in.dims
                    if dim in da_in.coords or dim in (lat_name, lon_name)
                }

                for start in range(0, split_size, chunk_len):
                    stop = min(start + chunk_len, split_size)
                    part = da_in.isel({split_dim: slice(start, stop)})
                    out_part = part.interp(
                        interp_coords,
                        method=method,
                        assume_sorted=True,
                    ).transpose(*da_in.dims)
                    part_values = np.asarray(out_part.values)

                    if mm is None:
                        tmp_parent = Path(tmp_dir).expanduser()
                        tmp_parent.mkdir(parents=True, exist_ok=True)
                        tmp = tempfile.NamedTemporaryFile(
                            prefix=f"earthml-regrid-{var_name}-",
                            suffix=".dat",
                            dir=tmp_parent,
                            delete=False,
                        )
                        tmp_path = Path(tmp.name)
                        tmp.close()

                        mm = np.memmap(
                            tmp_path,
                            dtype=part_values.dtype,
                            mode="w+",
                            shape=shape,
                        )

                        # POSIX keeps the mapping alive after unlinking the path.
                        # This gives us temporary on-disk storage without leaving
                        # cleanup files behind. On platforms that disallow this,
                        # the file remains and the debug log tells us where it is.
                        try:
                            tmp_path.unlink()
                        except OSError:
                            logger.debug(
                                "Could not unlink temporary regrid memmap %s immediately.",
                                tmp_path,
                            )

                    selector = [slice(None)] * len(shape)
                    selector[split_axis] = slice(start, stop)
                    mm[tuple(selector)] = part_values
                    mm.flush()

                if mm is None:
                    raise RuntimeError(f"Internal error while batching {var_name!r}.")

                return xr.DataArray(
                    mm,
                    dims=da_in.dims,
                    coords=out_coords,
                    attrs=da_in.attrs,
                    name=var_name,
                )

            data_vars_out: dict[str, xr.DataArray] = {}
            for name, da in ds_interp.data_vars.items():
                interp_coords: dict[str, xr.DataArray] = {}
                interp_dims: set[str] = set()
                if lat_name in da.dims:
                    interp_coords[lat_name] = lat_tgt_da
                    interp_dims.add(lat_name)
                if lon_name in da.dims:
                    interp_coords[lon_name] = lon_tgt_da
                    interp_dims.add(lon_name)

                if (
                    name in vars_to_regrid_list
                    and lat_name in da.dims
                    and lon_name in da.dims
                ):
                    data_vars_out[name] = interp_rectilinear_da(
                        name,
                        da,
                        interp_coords,
                        interp_dims,
                        method="linear",
                    )
                elif interp_coords:
                    # Keep all spatial variables on the target grid.
                    data_vars_out[name] = interp_rectilinear_da(
                        name,
                        da,
                        interp_coords,
                        interp_dims,
                        method="nearest",
                    )
                else:
                    data_vars_out[name] = da

            return xr.Dataset(data_vars_out, coords=coord_out, attrs=ds_interp.attrs)

        if lat_da.ndim != 2 or lon_da.ndim != 2:
            raise ValueError(
                "Regridding supports lat/lon layouts (1D, 1D) or (2D, 2D). "
                f"Got lat.ndim={lat_da.ndim}, lon.ndim={lon_da.ndim}."
            )

        if lat_da.dims != lon_da.dims:
            raise ValueError(
                "Curvilinear lat/lon must share the same dimensions. "
                f"Got lat.dims={lat_da.dims}, lon.dims={lon_da.dims}."
            )

        y_dim_src, x_dim_src = lat_da.dims
        source_spatial_dims = {y_dim_src, x_dim_src}

        lat_src_2d = np.asarray(lat_da.values, dtype=np.float64)
        lon_src_raw_2d = np.asarray(lon_da.values, dtype=np.float64)
        coord_invalid_2d = _invalid_curvilinear_coord_mask(lat_src_2d, lon_src_raw_2d)
        lon_src_2d = _shift_longitudes_near_reference(lon_src_raw_2d, lon_center)

        def copy_without_source_spatial_coords(da: xr.DataArray) -> xr.DataArray:
            drop_coords = [
                coord_name
                for coord_name, coord in da.coords.items()
                if coord_name in (lat_name, lon_name)
                and bool(source_spatial_dims & set(coord.dims))
            ]
            if drop_coords:
                return da.reset_coords(drop_coords, drop=True)
            return da

        coord_out = {
            name: coord
            for name, coord in ds.coords.items()
            if name not in (lat_name, lon_name)
            and not (source_spatial_dims & set(coord.dims))
        }
        coord_out[lat_name] = xr.DataArray(
            lat_target,
            dims=(lat_name,),
            name=lat_name,
            attrs=lat_da.attrs,
        )
        coord_out[lon_name] = xr.DataArray(
            lon_target,
            dims=(lon_name,),
            name=lon_name,
            attrs=lon_da.attrs,
        )

        if backend_normalized == "xesmf":
            log_info(
                "Regrid: curvilinear (2D) source -> rectilinear (1D) target via xESMF."
            )

            xe = _import_xesmf()

            lon_out_2d, lat_out_2d = np.meshgrid(lon_target, lat_target)
            ds_out_grid = _build_xesmf_grid(lat_out_2d, lon_out_2d)

            data_vars_out: dict[str, xr.DataArray] = {}
            for name, da in ds.data_vars.items():
                if name not in vars_to_regrid_list:
                    data_vars_out[name] = copy_without_source_spatial_coords(da)
                    continue
                if y_dim_src not in da.dims or x_dim_src not in da.dims:
                    data_vars_out[name] = copy_without_source_spatial_coords(da)
                    continue

                da_spatial = da.transpose(
                    *[d for d in da.dims if d not in (y_dim_src, x_dim_src)],
                    y_dim_src,
                    x_dim_src,
                )

                ny_src, nx_src = da_spatial.shape[-2:]
                if (ny_src, nx_src) != lat_src_2d.shape:
                    raise ValueError(
                        f"Variable {name!r} spatial shape {(ny_src, nx_src)} does not "
                        f"match lat/lon shape {lat_src_2d.shape}."
                    )

                # Intentional performance choice: one static source-support mask.
                leading_dims = da_spatial.dims[:-2]
                sample = (
                    da_spatial.isel({dim: 0 for dim in leading_dims})
                    if leading_dims
                    else da_spatial
                )
                data_mask_2d = np.isfinite(sample.values)

                lat_src_use, lon_src_use, mask_2d, da_spatial = _crop_curvilinear_valid_bbox(
                    lat_2d=lat_src_2d,
                    lon_2d=lon_src_2d,
                    coord_invalid_2d=coord_invalid_2d,
                    data_mask_2d=data_mask_2d,
                    da_spatial=da_spatial,
                    y_dim_src=y_dim_src,
                    x_dim_src=x_dim_src,
                )

                ds_in_grid = _build_xesmf_grid(lat_src_use, lon_src_use, mask_2d=mask_2d)

                weights_key = hashlib.blake2b(digest_size=16)
                weights_key.update(b"earthml-xesmf-bilinear-v2")
                for arr in (lat_src_use, lon_src_use, lat_target, lon_target):
                    weights_key.update(_hash_array(np.asarray(arr)).encode("ascii"))
                weights_key.update(_hash_array(mask_2d.astype(np.int8)).encode("ascii"))

                weights_path = (
                    _cache_dir("EARTHML_REGRID_XESMF_CACHE_DIR", _XESMF_CACHE_SUBDIR)
                    / f"{weights_key.hexdigest()}.nc"
                )
                weights_path.parent.mkdir(parents=True, exist_ok=True)

                with _get_xesmf_weights_lock(weights_path):
                    if weights_path.exists():
                        logger.debug("Regrid: using weights in %s", weights_path)
                    else:
                        logger.debug("Regrid: building weights in %s", weights_path)
                        _build_xesmf_weights_on_disk(
                            ds_in_grid=ds_in_grid,
                            ds_out_grid=ds_out_grid,
                            weights_path=weights_path,
                            method="bilinear",
                        )

                    try:
                        with xr.open_dataset(weights_path, engine="h5netcdf") as ds_weights:
                            ds_weights = ds_weights.load()
                    except OSError:
                        with xr.open_dataset(weights_path, engine="scipy") as ds_weights:
                            ds_weights = ds_weights.load()

                regridder = xe.Regridder(
                    ds_in_grid,
                    ds_out_grid,
                    "bilinear",
                    unmapped_to_nan=True,
                    reuse_weights=True,
                    weights=ds_weights,
                )

                rename_for_xesmf = {}
                if y_dim_src != "y":
                    rename_for_xesmf[y_dim_src] = "y"
                if x_dim_src != "x":
                    rename_for_xesmf[x_dim_src] = "x"

                da_for_xesmf = (
                    da_spatial.rename(rename_for_xesmf)
                    if rename_for_xesmf
                    else da_spatial
                )

                source_coord_names = [
                    coord_name
                    for coord_name in (lat_name, lon_name)
                    if coord_name in da_for_xesmf.coords
                ]
                if source_coord_names:
                    da_for_xesmf = da_for_xesmf.reset_coords(
                        source_coord_names,
                        drop=True,
                    )

                out = regridder(da_for_xesmf, keep_attrs=True)
                out_data = out.data if isinstance(out, xr.DataArray) else out

                leading_dims = da_spatial.dims[:-2]
                out_dims = leading_dims + (lat_name, lon_name)
                out_coords = {
                    **{
                        dim: da_spatial.coords[dim]
                        for dim in leading_dims
                        if dim in da_spatial.coords
                    },
                    lat_name: coord_out[lat_name],
                    lon_name: coord_out[lon_name],
                }

                data_vars_out[name] = xr.DataArray(
                    out_data,
                    dims=out_dims,
                    coords=out_coords,
                    attrs=da.attrs,
                    name=name,
                )

            return xr.Dataset(data_vars_out, coords=coord_out, attrs=ds.attrs)

        log_info(
            "Regrid: curvilinear (2D) source -> rectilinear (1D) target via scipy.griddata."
        )

        ny_target = lat_target.size
        nx_target = lon_target.size

        geometry_cache_key, geometry = _load_or_build_curvilinear_geometry(
            lat_src_2d=lat_src_2d,
            lon_src_2d=lon_src_2d,
            coord_invalid_2d=coord_invalid_2d,
            lat_target=lat_target,
            lon_target=lon_target,
            silent=silent,
        )
        coord_valid = geometry["coord_valid"]
        points = geometry["points"]
        xi = geometry["xi"]

        data_vars_out = {}
        for name, da in ds.data_vars.items():
            if name not in vars_to_regrid_list:
                data_vars_out[name] = copy_without_source_spatial_coords(da)
                continue
            if y_dim_src not in da.dims or x_dim_src not in da.dims:
                data_vars_out[name] = copy_without_source_spatial_coords(da)
                continue

            log_info("Regridding variable %r with scipy.griddata.", name)

            da_spatial = da.transpose(
                *[d for d in da.dims if d not in (y_dim_src, x_dim_src)],
                y_dim_src,
                x_dim_src,
            )
            data_np = np.asarray(da_spatial.values)

            leading_dims = da_spatial.dims[:-2]
            leading_shape = data_np.shape[:-2]
            ny_src, nx_src = data_np.shape[-2:]

            if (ny_src, nx_src) != lat_src_2d.shape:
                raise ValueError(
                    f"Variable {name!r} spatial shape {(ny_src, nx_src)} does not "
                    f"match lat/lon shape {lat_src_2d.shape}."
                )

            arr = data_np.reshape(-1, ny_src * nx_src)
            out_slices: list[np.ndarray] = []

            for zi in arr:
                zi_coord_valid = zi[coord_valid]
                data_valid = np.isfinite(zi_coord_valid)
                valid_count = int(data_valid.sum())

                if valid_count < 3:
                    zi_interp = np.full(xi.shape[0], np.nan, dtype=np.float64)
                else:
                    interp_map = _load_or_build_curvilinear_interp_map(
                        geometry_cache_key=geometry_cache_key,
                        points=points,
                        xi=xi,
                        data_valid_mask=data_valid,
                        silent=silent,
                    )

                    if bool(interp_map.get("usable", False)):
                        zi_interp = np.full(xi.shape[0], np.nan, dtype=np.float64)
                        inside_idx = np.asarray(interp_map["inside_idx"], dtype=np.int64)
                        vertices = np.asarray(interp_map["vertices"], dtype=np.int64)
                        weights = np.asarray(interp_map["weights"], dtype=np.float64)
                        value_indices = np.asarray(
                            interp_map["value_indices"],
                            dtype=np.int64,
                        )

                        if inside_idx.size:
                            zi_values = zi_coord_valid[value_indices]
                            zi_interp[inside_idx] = np.einsum(
                                "ij,ij->i",
                                zi_values[vertices],
                                weights,
                            )
                    else:
                        try:
                            zi_interp = griddata(
                                points[data_valid],
                                zi_coord_valid[data_valid],
                                xi,
                                method="linear",
                            )
                        except (QhullError, ValueError):
                            zi_interp = np.full(xi.shape[0], np.nan, dtype=np.float64)

                out_slices.append(zi_interp.reshape(ny_target, nx_target))

            out = np.stack(out_slices, axis=0).reshape(
                *leading_shape,
                ny_target,
                nx_target,
            )

            out_dims = leading_dims + (lat_name, lon_name)
            out_coords = {
                **{
                    dim: da_spatial.coords[dim]
                    for dim in leading_dims
                    if dim in da_spatial.coords
                },
                lat_name: coord_out[lat_name],
                lon_name: coord_out[lon_name],
            }

            data_vars_out[name] = xr.DataArray(
                out,
                dims=out_dims,
                coords=out_coords,
                attrs=da.attrs,
                name=name,
            )

        return xr.Dataset(data_vars_out, coords=coord_out, attrs=ds.attrs)
