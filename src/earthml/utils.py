from pathlib import Path
from typing import Sequence, Optional, Literal, Callable, List
from rich import print
from rich.pretty import pprint
from rich.table import Table as RichTable
from rich.highlighter import ReprHighlighter
from datetime import datetime, timedelta
from datetime import time as datetime_time
from dateutil.relativedelta import relativedelta
import cf_xarray
import xarray as xr
import pandas as pd
from earthkit.data.sources.empty import EmptySource
import os, psutil, multiprocessing, tempfile, logging, re, time
import dask
from dask.distributed import Client, LocalCluster
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import lightning as L
# import xesmf as xe
from scipy.interpolate import griddata
# Local imports
from .dataclasses import DataSelection, TimeRange

#---------------
# Helper classes
#---------------

class Dask:
    def __init__(self, base_port=8787, n_workers=None, processes=True, nanny=True):
        self.base_port = base_port
        self.n_workers = n_workers
        self.processes = processes
        self.nanny = nanny

        self.cluster = None
        self.client = None

    def start(self):
        logging.getLogger("tornado.application").setLevel(logging.ERROR)
        logging.getLogger("bokeh").setLevel(logging.ERROR)

        candidates = [
            os.environ.get("TMPDIR"),
            tempfile.gettempdir(),
            os.path.expanduser("~/.dask-tmp"),
            f"/scratch/{os.environ.get('USER')}",
            "/scratch",
        ]
        local_dir = next((p for p in candidates if p and os.path.exists(p)), tempfile.gettempdir())
        os.makedirs(local_dir, exist_ok=True)

        n_cores = multiprocessing.cpu_count()
        total_mem_gb = psutil.virtual_memory().total / 1e9
        n_workers = self.n_workers
        if n_workers is None:
            n_workers = min(n_cores, max(1, int(total_mem_gb // 4)))

        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=self.processes,
            timeout="600s",
            heartbeat_interval="10s",
            memory_limit="auto",
            local_directory=local_dir,
            dashboard_address=f":{self.base_port}",
            nanny=self.nanny,
        )
        self.client = Client(self.cluster)

        # Register cf_xarray
        self.client.run(lambda: __import__("cf_xarray"))

        import socket
        print(f"Dask dashboard running on {socket.gethostname()}:{self.cluster.scheduler.services['dashboard'].port}")
        print(f"Cores: {n_cores}, Mem: {total_mem_gb} GB -> Dask workers: {n_workers}")
        print(f"Write Dask local files in {local_dir}")

        return self

    def close(self):
        """
        Close in correct order: client first, then cluster.
        Cancel outstanding futures to avoid worker noise.
        """
        if self.client is not None:
            try:
                # prevents workers from continuing to run tasks during shutdown
                self.client.cancel(list(self.client.futures.values()), force=True)
            except Exception:
                pass
            try:
                self.client.close(timeout=10)
            except Exception:
                pass
            self.client = None

        if self.cluster is not None:
            try:
                self.cluster.close(timeout=10)
            except Exception:
                pass
            self.cluster = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don't suppress exceptions


class Table ():
    """Helper class to create rich Tables from multinested dicts"""
    def __init__ (self, data: dict, title: str = None, params_name:str = None, twocols: bool = False) -> RichTable:
        assert isinstance(data, dict)
        if len(data.keys()) == 1:
            assert isinstance(next(iter(data.values())), dict) # there must be data
            data_name = next(iter(data.keys()))
            data = data[data_name] # promote first inner dict to actual data
            title = data_name if title is None else title
        has_inner_dicts = self._has_inner_dicts(data)
        rich_params = {
            "title": title,
            "show_header": bool(has_inner_dicts and title and not twocols),
        }
        self.table = RichTable(**rich_params)
        rowheads = self._get_rowheads(data) # check only first inner level
        highligher = ReprHighlighter()
        params_name = 'params' if params_name is None else params_name
        if has_inner_dicts and rowheads and not twocols:
            self.table.add_column("params", style='magenta')
            for k in data.keys():
                self.table.add_column(str(k), style='cyan')
            row = {}
            for i,r in enumerate(rowheads):
                row[r] = []
                for v in data.values():
                    if isinstance(v, dict):
                        row[r].append(highligher(str(list(v.values())[i]))) if r in v.keys() else ""
            for r in rowheads:
                self.table.add_row(str(r), *row[r])
        else:
            self.table.add_column(title, style='magenta')
            self.table.add_column("", style='cyan')
            for k, v in data.items():
                self.table.add_row(k, highligher(str(v)))

    def _has_inner_dicts (self, d: dict) -> bool:
        for v in d.values():
            if isinstance(v, dict):
                return True or self.has_inner_dicts(v)
            elif isinstance(v, (list, tuple)):
                if any(isinstance(i, dict) and self.has_inner_dicts(i) for i in v):
                    return True
        return False

    def _get_rowheads (self, d: dict, recursive: bool = False) -> list:
        rowheads = []
        for v in d.values():
            if isinstance(v, dict):
                rowheads.extend(map(str, v.keys()))
                if recursive:
                    rowheads.extend(self._get_rowheads(v, recursive))
        return list(dict.fromkeys(rowheads))

#------------------------
# Module-level functions
#------------------------

def _extract_nc_path_from_oserror (e: Exception) -> Path | None:
    # netCDF4 often formats it like: "...: '/path/file.nc'"
    m = re.search(r"['\"](/[^'\"]+\.nc)['\"]", str(e))
    return Path(m.group(1)) if m else None


def retry_fetch_after_hdf_err_eks_source(
    fetch_fn: Callable,
    *,
    tries: int = 5,
    base_sleep: float = 1.5,
):
    for attempt in range(1, tries + 1):
        src = fetch_fn()
        return src

    raise ValueError("Couldn't fetch requested Earthkit source")

def _set_ekd_cache_dir (cache_dir: str = Path("/tmp/earthkit-cache/")):
    import earthkit.data as ekd
    ekd.config.set("cache-policy", "user")
    ekd.config.set("user-cache-directory", cache_dir)

def _get_ekd_cache_dir ():
    import earthkit.data as ekd
    return ekd.config.get("user-cache-directory")

def rmdir (directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()

def retry_fetch_after_hdf_err (
    fetch_fn: Callable[[], xr.Dataset | EmptySource],
    *,
    error_re: str | None = None,
    tries: int = 5,
    base_sleep: float = 1.5,
    delete_bad_file: bool = False,
    delete_bad_parent: bool = False,
):
    """fetch_fn(): should return an xr.Dataset"""
    pat = re.compile(error_re, re.I) if error_re else None
    last_e: Exception | None = None
    orig_ekd_cache_dir = _get_ekd_cache_dir()

    for attempt in range(1, tries + 1):
        try:
            data = fetch_fn()
            if isinstance(data, EmptySource):
                print(f"   EmptySource returned, setting tmp cache dir ({attempt}/{tries})")
                _set_ekd_cache_dir()
                # time.sleep(base_sleep * (2 ** (attempt - 1)))
                continue
            else:
                _set_ekd_cache_dir(orig_ekd_cache_dir)
                return data
        # except (OSError, RuntimeError, KeyError) as e:
        except Exception as e:
            print("   Attempt", attempt)
            last_e = e
            p = _extract_nc_path_from_oserror(e) # if e in (OSError, KeyError) else None
            if p:
                msg = str(e)
                if not pat.search(msg):
                    raise
                print(f"   HDF error opening {p}, wait {base_sleep}s (attempt {attempt}/{tries})")

                if delete_bad_file and p.exists():
                    try:
                        p.unlink()
                        print(f"   → deleted corrupt cache file: {p}")
                    except Exception as del_e:
                        print(f"   → failed to delete {p}: {del_e}")
                cache_subdir = p.parent

                if delete_bad_parent and cache_subdir.exists():
                    try:
                        rmdir(cache_subdir)
                        print(f"   → deleted corrupt cache parent subdir: {cache_subdir}")
                    except Exception as del_e:
                        print(f"   → failed to delete {cache_subdir}: {del_e}")
            else:
                print(e)
                # print(f"HDF error (attempt {attempt}/{tries}) but could not locate file path")

        time.sleep(base_sleep * (2 ** (attempt - 1)))

    raise RuntimeError

def generate_date_range (period: TimeRange):
    freq = period.freq
    start = period.start
    end = period.end
    shifted = period.shifted
    # Try to interpret freq as a Timedelta (works for H, D, etc., but not M/Y)
    try:
        freq_td = pd.to_timedelta(freq)
    except (TypeError, ValueError):
        freq_td = None
    # Only shift `end` forward for sub-daily frequencies
    if freq_td is not None and freq_td < pd.to_timedelta('24h'):
        end = end + freq_td
    dr = xr.date_range(
        start=start,
        end=end,
        freq=freq, # original string
        inclusive='both'
    )
    if shifted:
        dr = [d + relativedelta(**shifted) for d in dr]
    return dr

def _normalize_bounds (bounds):
    if bounds is None:
        return slice(None)
    if isinstance(bounds, (int, float, np.timedelta64)):
        return bounds
    return slice(*bounds)

def _guess_dim_or_coord_or_datavar_name (
    ds: xr.Dataset,
    cf_name: str,
    name_type: Literal["coord", "dim", "data_var"],
    fallback_names: list[str] | None = None,
) -> str | None:
    # Try cf_xarray key directly
    try:
        if name_type == 'dim':
            return ds.cf.indexes[cf_name].name
        elif name_type == 'coord':
            return ds.cf.coords[cf_name].name
        else:
            return ds.cf.data_vars[cf_name].name
    except KeyError:
        # print("key error")
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

def _guess_dim_name (
    ds: xr.Dataset,
    cf_name: str,
    fallback_names: list[str] | None = None,
):
    return _guess_dim_or_coord_or_datavar_name(ds, cf_name, 'dim', fallback_names)

def _guess_coord_name (
    ds: xr.Dataset,
    cf_name: str,
    fallback_names: list[str] | None = None,
):
    return _guess_dim_or_coord_or_datavar_name(ds, cf_name, 'coord', fallback_names)

def _guess_data_var_name (
    ds: xr.Dataset,
    cf_name: str,
    fallback_names: list[str] | None = None,
):
    return _guess_dim_or_coord_or_datavar_name(ds, cf_name, 'data_var', fallback_names)

def _dim_selection (
    ds: xr.Dataset,
    cf_name: str,
    fallback_names: list[str] | None = None,
    values: int | float | tuple | None = None,
) -> dict:
    sel_d: dict = {}
    dim_name = _guess_dim_name(ds, cf_name, fallback_names)

    if dim_name is not None and values is not None:
        sel_d[dim_name] = _normalize_bounds(values)

    return sel_d

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

def _get_grid_extremes (ds: xr.Dataset, lon_coord, lat_coord) -> tuple:
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

def subset_ds (
    data_selection: DataSelection,
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Apply CF-based:
    - lon/lat convention normalization
    - region selection
    - vertical level selection (if levhpa / levm provided)
    - time selection (if leadtime provided in data_selection; currently commented)

    This runs once on the full combined dataset, not per file.
    """
    # Leadtime selection
    leadtime = data_selection.variable.leadtime
    leadtime_sel_d = {}
    if leadtime is not None: # TODO refactor, it's used also in _preprocess
        # Build target timedelta from value + unit (e.g. "3 days", "12 hours")
        td = pd.to_timedelta(f"{leadtime.value} {leadtime.unit}")
        # Cast to same dtype as coord (usually timedelta64[ns])
        coord_dtype = ds[leadtime.name].dtype
        target = td.to_numpy().astype(coord_dtype)
        leadtime_sel_d = _dim_selection(
            ds, leadtime.name, ["lead_time", "leadtime"], target
        )
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
    level_sel_d = _dim_selection(ds, "vertical", ["level", "z"], level_value)

    lon_req = np.array(data_selection.region.lon, dtype=float)
    lat_req = np.array(data_selection.region.lat, dtype=float)

    lon_min_req, lon_max_req = float(lon_req.min()), float(lon_req.max())
    lat_min_req, lat_max_req = float(lat_req.min()), float(lat_req.max())

    # Lon/lat coordinate discovery
    lon_coord = _guess_coord_name(ds, "longitude", ["lon", "nav_lon"])
    lat_coord = _guess_coord_name(ds, "latitude", ["lat", "nav_lat"])
    if lon_coord is None or lat_coord is None:
        raise ValueError("Could not determine longitude/latitude coordinate names")
    lat_da, lon_da = ds[lat_coord], ds[lon_coord]
    # Get coordinate extremes
    (lon_grid_min, lon_grid_max), (lat_grid_min, lat_grid_max) = _get_grid_extremes(ds, lon_coord, lat_coord)
    # Normalize request to ds grid
    lon_min_req, lon_max_req = _normalize_lon_bounds(lon_min_req, lon_max_req, lon_grid_min)
    # Rectilinear vs curvilinear detection
    is_rectilinear = (
        lon_da.ndim == 1
        and lat_da.ndim == 1
        and lon_coord in ds.indexes
        and lat_coord in ds.indexes
    )
    grid_type = "rectilinear" if is_rectilinear else "curvilinear"
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
    # Roll if request crosses longitude cut-line (like dateline or Greenwich)
    ds = _roll_ds(ds, (lon_min_req, lon_max_req))
    # if 'sosaline' in ds.data_vars:
    #     quickplot(ds, 'sosaline', "/data/cmcc/jd19424/ML/experiments_earthML/", 'subset_ds_after_roll.png')
    (lon_grid_min, lon_grid_max), (lat_grid_min, lat_grid_max) = _get_grid_extremes(ds, lon_coord, lat_coord)
    lon_dim = _guess_dim_name(ds, 'longitude', ['lon', 'x'])
    _, cutting_lon = _get_cutting_lon(lon_da, lon_dim, (lon_min_req, lon_max_req))
    # print(f"subset_ds, cutting lon: {cutting_lon}")
    lon_min_req, lon_max_req = _wrap_longitudes((lon_min_req, lon_max_req), cutting_lon)
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
        # Respect latitude orientation (south->north or north->south)
        lat_vals_1d = lat_da.values
        if lat_vals_1d[0] < lat_vals_1d[-1]:
            # south -> north
            lat_slice = (lat_min_req, lat_max_req)
        else:
            # north -> south
            lat_slice = (lat_max_req, lat_min_req)

        lon_slice = (lon_min_req, lon_max_req)

        # print(
        #     f"lat requested after reorientation {lon_slice}, "
        #     f"lat requested after reorientation {lat_slice})"
        # )

        selection_d = {
            lon_coord: _normalize_bounds(lon_slice),
            lat_coord: _normalize_bounds(lat_slice),
        } | level_sel_d | leadtime_sel_d

        # Handle longitude wrap-around (e.g., [350, 10])
        if lon_slice[0] > lon_slice[1]:
            print("Longitude wrap-around on rectilinear grid...")
            ds1 = ds.sel({lon_coord: slice(lon_slice[0], lon_grid_max)})
            ds2 = ds.sel({lon_coord: slice(lon_grid_min, lon_slice[1])})
            ds = xr.concat([ds1, ds2], dim=lon_coord)
            selection_d.pop(lon_coord, None)

        # print(f"Selection: {selection_d}")
        if selection_d:
            ds = ds.sel(**selection_d)

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

    return ds

def get_lonlat_coords (ds: xr.Dataset):
    lon_coord = _guess_coord_name(ds, "longitude", ["lon", "nav_lon"])
    lat_coord = _guess_coord_name(ds, "latitude", ["lat", "nav_lat"])
    if lon_coord is None: lon_coord = _guess_data_var_name(ds, "longitude", ["lon", "nav_lon"])
    if lat_coord is None: lat_coord = _guess_data_var_name(ds, "latitude", ["lat", "nav_lat"])
    return lon_coord, lat_coord

def generate_hours (freq_str, output_type='string'):
    value = int(freq_str[:-1])
    if freq_str[-1] != 'h':
        raise ValueError("Only 'h' (hours) frequency supported")
    times = []
    current = datetime.strptime("00:00", "%H:%M")
    while current.hour < 24:
        if output_type == 'int':
            times.append(int(current.strftime("%H")))
        else:
            times.append(current.strftime("%H:%M"))
        current += timedelta(hours=value)
        if current.hour == 0:  # wrapped past midnight
            break
    return times

#--------
# Regrid
#--------

# TODO refactor regrid in one single optimized helper, possibly adding support for xesmf or other fast lib

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

def _is_regular_curvilinear(lat2d, lon2d, rel_tol=1e-2, min_points=10):
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

    lat_mean, lat_std, lat_cv = _spacing_stats_line(lat_line, is_lon=False)
    lon_mean, lon_std, lon_cv = _spacing_stats_line(lon_line, is_lon=True)

    n_lat = np.isfinite(lat_line).sum()
    n_lon = np.isfinite(lon_line).sum()

    if (
        np.isnan(lat_mean) or np.isnan(lon_mean)
        or n_lat < min_points or n_lon < min_points
    ):
        return False, (lat_mean, lon_mean)

    is_regular = (lat_cv < rel_tol) and (lon_cv < rel_tol)
    return is_regular, (abs(lat_mean), abs(lon_mean))

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

def get_ds_resolution(ds: xr.Dataset):
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
    lon_coord, lat_coord = get_lonlat_coords(ds)
    # print(f"Coords name lon: {lon_coord}, lat: {lat_coord}")

    lat_da = ds[lat_coord]
    lon_da = ds[lon_coord]

    # ---------- main logic ----------

    # Case 1: both 1D -> simple rectilinear
    if lat_da.ndim == 1 and lon_da.ndim == 1:
        lat_res = _coord_resolution_1d(lat_da.values, lat_coord)
        lon_res = _coord_resolution_1d(lon_da.values, lon_coord)
        return lat_res, lon_res

    # Case 2: both 2D -> combined detection logic
    if lat_da.ndim == 2 and lon_da.ndim == 2:
        lat2d = lat_da.values
        lon2d = lon_da.values

        # 2a. Rectilinear in disguise?
        if _is_rectilinear_disguised(lat2d, lon2d):
            print("Detected rectilinear grid stored as 2D.")
            lat_line = lat2d[:, 0]
            lon_line = lon2d[0, :]
            lat_res = _coord_resolution_1d(lat_line, lat_coord)
            lon_res = _coord_resolution_1d(lon_line, lon_coord)
            return lat_res, lon_res

        # 2b. Regular curvilinear?
        is_reg, (lat_reg, lon_reg) = _is_regular_curvilinear(lat2d, lon2d, rel_tol=0.5)
        if is_reg:
            print("Detected regular curvilinear grid.")
            return lat_reg, lon_reg

        # 2c. Fallback heuristic
        print("Grid appears non-regular; using fallback curvilinear heuristic.")
        lat_res, lon_res = _fallback_curvilinear_res(lat2d, lon2d)
        return lat_res, lon_res

    # Case 3: mixed dims (rare) – treat each independently
    print("Mixed lat/lon dimensions (one 1D, one 2D); treating separately.")

    # Latitude
    if lat_da.ndim == 1:
        lat_res = _coord_resolution_1d(lat_da.values, lat_coord)
    elif lat_da.ndim == 2:
        # use a simple central-column estimate
        lat_mean, _, _ = _spacing_stats_line(lat_da.values[:, lat_da.values.shape[1] // 2], is_lon=False)
        if np.isnan(lat_mean):
            raise ValueError("Cannot infer latitude resolution from 2D latitude.")
        lat_res = abs(lat_mean)
    else:
        raise ValueError("Latitude coordinate must be 1D or 2D")

    # Longitude
    if lon_da.ndim == 1:
        lon_res = _coord_resolution_1d(lon_da.values, lon_coord)
    elif lon_da.ndim == 2:
        lon_mean, _, _ = _spacing_stats_line(
            lon_da.values[lon_da.values.shape[0] // 2, :],
            is_lon=True,
        )
        if np.isnan(lon_mean):
            raise ValueError("Cannot infer longitude resolution from 2D longitude.")
        lon_res = abs(lon_mean)
    else:
        raise ValueError("Longitude coordinate must be 1D or 2D")

    return lat_res, lon_res

def _build_target_rect_grid (region, resolution):
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

def _wrap_longitudes(lon, center):
    """
    Wrap longitudes into a continuous interval centered at `center`,
    i.e. (center - 180, center + 180].
    """
    return ((lon - center + 180.0) % 360.0) - 180.0 + center

def _get_cutting_lon (lon_da: xr.DataArray, lon_dim: str, req_lon: tuple) -> tuple:
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

def _roll_ds (ds: xr.Dataset, req_lon: tuple) -> xr.Dataset:
    if req_lon[0] < req_lon[1]: # nothing to do
        return ds

    lon_coord, lat_coord = get_lonlat_coords(ds)
    lon_dim = _guess_dim_name(ds, 'longitude', ['lon', 'x'])
    lon_da, lat_da = ds[lon_coord], ds[lat_coord]

    cutting_lon_idx, cutting_lon = _get_cutting_lon(lon_da, lon_dim, req_lon)
    # print(f"_roll_ds: req_lon {req_lon}, cutting_lon_idx: {cutting_lon_idx}, cutting_lon: {cutting_lon}")
    # print(f"_roll_ds: lon_da.sizes {lon_da.sizes}, lat_da.sizes: {lat_da.sizes}")

    ds = ds.assign_coords(**{lon_coord: _wrap_longitudes(ds[lon_coord], cutting_lon)})
    return ds.roll(**{lon_dim: cutting_lon_idx}, roll_coords=True)

def regrid_to_rectilinear (
    src_ds: xr.Dataset,
    region,
    resolution,
    vars_to_regrid=None,
) -> xr.Dataset:
    """
    Regrid src_ds from its native grid (rectilinear or curvilinear) to a new
    rectilinear grid defined only by (region, resolution).

    region: has .lat and .lon with [north, south], [west, east]
    resolution: float or (lat_res, lon_res)
    vars_to_regrid: list of variable names, or None to auto-detect.
    """

    # src_ds = src_ds.load()  # make sure everything is in memory

    lon_name = _guess_coord_name(src_ds, "longitude", ["lon", "nav_lon"])
    lat_name = _guess_coord_name(src_ds, "latitude", ["lat", "nav_lat"])

    lat_da = src_ds[lat_name]
    lon_da = src_ds[lon_name]

    # --- Normalize region lon bounds to source grid convention -----------
    lon_grid_min = float(lon_da.min().values)
    lon_grid_max = float(lon_da.max().values)

    lat0, lat1 = region.lat  # [north, south]
    lon0, lon1 = region.lon  # [west, east]

    lon0_norm, lon1_norm = _normalize_lon_bounds(lon0, lon1, lon_grid_min)

    # if "sosaline" in src_ds.data_vars:
    #     quickplot(src_ds, "sosaline", "/data/cmcc/jd19424/ML/experiments_earthML/", "before_regrid.png")
    # Roll if request crosses longitude cut-line (like dateline or Greenwich)
    # src_ds = _roll_ds(src_ds, (lon0_norm, lon1_norm))
    # if "sosaline" in src_ds.data_vars:
    #     quickplot(src_ds, "sosaline", "/data/cmcc/jd19424/ML/experiments_earthML/", "rolled_sosaline.png")

    # --- Resolution handling ---------------------------------------------
    if isinstance(resolution, (tuple, list)):
        lat_res, lon_res = float(resolution[0]), float(resolution[1])
    else:
        lat_res = lon_res = float(resolution)

    # --- Build target 1D lat/lon in *source* convention ------------------
    eps = 1e-6

    # Latitude (ECMWF-style area: [north, south])
    if lat0 > lat1:
        # descending: north -> south
        lat_target = np.arange(lat0, lat1 - eps, -lat_res)
    else:
        # ascending
        lat_target = np.arange(lat0, lat1 + eps, lat_res)

    # Longitude
    if lon0_norm <= lon1_norm:
        lon_target = np.arange(lon0_norm, lon1_norm + eps, lon_res)
    else:
        lon_target = np.arange(lon0_norm, lon1_norm - eps, -lon_res)

    Ny, Nx = lat_target.size, lon_target.size

    # --- Which vars to regrid? ------------------------------------------
    if vars_to_regrid is None:
        vars_to_regrid = [
            name for name, da in src_ds.data_vars.items()
            if (lat_name in da.coords or lon_name in da.coords or
                lat_name in da.dims  or lon_name in da.dims)
        ]

    rectilinear_src = (lat_da.ndim == 1 and lon_da.ndim == 1)

    # ===== Case 1: rectilinear source (1D lat/lon) =====
    if rectilinear_src:
        print("Regrid: rectilinear (1D) source → rectilinear (1D) target via xarray.interp.")

        lat_tgt_da = xr.DataArray(lat_target, dims=(lat_name,), name=lat_name)
        lon_tgt_da = xr.DataArray(lon_target, dims=(lon_name,), name=lon_name)

        regridded = src_ds.interp(
            {lat_name: lat_tgt_da, lon_name: lon_tgt_da},
            method="linear",
        )

        # Bilinear inpainting on the rectilinear grid using index positions
        regridded = (
            regridded
            .interpolate_na(dim=lat_name, method="linear", use_coordinate=False)
            .interpolate_na(dim=lon_name, method="linear", use_coordinate=False)
        )

        # Force coords exactly to our target (avoid tiny FP diffs)
        regridded = regridded.assign_coords(
            {lat_name: lat_tgt_da, lon_name: lon_tgt_da}
        )
        return regridded

    # ===== Case 2: curvilinear (2D) source (lat/lon 2D) → rectilinear (1D) target =====
    print("Regrid: curvilinear (2D) source → rectilinear (1D) target via scipy.griddata.")

    if lat_da.ndim != 2 or lon_da.ndim != 2:
        raise ValueError(
            "Curvilinear regrid assumes lat/lon either (1D,1D) or (2D,2D). "
            f"Got lat.ndim={lat_da.ndim}, lon.ndim={lon_da.ndim}."
        )

    y_dim_src, x_dim_src = lat_da.dims  # e.g. ("y", "x")

    lat_src_2d = lat_da.values  # [Ny_src, Nx_src]
    lon_src_2d = lon_da.values

    # Ny_src, Nx_src = lat_src_2d.shape

    # Flatten source coords
    lat_flat = lat_src_2d.ravel()
    lon_flat = lon_src_2d.ravel()

    points = np.column_stack([lon_flat, lat_flat])

    # Target grid as flat xi
    lon_out_2d, lat_out_2d = np.meshgrid(lon_target, lat_target)  # [Ny, Nx]
    xi = np.column_stack([lon_out_2d.ravel(), lat_out_2d.ravel()])  # [Ntgt, 2]

    data_vars_out = {}

    for name, da in src_ds.data_vars.items():
        if name not in vars_to_regrid:
            data_vars_out[name] = da
            continue

        if y_dim_src not in da.dims or x_dim_src not in da.dims:
            data_vars_out[name] = da
            continue

        print(f"  Regridding variable '{name}' with griddata...")

        # Move (y,x) to the end
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
            zi = arr[i, :]          # [Nsrc]

            zi_interp = griddata(points, zi, xi, method="linear")  # [Ntgt]

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

        da_out = xr.DataArray(
            out,
            dims=out_dims,
            coords=out_coords,
            attrs=da.attrs,
        )

        # Bilinear inpainting on the rectilinear grid using index positions
        da_out = (
            da_out
            .interpolate_na(dim=lat_name, method="linear", use_coordinate=False)
            .interpolate_na(dim=lon_name, method="linear", use_coordinate=False)
        )

        data_vars_out[name] = da_out

    # Dataset coords: keep non-lat/lon coords from src, override lat/lon
    coord_out = {
        k: v for k, v in src_ds.coords.items()
        if k not in (lat_name, lon_name)
    }
    coord_out[lat_name] = xr.DataArray(lat_target, dims=(lat_name,))
    coord_out[lon_name] = xr.DataArray(lon_target, dims=(lon_name,))

    out_ds = xr.Dataset(data_vars_out, coords=coord_out, attrs=src_ds.attrs)
    return out_ds

def _inpaint_nans_bilinear_2d(arr_2d):
    """
    Inpaint NaNs in a 2D array using bilinear interpolation in index space.
    This is: linear along rows, then along columns. Works on a regular grid.
    """
    import numpy as np
    from scipy.interpolate import griddata

    if not np.isnan(arr_2d).any():
        return arr_2d

    ny, nx = arr_2d.shape
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))  # (ny, nx)

    mask_valid = ~np.isnan(arr_2d)
    if mask_valid.sum() < 4:
        # Not enough valid points to do any meaningful bilinear interpolation
        return arr_2d

    pts_valid = np.column_stack([
        ii[mask_valid].ravel(),
        jj[mask_valid].ravel(),
    ])
    vals_valid = arr_2d[mask_valid].ravel()

    pts_all = np.column_stack([
        ii.ravel(),
        jj.ravel(),
    ])

    arr_interp = griddata(
        pts_valid,
        vals_valid,
        pts_all,
        method="linear",
    ).reshape(ny, nx)

    # If still NaNs (e.g., outside convex hull), fall back to nearest
    if np.isnan(arr_interp).any():
        arr_nn = griddata(
            pts_valid,
            vals_valid,
            pts_all,
            method="nearest",
        ).reshape(ny, nx)
        mask = np.isnan(arr_interp)
        arr_interp[mask] = arr_nn[mask]

    return arr_interp

def _build_torch_sampling_grid(lat_src, lon_src, lat_tgt, lon_tgt, device):
    """
    Build a grid for torch.nn.functional.grid_sample.

    lat_src, lon_src: 1D numpy arrays for source grid (size H, W)
    lat_tgt, lon_tgt: 1D numpy arrays for target grid (size H2, W2)

    Returns: torch tensor of shape [1, H2, W2, 2] on `device`,
            with coordinates in [-1, 1] (grid_sample convention).
    """
    lat_src = np.asarray(lat_src, dtype=np.float32)
    lon_src = np.asarray(lon_src, dtype=np.float32)
    lat_tgt = np.asarray(lat_tgt, dtype=np.float32)
    lon_tgt = np.asarray(lon_tgt, dtype=np.float32)

    H = lat_src.size
    W = lon_src.size

    # Assume approximately uniform spacing. If needed, replace with something
    # more robust (e.g., searchsorted and per-point mapping).
    lat_min, lat_max = lat_src[0], lat_src[-1]
    lon_min, lon_max = lon_src[0], lon_src[-1]

    # Index coordinates in [0, H-1] / [0, W-1]
    lat_idx = (lat_tgt - lat_min) / (lat_max - lat_min) * (H - 1)
    lon_idx = (lon_tgt - lon_min) / (lon_max - lon_min) * (W - 1)

    # Meshgrid in index space
    lon_idx_2d, lat_idx_2d = np.meshgrid(lon_idx, lat_idx)  # [H2, W2]

    # Convert to normalized coords in [-1, 1]
    # grid_sample expects (x, y) with:
    #   x in [-1, 1] over W dimension
    #   y in [-1, 1] over H dimension
    x_norm = 2.0 * (lon_idx_2d / (W - 1)) - 1.0
    y_norm = 2.0 * (lat_idx_2d / (H - 1)) - 1.0

    grid = np.stack([x_norm, y_norm], axis=-1)  # [H2, W2, 2]
    grid = torch.from_numpy(grid).to(device=device, dtype=torch.float32)
    grid = grid.unsqueeze(0)  # [1, H2, W2, 2]

    return grid

def _regrid_rectilinear_torch(
        input_ds: xr.Dataset,
        target_ds: xr.Dataset,
        vars_to_regrid=None,
        device: str = "cuda"
    ) -> xr.Dataset:
    """
    Regrid from target_ds's rectilinear lat/lon onto input_ds's rectilinear lat/lon
    using PyTorch grid_sample on GPU.

    NOTE: This mirrors your current rectilinear branch, but does the interpolation
    with torch instead of xarray.interp.
    """

    device = torch.device(device)

    lat_in = input_ds.cf["latitude"].name   # "output" lat (your input_ds)
    lon_in = input_ds.cf["longitude"].name
    lat_tg = target_ds.cf["latitude"].name  # "source" lat (your target_ds)
    lon_tg = target_ds.cf["longitude"].name

    # If grids are already identical, just return target_ds as-is
    same_dims = (lat_in == lat_tg) and (lon_in == lon_tg)
    same_shape = (
        input_ds[lat_in].shape == target_ds[lat_tg].shape and
        input_ds[lon_in].shape == target_ds[lon_tg].shape
    )
    same_coords = (
        np.array_equal(input_ds[lat_in].values, target_ds[lat_tg].values) and
        np.array_equal(input_ds[lon_in].values, target_ds[lon_tg].values)
    )

    if same_dims and same_shape and same_coords:
        print("Regrid (torch rectilinear): grids already identical, nothing to do.")
        return target_ds

    # Choose variables to regrid if not explicitly given
    if vars_to_regrid is None:
        vars_to_regrid = [
            name for name, da in target_ds.data_vars.items()
            if (lat_tg in da.coords or lon_tg in da.coords or
                lat_tg in da.dims or lon_tg in da.dims)
        ]

    # Sanity: require 1D lat/lon
    if (input_ds[lat_in].ndim != 1 or input_ds[lon_in].ndim != 1 or
        target_ds[lat_tg].ndim != 1 or target_ds[lon_tg].ndim != 1):
        raise ValueError("regrid_rectilinear_torch only supports 1D lat/lon on both grids.")

    # Source grid (where we have data)
    lat_src = target_ds[lat_tg].values
    lon_src = target_ds[lon_tg].values

    # Target grid (where we want data)
    lat_out = input_ds[lat_in].values
    lon_out = input_ds[lon_in].values

    H2 = lat_out.size
    W2 = lon_out.size

    # Build a single sampling grid and reuse for all variables
    sampling_grid = _build_torch_sampling_grid(
        lat_src, lon_src, lat_out, lon_out, device=device
    )  # [1, H2, W2, 2]

    data_vars_out = {}

    for name, da in target_ds.data_vars.items():
        if name not in vars_to_regrid:
            data_vars_out[name] = da
            continue

        # Require lat/lon dims
        if (lat_tg not in da.dims) or (lon_tg not in da.dims):
            data_vars_out[name] = da
            continue

        print(f"  Torch regridding variable '{name}' on device {device}...")

        # Move spatial dims to the end
        da_spatial = da.transpose(
            *[d for d in da.dims if d not in (lat_tg, lon_tg)],
            lat_tg, lon_tg,
        )

        data_np = da_spatial.values.astype("float32")
        leading_dims = da_spatial.dims[:-2]
        leading_shape = data_np.shape[:-2]
        H, W = data_np.shape[-2:]

        # Flatten leading dims into "channels"
        data_t = torch.from_numpy(data_np).to(device=device)
        data_t = data_t.reshape(1, -1, H, W)  # [N=1, C=prod(leading), H, W]

        # Bilinear sampling on GPU
        with torch.no_grad():
            out_t = F.grid_sample(
                data_t,
                sampling_grid,
                mode="bilinear",
                padding_mode="border",  # or "zeros" depending on your preference
                align_corners=True,
            )

        out_t = out_t.reshape(*leading_shape, H2, W2)  # [*leading, H2, W2]
        out_np = out_t.cpu().numpy()

        # Build output DataArray
        out_dims = leading_dims + (lat_in, lon_in)
        out_coords = {
            **{d: da_spatial.coords[d] for d in leading_dims},
            lat_in: input_ds[lat_in],
            lon_in: input_ds[lon_in],
        }

        data_vars_out[name] = xr.DataArray(
            out_np,
            dims=out_dims,
            coords=out_coords,
            attrs=da.attrs,
        )

    # Carry over non-lat/lon coordinates (override lat/lon with new ones)
    coord_out = {
        k: v for k, v in target_ds.coords.items()
        if k not in (lat_tg, lon_tg)
    }
    coord_out[lat_in] = input_ds[lat_in]
    coord_out[lon_in] = input_ds[lon_in]

    out_ds = xr.Dataset(data_vars_out, coords=coord_out, attrs=target_ds.attrs)
    return out_ds

def regrid_torch_or_scipy (input_ds: xr.Dataset, target_ds: xr.Dataset, vars_to_regrid=None) -> xr.Dataset:
    lat_in = input_ds.cf["latitude"].name
    lon_in = input_ds.cf["longitude"].name
    lat_tg = target_ds.cf["latitude"].name
    lon_tg = target_ds.cf["longitude"].name

    # print(f"Regrid (lat, lon): input ({lat_in}, {lon_in}), target ({lat_tg}, {lon_tg})")

    same_dims = (lat_in == lat_tg) and (lon_in == lon_tg)
    same_shape = (
        input_ds[lat_in].shape == target_ds[lat_tg].shape and
        input_ds[lon_in].shape == target_ds[lon_tg].shape
    )
    same_coords = (
        np.array_equal(input_ds[lat_in].values, target_ds[lat_tg].values) and
        np.array_equal(input_ds[lon_in].values, target_ds[lon_tg].values)
    )

    if same_dims and same_shape and same_coords:
        print("Regrid: grids already identical, nothing to do.")
        return target_ds

    if vars_to_regrid is None:
        vars_to_regrid = [
            name for name, da in target_ds.data_vars.items()
            if (lat_tg in da.coords or lon_tg in da.coords or
                lat_tg in da.dims or lon_tg in da.dims)
        ]

    # ---------- rectilinear: interp is bilinear ----------
    rectilinear = (
        input_ds[lat_in].ndim == 1 and input_ds[lon_in].ndim == 1 and
        target_ds[lat_tg].ndim == 1 and target_ds[lon_tg].ndim == 1
    )

    if rectilinear:
        print("Regrid: using torch (bilinear on rectilinear grid, GPU).")
        return _regrid_rectilinear_torch(input_ds, target_ds, vars_to_regrid, device="cuda")

    # ---------- curvilinear: scipy.griddata ----------
    print("Regrid: detected curvilinear source grid, using scipy.interpolate.griddata (linear).")

    lat_src = target_ds[lat_tg]
    lon_src = target_ds[lon_tg]
    if lat_src.ndim != 2 or lon_src.ndim != 2:
        raise ValueError(
            "Curvilinear regrid currently assumes 2D lat/lon (y,x). "
            f"Got lat ndim={lat_src.ndim}, lon ndim={lon_src.ndim}."
        )
    y_dim, x_dim = lat_src.dims  # e.g. ("y", "x")

    # Output grid: we assume input grid is rectilinear 1D lat/lon
    lat_out = input_ds[lat_in]
    lon_out = input_ds[lon_in]
    if lat_out.ndim != 1 or lon_out.ndim != 1:
        raise ValueError(
            "Curvilinear->rectilinear regrid assumes input lat/lon are 1D. "
            f"Got lat_in.ndim={lat_out.ndim}, lon_in.ndim={lon_out.ndim}."
        )

    Ny = lat_out.size
    Nx = lon_out.size

    # Build 2D (lon, lat) arrays for the output grid
    lon_out_2d, lat_out_2d = np.meshgrid(lon_out.values, lat_out.values)  # shapes (Ny, Nx)

    # Flatten source and target points
    points = np.column_stack([
        lon_src.values.ravel(),
        lat_src.values.ravel(),
    ])                               # shape (Ny_src * Nx_src, 2)

    xi = np.column_stack([
        lon_out_2d.ravel(),
        lat_out_2d.ravel(),
    ])                               # shape (Ny * Nx, 2)

    data_vars_out = {}
    for name, da in target_ds.data_vars.items():
        if name not in vars_to_regrid:
            data_vars_out[name] = da
            continue
        if y_dim not in da.dims or x_dim not in da.dims:
            data_vars_out[name] = da
            continue

        print(f"  Regridding variable '{name}'")

        # Move spatial dims to end
        da_spatial = da.transpose(
            *[d for d in da.dims if d not in (y_dim, x_dim)],
            y_dim, x_dim,
        )
        data_np = da_spatial.values  # already NumPy (we loaded in __init__)

        leading_dims = da_spatial.dims[:-2]
        leading_shape = data_np.shape[:-2]
        ny_src, nx_src = data_np.shape[-2:]

        arr = data_np.reshape(-1, ny_src * nx_src)  # [Nlead, Ny_src*Nx_src]

        out_slices = []
        for i in range(arr.shape[0]):
            zi = arr[i, :]
            zi_interp = griddata(points, zi, xi, method="linear")

            # Fill NaNs outside convex hull with nearest
            if np.any(np.isnan(zi_interp)):
                zi_nn = griddata(points, zi, xi, method="nearest")
                mask = np.isnan(zi_interp)
                zi_interp[mask] = zi_nn[mask]

            out_slices.append(zi_interp.reshape(Ny, Nx))

        out = np.stack(out_slices, axis=0).reshape(
            *leading_shape,
            Ny,
            Nx,
        )

        out_dims = leading_dims + (lat_in, lon_in)
        out_coords = {
            **{d: da_spatial.coords[d] for d in leading_dims},
            lat_in: lat_out,
            lon_in: lon_out,
        }

        data_vars_out[name] = xr.DataArray(
            out,
            dims=out_dims,
            coords=out_coords,
            attrs=da.attrs,
        )

    # Carry over non-lat/lon coords
    coord_out = {
        k: v for k, v in target_ds.coords.items()
        if k not in (lat_tg, lon_tg)
    }
    coord_out[lat_in] = lat_out
    coord_out[lon_in] = lon_out

    out_ds = xr.Dataset(data_vars_out, coords=coord_out, attrs=target_ds.attrs)
    return out_ds

import matplotlib.pyplot as plt
def quickplot (ds, varname="", folder="./", filename="rolled.png", t_idx=0):
    """Quick diagnostic plot of a 2D field in ds."""
    da = ds[varname] if isinstance(ds, xr.Dataset) else ds

    # Select time slice
    while da.ndim > 2:
        da = da.isel({da.dims[0]: t_idx})

    # Get lon/lat
    lon_coord, lat_coord = get_lonlat_coords(ds)
    lon = ds[lon_coord]
    lat = ds[lat_coord]

    plt.figure(figsize=(10,5))
    plt.pcolormesh(lon, lat, da, shading="auto")
    plt.colorbar(label=varname)
    plt.title(f"{varname}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    name, _, ext = filename.rpartition('.')
    plt.savefig(f"{folder}/{name}_{t_idx}.{ext}", dpi=150)
    plt.close()

def print_ds_info (
    ds: xr.Dataset,
    quantiles: Optional[Sequence[float]] = None,
    regimes: Optional[Sequence[float]] = None,
) -> None:
    for name, da in ds.data_vars.items():
        # Basic stats
        mean = float(da.mean().values)
        std = float(da.std().values)
        vmin = float(da.min().values)
        vmax = float(da.max().values)
        print(f"var {name}")
        print(f"  shape:    {da.shape}")
        print(f"  dtype:    {da.dtype}")
        print(f"  mean:     {mean:.4g}")
        print(f"  std:      {std:.4g}")
        print(f"  min/max:  {vmin:.4g} / {vmax:.4g}")
        print(f"  attrs:    {da.attrs}")
        print(f"  encoding: {da.encoding}")
        # Quantiles
        if quantiles is not None:
            default_qs = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
            qs = da.quantile(default_qs if len(quantiles) == 0 else quantiles)
            print("  quantiles:")
            for q, v in zip(qs["quantile"].values, qs.values):
                print(f"    q={float(q):.3f}: {float(v):.4g}")
        # Separate regimes # TODO test
        if regimes is not None and len(regimes) > 1:
            edges = np.sort(np.asarray(regimes))
            loc_regimes = []
            # lower tail
            loc_regimes.append(da.where(da <= edges[0]))
            # interior bins
            for low, high in zip(edges[:-1], edges[1:]):
                loc_regimes.append(da.where((da > low) & (da <= high)))
            # upper tail
            loc_regimes.append(da.where(da > edges[-1]))
            for i, r in enumerate(loc_regimes):
                r_mean = float(r.mean().values)
                r_std = float(r.std().values)
                print(f"  regime {i} mean/std: {r_mean:.4g} / {r_std:.4g}")

def _floor_to_midnight (dt: datetime) -> datetime:
    return datetime.combine(dt.date(), datetime_time.min, tzinfo=dt.tzinfo)

def half_train_periods_days (
    base: TimeRange,
    min_months: int = 3,
    anchor: str = "end",  # "end" or "start"
) -> List[TimeRange]:
    if base.end <= base.start:
        raise ValueError("base.end must be after base.start")
    if anchor not in {"end", "start"}:
        raise ValueError("anchor must be 'end' or 'start'")

    # Normalize endpoints to midnight to avoid hour drift
    start0 = _floor_to_midnight(base.start)
    end0   = _floor_to_midnight(base.end)

    # Minimum length in days, calendar months threshold
    if anchor == "end":
        min_start = end0 - relativedelta(months=min_months)
        min_days = (end0 - _floor_to_midnight(min_start)).days
    else:
        min_end = start0 + relativedelta(months=min_months)
        min_days = (_floor_to_midnight(min_end) - start0).days

    total_days = (end0 - start0).days
    if total_days <= 0:
        raise ValueError("After midnight alignment, range has no full days.")

    out: List[TimeRange] = []
    days = total_days
    while days >= min_days:
        if anchor == "end":
            tr_start = end0 - timedelta(days=days)
            tr_end = end0
        else:
            tr_start = start0
            tr_end = start0 + timedelta(days=days)

        out.append(TimeRange(start=tr_start, end=tr_end, freq=base.freq, shifted=base.shifted))
        days //= 2  # day-granular

    return out

def halved_windows_split_by_cutoff (
    base: "TimeRange",
    cutoff_end: datetime,          # e.g. datetime(2014, 12, 31)
    min_months: int = 3,
    anchor: str = "end",           # "end" (default) or "start" for the halved window
    post_starts_next_day: bool = True,
) -> List[List["TimeRange"]]:
    """
    Builds progressively halved windows (day-granular). For each window:
      - if cutoff_end lies inside the window (inclusive), returns [pre, post]
      - otherwise returns [window] only

    Output is a list of lists. Each inner list has length 1 or 2.
    """

    if anchor not in {"end", "start"}:
        raise ValueError("anchor must be 'end' or 'start'")

    # Day-align everything to avoid hour drift
    base_start = _floor_to_midnight(base.start)
    base_end   = _floor_to_midnight(base.end)
    cutoff0    = _floor_to_midnight(cutoff_end)

    if base_end <= base_start:
        raise ValueError("base.end must be after base.start")

    # Minimum window length in whole days (using calendar months)
    if anchor == "end":
        min_start = base_end - relativedelta(months=min_months)
        min_days = (base_end - _floor_to_midnight(min_start)).days
    else:
        min_end = base_start + relativedelta(months=min_months)
        min_days = (_floor_to_midnight(min_end) - base_start).days

    total_days = (base_end - base_start).days
    days = total_days

    out: List[List[TimeRange]] = []

    while days >= min_days and days > 0:
        # Build the halved window (day-only)
        if anchor == "end":
            win_start = base_end - timedelta(days=days)
            win_end = base_end
        else:
            win_start = base_start
            win_end = base_start + timedelta(days=days)

        window = TimeRange(start=win_start, end=win_end, freq=base.freq, shifted=base.shifted)

        # If cutoff is inside, split; else return just the window
        if win_start <= cutoff0 <= win_end:
            pre = TimeRange(start=win_start, end=cutoff0, freq=base.freq, shifted=base.shifted)

            post_start = cutoff0 + timedelta(days=1) if post_starts_next_day else cutoff0
            if post_start <= win_end:
                post = TimeRange(start=post_start, end=win_end, freq=base.freq, shifted=base.shifted)
                out.append([pre, post])
            else:
                # cutoff is at/near the end so post would be empty
                out.append([pre])
        else:
            out.append([window])

        days //= 2

    return out
