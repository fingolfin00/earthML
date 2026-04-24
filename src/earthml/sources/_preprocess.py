import cf_xarray  # ensure .cf accessor on workers
import time
import xarray as xr
import pandas as pd
from rich import print

from .dataclasses import DataSelection
from ..logging import get_logger


logger = get_logger(__name__)


def _status_da(ds: xr.Dataset, time_coord: str | None, ok: bool, name: str = "_has_var"):
    import xarray as xr

    # ensure a length-1 time dim if scalar
    if time_coord is not None and time_coord not in ds.dims and time_coord in ds.coords:
        coord = ds[time_coord]
        ds = ds.expand_dims({time_coord: [coord.values]})

    if time_coord is not None and time_coord in ds.sizes:
        n = ds.sizes[time_coord]
        return xr.DataArray(
            [ok] * n,
            dims=(time_coord,),
            coords={time_coord: ds[time_coord]},
            name=name,
        )

    # fallback: scalar # TODO verify this is acceptable downstream
    return xr.DataArray(ok, name=name)


def ensure_time_dim(ds: xr.Dataset) -> tuple[xr.Dataset, str]:
    import numpy as np
    import xarray as xr

    time_name = ds.earthml.guessed_dims.time or ds.earthml.guessed_coords.time

    if time_name is not None:
        ds2 = ds.earthml.promote_to_dim(
            time_name,
            standard_name="time",
            axis="T",
        )
        return ds2, time_name

    # fallback: create placeholder time dim
    tname = "time"
    if tname in ds.dims or tname in ds.coords or tname in ds.data_vars:
        base = tname
        i = 1
        while tname in ds.dims or tname in ds.coords or tname in ds.data_vars:
            tname = f"{base}_{i}"
            i += 1

    ds2 = ds.expand_dims({tname: 1})
    ds2 = ds2.assign_coords({
        tname: xr.DataArray(
            np.array([np.datetime64("NaT")], dtype="datetime64[ns]"),
            dims=(tname,),
        )
    })
    ds2[tname].attrs.update({"standard_name": "time", "axis": "T"})
    return ds2, tname


def _select_vertical_level(
    ds: xr.Dataset,
    da: xr.DataArray,
    data: DataSelection,
) -> tuple[xr.Dataset, xr.DataArray, float]:
    import numpy as np

    t0 = time.perf_counter()
    var0 = data.variable[0] if isinstance(data.variable, list) else data.variable
    level_value = next((lv for lv in (var0.levhpa, var0.levm) if lv is not None), None)
    if level_value is None:
        return ds, da, 0.0

    level_dim = ds.earthml.guessed_dims.level
    level_coord = ds.earthml.guessed_coords.level

    if level_dim is None or level_dim not in da.dims:
        return ds, da, 0.0

    if level_coord is not None and level_coord in ds.coords:
        coord = ds[level_coord]
    elif level_dim in ds.coords:
        coord = ds[level_dim]
    else:
        coord = None

    if coord is not None and coord.ndim == 1 and level_dim in coord.dims:
        coord_vals = np.asarray(coord.values, dtype=np.float64)
        idx = int(np.abs(coord_vals - float(level_value)).argmin())
        da = da.isel({level_dim: idx})
        ds = ds.isel({level_dim: idx})
    else:
        try:
            da = da.sel({level_dim: level_value}, method="nearest")
            ds = ds.sel({level_dim: level_value}, method="nearest")
        except Exception:
            da = da.isel({level_dim: 0})
            ds = ds.isel({level_dim: 0})

    return ds, da, time.perf_counter() - t0


def preprocess_mfdataset(
    ds: xr.Dataset,
    data: DataSelection,
    var_name: str | None = None,
    date = None,
    fill_value: float | None = None,
) -> xr.Dataset:
    import numpy as np
    import cf_xarray, earthml  # noqa
    import xarray as xr
    import pandas as pd

    t0 = time.perf_counter()
    var0 = data.variable[0] if isinstance(data.variable, list) else data.variable
    leadtime = var0.leadtime

    # Ensure ds a=has a time dim
    t_ensure0 = time.perf_counter()
    ds, time_coord = ensure_time_dim(ds)
    t_ensure1 = time.perf_counter()

    t_coords0 = time.perf_counter()
    lon_coord, lat_coord = ds.earthml.guessed_coords.longitude, ds.earthml.guessed_coords.latitude
    t_coords1 = time.perf_counter()

    # Require both spatial coordinates; otherwise mark sample invalid
    if lon_coord is None or lat_coord is None:
        out = xr.Dataset()
        out["_has_var"] = _status_da(ds, time_coord, False, name="_has_var")
        out.attrs["_missing_spatial_coords"] = {
            "longitude": lon_coord,
            "latitude": lat_coord,
        }
        out.attrs["_source"] = ds.encoding.get("source", "")
        if date is not None and time_coord is not None and time_coord in out.dims and out.sizes[time_coord] == 1:
            out = out.assign_coords({time_coord: [np.datetime64(date)]})
        logger.debug(
            "preprocess timing date=%s var=%s missing_spatial ensure_time=%.3fs guessed_coords=%.3fs total=%.3fs",
            date,
            var_name or var0.name,
            t_ensure1 - t_ensure0,
            t_coords1 - t_coords0,
            time.perf_counter() - t0,
        )
        return out

    # Decide desired variable name
    var_name = var_name or var0.name

    # Fallback if variable not present in ds
    if var_name not in ds.data_vars and var_name not in ds.variables:
        t_missing0 = time.perf_counter()
        # propagate “missing var” info in a concat-safe way
        out = xr.Dataset()
        out["_has_var"] = _status_da(ds, time_coord, False, name="_has_var")
        # keep coords lon and lat coords
        if lon_coord and lat_coord:
            out = out.assign_coords({lon_coord: ds[lon_coord], lat_coord: ds[lat_coord]})
        # keep var name and source file name as attributes
        out.attrs["_missing_var_name"] = var_name
        out.attrs["_source"] = ds.encoding.get("source", "")
        # Assign at least time coord
        # if not time_coord:
        #     # print(date)
        #     out = out.assign_coords({"source_time": date})
        #     out = out.expand_dims("source_time")
        if date is not None and time_coord is not None and time_coord in out.dims and out.sizes[time_coord] == 1:
            out = out.assign_coords({time_coord: [np.datetime64(date)]})
        # print(out)
        logger.debug(
            "preprocess timing date=%s var=%s missing_var ensure_time=%.3fs guessed_coords=%.3fs missing_build=%.3fs total=%.3fs",
            date,
            var_name,
            t_ensure1 - t_ensure0,
            t_coords1 - t_coords0,
            time.perf_counter() - t_missing0,
            time.perf_counter() - t0,
        )
        return out

    # Normal case
    t_var0 = time.perf_counter()
    da = ds[var_name]
    t_var1 = time.perf_counter()

    # Select vertical level
    ds, da, level_select_s = _select_vertical_level(ds, da, data)

    # Handle NaNs
    if fill_value is not None:
        da = da.where(da != fill_value)

    # Select leadtime if present
    leadtime_select_s = 0.0
    if leadtime is not None and leadtime.name in da.coords: # TODO need to understand this bit better
        t_lt0 = time.perf_counter()
        td = leadtime.to_timedelta()
        coord = da[leadtime.name]
        target = td.to_numpy().astype(coord.dtype)
        if coord.ndim != 1:
            raise ValueError(
                f"Expected 1D leadtime coordinate {leadtime.name!r}, got dims={coord.dims}"
            )
        lead_dim = coord.dims[0]
        # idx = int((abs(coord - target)).argmin(dim=lead_dim).compute().item())
        ## speed-up attempt
        coord_vals = np.asarray(coord.values)
        idx = int(np.abs(coord_vals - target).argmin())
        ##
        da = da.isel({lead_dim: idx})
        t_lt1 = time.perf_counter()
        leadtime_select_s = t_lt1 - t_lt0

    t_out0 = time.perf_counter()
    out = xr.Dataset({da.name or var_name: da})
    out["_has_var"] = _status_da(out, time_coord, True, name="_has_var")

    if lon_coord and lat_coord:
        out = out.assign_coords({lon_coord: ds[lon_coord], lat_coord: ds[lat_coord]})
    out.attrs["_source"] = ds.encoding.get("source", "")

    # In the Juno local workflow, `date` is the canonical sample/valid time we
    # want to train on. NetCDF inputs may carry their own internal time stamp
    # (often init/issue time or another provider-specific convention), which can
    # visually and semantically shift the sample if we leave it untouched.
    if date is not None and time_coord is not None and time_coord in out.dims and out.sizes[time_coord] == 1:
        out = out.assign_coords({time_coord: [np.datetime64(date)]})

    # Drop non-dim extra "time" coord
    if "time" in out.coords and "time" not in out.dims and "time" != time_coord:
        # print("Reset extra time coordinate")
        out = out.reset_coords("time", drop=True)

    t_out1 = time.perf_counter()
    logger.debug(
        "preprocess timing date=%s var=%s ensure_time=%.3fs guessed_coords=%.3fs get_var=%.3fs level_select=%.3fs leadtime_select=%.3fs build_out=%.3fs total=%.3fs",
        date,
        var_name,
        t_ensure1 - t_ensure0,
        t_coords1 - t_coords0,
        t_var1 - t_var0,
        level_select_s,
        leadtime_select_s,
        t_out1 - t_out0,
        t_out1 - t0,
    )

    # print("preprocess", out.dims)
    return out
