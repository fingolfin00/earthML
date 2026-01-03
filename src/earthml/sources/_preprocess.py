import cf_xarray  # ensure .cf accessor on workers
import xarray as xr
import pandas as pd
from rich import print

from ..dataclasses import DataSelection
from ..utils import _guess_coord_name, get_lonlat_coords

def _status_da (ds: xr.Dataset, time_coord: str | None, ok: bool, name: str = "_has_var"):
    import xarray as xr

    # ensure a length-1 time dim if scalar
    if time_coord not in ds.dims and time_coord in ds.coords:
        coord = ds[time_coord]
        ds = ds.expand_dims({time_coord: [coord.values]})

    if time_coord in ds.sizes:
        n = ds.sizes[time_coord]
        return xr.DataArray(
            [ok] * n,
            dims=(time_coord,),
            coords={time_coord: ds[time_coord]},
            name=name,
        )

    # fallback: scalar
    return xr.DataArray(ok, name=name)

def ensure_time_dim (
    ds: xr.Dataset,
    time_coord: str | None,
    *,
    standard_name="time",
) -> tuple[xr.Dataset, str]:
    """
    Ensure there is a time dimension. Returns (ds2, time_name_used).

    Strategy:
    - If time_coord is None, pick the first candidate that exists as coord/var.
    - If chosen time exists but is scalar coord -> attach along a dummy dim and swap.
    - If chosen time exists 1D on another dim -> swap that dim to time.
    - If no time-like variable exists -> create a placeholder time dim with NaT length 1.
    """
    import numpy as np
    import xarray as xr

    tname = time_coord
    # If no tname, return
    if tname is None:
        return ds, tname

    # If already a dimension, just ensure attrs (and if scalar coord exists too, leave it)
    if tname in ds.dims:
        ds2 = ds
        if tname in ds2.coords:
            ds2[tname].attrs.update({"standard_name": standard_name, "axis": "T"})
        return ds2, tname

    # Get time "values" as a DataArray (coord preferred, else data_var)
    t = ds.coords.get(tname, None)
    if t is None:
        t = ds[tname]  # data_var

    # Promote scalar time to a dimension (no name collision)
    if t.ndim == 0:
        # Create a dummy dim that definitely doesn't collide
        dummy = "__time_dummy__"
        while dummy in ds.dims or dummy in ds.coords or dummy in ds.data_vars:
            dummy = "_" + dummy

        # Expand along dummy, then attach time values along dummy, then swap dims
        ds2 = ds.expand_dims({dummy: 1})
        ds2 = ds2.assign_coords(
            {tname: xr.DataArray(np.array([t.values], dtype="datetime64[ns]"), dims=(dummy,))}
        )
        ds2 = ds2.swap_dims({dummy: tname})
        ds2[tname].attrs.update({"standard_name": standard_name, "axis": "T"})
        return ds2, tname

    # If time is 1D on some other dim, swap that dim to time
    if t.ndim == 1:
        base_dim = t.dims[0]
        ds2 = ds.swap_dims({base_dim: tname})
        ds2[tname].attrs.update({"standard_name": standard_name, "axis": "T"})
        return ds2, tname

    print(f"Don't know how to promote {tname!r} with ndim={t.ndim} to a time dimension.")
    return ds, tname

def preprocess_mfdataset (ds: xr.Dataset, data: DataSelection, var_name: str | None = None, date = None) -> xr.Dataset:
    import numpy as np
    import cf_xarray  # noqa
    import xarray as xr
    import pandas as pd

    var0 = data.variable[0] if isinstance(data.variable, list) else data.variable
    leadtime = var0.leadtime

    time_coord = _guess_coord_name(ds, "time", ["valid_time", "time_counter"])
    # Ensure time_coord is a dimension
    ds, time_coord = ensure_time_dim(ds, time_coord)

    lon_coord, lat_coord = get_lonlat_coords(ds)

    # Decide desired variable name
    var_name = var_name or var0.name

    # Fallback if variable not present in ds
    if var_name not in ds.data_vars and var_name not in ds.variables:
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
        if not time_coord:
            # print(date)
            out = out.assign_coords({"source_time": date})
            out = out.expand_dims("source_time")
        # print(out)
        return out

    # Normal case
    da = ds[var_name]

    # Select leadtime if present
    if leadtime is not None and leadtime.name in ds.coords:
        td = pd.to_timedelta(f"{leadtime.value} {leadtime.unit}")
        coord_dtype = ds[leadtime.name].dtype
        target = td.to_numpy().astype(coord_dtype)
        dist = abs(da[leadtime.name] - target)
        idx = int(dist.argmin(time_coord).compute())
        da = da.isel({time_coord: idx})

    out = xr.Dataset({da.name or var_name: da})
    out["_has_var"] = _status_da(out, time_coord, True, name="_has_var")

    if lon_coord and lat_coord:
        out = out.assign_coords({lon_coord: ds[lon_coord], lat_coord: ds[lat_coord]})
    out.attrs["_source"] = ds.encoding.get("source", "")

    # Drop non-dim extra "time" coord
    if "time" in out.coords and "time" not in out.dims and "time" != time_coord:
        # print("Reset extra time coordinate")
        out = out.reset_coords("time", drop=True)

    # print(out)
    return out
