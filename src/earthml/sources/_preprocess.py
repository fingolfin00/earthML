import cf_xarray  # ensure .cf accessor on workers
import xarray as xr
import pandas as pd
from rich import print

from ..dataclasses import DataSelection
from ..utils import _guess_coord_name, get_lonlat_coords

def _status_da(ds: xr.Dataset, time_coord: str, ok: bool, name: str = "_has_var"):
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

def preprocess_mfdataset (ds: xr.Dataset, data: DataSelection, var_name: str | None = None, date = None) -> xr.Dataset:
    import cf_xarray  # noqa
    import numpy as np
    import xarray as xr
    import pandas as pd

    var0 = data.variable[0] if isinstance(data.variable, list) else data.variable
    leadtime = var0.leadtime

    if leadtime is not None:
        time_coord = _guess_coord_name(ds, leadtime.name)
    else:
        time_coord = _guess_coord_name(ds, "time", ["valid_time", "time_counter"])

    # if not time_coord:
    #     print("Could not find a time coord or CF time axis, try to assign it")

    # Ensure time_coord is a dimension
    if time_coord not in ds.dims and time_coord in ds.coords:
        values = np.asarray(ds[time_coord].values).ravel()[0].astype("datetime64[ns]")
        ds = ds.expand_dims({time_coord: [values]})
        ds = ds.assign_coords(
            **{time_coord: ds[time_coord].assign_attrs(standard_name="time", axis="T")}
        )

    lon_coord, lat_coord = get_lonlat_coords(ds)

    # Decide desired variable name
    target_name = var_name or var0.name

    # Select variable
    if target_name not in ds.data_vars and target_name not in ds.variables:
        # propagate “missing var” info in a concat-safe way
        out = xr.Dataset()
        out["_has_var"] = _status_da(ds, time_coord, False, name="_has_var")
        # keep coords you care about
        if lon_coord and lat_coord:
            out = out.assign_coords({lon_coord: ds[lon_coord], lat_coord: ds[lat_coord]})
        # keep a useful breadcrumb (often set by xarray backends)
        out.attrs["_missing_var_name"] = target_name
        out.attrs["_source"] = ds.encoding.get("source", "")
        # Assign at least time coord
        if not time_coord:
            # print(date)
            out["source_time"] = date
        return out

    # Normal case
    da = ds[target_name]

    # Select leadtime if present
    if leadtime is not None and leadtime.name in ds.coords:
        td = pd.to_timedelta(f"{leadtime.value} {leadtime.unit}")
        coord_dtype = ds[leadtime.name].dtype
        target = td.to_numpy().astype(coord_dtype)
        da = da.sel({leadtime.name: target}, method="nearest")

    out = xr.Dataset({da.name or target_name: da})
    out["_has_var"] = _status_da(ds, time_coord, True, name="_has_var")

    if lon_coord and lat_coord:
        out = out.assign_coords({lon_coord: ds[lon_coord], lat_coord: ds[lat_coord]})
    out.attrs["_source"] = ds.encoding.get("source", "")

    # Drop non-dim extra "time" coord
    if "time" in out.coords and "time" not in out.dims and "time" != time_coord:
        # print("Reset extra time coordinate")
        out = out.reset_coords("time", drop=True)

    return out
