import cf_xarray  # ensure .cf accessor on workers
import xarray as xr
import pandas as pd
from rich import print

from .dataclasses import DataSelection


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


def preprocess_mfdataset(ds: xr.Dataset, data: DataSelection, var_name: str | None = None, date = None) -> xr.Dataset:
    import numpy as np
    import cf_xarray, earthml  # noqa
    import xarray as xr
    import pandas as pd

    var0 = data.variable[0] if isinstance(data.variable, list) else data.variable
    leadtime = var0.leadtime

    # Ensure ds a=has a time dim
    ds, time_coord = ensure_time_dim(ds)

    lon_coord, lat_coord = ds.earthml.guessed_coords.longitude, ds.earthml.guessed_coords.latitude

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
        return out

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
        # if not time_coord:
        #     # print(date)
        #     out = out.assign_coords({"source_time": date})
        #     out = out.expand_dims("source_time")
        if date is not None and time_coord is not None and time_coord in out.dims and out.sizes[time_coord] == 1:
            out = out.assign_coords({time_coord: [np.datetime64(date)]})
        # print(out)
        return out

    # Normal case
    da = ds[var_name]

    # Select leadtime if present
    if leadtime is not None and leadtime.name in da.coords: # TODO need to understand this bit better
        td = pd.to_timedelta(f"{leadtime.value} {leadtime.unit}")
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

    out = xr.Dataset({da.name or var_name: da})
    out["_has_var"] = _status_da(out, time_coord, True, name="_has_var")

    if lon_coord and lat_coord:
        out = out.assign_coords({lon_coord: ds[lon_coord], lat_coord: ds[lat_coord]})
    out.attrs["_source"] = ds.encoding.get("source", "")

    # Drop non-dim extra "time" coord
    if "time" in out.coords and "time" not in out.dims and "time" != time_coord:
        # print("Reset extra time coordinate")
        out = out.reset_coords("time", drop=True)

    # print("preprocess", out.dims)
    return out
