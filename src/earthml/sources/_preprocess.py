import cf_xarray  # ensure .cf accessor on workers
import xarray as xr
import pandas as pd
from rich import print

from ..dataclasses import DataSelection
from ..utils import _guess_coord_name, get_lonlat_coords

def preprocess_mfdataset (
    ds: xr.Dataset,
    data: DataSelection,
    var_name: str | None = None,
) -> xr.Dataset:
    """
    Lightweight per-file preprocess for open_mfdataset.

    Responsibilities:
    - optionally select a single variable
    - ensure a time-like dimension exists (for concatenation)
    - optionally select a specific leadtime if present
    """
    import cf_xarray  # ensure .cf accessor on workers
    import xarray as xr
    import pandas as pd

    # --- Leadtime handling -----------------------------------------
    # Use first variable's leadtime if `data.variable` is a list
    var0 = data.variable[0] if isinstance(data.variable, list) else data.variable
    leadtime = var0.leadtime

    # --- Guess time-like dimension name -----------------------------
    # If we have a leadtime axis, we often want to use that name directly
    if leadtime is not None:
        # e.g. leadtime.name == "leadtime"
        time_coord = _guess_coord_name(ds, leadtime.name)
    else:
        # Fall back to CF time / common time dim names
        time_coord = _guess_coord_name(ds, "time", ["valid_time", "time_counter"])

    if not time_coord:
        print("Could not find a time coord or CF time axis, try to assign it")

    # print("time coord", time_coord)
    # --- Ensure this dimension exists in ds.dims --------------------
    # Some files may have a scalar time coordinate: make it a length-1 dimension
    if time_coord not in ds.dims and time_coord in ds.coords:
        coord = ds[time_coord]
        ds = ds.expand_dims({time_coord: [coord.values]})
        ds = ds.assign_coords(
            **{
                time_coord: ds[time_coord].assign_attrs(
                    standard_name="time",
                    axis="T",
                )
            }
        )

    lon_coord, lat_coord = get_lonlat_coords(ds)
    # print("Time coord", time_coord, "Lon coord:", lon_coord, ", lat coord:", lat_coord)
    # if ds[lon_coord].ndim > 1:
    #     print(ds[lon_coord].values)

    # --- Select variable --------------------------------------------
    if var_name is not None:
        da = ds[var_name]
    else:
        if isinstance(data.variable, list):
            da = ds[var0.name]  # use first variable's name
        else:
            da = ds[var0.name]

    # --- Select leadtime if present ---------------------------------
    if leadtime is not None:
        # Build target timedelta from value + unit (e.g. "3 days", "12 hours")
        td = pd.to_timedelta(f"{leadtime.value} {leadtime.unit}")

        # Cast to same dtype as coord (usually timedelta64[ns])
        coord_dtype = ds[leadtime.name].dtype
        target = td.to_numpy().astype(coord_dtype)

        da = da.sel({leadtime.name: target}, method="nearest")

    # We need to return a xr.Dataset otherwise xr.open_mfdataset with combine="nested"
    # returns a DataArray, which would break consistency further ahead
    return xr.Dataset(
        {da.name or var_name or var0.name: da},
        {lon_coord: ds[lon_coord], lat_coord: ds[lat_coord]} if lon_coord and lat_coord else None
    )
