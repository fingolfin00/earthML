from pathlib import Path

import pandas as pd
import xarray as xr


def ensure_time_coord(
    ds: xr.Dataset,
    *,
    start: str,
    freq: str = "YS",
) -> xr.Dataset:
    if "time" not in ds.dims or "time" in ds.coords:
        return ds

    start = pd.Timestamp(start)

    if freq == "YS" and start.month != 1:
        freq = f"YS-{start.strftime('%b').upper()}"

    dates = pd.date_range(start=start, periods=ds.sizes["time"], freq=freq)

    return ds.assign_coords(time=dates)

def normalize_lon_range(
    lon_range: tuple[float, float],
    lon_coord: xr.DataArray,
) -> tuple[float, float]:
    lon_min = float(lon_coord.min())
    lon_max = float(lon_coord.max())

    # Dataset uses 0..360 but requested -180..180
    if lon_min >= 0 and lon_max > 180:
        return tuple(lon % 360 for lon in lon_range)

    return lon_range
