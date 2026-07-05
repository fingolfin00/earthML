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
) -> tuple[tuple[float, float], bool]:
    lon_min = float(lon_coord.min())
    lon_max = float(lon_coord.max())

    wrap = False

    if lon_min >= 0 and lon_max > 180:
        lon0 = lon_range[0] % 360
        lon1 = lon_range[1] % 360
        wrap = lon0 > lon1
        return (lon0, lon1), wrap

    wrap = lon_range[0] > lon_range[1]
    return lon_range, wrap
