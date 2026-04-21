import xarray as xr


_SECONDS_PER_DAY = 24 * 60 * 60
_DEFAULT_MONTH_LENGTH_DAYS = 30


def celsius_to_kelvin(x):
    return x + 273.15


def _get_time_coord(da: xr.DataArray) -> xr.DataArray | None:
    time_dim = da.earthml.guessed_dims.time
    time_coord_name = da.earthml.guessed_coords.time
    if time_dim is None or time_coord_name is None:
        return None

    coord = da.coords.get(time_coord_name)
    if coord is None:
        return None
    if coord.ndim != 1 or time_dim not in da.dims or time_dim not in coord.dims:
        return None
    return coord


def _seconds_per_timestep(da: xr.DataArray) -> xr.DataArray | float:
    time_coord = _get_time_coord(da)
    if time_coord is None:
        return float(_DEFAULT_MONTH_LENGTH_DAYS * _SECONDS_PER_DAY)

    try:
        return time_coord.dt.days_in_month.astype("float32") * float(_SECONDS_PER_DAY)
    except Exception:
        return float(_DEFAULT_MONTH_LENGTH_DAYS * _SECONDS_PER_DAY)


def total_precipitation_to_rate(x: xr.DataArray) -> xr.DataArray:
    """
    Convert accumulated total precipitation [m] to mean precipitation rate [m/s].

    For monthly data we infer the number of seconds from the time coordinate month
    length when available; otherwise we fall back to the project-wide 30-day month
    convention already used for monthly leadtimes.
    """
    return x / _seconds_per_timestep(x)


def precipitation_rate_to_total_precipitation(x: xr.DataArray) -> xr.DataArray:
    """
    Convert mean precipitation rate [m/s] to accumulated total precipitation [m].
    """
    return x * _seconds_per_timestep(x)
