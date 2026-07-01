from typing import Sequence, Literal, TypeVar, cast

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dask.diagnostics.progress import ProgressBar

from .settings import Settings
from .coords import ensure_time_coord, normalize_lon_range


T_Xarray = TypeVar("T_Xarray", xr.DataArray, xr.Dataset)


def open_zarr(
    path: str | Path,
) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)

def open_zarr_var(
    path: str | Path,
    var: str,
) -> xr.DataArray:
    return open_zarr(path)[var]

def open_nc(
    path: str | Path,
    engine: Literal["netcdf4", "h5netcdf"] = "netcdf4",
) -> xr.Dataset | None:
    ds = xr.open_dataset(
        path,
        engine=engine,
        chunks={},
        decode_times=False,
    )

    if len(ds.data_vars) == 0:
        return None

    return ds

def open_nc_var(
    path: str | Path,
    var: str,
    engine: Literal["netcdf4", "h5netcdf"] = "netcdf4",
) -> xr.DataArray:
    ds = open_nc(path, engine)

    if ds is None:
        raise ValueError(f"No data variables found in {path}")

    return ds[var]


def get_and_subset_datasets(
    s: Settings,
    period: Literal["hours", "days", "months", "years"],
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    time_start: str | None = None,
    interpolate: bool = True,
    engine: Literal["netcdf", "zarr"] = "zarr",
    build_analysis: bool = True,
    coord_rename_fc: Sequence[Sequence[str]] | None = None,
    coord_rename_an: Sequence[Sequence[str]] | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset | None]:
    if engine == "zarr":
        open_engine = open_zarr
        open_engine_var = open_zarr_var
    elif engine == "netcdf":
        open_engine = open_nc
        open_engine_var = open_nc_var
    else:
        raise ValueErrot(f"Engine {engine} not supported, choose between [netcdf, zarr]")

    fc_da = open_engine_var(s.input_fc, s.var_fc)
    an_da = open_engine_var(s.input_an, s.var_an)

    if coord_rename_fc is not None:
        for rename_seq in coord_rename_fc:
            print(rename_seq)
            if len(list(rename_seq)) == 2:
                fc_da = fc_da.earthml.rename_dim_and_coord(rename_seq[0], rename_seq[1])
    if coord_rename_an is not None:
        for rename_seq in coord_rename_an:
            if len(list(rename_seq)) == 2:
                an_da = an_da.earthml.rename_dim_and_coord(rename_seq[0], rename_seq[1])

    time_dim = fc_da.earthml.guessed_dims.time
    leadtime_dim = fc_da.earthml.guessed_dims.leadtime
    lon_dim = fc_da.earthml.guessed_dims.longitude
    lat_dim = fc_da.earthml.guessed_dims.latitude

    if build_analysis:
        an_da = build_analysis_for_forecast_leadtimes(
            fc_all=fc_da,
            an_raw=an_da,
            leadtime_dim=leadtime_dim,
            period=period,
            lead_period_offset=s.lead_period_offset,
        )
    fc = subset_dataset(
        fc_da.to_dataset(name=s.var_file_an),
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=time_range,
        time_start=time_start,
    )

    an = subset_dataset(
        an_da.to_dataset(name=s.var_file_fc),
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=time_range,
        time_start=time_start,
    )

    # Regrid analysis into forecast
    if interpolate:
        an = an.interp(
            {
                lat_dim: fc[lat_dim],
                lon_dim: fc[lon_dim],
            }
        )

    mlfc = None
    if s.input_mlfc_train.exists() and s.input_mlfc_test.exists():
        mlfc = xr.concat(
            [open_engine(s.input_mlfc_train), open_engine(s.input_mlfc_test)],
            dim=time_dim,
            coords="minimal",
            compat="override",
        ).sortby(time_dim)
    elif s.input_mlfc_test.exists():
        mlfc = open_engine(s.input_mlfc_test)

    if mlfc is not None:
        mlfc = subset_dataset(
            mlfc,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            time_start=time_start,
        )

    return (fc, an, mlfc)


def subset_dataset(
    ds: xr.Dataset,
    *,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
    time_start: str | None = None,
    freq: str = "YS",
) -> xr.Dataset:
    time_start = time_range[0] if (time_start is None and time_range is not None) else time_start

    if time_start is not None:
        ds = ensure_time_coord(ds, start=time_start, freq=freq)

    selectors = {}

    if lat_range is not None:
        lat_dim = ds.earthml.guessed_dims.latitude
        if lat_dim in ds.coords and ds.sizes[lat_dim] > 0:
            lat_coord = ds[lat_dim]
            ascending = bool(lat_coord[0] < lat_coord[-1])

            selectors[lat_dim] = (
                slice(min(lat_range), max(lat_range))
                if ascending
                else slice(max(lat_range), min(lat_range))
            )

    if lon_range is not None:
        lon_dim = ds.earthml.guessed_dims.longitude
        if lon_dim in ds.coords and ds.sizes[lon_dim] > 0:
            lon_coord = ds[lon_dim]
            lon_range = normalize_lon_range(lon_range, lon_coord)

            ascending = bool(lon_coord[0] < lon_coord[-1])

            selectors[lon_dim] = (
                slice(min(lon_range), max(lon_range))
                if ascending
                else slice(max(lon_range), min(lon_range))
            )

    time_dim = ds.earthml.guessed_dims.time
    if time_range is not None and time_dim in ds.coords:
        selectors[time_dim] = slice(*time_range)

    return ds.sel(selectors)


def valid_times_from_init_times(
    init_times: xr.DataArray | pd.DatetimeIndex,
    lead: int,
    period: Literal["hours", "days", "months", "years"],
    lead_period_offset: int = -1,
) -> xr.DataArray:
    init_values = init_times.values if isinstance(init_times, xr.DataArray) else init_times

    valid_times = [
        pd.Timestamp(t) + pd.DateOffset(**{period: lead + lead_period_offset})
        for t in init_values
    ]

    return xr.DataArray(
        valid_times,
        dims="time",
        coords={"time": init_values},
    )

def select_target_for_lead(
    an: xr.DataArray | xr.Dataset,
    lead: int,
    period: Literal["hours", "days", "months", "years"],
    fc: xr.DataArray | xr.Dataset | None = None,
    start: str | None = None,
    end: str | None = None,
    lead_period_offset: int = -1,
) -> T_Xarray:
    """Select analysis/an targets matching init time + lead."""

    if fc is not None:
        init_times = fc.time
    else:
        if start is None and end is None:
            raise ValueError("Pass either fc or start/end dates.")

        init_times = an.sel(time=slice(start, end)).time

    valid_times = valid_times_from_init_times(
        init_times,
        lead,
        period=period,
        lead_period_offset=lead_period_offset,
    )

    target_match = an.sel(time=valid_times.values)

    target_match = target_match.assign_coords(time=valid_times.time.values)

    return cast(T_Xarray, target_match)


def build_analysis_for_forecast_leadtimes(
    fc_all: xr.DataArray,
    an_raw: xr.DataArray,
    leadtime_dim: str,
    period: Literal["hours", "days", "months", "years"],
    lead_period_offset: int = -1,
) -> xr.DataArray:
    """
    Build an analysis dataset aligned with each forecast lead time.

    For every lead time in ``fc_all``, this function selects the
    corresponding analysis targets from ``an_raw`` using
    ``select_target_for_lead`` and returns a single DataArray with
    ``leadtime_dim`` as an explicit dimension.

    The resulting array can be used for verification, climatology
    computation, or training workflows that require analysis fields
    aligned with forecast lead times.

    Parameters
    ----------
    fc_all : xr.DataArray
        Forecast dataset containing a lead-time dimension.
    an_raw : xr.DataArray
        Analysis dataset from which target values are selected.
    leadtime_dim : str
        Name of the lead-time dimension.
    lead_period_offset: int
        The monthly offset for forecast leadtime naming convention.

    Returns
    -------
    xr.DataArray
        Analysis targets indexed by lead time, with dimensions
        matching the forecast-valid times for each lead.
    """
    out: list[xr.DataArray] = []

    for lead in fc_all[leadtime_dim].values:
        lead = int(lead)

        fc = fc_all.sel({leadtime_dim: lead})
        fc = common_time_range(
            fc=fc,
            an=an_raw,
            max_lead=lead,
            period=period,
            lead_period_offset=lead_period_offset,
        )

        an = select_target_for_lead(
            an=an_raw,
            lead=lead,
            fc=fc,
            period=period,
            lead_period_offset=lead_period_offset,
        )

        if leadtime_dim in an.dims:
            an = an.squeeze(leadtime_dim, drop=True)

        an = an.expand_dims({leadtime_dim: [lead]})
        out.append(an)

    return xr.concat(
        out,
        dim=leadtime_dim,
        coords="minimal",
        compat="override",
        join="inner",
        combine_attrs="override",
    )


def valid_times_from_forecast(
    fc: xr.DataArray | xr.Dataset,
    lead: int,
    period: Literal["hours", "days", "months", "years"],
    lead_period_offset: int = -1,
) -> xr.DataArray:
    return valid_times_from_init_times(
        init_times=fc.time,
        lead=lead,
        period=period,
        lead_period_offset=lead_period_offset,
    )


def common_time_range(
    fc: xr.DataArray | xr.Dataset,
    an: xr.DataArray | xr.Dataset,
    max_lead: int,
    period: Literal["hours", "days", "months", "years"],
    lead_period_offset: int = -1,
) -> T_Xarray:
    """Drop forecast initializations whose verifying an time is unavailable."""
    valid_times = valid_times_from_forecast(
        fc=fc,
        lead=max_lead,
        period=period,
        lead_period_offset=lead_period_offset,
    )
    available_an_times = set(pd.to_datetime(an.time.values))
    keep = [pd.Timestamp(t.item()) in available_an_times for t in valid_times.values]

    return cast(T_Xarray, fc.isel(time=keep))
