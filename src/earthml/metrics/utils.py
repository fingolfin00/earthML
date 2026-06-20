from typing import Sequence, TypeVar, cast

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from dask.diagnostics.progress import ProgressBar

from ..settings import Settings


T_Xarray = TypeVar("T_Xarray", xr.DataArray, xr.Dataset)


def safe_percent(num: xr.DataArray, den: xr.DataArray) -> xr.DataArray:
    return xr.where(den != 0, 100 * num / den, np.nan)

def safe_div(num: xr.DataArray, den: xr.DataArray) -> xr.DataArray:
    return xr.where(den != 0, num / den, np.nan)


def get_experiment_configs(
    experiments_root: str | Path,
    variables: Sequence | None = None,
    regions: Sequence | None = None,
) -> list[Settings]:
    config_paths = list(Path(experiments_root).rglob("config.json"))

    matching_settings: list[Settings] = []

    for config_path in config_paths:
        s = Settings.from_json(config_path)

        if variables is not None and s.var_fc not in variables:
            continue

        if regions is not None and s.region_name not in regions:
            continue

        matching_settings.append(s)

    return matching_settings


def get_and_subset_datasets(
    s: Settings,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset | None]:
    fc_da = open_zarr_var(s.input_fc, s.var_fc)
    an_da = open_zarr_var(s.input_an, s.var_an)

    leadtime_dim = fc_da.earthml.guessed_dims.leadtime
    lon_dim = fc_da.earthml.guessed_dims.longitude
    lat_dim = fc_da.earthml.guessed_dims.latitude

    an_da = build_analysis_for_forecast_leadtimes(fc_da, an_da, leadtime_dim)

    fc = subset_dataset(
        fc_da.to_dataset(name=s.var_fc),
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=time_range,
    )

    an = subset_dataset(
        an_da.to_dataset(name=s.var_an),
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=time_range,
    )

    # Regrid analysis into forecast
    an = an.interp(
        {
            lat_dim: fc[lat_dim],
            lon_dim: fc[lon_dim],
        }
    )

    mlfc = None
    if s.input_mlfc_train.exists() and s.input_mlfc_test.exists():
        mlfc = xr.concat(
            [open_zarr(s.input_mlfc_train), open_zarr(s.input_mlfc_test)],
            dim="time",
            coords="minimal",
            compat="override",
        ).sortby("time")
    elif s.input_mlfc_test.exists():
        mlfc = open_zarr(s.input_mlfc_test)

    if mlfc is not None:
        mlfc = subset_dataset(
            mlfc,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
        )

    return (fc, an, mlfc)


def calculate_climatology(
    da: xr.DataArray | xr.Dataset,
    time_dim = "time",
    clim_period: str = "month",
) -> T_Xarray:
    return cast(T_Xarray, da.groupby(f"{time_dim}.{clim_period}").mean(time_dim))

def calculate_save_and_subset_climatologies(
    s: Settings,
    force: bool = False,
    time_dim: str = "time",
    clim_period: str = "month",
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset | None]:
    # Original forecast climatology
    fc_path = s.input_dir / f"{s.model_fc}_{s.var_fc}.zarr"
    fc_clim_path = s.input_clim_dir / f"{s.model_fc}_{s.var_fc}_train_clim.zarr"

    # Analysis climatology
    an_path = s.input_dir / f"{s.model_an}_{s.var_an}.zarr"
    an_clim_path = s.input_clim_dir / f"{s.model_an}_{s.var_an}_train_clim.zarr"

    if not fc_path.exists():
        raise FileNotFoundError(f"Couldn't find forecast dataset {fc_path}")
    if not an_path.exists():
        raise FileNotFoundError(f"Couldn't find analysis dataset {an_path}")

    # Corrected forecast climatology
    train_pred_path = s.output_dir / "train_corrected.zarr"
    train_pred_clim_path = s.output_clim_dir / "train_corrected_clim.zarr"

    print(f"Get climatologies ({s.train_start} - {s.train_end}) for experiment {s.output_name}")

    s.make_dirs()

    mlfc_clim = None
    if train_pred_path.exists():
        should_compute = force or not train_pred_clim_path.exists()
        if should_compute:
            print("Save ML-corrected forecast climatology to:", train_pred_clim_path.name)

            train_pred = open_zarr_var(train_pred_path, s.var_fc)

            with ProgressBar():
                mlfc_clim = calculate_climatology(train_pred, time_dim=time_dim, clim_period=clim_period).compute()

            mlfc_clim.to_dataset(name=s.var_fc).to_zarr(
                train_pred_clim_path,
                mode="w",
                consolidated=False,
                zarr_format=2,
            )
        else:
            mlfc_clim = open_zarr_var(train_pred_clim_path, s.var_fc)
    else:
        print(f"Skipping ML-corrected forecast climatology")

    should_compute = force or not fc_clim_path.exists() or not an_clim_path.exists()
    if should_compute:
        print("Save original forecast climatology to:", fc_clim_path.name)

        fc = open_zarr_var(fc_path, s.var_fc)
        fc_train = fc.sel({time_dim: slice(s.train_start, s.train_end)})

        with ProgressBar():
            print("Calculate original forecast climatology")
            fc_clim = calculate_climatology(fc_train).compute()

        fc_clim.to_dataset(name=s.var_fc).to_zarr(
            fc_clim_path,
            mode="w",
            consolidated=False,
            zarr_format=2,
        )

        print("Save analysis climatology to:", an_clim_path.name)

        an = open_zarr_var(an_path, s.var_an)

        an_clims: list[xr.DataArray] = []

        leadtime_dim = fc.earthml.guessed_dims.leadtime
        for lead in fc[leadtime_dim].values:
            lead = int(lead)

            fc_train_lead = fc.sel(
                indexers={
                    leadtime_dim: lead,
                    time_dim: slice(s.train_start, s.train_end),
                }
            )

            an_train_lead = select_target_for_lead(an, lead, fc_train_lead)

            an_clims.append(
                calculate_climatology(an_train_lead).expand_dims(
                    {leadtime_dim: [lead]}
                )
            )

        with ProgressBar():
            print("Calculate target climatology")
            an_clim = xr.concat(
                an_clims,
                dim=leadtime_dim,
                coords="minimal",
                compat="override",
            ).compute()

        an_clim.to_dataset(name=s.var_an).to_zarr(
            an_clim_path,
            mode="w",
            consolidated=False,
            zarr_format=2,
        )
    else:
        fc_clim = open_zarr_var(fc_clim_path, s.var_fc)
        an_clim = open_zarr_var(an_clim_path, s.var_an)


    lon_dim = fc_clim.earthml.guessed_dims.longitude
    lat_dim = fc_clim.earthml.guessed_dims.latitude

    an_clim = an_clim.interp(
        {
            lat_dim: fc_clim[lat_dim],
            lon_dim: fc_clim[lon_dim],
        }
    )

    return (
        subset_dataset(
            fc_clim.to_dataset(name=s.var_fc),
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
        ),
        subset_dataset(
            an_clim.to_dataset(name=s.var_an),
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
        ),
        subset_dataset(
            mlfc_clim.to_dataset(name=s.var_fc),
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
        ) if mlfc_clim is not None else None,
    )


def valid_times_from_init_times(
    init_times: xr.DataArray | pd.DatetimeIndex,
    lead: int,
) -> xr.DataArray:
    s = Settings()

    init_values = init_times.values if isinstance(init_times, xr.DataArray) else init_times

    valid_times = [
        pd.Timestamp(t) + pd.DateOffset(months=lead + s.lead_month_offset)
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
    fc: xr.DataArray | xr.Dataset | None = None,
    start: str | None = None,
    end: str | None = None,
) -> T_Xarray:
    """Select analysis/an targets matching init time + lead."""

    if fc is not None:
        init_times = fc.time
    else:
        if start is None and end is None:
            raise ValueError("Pass either fc or start/end dates.")

        init_times = an.sel(time=slice(start, end)).time

    valid_times = valid_times_from_init_times(init_times, lead)

    target_match = an.sel(time=valid_times.values)

    target_match = target_match.assign_coords(time=valid_times.time.values)

    return cast(T_Xarray, target_match)

def build_analysis_for_forecast_leadtimes(
    fc_all: xr.DataArray,
    an_raw: xr.DataArray,
    leadtime_dim: str,
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
        fc = common_time_range(fc, an_raw, max_lead=lead)

        an = select_target_for_lead(
            an=an_raw,
            lead=lead,
            fc=fc,
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
) -> xr.DataArray:
    return valid_times_from_init_times(fc.time, lead)


def common_time_range(
    fc: xr.DataArray | xr.Dataset,
    an: xr.DataArray | xr.Dataset,
    max_lead: int,
) -> T_Xarray:
    """Drop forecast initializations whose verifying an time is unavailable."""
    valid_times = valid_times_from_forecast(fc, max_lead)
    available_an_times = set(pd.to_datetime(an.time.values))
    keep = [pd.Timestamp(t.item()) in available_an_times for t in valid_times.values]

    return cast(T_Xarray, fc.isel(time=keep))


def subset_dataset(
    ds: xr.Dataset,
    *,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    time_range: tuple[str, str] | None = None,
) -> xr.Dataset:
    selectors = {}

    if lat_range is not None:
        lat = "latitude" if "latitude" in ds.dims else "lat"
        if lat in ds.coords:
            selectors[lat] = slice(*lat_range)

    if lon_range is not None:
        lon = "longitude" if "longitude" in ds.dims else "lon"
        if lon in ds.coords:
            selectors[lon] = slice(*lon_range)

    if time_range is not None and "time" in ds.coords:
        selectors["time"] = slice(*time_range)

    return ds.sel(selectors)


def aggregate_leadtime_da(
    da: xr.DataArray,
    windows: dict[str, list[int]],
    leadtime_dim: str = "leadtime",
    leadtime_agg_coord: str = "seasonal_leadtime",
) -> xr.DataArray:
    """
    Average data over predefined lead-time windows.

    For each window, computes the mean across the selected lead times and
    concatenates the results along the lead-time dimension. Window labels are
    stored as a coordinate given by ``agg_leadtime_coord``.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a lead-time dimension.
    windows : dict[str, list[int]]
        Mapping of window labels to lead times.
    leadtime_dim : str, default="leadtime"
        Name of the lead-time dimension.
    leadtime_agg_coord : str, default="seasonal_leadtime"
        Name of the coordinate containing window labels.

    Returns
    -------
    xr.DataArray
        Data aggregated over lead-time windows.
    """
    available = set(int(x) for x in da[leadtime_dim].values)

    out: list[xr.DataArray] = []
    labels: list[str] = []
    centers: list[float] = []

    for name, leads in windows.items():
        leads = [int(x) for x in leads]

        if not set(leads).issubset(available):
            continue

        x = da.sel({leadtime_dim: leads}).mean(leadtime_dim)

        labels.append(name)
        centers.append(float(np.mean(leads)))
        out.append(x)

    if not out:
        raise ValueError("No valid lead windows found.")

    result = xr.concat(
        out,
        dim=xr.IndexVariable(leadtime_dim, centers),
        coords="minimal",
        compat="override",
        join="inner",
        combine_attrs="override",
    )

    result = result.assign_coords({
        leadtime_agg_coord: (leadtime_dim, np.asarray(labels, dtype="U")),
    })

    return result

def aggregate_leadtime_ds(
    ds: xr.Dataset,
    windows: dict[str, list[int]],
    leadtime_dim: str = "leadtime",
    leadtime_agg_coord: str = "seasonal_leadtime",
) -> xr.Dataset:
    return xr.Dataset(
        {
            var: (
                aggregate_leadtime_da(ds[var], windows, leadtime_dim, leadtime_agg_coord)
                if leadtime_dim in ds[var].dims
                else ds[var]
            )
            for var in ds.data_vars
        },
        attrs=ds.attrs,
    )

def convert_to_da_list(
    ds: xr.DataArray | xr.Dataset | Sequence[xr.DataArray | xr.Dataset | None] | None,
    var: str,
) -> list[xr.DataArray]:
    if ds is None:
        return []

    if isinstance(ds, xr.DataArray):
        return [ds]

    if isinstance(ds, xr.Dataset):
        return [ds[var]]

    if isinstance(ds, Sequence) and not isinstance(ds, (str, bytes)):
        result: list[xr.DataArray] = []

        for d in ds:
            if d is None:
                continue
            if isinstance(d, xr.Dataset):
                result.append(d[var])
            elif isinstance(d, xr.DataArray):
                result.append(d)
            else:
                raise TypeError(f"Type {type(d)} not supported.")

        return result

    raise TypeError(f"Type {type(ds)} not supported.")


def open_zarr(
    path: str | Path,
) -> xr.Dataset:
    return xr.open_zarr(path, consolidated=False)

def open_zarr_var(
    path: str | Path,
    var: str
) -> xr.DataArray:
    return open_zarr(path)[var]
