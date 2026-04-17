from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import earthml
import earthkit.data as ekd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from earthml.experiments.mlbc.catalog import make_catalog
from rich import print

# LEADTIME_MONTHS = ["1", "2", "3", "4", "5", "6"]
LEADTIME_MONTHS = ["1"]
# YEARS = ["1994", "1995"]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
YEARS = ["1994", "1995"]
VAR_LONG = "2m_temperature" # 2m_temperature, sea_surface_salinity, sea_surface_temperature
# MONTHS = ["01"]
# EXP = "ocean"
EXP = "atmo"
# EXP = "sst"
PLOT_REGION = make_catalog().region.natlantic

SEASONAL_ATMO_DATASET = "seasonal-monthly-single-levels"
SEASONAL_ATMO_REQUEST = {
    "variable": [VAR_LONG],
    "year": YEARS,
    "month": MONTHS,
    "product_type": ["monthly_mean"],
    "originating_centre": "cmcc",
    "system": "4",
    "leadtime_month": LEADTIME_MONTHS,
    "data_format": "netcdf",
}

SEASONAL_OCEAN_DATASET = "seasonal-monthly-ocean"
SEASONAL_OCEAN_REQUEST = {
    "variable": [VAR_LONG],
    "year": YEARS,
    "month": MONTHS,
    "forecast_type": ["hindcast"],
    "originating_centre": "cmcc",
    "system": "4",
    "leadtime_month": LEADTIME_MONTHS,
    "data_format": "netcdf",
}

ERA5_DATASET = "reanalysis-era5-single-levels-monthly-means"
ERA5_REQUEST = {
    "variable": [VAR_LONG],
    "year": YEARS,
    "month": MONTHS,
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "product_type": ["monthly_averaged_reanalysis"],
}

ORAS5_DATASET = "reanalysis-oras5"
ORAS5_REQUEST = {
    "variable": [VAR_LONG],
    "year": YEARS,
    "month": MONTHS,
    "product_type": ["consolidated"],
    "vertical_resolution": "single_level",
}


if EXP=="atmo":
    FORECAST_DATASET = SEASONAL_ATMO_DATASET
    FORECAST_REQUEST = SEASONAL_ATMO_REQUEST
    ANALYSIS_DATASET = ERA5_DATASET
    ANALYSIS_REQUEST = ERA5_REQUEST
elif EXP=="ocean":
    FORECAST_DATASET = SEASONAL_OCEAN_DATASET
    FORECAST_REQUEST = SEASONAL_OCEAN_REQUEST
    ANALYSIS_DATASET = ORAS5_DATASET
    ANALYSIS_REQUEST = ORAS5_REQUEST
elif EXP=="sst":
    FORECAST_DATASET = SEASONAL_ATMO_DATASET
    FORECAST_REQUEST = SEASONAL_ATMO_REQUEST
    ANALYSIS_DATASET = ORAS5_DATASET
    ANALYSIS_REQUEST = ORAS5_REQUEST
else:
    raise ValueError(f"Not suppoerted experiment {EXP}")


def _netcdf_safe_attrs(attrs: dict) -> dict:
    valid_scalar_types = (str, bytes, int, float, bool)
    safe_attrs: dict = {}
    for key, value in attrs.items():
        if value is None or isinstance(value, valid_scalar_types):
            safe_attrs[key] = value
        elif isinstance(value, (list, tuple)):
            if all(item is None or isinstance(item, valid_scalar_types) for item in value):
                safe_attrs[key] = value
        elif hasattr(value, "dtype"):
            safe_attrs[key] = value
    return safe_attrs


def _make_netcdf_safe(ds: xr.Dataset) -> xr.Dataset:
    ds_safe = ds.copy(deep=False)
    ds_safe.attrs = _netcdf_safe_attrs(dict(ds.attrs))

    for coord_name in ds_safe.coords:
        ds_safe[coord_name].attrs = _netcdf_safe_attrs(dict(ds_safe[coord_name].attrs))

    for var_name in ds_safe.data_vars:
        ds_safe[var_name].attrs = _netcdf_safe_attrs(dict(ds_safe[var_name].attrs))

    return ds_safe


def _print_xarray_summary(label: str, ds: xr.Dataset) -> None:
    print(f"\n=== {label} ===")
    print("dims:")
    print(dict(ds.sizes))
    print("data_vars:")
    print(list(ds.data_vars))
    print("coords:")
    print(list(ds.coords))

    for coord_name in ds.coords:
        coord = ds[coord_name]
        print(f"\ncoord {coord_name}:")
        print(" dims:")
        print(coord.dims)
        print(" dtype:")
        print(coord.dtype)
        print(" attrs:")
        print(dict(coord.attrs))

        values = coord.values.reshape(-1)
        len_val = len(values)
        print(" len:")
        print(len_val)
        if "datetime64" in str(coord.dtype):
            head = pd.to_datetime(values[-min(8, len(values)):]).tolist()
            tail = pd.to_datetime(values[-min(8, len(values)):]).tolist()
        else:
            head = values[: min(8, len(values))].tolist()
            tail = values[-min(8, len(values)):].tolist()
        print(" head:")
        print(head)
        print(" tail:")
        print(tail)
        print(f" unique vals [{len(np.unique(coord.values))}]:")
        print(np.unique(coord.values))


def _make_unique_output_path(output_dir: Path, filename: str) -> Path:
    base = Path(filename)
    request_tag = _build_request_tag()
    return output_dir / f"{base.stem}_{request_tag}{base.suffix}"


def _sanitize_name_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower()


def _build_request_tag() -> str:
    start_date = f"{min(YEARS)}{min(MONTHS)}"
    end_date = f"{max(YEARS)}{max(MONTHS)}"
    parts = [
        EXP,
        VAR_LONG,
        f"lt{LEADTIME_MONTHS[0]}",
        f"{start_date}-{end_date}",
    ]
    return "_".join(_sanitize_name_part(part) for part in parts)


def _open_cds(dataset: str, request: dict) -> xr.Dataset:
    print(f"\ndataset: {dataset}")
    print("request:")
    print(request)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*xarray will not decode the variable 'leadtime' into a timedelta64 dtype.*",
            category=FutureWarning,
            module=r"earthkit\.data\.readers\.netcdf\.fieldlist",
        )
        src = ekd.from_source("cds", dataset, **request)
    paths = [Path(index.path) for index in getattr(src, "_indexes", [])]

    if len(paths) <= 1:
        return src.to_xarray(
            decode_timedelta=True,
            data_vars="all",
            coords="minimal",
            compat="override",
        )

    with xr.open_dataset(paths[0], decode_timedelta=True) as first_ds:
        concat_dim = next(
            (dim for dim in ("forecast_reference_time", "time", "valid_time", "leadtime") if dim in first_ds.dims),
            None,
        )

    if concat_dim is None:
        raise ValueError(f"Could not find a concat dimension for {dataset}. First file: {paths[0]}")

    ds = xr.open_mfdataset(
        paths,
        combine="nested",
        concat_dim=concat_dim,
        decode_timedelta=True,
        data_vars="all",
        coords="minimal",
        compat="override",
    )
    if concat_dim in ds.dims:
        ds = ds.sortby(concat_dim)
    return ds

def _save_mean_timeseries_plot(
    seasonal_ds: xr.Dataset,
    era5_ds: xr.Dataset,
    output_path: Path,
) -> None:
    seasonal_ds = seasonal_ds.earthml.regrid_to_rectilinear(
        region=PLOT_REGION,
        resolution=0.25,
        vars_to_regrid=None,
    )
    era5_ds = era5_ds.earthml.regrid_to_rectilinear(
        region=PLOT_REGION,
        resolution=0.25,
        vars_to_regrid=None,
    )
    seasonal_da = seasonal_ds[next(iter(seasonal_ds.data_vars))]
    era5_da = era5_ds[next(iter(era5_ds.data_vars))]

    time_dim_seasonal = (
        "forecast_reference_time"
        if "forecast_reference_time" in seasonal_da.dims
        else seasonal_da.earthml.guess_time_dim
    )
    leadtime_dim_seasonal = (
        "forecastMonth"
        if "forecastMonth" in seasonal_da.dims
        else seasonal_da.earthml.guess_leadtime_dim
    )
    time_dim_era5 = era5_da.earthml.guess_time_dim

    print("Plotting, forecast da dims:", seasonal_da.dims)
    print("Plotting, analysis da dims:", era5_da.dims)

    if time_dim_seasonal is None:
        raise ValueError(f"Could not determine the forecast reference-time dimension from {seasonal_da.dims}")

    seasonal_keep_dims = {time_dim_seasonal}
    if leadtime_dim_seasonal is not None:
        seasonal_keep_dims.add(leadtime_dim_seasonal)
    seasonal_ts = seasonal_da.earthml.geo_mean(dim=[d for d in seasonal_da.dims if d not in seasonal_keep_dims])
    era5_ts = era5_da.earthml.geo_mean(dim=[d for d in era5_da.dims if d != time_dim_era5])
    if EXP == "sst":
        era5_ts = era5_ts + 273.15
        era5_ts.attrs["units"] = "K"

    if leadtime_dim_seasonal is not None and leadtime_dim_seasonal in seasonal_ts.dims:
        leadtime_values = np.asarray(seasonal_ts[leadtime_dim_seasonal].values).reshape(-1)
        if len(leadtime_values) != 1:
            raise ValueError(
                f"Expected a single leadtime for plotting, got {leadtime_values.tolist()} "
                f"on dimension {leadtime_dim_seasonal!r}"
            )
        leadtime_months = int(leadtime_values[0])
        seasonal_ts = seasonal_ts.squeeze(leadtime_dim_seasonal, drop=True)
    else:
        leadtime_months = int(LEADTIME_MONTHS[0])

    seasonal_reference_time = pd.to_datetime(seasonal_da.indexes[time_dim_seasonal].values)
    seasonal_time_shifted = pd.DatetimeIndex(
        [timestamp + pd.DateOffset(months=leadtime_months) for timestamp in seasonal_reference_time]
    )
    seasonal_ts = seasonal_ts.assign_coords(time=(time_dim_seasonal, seasonal_time_shifted.to_numpy()))

    seasonal_time = pd.to_datetime(seasonal_ts.coords["time"].values)
    era5_time = pd.to_datetime(era5_da.indexes[time_dim_era5].values)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(seasonal_time, seasonal_ts.values, label="Seasonal mean", linewidth=2)
    ax.plot(era5_time, era5_ts.values, label="Analysis mean", linewidth=2)
    ax.set_title("Spatial + Realization Mean Timeseries")
    ax.set_xlabel("Time")
    y_label = str(seasonal_ts.name or "value")
    if era5_ts.attrs.get("units") == "K":
        y_label = f"{y_label} [K]"
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect two raw CDS datasets returned by earthkit-data.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    seasonal_ds = _open_cds(FORECAST_DATASET, FORECAST_REQUEST)
    _print_xarray_summary("Seasonal Forecast Raw Object", seasonal_ds)

    era5_ds = _open_cds(ANALYSIS_DATASET, ANALYSIS_REQUEST)
    _print_xarray_summary("ERA5 Monthly Raw Object", era5_ds)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        seasonal_path = _make_unique_output_path(args.output_dir, "forecast_raw.nc")
        era5_path = _make_unique_output_path(args.output_dir, "analysis_raw.nc")
        plot_path = _make_unique_output_path(args.output_dir, "mean_timeseries.png")
        _make_netcdf_safe(seasonal_ds).to_netcdf(seasonal_path)
        _make_netcdf_safe(era5_ds).to_netcdf(era5_path)
        _save_mean_timeseries_plot(seasonal_ds, era5_ds, plot_path)
        print("\nsaved:")
        print(" ", seasonal_path)
        print(" ", era5_path)
        print(" ", plot_path)


if __name__ == "__main__":
    main()
