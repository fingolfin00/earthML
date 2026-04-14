from __future__ import annotations

import argparse
from pathlib import Path

import earthkit.data as ekd
import pandas as pd
import xarray as xr


def _print_xarray_summary(label: str, ds: xr.Dataset) -> None:
    print(f"\n=== {label} ===")
    print("dims:", dict(ds.sizes))
    print("data_vars:", list(ds.data_vars))
    print("coords:", list(ds.coords))

    for coord_name in ("time", "valid_time", "forecast_reference_time", "step", "number"):
        if coord_name not in ds.coords:
            continue
        coord = ds[coord_name]
        print(f"\ncoord {coord_name}:")
        print(" dims:", coord.dims)
        print(" dtype:", coord.dtype)
        print(" attrs:", dict(coord.attrs))

        if "datetime64" in str(coord.dtype):
            values = pd.to_datetime(coord.values.reshape(-1))
            print(" head:", values[: min(8, len(values))].tolist())
        else:
            flat = coord.values.reshape(-1)
            print(" head:", flat[: min(8, len(flat))].tolist())


def _coord_head(ds: xr.Dataset, coord_name: str) -> list:
    if coord_name not in ds.coords:
        return []
    coord = ds[coord_name]
    values = coord.values.reshape(-1)
    if "datetime64" in str(coord.dtype):
        return pd.to_datetime(values[: min(8, len(values))]).tolist()
    return values[: min(8, len(values))].tolist()


def _print_direct_comparison(seasonal_ds: xr.Dataset, era5_ds: xr.Dataset) -> None:
    print("\n=== Direct Comparison ===")
    seasonal_time = _coord_head(seasonal_ds, "time")
    seasonal_valid_time = _coord_head(seasonal_ds, "valid_time")
    era5_time = _coord_head(era5_ds, "time")
    era5_valid_time = _coord_head(era5_ds, "valid_time")

    print("seasonal time       :", seasonal_time[:4])
    print("seasonal valid_time :", seasonal_valid_time[:4])
    print("era5 time           :", era5_time[:4])
    print("era5 valid_time     :", era5_valid_time[:4])

    if seasonal_valid_time and era5_time:
        same_clock = all(a == b for a, b in zip(seasonal_valid_time[:4], era5_time[:4]))
        print("seasonal valid_time == era5 time :", same_clock)

    if seasonal_time and seasonal_valid_time:
        delta = seasonal_valid_time[0] - seasonal_time[0]
        print("seasonal lead from time to valid_time :", delta)

    if era5_time and era5_valid_time:
        delta = era5_valid_time[0] - era5_time[0]
        print("era5 valid_time - time :", delta)

    if seasonal_valid_time and era5_time:
        delta = era5_time[0] - seasonal_valid_time[0]
        print("era5 time - seasonal valid_time :", delta)


def _seasonal_request(
    variable: str,
    year: str,
    month: str,
    leadtime_month: str,
    centre: str,
    system: str,
) -> dict:
    return dict(
        variable=[variable],
        year=[year],
        month=[month],
        product_type=["monthly_mean"],
        originating_centre=centre,
        system=system,
        leadtime_month=[leadtime_month],
        data_format="grib",
    )


def _era5_request(
    variable: str,
    year: str,
    month: str,
    product_type: str,
) -> dict:
    return dict(
        variable=[variable],
        year=[year],
        month=[month],
        product_type=product_type,
        vertical_resolution="single_level",
    )


def _target_month(year: str, month: str, leadtime_month: str) -> tuple[str, str]:
    init = pd.Timestamp(f"{year}-{month}-01")
    target = init + pd.DateOffset(months=int(leadtime_month))
    return f"{target.year:04d}", f"{target.month:02d}"


def _open_cds(dataset: str, request: dict) -> xr.Dataset:
    src = ekd.from_source("cds", dataset, **request)
    return src.to_xarray()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch one seasonal CDS forecast object and one ERA5 monthly object, "
            "then print their native time conventions for inspection."
        )
    )
    parser.add_argument("--variable", default="2m_temperature")
    parser.add_argument("--forecast-year", default="1993")
    parser.add_argument("--forecast-month", default="06")
    parser.add_argument("--leadtime-month", default="1")
    parser.add_argument("--centre", default="cmcc")
    parser.add_argument("--system", default="4")
    parser.add_argument("--era5-product-type", default="reanalysis")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    target_year, target_month = _target_month(
        args.forecast_year,
        args.forecast_month,
        args.leadtime_month,
    )

    seasonal_request = _seasonal_request(
        variable=args.variable,
        year=args.forecast_year,
        month=args.forecast_month,
        leadtime_month=args.leadtime_month,
        centre=args.centre,
        system=args.system,
    )
    era5_request = _era5_request(
        variable=args.variable,
        year=target_year,
        month=target_month,
        product_type=args.era5_product_type,
    )

    print("seasonal dataset: seasonal-monthly-single-levels")
    print("seasonal request:", seasonal_request)
    seasonal_ds = _open_cds("seasonal-monthly-single-levels", seasonal_request)
    _print_xarray_summary("Seasonal Forecast Raw Object", seasonal_ds)

    print("\nera5 dataset: reanalysis-era5-single-levels-monthly-means")
    print("era5 request:", era5_request)
    era5_ds = _open_cds("reanalysis-era5-single-levels-monthly-means", era5_request)
    _print_xarray_summary("ERA5 Monthly Raw Object", era5_ds)
    _print_direct_comparison(seasonal_ds, era5_ds)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        seasonal_path = args.output_dir / "seasonal_raw.nc"
        era5_path = args.output_dir / "era5_raw.nc"
        seasonal_ds.to_netcdf(seasonal_path)
        era5_ds.to_netcdf(era5_path)
        print("\nsaved:")
        print(" ", seasonal_path)
        print(" ", era5_path)


if __name__ == "__main__":
    main()
