from dateutil.relativedelta import relativedelta
from pathlib import Path

from ..dataclasses import ProviderSpec
from .base import merge
from .registry import register_provider

@register_provider("atmo.juno.ecmwf.forecast.hourly")
def juno_forecast_hourly (
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    root_path: str | Path = "/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/1.0forecast/1h/grib/",
    concat_dim: str = "valid_time",
    engine: str = "cfgrib",
    file_path_date_format: str = "%Y%m%d",
    file_header: str = "JLS",
    file_suffix: str = "*",
    file_date_format: str = "%m%d%H%M",
    both_data_and_previous_date_in_file: bool = True,
    minus_hours: int = 1,
    plus_hours: int = 1,
    overrides: dict | None = None,
    **kw,
) -> ProviderSpec:
    base = dict(
        root_path=Path(root_path),
        engine=engine,
        file_path_date_format=file_path_date_format,
        file_header=file_header,
        file_suffix=file_suffix,
        file_date_format=file_date_format,
        both_data_and_previous_date_in_file=both_data_and_previous_date_in_file,
        lead_time=relativedelta(**{leadtime_unit: leadtime_value}),
        minus_timedelta=relativedelta(hours=minus_hours),
        plus_timedelta=relativedelta(hours=plus_hours),
        concat_dim=concat_dim,
    )
    return ProviderSpec('juno-local', merge(base, overrides, **kw))

@register_provider("atmo.juno.ecmwf.analysis.6hourly")
def juno_analysis_6hourly (
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    root_path: str | Path = "/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010/analysis/6h/grib/",
    concat_dim: str = "valid_time",
    engine: str = "cfgrib",
    file_path_date_format: str = "%Y/%m",
    file_header: str = "JLD",
    file_suffix: str = "*",
    file_date_format: str = "%m%d%H%M",
    both_data_and_previous_date_in_file: bool = True,
    minus_hours: int = 1,
    plus_hours: int = 1,
    overrides: dict | None = None,
    **kw,
) -> ProviderSpec:
    base = dict(
        root_path=Path(root_path),
        engine=engine,
        file_path_date_format=file_path_date_format,
        file_header=file_header,
        file_suffix=file_suffix,
        file_date_format=file_date_format,
        both_data_and_previous_date_in_file=both_data_and_previous_date_in_file,
        lead_time=relativedelta(hours=0),
        minus_timedelta=relativedelta(hours=minus_hours),
        plus_timedelta=relativedelta(hours=plus_hours),
        concat_dim=concat_dim,
    )
    return ProviderSpec('juno-local', merge(base, overrides, **kw))

@register_provider("atmo.earthkit.era5.reanalysis.6hourly")
def earthkit_cds_era5_single_levels (
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    split_request: bool = True,
    product_type: str = "reanalysis",
    time_dim_mode: str = "valid_time",
    chunks: dict | None = None,
    add_earthkit_attrs: bool = False,
    overrides: dict | None = None,
    **kw,
) -> ProviderSpec:
    base = dict(
        provider="cds",
        lead_time=relativedelta(hours=0),
        dataset="reanalysis-era5-single-levels",
        split_request=split_request,
        request_extra_args=dict(product_type=product_type),
        to_xarray_args=dict(
            time_dim_mode=time_dim_mode,
            chunks=chunks or {"valid_time": 1},
            add_earthkit_attrs=add_earthkit_attrs,
        ),
    )
    return ProviderSpec('earthkit', merge(base, overrides, **kw))

@register_provider("atmo.earthkit.cmcc.hindcast.monthly")
def earthkit_cmcc_monthly_hindcast_atmo (
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    originating_centre: str = "cmcc",
    system: str = "4",
    regrid_resolution: float = 0.25,
    leadtime_month: list[str] = ["1", "2", "3", "4", "5", "6"],
    split_month: int = 1,
    split_month_jump: list[str] | None = None,
    data_format: str = "grib", # netcdf (experimental)
    earthkit_cache_dir: str  = Path("/tmp/earthkit-cache/"),
    overrides: dict | None = None,
    **kw,
) -> ProviderSpec:
    if data_format == "netcdf":
        to_xarray_args = dict(
            engine="h5netcdf",
            decode_timedelta=True,
            data_vars="all",
            coords="minimal",
            compat="override",
            concat_dim="leadtime",
            combine="nested",
        )
    else:
        to_xarray_args = dict(
            engine="cfgrib",
            decode_timedelta=True,
        )
    base = dict(
        provider="cds",
        lead_time=relativedelta(**{leadtime_unit: leadtime_value}),
        dataset="seasonal-monthly-single-levels",
        regrid_resolution=regrid_resolution,
        split_request=True,
        split_month=split_month,
        split_month_jump=split_month_jump or [],
        request_type="monthly",
        request_extra_args=dict(
            product_type=["monthly_mean"],
            originating_centre=originating_centre,
            system=system,
            leadtime_month=leadtime_month,
            data_format=data_format,
        ),
        to_xarray_args=to_xarray_args,
        xarray_concat_dim="time",
        xarray_concat_extra_args=dict(coords="minimal", compat="override"),
        earthkit_cache_dir=earthkit_cache_dir,
    )
    return ProviderSpec('earthkit', merge(base, overrides, **kw))
