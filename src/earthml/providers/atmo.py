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
