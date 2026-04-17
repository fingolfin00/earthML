from pathlib import Path

from dateutil.relativedelta import relativedelta
from ...base import Leadtime, LeadtimeUnit
from .. import SourceConfigContainer, RegridConfig, JunoLocalSourceConfig, JunoLocalSourceFileNameConfig, EarthkitSourceConfig

from .registry import register_source_config_provider as register_provider

@register_provider("atmo.juno.ecmwf.forecast.hourly")
def juno_forecast_hourly(
    var_name: str,
    leadtime_value: int,
    leadtime_unit: LeadtimeUnit,
    root_path: str | Path = "/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/1.0forecast/1h/grib/",
    engine: str = "cfgrib",
    cfgrib_idx_path: str = "",
    regrid_resolution: float | None = None,
    file_path_date_format: str = "%Y%m%d",
    file_header: str = "JLS",
    file_suffix: str = "*",
    file_date_format: str = "%m%d%H%M",
    both_data_and_previous_date_in_file: bool = True,
    minus_hours: int = 1,
    plus_hours: int = 1,
    file_open_workers: int | None = 1,
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, leadtime_value)

    return SourceConfigContainer(
        source="juno-local",
        config=JunoLocalSourceConfig(
            leadtime=leadtime,
            root_path=Path(root_path),
            engine=engine,
            cfgrib_idx_path=cfgrib_idx_path,
            file_open_workers=file_open_workers,
            file_name_config=JunoLocalSourceFileNameConfig(
                file_path_date_format=file_path_date_format,
                file_header=file_header,
                file_suffix=file_suffix,
                file_date_format=file_date_format,
                both_data_and_previous_date_in_file=both_data_and_previous_date_in_file,
                realizations=1,
                minus_timedelta=relativedelta(hours=minus_hours),
                plus_timedelta=relativedelta(hours=plus_hours),
            ),
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
        ),
    )


@register_provider("atmo.juno.ecmwf.analysis.6hourly")
def juno_analysis_6hourly(
    var_name: str,
    leadtime_value: int, # ignored
    leadtime_unit: LeadtimeUnit,
    root_path: str | Path = "/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010/analysis/6h/grib/",
    engine: str = "cfgrib",
    cfgrib_idx_path: str = "",
    regrid_resolution: float | None = None,
    file_path_date_format: str = "%Y/%m",
    file_header: str = "JLD",
    file_suffix: str = "*",
    file_date_format: str = "%m%d%H%M",
    both_data_and_previous_date_in_file: bool = True,
    minus_hours: int = 1,
    plus_hours: int = 1,
    file_open_workers: int | None = 1,
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, 0)

    return SourceConfigContainer(
        source="juno-local",
        config=JunoLocalSourceConfig(
            leadtime=leadtime,
            root_path=Path(root_path),
            engine=engine,
            cfgrib_idx_path=cfgrib_idx_path,
            file_open_workers=file_open_workers,
            file_name_config=JunoLocalSourceFileNameConfig(
                file_path_date_format=file_path_date_format,
                file_header=file_header,
                file_suffix=file_suffix,
                file_date_format=file_date_format,
                both_data_and_previous_date_in_file=both_data_and_previous_date_in_file,
                realizations=1,
                minus_timedelta=relativedelta(hours=minus_hours),
                plus_timedelta=relativedelta(hours=plus_hours),
            ),
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
        ),
    )


@register_provider("atmo.earthkit.era5.reanalysis.6hourly")
def earthkit_cds_era5_single_levels(
    var_name: str,
    leadtime_value: int, # ignored
    leadtime_unit: str,
    product_type: str = "reanalysis",
    regrid_resolution: float = 0.25,
    select_area_after_request: bool = True,
    earthkit_cache_dir: str  = Path("/tmp/earthkit-cache/"),
    convert_unit: dict | None = None, # dict of var_name: (func, target_unit) to convert variable unit (e.g. {"temperature": (lambda x: x - 273.15, "C")})
    time_dim_mode: str = "valid_time",
    add_earthkit_attrs: bool = False,
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, 0)

    return SourceConfigContainer(
        source="earthkit",
        config=EarthkitSourceConfig(
            leadtime=leadtime,
            provider="cds",
            dataset="reanalysis-era5-single-levels",
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
            split_request=True,
            select_area_after_request=select_area_after_request,
            request_type="monthly",
            request_extra_args=dict(
                product_type=product_type,
                vertical_resolution="single_level",
            ),
            to_xarray_args=dict(
                engine="h5netcdf", # h5netcdf, netcdf4
                time_dim_mode=time_dim_mode, # TODO check if still used
                decode_timedelta=True,
                data_vars="all",
                combine="by_coords",
                coords="minimal",
                compat="override",
                parallel=True,
                add_earthkit_attrs=add_earthkit_attrs, # TODO check if still used
                # chunks=None, # disable dask
            ),
            xarray_concat_dim=None,
            xarray_concat_extra_args=dict(coords="minimal", compat="override"),
            snap_time_index="same", # account for ERA5 convention of start of the month at 12:00 (snap it to 00:00)
            convert_unit=convert_unit,
            earthkit_cache_dir=earthkit_cache_dir,
        ), 
    )


@register_provider("atmo.earthkit.cmcc.hindcast.monthly")
def earthkit_cmcc_monthly_hindcast_atmo(
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    originating_centre: str = "cmcc",
    system: str = "4",
    regrid_resolution: float = 0.25,
    select_area_after_request: bool = True,
    leadtime_month: list[str] = ["1", "2", "3", "4", "5", "6"],
    split_month: int = 12, # 12: one single request, 1: 12 requests (one per month)
    split_month_jump: list[str] | None = None,
    data_format: str = "grib", # netcdf (experimental)
    earthkit_cache_dir: str  = Path("/tmp/earthkit-cache/"),
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, leadtime_value)
    
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

    return SourceConfigContainer(
        source="earthkit",
        config=EarthkitSourceConfig(
            leadtime=leadtime,
            provider="cds",
            dataset="seasonal-monthly-single-levels",
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
            split_request=True,
            split_month=split_month,
            split_month_jump=split_month_jump or [],
            select_area_after_request=select_area_after_request,
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
            snap_time_index="nearest", # account for SPS4 atmo use of "valid_time" to set the forecast target time
            earthkit_cache_dir=earthkit_cache_dir,
        ),
    )

@register_provider("atmo.earthkit.era5.reanalysis.monthly")
def earthkit_cds_era5_single_levels_monthly(
    var_name: str,
    leadtime_value: int, # ignored
    leadtime_unit: str,
    product_type: str = "reanalysis",
    regrid_resolution: float = 0.25,
    select_area_after_request: bool = True,
    earthkit_cache_dir: str  = Path("/tmp/earthkit-cache/"),
    convert_unit: dict | None = None, # dict of var_name: (func, target_unit) to convert variable unit (e.g. {"temperature": (lambda x: x - 273.15, "C")})
    add_earthkit_attrs: bool = False,
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, 0)

    return SourceConfigContainer(
        source="earthkit",
        config=EarthkitSourceConfig(
            leadtime=leadtime,
            provider="cds",
            dataset="reanalysis-era5-single-levels-monthly-means",
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
            split_request=True,
            select_area_after_request=select_area_after_request,
            request_type="monthly",
            request_extra_args=dict(
                product_type=product_type,
                vertical_resolution="single_level",
            ),
            to_xarray_args=dict(
                engine="cfgrib", # cfgrib, earthkit
                decode_timedelta=True,
                # chunks=None, # disable dask
            ),
            xarray_concat_dim=None,
            xarray_concat_extra_args=dict(coords="minimal", compat="override"),
            convert_unit=convert_unit,
            earthkit_cache_dir=earthkit_cache_dir,
        ),
    )
