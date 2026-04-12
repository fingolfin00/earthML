from pathlib import Path
from dateutil.relativedelta import relativedelta

from .. import SourceConfigContainer, RegridConfig, JunoLocalSourceConfig, JunoLocalSourceFileNameConfig, EarthkitSourceConfig, CopernicusmarineSourceConfig

from .registry import register_source_config_provider as register_provider


# Seasonal ocean
@register_provider("ocean.juno.cmcc.hindcast") # monthly, 6-hourly
def juno_monthly_hindcast_ocean_netcdf(
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    root_path: str = "/work/cmcc/cp1/CMCC-CM/archive/C3S/",
    realizations: str | int = "all",
    engine: str = "h5netcdf", # h5netcdf, netcdf4
    regrid_resolution: float = 0.25,
    file_path_date_format: str = "%Y%m",
    file_header: str = "cmcc_CMCC-CM3-v20231101_hindcast_S",
    file_path_var_prefix: str = "00_ocean_mon_ocean2d_", # _ocean_6hr_surface_
    file_date_format: str = "%Y%m%d",
    both_data_and_previous_date_in_file: bool = False,
) -> SourceConfigContainer:
    members = str(realizations) if isinstance(realizations, int) else "*"
    leadtime = relativedelta(**{leadtime_unit: leadtime_value})

    return SourceConfigContainer(
        source="juno-local",
        config=JunoLocalSourceConfig(
            leadtime=leadtime,
            root_path=root_path,
            engine=engine,
            file_name_config=JunoLocalSourceFileNameConfig(
                file_path_date_format=file_path_date_format,
                file_header=file_header,
                file_suffix=f"{file_path_var_prefix}{var_name}_r{members}i00p00.nc", # *: mon/6hr, *: ocean2d/surface
                file_date_format=file_date_format,
                both_data_and_previous_date_in_file=both_data_and_previous_date_in_file,
                realizations=realizations,
            ),
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
        ),
    )


@register_provider("ocean.earthkit.cmcc.hindcast.monthly")
def earthkit_cmcc_monthly_hindcast_ocean(
    var_name: str,
    leadtime_value: int, # months
    leadtime_unit: str,
    originating_centre: str = "cmcc",
    system: str = "4",
    regrid_resolution: float = 0.25,
    select_area_after_request: bool = True,
    split_month: int = 1,
    split_month_jump: list[str] | None = None, # ['03', '04', '06', '07'],
    earthkit_cache_dir: str  = Path("/tmp/earthkit-cache/"),
) -> SourceConfigContainer:
    leadtime = relativedelta(**{leadtime_unit: leadtime_value})

    return SourceConfigContainer(
        source="earthkit",
        config=EarthkitSourceConfig(
            leadtime=leadtime,
            provider="cds",
            dataset="seasonal-monthly-ocean",
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
                forecast_type="hindcast",
                originating_centre=originating_centre,
                system=system,
            ),
            to_xarray_args=dict(
                engine="h5netcdf", # h5netcdf, netcdf4
                decode_timedelta=True,
                data_vars="all",
                coords="minimal",
                compat="override",
                concat_dim="leadtime",
                combine="nested",
                parallel=True,
                # chunks=None, # disable dask
            ),
            xarray_concat_dim="time",
            xarray_concat_extra_args=dict(coords="minimal", compat="override"),
            earthkit_cache_dir=earthkit_cache_dir,
        ),
    )


@register_provider("ocean.earthkit.oras5.reanalysis.monthly")
def earthkit_cds_oras5(
    var_name: str,
    leadtime_value: int, # ignored
    leadtime_unit: str,
    product_type: str = "consolidated", # or "operational"
    regrid_resolution: float = 0.25,
    select_area_after_request: bool = True,
    earthkit_cache_dir: str  = Path("/tmp/earthkit-cache/"),
    convert_unit: dict | None = None, # dict of var_name: (func, target_unit) to convert variable unit (e.g. {"temperature": (lambda x: x - 273.15, "C")})
) -> SourceConfigContainer:
    leadtime = relativedelta(**{leadtime_unit: 0})

    return SourceConfigContainer(
        source="earthkit",
        config=EarthkitSourceConfig(
            leadtime=leadtime,
            provider="cds",
            dataset="reanalysis-oras5",
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
                decode_timedelta=True,
                data_vars="all",
                combine="by_coords",
                coords="minimal",
                compat="override",
                parallel=True,
                # chunks=None, # disable dask
            ),
            xarray_concat_dim=None,
            xarray_concat_extra_args=dict(coords="minimal", compat="override"),
            convert_unit=convert_unit,
            earthkit_cache_dir=earthkit_cache_dir,
        ), 
    )


# Weather ocean, copernicusmarine, hourly, analysis before TODAY
@register_provider("ocean.copernicusmarine.gopaf.analysis.hourly")
def copernicusmarine_global_ocean_physics_analysis_hourly(
    var_name: str,
    leadtime_value: int, # unused
    leadtime_unit: str,
    username: str,
    password: str,
    regrid_resolution: float = 0.08,
    select_area_after_request: bool = True,
    # copernicusmarine_cache_dir: str  = Path("/tmp/copernicusmarine-cache/"),
    convert_unit: dict | None = None, # dict of var_name: (func, target_unit) to convert variable unit (e.g. {"temperature": (lambda x: x - 273.15, "C")})
) -> SourceConfigContainer:
    leadtime = relativedelta(**{leadtime_unit: 0})

    return SourceConfigContainer(
        source="copernicusmarine",
        config=CopernicusmarineSourceConfig(
            leadtime=leadtime,
            username=username,
            password=password,
            dataset="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m", # hourly
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
            select_area_after_request=select_area_after_request,
            convert_unit=convert_unit,
            # cache_dir=copernicusmarine_cache_dir,
        ),
    )

# Weather ocean, copernicusmarine, daily, analysis before TODAY
@register_provider("ocean.copernicusmarine.gopaf.analysis.daily")
def copernicusmarine_global_ocean_physics_analysis_daily(
    var_name: str,
    leadtime_value: int, # unused
    leadtime_unit: str,
    username: str,
    password: str,
    regrid_resolution: float = 0.08,
    select_area_after_request: bool = True,
    # copernicusmarine_cache_dir: str  = Path("/tmp/copernicusmarine-cache/"),
    convert_unit: dict | None = None, # dict of var_name: (func, target_unit) to convert variable unit (e.g. {"temperature": (lambda x: x - 273.15, "C")})
) -> SourceConfigContainer:
    leadtime = relativedelta(**{leadtime_unit: 0})

    if var_name in ("thetao", "so"):
        var_dataset_name = f"-{var_name}"
    else:
        var_dataset_name = ""

    return SourceConfigContainer(
        source="copernicusmarine",
        config=CopernicusmarineSourceConfig(
            leadtime=leadtime,
            username=username,
            password=password,
            dataset=f"cmems_mod_glo_phy{var_dataset_name}_anfc_0.083deg_P1D-m", # hourly
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
            select_area_after_request=select_area_after_request,
            convert_unit=convert_unit,
            # cache_dir=copernicusmarine_cache_dir,
        ),
    )

# Weather ocean, local Juno, Atlantic box lon: [-18, 1], lat: [30, 46]
@register_provider("ocean.juno.gopaf.forecast.daily.atlantic")
def juno_global_ocean_physics_forecast_daily_atlantic(
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    root_path: str | Path = "/data/inputs/METOCEAN/rolling/model/ocean/CMS/GlobOce/MERCATOR/1.0forecast/day/",
    engine: str = "h5netcdf",
    cfgrib_idx_path: str = "",
    regrid_resolution: float = 0.08,
    file_path_date_format: str = "%Y%m%d",
    file_header: str = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m_MEDATL_*",
    file_suffix: str = ".nc",
    file_date_format: str = "%Y%m%d",
    both_data_and_previous_date_in_file: bool = True,
) -> SourceConfigContainer:
    leadtime = relativedelta(**{leadtime_unit: leadtime_value})

    return SourceConfigContainer(
        source="juno-local",
        config=JunoLocalSourceConfig(
            leadtime=leadtime,
            root_path=Path(root_path),
            engine=engine,
            cfgrib_idx_path=cfgrib_idx_path,
            file_name_config=JunoLocalSourceFileNameConfig(
                file_path_date_format=file_path_date_format,
                file_header=file_header,
                file_suffix=file_suffix,
                file_date_format=file_date_format,
                both_data_and_previous_date_in_file=both_data_and_previous_date_in_file,
                date_order="current_previous",
                realizations=1,
            ),
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
        ),
    )
