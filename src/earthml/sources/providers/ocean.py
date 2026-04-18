from typing import Literal, Any, TypeAlias

from pathlib import Path

from dateutil.relativedelta import relativedelta

from ...base import Leadtime, LeadtimeUnit
from .. import SourceConfigContainer, RegridConfig, JunoLocalSourceConfig, JunoLocalSourceFileNameConfig, JunoLocalSourcePathConfig, EarthkitSourceConfig, CopernicusmarineSourceConfig

from .registry import register_source_config_provider as register_provider


# Seasonal ocean
@register_provider("ocean.juno.cmcc.hindcast") # monthly, 6-hourly
def juno_monthly_hindcast_ocean_netcdf(
    var_name: str,
    leadtime_value: int,
    leadtime_unit: LeadtimeUnit,
    root_path: str = "/work/cmcc/cp1/CMCC-CM/archive/C3S/",
    realizations: str | int = "all",
    engine: str = "h5netcdf", # h5netcdf, netcdf4
    regrid_resolution: float = 0.25,
    file_path_date_format: str = "%Y%m",
    file_header: str = "cmcc_CMCC-CM3-v20231101_hindcast_S",
    file_path_var_prefix: str = "00_ocean_mon_ocean2d_", # _ocean_6hr_surface_
    file_date_format: str = "%Y%m%d",
    both_data_and_previous_date_in_file: bool = False,
    file_open_workers: int | None = 1,
) -> SourceConfigContainer:
    members = str(realizations) if isinstance(realizations, int) else "*"
    leadtime = Leadtime("leadtime", leadtime_unit, leadtime_value)

    return SourceConfigContainer(
        source="juno-local",
        config=JunoLocalSourceConfig(
            leadtime=leadtime,
            root_path=root_path,
            engine=engine,
            file_open_workers=file_open_workers,
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
    leadtime_unit: LeadtimeUnit,
    originating_centre: str = "cmcc",
    system: str = "4",
    regrid_resolution: float = 0.25,
    select_area_after_request: bool = True,
    leadtime_month: list[str] = ["1", "2", "3", "4", "5", "6"],
    split_request: bool = True,
    split_month: int = 6, # 12: one single request, 1: 12 requests (one per month)
    split_month_jump: list[str] | None = None, # ['03', '04', '06', '07'],
    data_format: str = "grib", # netcdf (experimental)
    earthkit_cache_dir: str | Path = Path("/tmp/earthkit-cache/"),
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, leadtime_value)

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
            split_request=split_request,
            split_month=split_month,
            split_month_jump=split_month_jump or [],
            select_area_after_request=select_area_after_request,
            request_type="monthly",
            request_extra_args=dict(
                forecast_type="hindcast",
                originating_centre=originating_centre,
                system=system,
                leadtime_month=leadtime_month,
                data_format=data_format,
            ),
            # request_delta_hack=relativedelta(months=1),
            to_xarray_args=dict( # passed to earthkit-data to_xarray method
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
            snap_time_index="next", # forecast at half month corresponds to following month
            earthkit_cache_dir=earthkit_cache_dir,
        ),
    )


@register_provider("ocean.earthkit.oras5.reanalysis.monthly")
def earthkit_cds_oras5(
    var_name: str,
    leadtime_value: int, # ignored
    leadtime_unit: LeadtimeUnit,
    product_type: str = "consolidated", # or "operational"
    regrid_resolution: float = 0.25,
    select_area_after_request: bool = True,
    earthkit_cache_dir: str | Path = Path("/tmp/earthkit-cache/"),
    convert_unit: dict | None = None, # dict of var_name: (func, target_unit) to convert variable unit (e.g. {"temperature": (lambda x: x - 273.15, "C")})
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, 0)

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
            request_delta_hack={"sosstsst": -relativedelta(days=15)}, # account for sosstsst ORAS5 (possible) convention that half of the month corresponds to the following month
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
            snap_time_index="next" if var_name=="sosstsst" else "same",
            convert_unit=convert_unit,
            earthkit_cache_dir=earthkit_cache_dir,
        ), 
    )


# Weather ocean, copernicusmarine, hourly, analysis before TODAY
@register_provider("ocean.copernicusmarine.gopaf.analysis.hourly")
def copernicusmarine_global_ocean_physics_analysis_hourly(
    var_name: str,
    leadtime_value: int, # unused
    leadtime_unit: LeadtimeUnit,
    username: str,
    password: str,
    regrid_resolution: float = 0.08,
    select_area_after_request: bool = True,
    # copernicusmarine_cache_dir: str  = Path("/tmp/copernicusmarine-cache/"),
    convert_unit: dict | None = None, # dict of var_name: (func, target_unit) to convert variable unit (e.g. {"temperature": (lambda x: x - 273.15, "C")})
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, 0)

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
    leadtime_unit: LeadtimeUnit,
    username: str,
    password: str,
    regrid_resolution: float = 0.08,
    select_area_after_request: bool = True,
    # copernicusmarine_cache_dir: str  = Path("/tmp/copernicusmarine-cache/"),
    convert_unit: dict | None = None, # dict of var_name: (func, target_unit) to convert variable unit (e.g. {"temperature": (lambda x: x - 273.15, "C")})
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, 0)

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
@register_provider("ocean.juno.gopaf.forecast.daily")
def juno_global_ocean_physics_forecast_daily(
    var_name: str,
    leadtime_value: int,
    leadtime_unit: LeadtimeUnit,
    root_path: str | Path = "/work/cmcc/jd19424/test-ML/dataML/mercator/1.0forecast/day/",
    engine: str = "h5netcdf",
    cfgrib_idx_path: str = "",
    regrid_resolution: float = 0.08,
    file_path_date_format: str = "%Y%m%d",
    region_str: Literal["MEDATL", "MEDDRD"] = "MEDATL",
    file_header: str = "cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
    file_suffix: str = ".nc",
    file_date_format: str = "%Y%m%d",
    both_data_and_previous_date_in_file: bool = True,
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
                file_header=f"{file_header}_{region_str}_*",
                file_suffix=file_suffix,
                file_date_format=file_date_format,
                both_data_and_previous_date_in_file=both_data_and_previous_date_in_file,
                date_order="current_previous",
                date_separator="_",
                realizations=1,
            ),
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
        ),
    )


# MedFS
MedFSGridToken: TypeAlias = Literal["T", "T_h", "U", "U_h", "V", "V_h", "W"]
MedFSVersion: TypeAlias = Literal["6.8", "6.82", "7.1", "8", "9", "10"]

def _medfs_grid_token(var_name: str, medfs_ver: MedFSVersion) -> MedFSGridToken:
    OLD_VERSIONS = {"6.8", "6.82", "7.1"}
    if var_name in {"sozotaux", "vozocrtx"}:
        return "U"
    if var_name == "ssu":
        return "U_h" if medfs_ver in OLD_VERSIONS else "U"
    if var_name in {"sometauy", "vomecrty"}:
        return "V"
    if var_name == "ssv":
        return "V_h" if medfs_ver in OLD_VERSIONS else "V"
    if var_name in {"votkeavt", "vovecrtz"}:
        return "W"
    if var_name in {
        "votemper",
        "vosaline",
        "somxl010",
        "solafldo",
        "sosefldo",
        "solofldo",
        "sohefldo",
        "soshfldo",
        "sorunoff",
        "soprecip",
        "soevapor",
        "sowaflup",
    }:
        return "T"
    if var_name == "sossheig":
        return "T_h" if medfs_ver in OLD_VERSIONS else "T"
    raise ValueError(f"Unsupported var_name: {var_name}")

# MedFS forecasts
@register_provider("ocean.juno.medfs.forecast.daily")
def juno_medfs_forecast_daily(
    var_name: str,
    leadtime_value: int,
    leadtime_unit: LeadtimeUnit,
    root_path: str | Path = "/data/products/MFS/MFS_EAS6v8/bulletin",
    engine: str = "h5netcdf",
    regrid_resolution: float | None = None,
    file_path_date_format: str = "%Y%m%d",
    file_date_format: str = "%Y%m%d",
    file_open_workers: int | None = 1,
    chunk_option: Any = {},
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, leadtime_value)

    data_type_suf = "f"
    file_path_header="b"
    file_path_suffix=""
    file_name_config = JunoLocalSourceFileNameConfig(
        realizations=1,
    )

    return SourceConfigContainer(
        source="juno-local",
        config=JunoLocalSourceConfig(
            leadtime=leadtime,
            root_path=Path(root_path),
            engine=engine,
            file_open_workers=file_open_workers,
            file_name_config=file_name_config,
            path_configs=[
                JunoLocalSourcePathConfig(
                    start_date="2022-01-01",
                    end_date="2022-10-17",
                    root_path="/data/products/MFS/MFS_EAS6v8/bulletin",
                    file_name_config=JunoLocalSourceFileNameConfig(
                        file_path_header=file_path_header,
                        file_path_suffix=file_path_suffix,
                        file_path_date_format=file_path_date_format,
                        file_header="mfs_eas6v8-",
                        file_suffix=f"-{data_type_suf}-{_medfs_grid_token(var_name, 6.8)}.nc",
                        file_date_format=file_date_format,
                        both_data_and_previous_date_in_file=True,
                        date_separator="-",
                        realizations=1,
                    ),
                ),
                JunoLocalSourcePathConfig(
                    start_date="2022-10-18",
                    end_date="2022-11-14",
                    root_path="/data/products/MFS/MFS_EAS6v82/bulletin",
                    file_name_config=JunoLocalSourceFileNameConfig(
                        file_path_header=file_path_header,
                        file_path_suffix=file_path_suffix,
                        file_path_date_format=file_path_date_format,
                        file_header="mfs_eas6v8-",
                        file_suffix=f"-{data_type_suf}-{_medfs_grid_token(var_name, 6.82)}.nc",
                        file_date_format=file_date_format,
                        both_data_and_previous_date_in_file=True,
                        date_separator="-",
                        realizations=1,
                    ),
                ),
                JunoLocalSourcePathConfig(
                    start_date="2022-11-15",
                    end_date="2023-11-21",
                    root_path="/data/products/MFS/MFS_EAS7v1/bulletin",
                    file_name_config=JunoLocalSourceFileNameConfig(
                        file_path_header=file_path_header,
                        file_path_suffix=file_path_suffix,
                        file_path_date_format=file_path_date_format,
                        file_header="mfs_eas7-",
                        file_suffix=f"-{data_type_suf}-{_medfs_grid_token(var_name, 7.1)}.nc",
                        file_date_format=file_date_format,
                        both_data_and_previous_date_in_file=True,
                        date_separator="-",
                        realizations=1,
                    ),
                ),
                JunoLocalSourcePathConfig(
                    start_date="2023-11-22",
                    end_date="2024-11-11",
                    root_path="/data/products/MFS/MFS_EAS8/bulletin",
                    file_name_config=JunoLocalSourceFileNameConfig(
                        file_path_header=file_path_header,
                        file_path_suffix=file_path_suffix,
                        file_path_date_format=file_path_date_format,
                        file_header="medfs-eas8_1d_",
                        file_suffix=f"_ALL_grid_{_medfs_grid_token(var_name, 8)}.nc",
                        file_date_format=file_date_format,
                        both_data_and_previous_date_in_file=True,
                        date_separator="_",
                        realizations=1,
                    ),
                ),
                JunoLocalSourcePathConfig(
                    start_date="2024-11-12",
                    end_date="2025-11-4",
                    root_path="/data/products/MFS/MFS_EAS9/bulletin",
                    file_name_config=JunoLocalSourceFileNameConfig(
                        file_path_header=file_path_header,
                        file_path_suffix=file_path_suffix,
                        file_path_date_format=file_path_date_format,
                        file_header="medfs-eas9_1d_",
                        file_suffix=f"_ALL_grid_{_medfs_grid_token(var_name, 9)}.nc",
                        file_date_format=file_date_format,
                        both_data_and_previous_date_in_file=True,
                        date_separator="_",
                        realizations=1,
                    ),
                ),
                JunoLocalSourcePathConfig(
                    start_date="2025-11-5",
                    end_date=None,
                    root_path="/data/products/MFS/MedFS_EAS10/bulletin",
                    file_name_config=JunoLocalSourceFileNameConfig(
                        file_path_header=file_path_header,
                        file_path_suffix=file_path_suffix,
                        file_path_date_format=file_path_date_format,
                        file_header="medfs-eas10_1d_",
                        file_suffix=f"_ALL_grid_{_medfs_grid_token(var_name, 10)}.nc",
                        file_date_format=file_date_format,
                        both_data_and_previous_date_in_file=True,
                        date_separator="_",
                        realizations=1,
                    ),
                ),
            ],
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
            chunk_option=chunk_option,
        ),
    )

# MedFS analysis
@register_provider("ocean.juno.medfs.analysis.daily")
def juno_medfs_analysis_daily(
    var_name: str,
    leadtime_value: int,
    leadtime_unit: LeadtimeUnit,
    root_path: str | Path = "/data/products/MFS/MFS_EAS9/analysis_daily_mean",
    engine: str = "h5netcdf",
    regrid_resolution: float | None = None,
    file_path_date_format: str = "%Y/%m",
    file_date_format: str = "%Y%m%d",
    file_open_workers: int | None = 1,
    chunk_option: Any = {},
) -> SourceConfigContainer:
    leadtime = Leadtime("leadtime", leadtime_unit, 0)

    file_path_header="b"
    file_path_suffix=""
    file_name_config = JunoLocalSourceFileNameConfig(
        file_path_header=file_path_header,
        file_path_suffix=file_path_suffix,
        file_path_date_format=file_path_date_format,
        file_header="medfs-eas10_1d_",
        file_suffix=f"_ALL_grid_{_medfs_grid_token(var_name, 10)}.nc",
        file_date_format=file_date_format,
        both_data_and_previous_date_in_file=False,
        date_separator="-",
        realizations=1,
    )

    return SourceConfigContainer(
        source="juno-local",
        config=JunoLocalSourceConfig(
            leadtime=leadtime,
            root_path=Path(root_path),
            engine=engine,
            file_open_workers=file_open_workers,
            file_name_config=file_name_config,
            regrid_config=RegridConfig(
                regrid_resolution=regrid_resolution,
                regrid_vars=[var_name],
            ),
            chunk_option=chunk_option,
        ),
    )
