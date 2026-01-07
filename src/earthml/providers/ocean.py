from pathlib import Path
from dateutil.relativedelta import relativedelta

from ..dataclasses import ProviderSpec
from .base import merge
from .registry import register_provider

@register_provider("ocean.juno.cmcc.hindcast.monthly")
def juno_monthly_hindcast_ocean_netcdf (
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    root_path: str = "/work/cmcc/cp1/CMCC-CM/archive/C3S/",
    realizations: str | int = "all",
    engine: str = "h5netcdf",
    regrid_resolution: float = 0.25,
    concat_dim: str = "time",
    file_path_date_format: str = "%Y%m",
    file_header: str = "cmcc_CMCC-CM3-v20231101_hindcast_S",
    file_date_format: str = "%Y%m%d",
    both_data_and_previous_date_in_file: bool = False,
    overrides: dict | None = None,
    **kw,
) -> ProviderSpec:
    members = str(realizations) if isinstance(realizations, int) else "*"
    base = dict(
        root_path=root_path,
        engine=engine,
        file_path_date_format=file_path_date_format,
        file_header=file_header,
        file_suffix=f"*ocean_mon_ocean2d_{var_name}_r{members}i00p00.nc",
        file_date_format=file_date_format,
        both_data_and_previous_date_in_file=both_data_and_previous_date_in_file,
        realizations=realizations,
        lead_time=relativedelta(**{leadtime_unit: leadtime_value}),
        regrid_resolution=regrid_resolution,
        concat_dim=concat_dim,
    )
    return ProviderSpec('juno-local', merge(base, overrides, **kw))

@register_provider("ocean.earthkit.cmcc.hindcast.monthly")
def earthkit_cmcc_monthly_hindcast_ocean_netcdf (
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    originating_centre: str = "cmcc",
    system: str = "4",
    regrid_resolution: float = 0.25,
    split_month: int = 1,
    split_month_jump: list[str] | None = ['03', '04', '06', '07'],
    earthkit_cache_dir: str  = Path("/tmp/earthkit-cache/"),
    overrides: dict | None = None,
    **kw,
) -> ProviderSpec:
    base = dict(
        provider="cds",
        lead_time=relativedelta(**{leadtime_unit: leadtime_value}),
        dataset="seasonal-monthly-ocean",
        regrid_resolution=regrid_resolution,
        split_request=True,
        split_month=split_month,
        split_month_jump=split_month_jump or [],
        request_type="monthly",
        request_extra_args=dict(
            forecast_type="hindcast",
            originating_centre=originating_centre,
            system=system,
        ),
        to_xarray_args=dict(
            engine="h5netcdf",
            decode_timedelta=True,
            data_vars="all",
            coords="minimal",
            compat="override",
            concat_dim="leadtime",
            combine="nested",
        ),
        xarray_concat_dim="time",
        xarray_concat_extra_args=dict(coords="minimal", compat="override"),
        earthkit_cache_dir=earthkit_cache_dir,
    )
    return ProviderSpec('earthkit', merge(base, overrides, **kw))

@register_provider("ocean.earthkit.oras5.reanalysis.monthly")
def earthkit_cds_oras5 (
    var_name: str,
    leadtime_value: int,
    leadtime_unit: str,
    product_type: str = "consolidated",  # or "operational"
    regrid_resolution: float = 0.25,
    select_area_after_request: bool = True,
    earthkit_cache_dir: str  = Path("/tmp/earthkit-cache/"),
    overrides: dict | None = None,
    **kw,
) -> ProviderSpec:
    base = dict(
        provider="cds",
        lead_time=relativedelta(hours=0),
        dataset="reanalysis-oras5",
        split_request=True,
        select_area_after_request=select_area_after_request,
        regrid_resolution=regrid_resolution,
        request_type="monthly",
        request_extra_args=dict(
            product_type=product_type,
            vertical_resolution="single_level",
        ),
        to_xarray_args=dict(
            engine="h5netcdf",
            decode_timedelta=True,
            data_vars="all",
            combine="by_coords",
            coords="minimal",
            compat="override",
            parallel=True,
        ),
        xarray_concat_dim=None,
        xarray_concat_extra_args=dict(coords="minimal", compat="override"),
        earthkit_cache_dir=earthkit_cache_dir,
    )
    return ProviderSpec('earthkit', merge(base, overrides, **kw))
