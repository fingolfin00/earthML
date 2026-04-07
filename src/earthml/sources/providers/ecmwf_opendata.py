from dateutil.relativedelta import relativedelta

from .. import SourceConfig, RegridConfig, EarthkitSourceConfig

from .registry import register_provider


@register_provider("earthkit.ecmwf_open_data.forecast.sfc")
def earthkit_ecmwf_open_data_fc_sfc(
    var_name: str,
    leadtime_value: int, # ignored
    leadtime_unit: str,
    levtype: str = "sfc",
    split_request: bool = True,
    select_area_after_request: bool = True,
    time_dim_mode: str = "valid_time",
    chunks: dict | None = None,
    add_earthkit_attrs: bool = False,
) -> SourceConfig:
    leadtime = relativedelta(**{leadtime_unit: 0})

    return SourceConfig(
        source="earthkit",
        config=EarthkitSourceConfig(
            leadtime=leadtime,
            provider="ecmwf-open-data",
            split_request=split_request,
            select_area_after_request=select_area_after_request,
            request_extra_args=dict(request=dict(
                param=var_name,
                levtype=levtype,
            )),
            to_xarray_args=dict(
                time_dim_mode=time_dim_mode,
                chunks=chunks or {"valid_time": 1},
                add_earthkit_attrs=add_earthkit_attrs,
            ),
        ),
    )
