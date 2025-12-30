from ..dataclasses import ProviderSpec
from .base import merge
from .registry import register

@register("earthkit.ecmwf_open_data.fc.sfc")
def earthkit_ecmwf_open_data_fc_sfc (
    param: str,
    levtype: str = "sfc",
    split_request: bool = True,
    select_area_after_request: bool = True,
    time_dim_mode: str = "valid_time",
    chunks: dict | None = None,
    add_earthkit_attrs: bool = False,
    overrides: dict | None = None,
    **kw,
) -> ProviderSpec:
    base = dict(
        provider="ecmwf-open-data",
        split_request=split_request,
        select_area_after_request=select_area_after_request,
        request_extra_args=dict(request=dict(
            param=param,
            levtype=levtype,
        )),
        to_xarray_args=dict(
            time_dim_mode=time_dim_mode,
            chunks=chunks or {"valid_time": 1},
            add_earthkit_attrs=add_earthkit_attrs,
        ),
    )
    return ProviderSpec('earthkit', merge(base, overrides, **kw))
