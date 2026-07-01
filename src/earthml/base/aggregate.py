from typing import Sequence

import numpy as np
import xarray as xr


def aggregate_leadtime_da(
    da: xr.DataArray,
    windows: dict[str, Sequence[int]],
    leadtime_dim: str = "leadtime",
    leadtime_agg_coord: str = "seasonal_leadtime",
) -> xr.DataArray:
    """
    Average data over predefined lead-time windows.

    For each window, computes the mean across the selected lead times and
    concatenates the results along the lead-time dimension. Window labels are
    stored as a coordinate given by ``agg_leadtime_coord``.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a lead-time dimension.
    windows : dict[str, list[int]]
        Mapping of window labels to lead times.
    leadtime_dim : str, default="leadtime"
        Name of the lead-time dimension.
    leadtime_agg_coord : str, default="seasonal_leadtime"
        Name of the coordinate containing window labels.

    Returns
    -------
    xr.DataArray
        Data aggregated over lead-time windows.
    """
    available = set(int(x) for x in da[leadtime_dim].values)

    out: list[xr.DataArray] = []
    labels: list[str] = []
    centers: list[float] = []

    print(da[leadtime_dim])
    print(da[leadtime_dim].dtype)
    print(da[leadtime_dim].values[:10])

    for name, leads in windows.items():
        leads = [int(x) for x in leads]
        print("name:", name)
        print("available:", sorted(available))
        print("requested:", leads)
        print("missing:", set(leads) - available)

        if not set(leads).issubset(available):
            continue

        x = da.sel({leadtime_dim: leads}).mean(leadtime_dim, skipna=False)
        print("agg:", name, x)

        labels.append(name)
        centers.append(float(np.mean(leads)))
        out.append(x)

    if not out:
        raise ValueError("No valid lead windows found.")

    result = xr.concat(
        out,
        dim=xr.IndexVariable(leadtime_dim, centers),
        coords="minimal",
        compat="override",
        join="inner",
        combine_attrs="override",
    )

    result = result.assign_coords({
        leadtime_agg_coord: (leadtime_dim, np.asarray(labels, dtype="U")),
    })

    # import sys; sys.exit("stop at aggregate")

    return result


def aggregate_leadtime_da_dayweighted(
    da: xr.DataArray,
    windows: dict[str, Sequence[int]],
    init_month: int | float,
    leadtime_dim: str = "leadtime",
    leadtime_agg_coord: str = "seasonal_leadtime",
) -> xr.DataArray:
    available = set(int(x) for x in da[leadtime_dim].values)

    out = []
    labels = []
    centers = []

    for name, leads in windows.items():
        leads = [int(x) for x in leads]

        if not set(leads).issubset(available):
            continue

        # valid months for this init month + leadtime
        valid_months = [((init_month - 1 + lead) % 12) + 1 for lead in leads]

        weights = xr.DataArray(
            [31 if m in [1, 3, 5, 7, 8, 10, 12]
             else 30 if m in [4, 6, 9, 11]
             else 28
             for m in valid_months],
            dims=[leadtime_dim],
            coords={leadtime_dim: leads},
        )

        x = da.sel({leadtime_dim: leads}).weighted(weights).mean(leadtime_dim)

        out.append(x)
        labels.append(name)
        centers.append(float(np.mean(leads)))

    result = xr.concat(
        out,
        dim=xr.IndexVariable(leadtime_dim, centers),
        coords="minimal",
        compat="override",
        join="inner",
        combine_attrs="override",
    )

    result = result.assign_coords({
        leadtime_agg_coord: (leadtime_dim, np.asarray(labels, dtype="U")),
    })

    return result

def aggregate_leadtime_ds(
    ds: xr.Dataset,
    windows: dict[str, Sequence[int]],
    init_month: int | float,
    leadtime_dim: str = "leadtime",
    leadtime_agg_coord: str = "seasonal_leadtime",
) -> xr.Dataset:
    return xr.Dataset(
        {
            var: (
                aggregate_leadtime_da_dayweighted(ds[var], windows, init_month, leadtime_dim, leadtime_agg_coord)
                if leadtime_dim in ds[var].dims
                else ds[var]
            )
            for var in ds.data_vars
        },
        attrs=ds.attrs,
    )
