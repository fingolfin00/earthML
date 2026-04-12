from typing import Literal, Sequence

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def _as_dataarray(data: xr.DataArray | xr.Dataset) -> xr.DataArray:
    if isinstance(data, xr.DataArray):
        return data
    if len(data.data_vars) != 1:
        raise ValueError("Dataset must contain exactly one variable for timeseries plotting.")
    return data[next(iter(data.data_vars))]


def _reduce_for_timeseries(
    da: xr.DataArray | xr.Dataset,
    *,
    keep_dims: Sequence[str],
) -> xr.DataArray:
    da = _as_dataarray(da)
    reduce_dims = tuple(dim for dim in da.dims if dim not in keep_dims)
    if not reduce_dims:
        return da
    return da.earthml.geo_mean(reduce_dims)


def plot_realization_timeseries(
    series_mean: xr.DataArray | xr.Dataset,
    members: xr.DataArray | xr.Dataset | None = None,
    extra: xr.DataArray | xr.Dataset | None = None,
    *,
    x_dim: str = "time",
    ens_dim: str = "realization",
    ax: Axes | None = None,
    x_label: str = "Time",
    label: str = "",
    mean_label: str | None = None,
    color: str | None = None,
    spread: Literal["std", "minmax"] = "std",
    nsigma: float = 1.0,
    plot_members: bool = True,
    members_alpha: float = 0.15,
    members_lw: float = 0.5,
    members_ls: str = "--",
    spread_alpha: float = 0.25,
    mean_lw: float = 1.8,
    extra_label: str = "",
    extra_lw: float = 1.5,
    extra_ls: str = ":",
) -> Axes:
    """
    Plot a 1D series with optional realization members and shaded spread.

    Any dimensions other than ``x_dim`` and ``ens_dim`` are reduced before
    plotting using ``earthml.geo_mean(...)``. This means spatial fields like
    ``(time, lat, lon)`` or ``(time, realization, lat, lon)`` can be passed
    directly and will be collapsed to a timeseries first.

    Parameters
    ----------
    series_mean : xarray.DataArray or xarray.Dataset
        Mean series with dim ``(x_dim,)`` after optional reduction.
        If a Dataset is passed, it must contain exactly one variable.
    members : xarray.DataArray or xarray.Dataset, optional
        Realization series with dims ``(x_dim, ens_dim)`` after optional reduction.
        If a Dataset is passed, it must contain exactly one variable.
    extra : xarray.DataArray or xarray.Dataset, optional
        Additional line with dim ``(x_dim,)`` after optional reduction.
        If a Dataset is passed, it must contain exactly one variable.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    series_mean = _reduce_for_timeseries(series_mean, keep_dims=(x_dim,))
    series_mean = series_mean.transpose(x_dim)
    x = series_mean[x_dim].values

    if members is not None:
        members = _reduce_for_timeseries(members, keep_dims=(x_dim, ens_dim))
        members = members.transpose(x_dim, ens_dim)

        if plot_members:
            y_members = members.values
            for i in range(y_members.shape[1]):
                ax.plot(
                    x,
                    y_members[:, i],
                    linestyle=members_ls,
                    linewidth=members_lw,
                    alpha=members_alpha,
                    color=color,
                )

        if spread == "std":
            sd = members.std(ens_dim)
            lower = (series_mean - nsigma * sd).values
            upper = (series_mean + nsigma * sd).values
        elif spread == "minmax":
            lower = members.min(ens_dim).values
            upper = members.max(ens_dim).values
        else:
            raise ValueError("spread must be 'std' or 'minmax'")

        ax.fill_between(
            x,
            lower,
            upper,
            alpha=spread_alpha,
            color=color,
            label=f"{label} spread" if label else "spread",
        )

    ax.plot(
        x,
        series_mean.values,
        linewidth=mean_lw,
        color=color,
        label=mean_label if mean_label is not None else (f"{label} mean" if label else "mean"),
    )

    if extra is not None:
        extra = _reduce_for_timeseries(extra, keep_dims=(x_dim,))
        extra = extra.transpose(x_dim)
        ax.plot(
            x,
            extra.values,
            linewidth=extra_lw,
            linestyle=extra_ls,
            color=color,
            label=f"{extra_label}" if extra_label else "extra",
        )

    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.3)

    return ax
