from typing import Literal, Sequence

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def _snap_to_nearest_month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    month_start = idx.to_period("M").to_timestamp(how="start")
    next_month = (idx.to_period("M") + 1).to_timestamp(how="start")
    return month_start.where((idx - month_start) <= (next_month - idx), next_month)


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
    eager_compute: bool = False,
) -> xr.DataArray:
    da = _as_dataarray(da)
    if eager_compute and np.issubdtype(da.dtype, np.floating) and da.dtype.itemsize < np.dtype(np.float32).itemsize:
        # Flox/numbagg reductions do not support float16 inputs reliably; promote
        # plotting reductions to float32 without forcing eager computation.
        da = da.astype(np.float32)
    reduce_dims = tuple(dim for dim in da.dims if dim not in keep_dims)
    if not reduce_dims:
        return da
    with xr.set_options(use_numbagg=False):
        if eager_compute:
            # Plotting only needs one reduced timeseries at a time, so materialize
            # first and reduce eagerly to avoid dask->flox/numbagg float16 failures.
            reduced = da.load().earthml.geo_mean(reduce_dims)
        else:
            reduced = da.earthml.geo_mean(reduce_dims)
            reduced = reduced.load()
    return reduced


def _plot_x_values_raw(x):
    x = np.asarray(x)

    if not np.issubdtype(x.dtype, np.datetime64):
        return x

    try:
        idx = pd.DatetimeIndex(pd.to_datetime(x))
    except Exception:
        return x

    if len(idx) < 2:
        return x

    try:
        freq = getattr(idx, "freqstr", None) or xr.infer_freq(idx)
    except Exception:
        freq = None

    is_monthly = False
    if freq is not None:
        freq_s = str(freq).upper()
        is_monthly = ("MS" in freq_s) or (freq_s == "M")
    else:
        try:
            diffs = np.diff(idx.values.astype("datetime64[ns]")).astype("timedelta64[D]").astype(np.int64)
            if diffs.size:
                med = int(np.median(diffs))
                is_monthly = 27 <= med <= 32
        except Exception:
            is_monthly = False

    if not is_monthly:
        return x

    month_start = _snap_to_nearest_month_start(idx)
    next_month = (idx.to_period("M") + 1).to_timestamp(how="start")
    midpoint = month_start + (next_month - month_start) / 2
    return midpoint.to_numpy()


def _plot_x_values(da: xr.DataArray, x_dim: str):
    return _plot_x_values_raw(da[x_dim].values)


def _plot_coord_name(da: xr.DataArray, x_dim: str) -> str:
    if x_dim != "time":
        return x_dim
    valid_time = da.coords.get("valid_time")
    if valid_time is None:
        return x_dim
    if valid_time.dims != (x_dim,):
        return x_dim
    if valid_time.sizes.get(x_dim, 0) != da.sizes.get(x_dim, 0):
        return x_dim
    return "valid_time"


def plot_realization_timeseries(
    series_mean: xr.DataArray | xr.Dataset,
    members: xr.DataArray | xr.Dataset | None = None,
    extra: xr.DataArray | xr.Dataset | None = None,
    *,
    x_dim: str = "time",
    ens_dim: str = "realization",
    ax: Axes | None = None,
    x_label: str = "Time",
    y_label: str | None = None,
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
    eager_compute: bool = False,
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

    series_mean = _reduce_for_timeseries(series_mean, keep_dims=(x_dim,), eager_compute=eager_compute)
    series_mean = series_mean.transpose(x_dim)
    x_coord = _plot_coord_name(series_mean, x_dim)
    x = _plot_x_values_raw(series_mean[x_coord].values)

    if members is not None:
        members = _reduce_for_timeseries(members, keep_dims=(x_dim, ens_dim), eager_compute=eager_compute)
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
        extra = _reduce_for_timeseries(extra, keep_dims=(x_dim,), eager_compute=eager_compute)
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
    if y_label is not None:
        ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)

    return ax
