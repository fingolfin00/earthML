from typing import Literal

import xarray as xr
from matplotlib.axes import Axes

from .timeseries import plot_realization_timeseries



def plot_ensemble_profile(
    ens_mean: xr.DataArray,
    members: xr.DataArray | None = None,
    extra: xr.DataArray | None = None,
    *,
    x_dim: str = "leadtime",
    ens_dim: str = "realization",
    ax: Axes | None = None,
    label: str = "",
    color: str | None = None,
    spread: Literal["std", "minmax"] = "std",
    nsigma: float = 1.0,
    plot_members: bool = True,
    members_alpha: float = 0.15,
    members_lw: float = 0.8,
    members_ls: str = "--",
    spread_alpha: float = 0.25,
    mean_lw: float = 2.5,
    extra_label: str = "",
    extra_lw: float = 2.0,
    extra_ls: str = ":",
) -> Axes:
    """
    Plot ensemble mean, optionally members and spread, against a chosen x-axis.

    Parameters
    ----------
    ens_mean : xarray.DataArray
        Ensemble mean with dim (x_dim,)
    members : xarray.DataArray, optional
        Ensemble realizations with dims (x_dim, ens_dim)
    extra : xarray.DataArray, optional
        Extra line with dim (x_dim,)
    """

    ax = plot_realization_timeseries(
        ens_mean,
        members=members,
        extra=extra,
        x_dim=x_dim,
        ens_dim=ens_dim,
        ax=ax,
        x_label=x_dim.replace("_", " ").title(),
        label=label,
        mean_label=f"{label} ens mean" if label else "ens mean",
        color=color,
        spread=spread,
        nsigma=nsigma,
        plot_members=plot_members,
        members_alpha=members_alpha,
        members_lw=members_lw,
        members_ls=members_ls,
        spread_alpha=spread_alpha,
        mean_lw=mean_lw,
        extra_label=extra_label,
        extra_lw=extra_lw,
        extra_ls=extra_ls,
    )

    return ax
