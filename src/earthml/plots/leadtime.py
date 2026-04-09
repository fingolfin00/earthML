from typing import List, Optional, Sequence
from numbers import Number
from pathlib import Path
from itertools import product

from rich import print

import numpy as np
import pandas as pd
import cf_xarray
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



def plot_ensemble_leadtime(
    ens_mean,
    members=None,
    extra=None,
    *,
    lead_dim="leadtime",
    ens_dim="realization",
    ax=None,
    label="",
    color=None,
    spread="std",          # "std" or "minmax"
    nsigma=1.0,
    plot_members=True,
    members_alpha=0.15,
    members_lw=0.8,
    members_ls="--",
    spread_alpha=0.25,
    mean_lw=2.5,
    extra_label="",
    extra_lw=2.0,
    extra_ls=":",
):
    """
    Plot ensemble mean, optionally members and spread vs lead time.

    Parameters
    ----------
    ens_mean : xarray.DataArray
        Ensemble mean with dim (lead_dim,)
    members : xarray.DataArray, optional
        Ensemble realizations with dims (lead_dim, ens_dim)
    extra : xarray.DataArray, optional
        Extra line with dim (lead_dim,)
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Ensure mean has correct dimension order
    ens_mean = ens_mean.transpose(lead_dim)
    x = ens_mean[lead_dim].values

    # Handle members if provided
    if members is not None:
        members = members.transpose(lead_dim, ens_dim)

        # Plot individual members
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

        # Compute spread
        if spread == "std":
            sd = members.std(ens_dim)
            lower = (ens_mean - nsigma * sd).values
            upper = (ens_mean + nsigma * sd).values
        elif spread == "minmax":
            lower = members.min(ens_dim).values
            upper = members.max(ens_dim).values
        else:
            raise ValueError("spread must be 'std' or 'minmax'")

        # Plot spread
        ax.fill_between(
            x,
            lower,
            upper,
            alpha=spread_alpha,
            color=color,
            label=f"{label} spread" if label else "spread",
        )

    # Always plot mean
    ax.plot(
        x,
        ens_mean.values,
        linewidth=mean_lw,
        color=color,
        label=f"{label} ens mean" if label else "ens mean",
    )

    # Optional extra line
    if extra is not None:
        extra = extra.transpose(lead_dim)
        ax.plot(
            x,
            extra.values,
            linewidth=extra_lw,
            linestyle=extra_ls,
            color=color,
            label=f"{extra_label}" if extra_label else "extra",
        )

    ax.set_xlabel("Lead time")
    ax.grid(True, alpha=0.3)

    return ax
