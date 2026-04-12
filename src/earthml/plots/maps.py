from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs


def _reduce_for_temporal_mean_map(data: xr.Dataset | xr.DataArray) -> xr.DataArray:
    if isinstance(data, xr.Dataset):
        if len(data.data_vars) != 1:
            raise ValueError("Dataset must contain exactly one variable for map plotting.")
        da = data[next(iter(data.data_vars))]
    else:
        da = data

    time_dim = da.earthml.guessed_dims.time
    realization_dim = da.earthml.guessed_dims.realization

    if realization_dim is not None and realization_dim in da.dims:
        da = da.mean(dim=realization_dim, skipna=True)
    if time_dim is not None and time_dim in da.dims:
        da = da.mean(dim=time_dim, skipna=True)

    return da


def _sort_lon_for_plot(da: xr.DataArray) -> xr.DataArray:
    lon_name = da.earthml.guessed_dims.longitude
    if lon_name is None or lon_name not in da.coords:
        return da
    lon = da[lon_name]
    lon_plot = (np.asarray(lon.values) + 360) % 360
    return da.assign_coords({lon_name: lon_plot}).sortby(lon_name)


def plot_temporal_mean_map(
    data: xr.Dataset | xr.DataArray,
    *,
    save_path: str | Path,
    cbar_label: str = "",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 4.5),
    dpi: int = 200,
) -> None:
    da = _reduce_for_temporal_mean_map(data)
    da = _sort_lon_for_plot(da)

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    da.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=True,
        cbar_kwargs={"label": cbar_label},
    )
    ax.coastlines(linewidth=0.6)
    ax.gridlines(draw_labels=False, linewidth=0.35, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
