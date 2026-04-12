from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs


def _continuous_lon_values(lon_values: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_values, dtype=np.float64).copy()
    if lon.ndim != 1 or lon.size <= 1:
        return lon

    for i in range(1, lon.size):
        if not (np.isfinite(lon[i - 1]) and np.isfinite(lon[i])):
            continue
        delta = lon[i] - lon[i - 1]
        while delta <= -180.0:
            lon[i] += 360.0
            delta = lon[i] - lon[i - 1]
        while delta > 180.0:
            lon[i] -= 360.0
            delta = lon[i] - lon[i - 1]

    finite = lon[np.isfinite(lon)]
    if finite.size:
        mean_lon = float(finite.mean())
        if mean_lon > 180.0:
            lon -= 360.0
        elif mean_lon <= -180.0:
            lon += 360.0

    return lon


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
    lon_plot = _continuous_lon_values(da[lon_name].values)
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
