from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, TwoSlopeNorm
import xarray as xr
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


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


def _extent_from_da(
    da: xr.DataArray,
    *,
    pad_deg: float = 2.0,
) -> tuple[float, float, float, float] | None:
    lon_name = da.earthml.guessed_dims.longitude
    lat_name = da.earthml.guessed_dims.latitude
    if lon_name is None or lat_name is None:
        return None

    lon = np.asarray(da[lon_name].values, dtype=np.float64).ravel()
    lat = np.asarray(da[lat_name].values, dtype=np.float64).ravel()

    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]
    if lon.size == 0 or lat.size == 0:
        return None

    lon_cont = _continuous_lon_values(lon)
    return (
        float(lon_cont.min()) - pad_deg,
        float(lon_cont.max()) + pad_deg,
        float(lat.min()) - pad_deg,
        float(lat.max()) + pad_deg,
    )


def _center_lon_from_extent(
    extent: tuple[float, float, float, float] | None,
    *,
    global_width_threshold: float = 300.0,
) -> float:
    if extent is None:
        return 0.0

    lon_min, lon_max, _, _ = extent
    if not (np.isfinite(lon_min) and np.isfinite(lon_max)):
        return 0.0

    width = lon_max - lon_min
    if width >= global_width_threshold:
        return 0.0

    center_lon = 0.5 * (lon_min + lon_max)
    return float(center_lon % 360.0)


def plot_temporal_mean_map(
    data: xr.Dataset | xr.DataArray,
    *,
    save_path: str | Path,
    title: str | None = None,
    subtitle: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    cbar_label: str = "",
    cmap: str = "jet",
    vmin: float | None = None,
    vmax: float | None = None,
    lon_tick_step: float = 10.0,
    lat_tick_step: float = 10.0,
    figsize: tuple[float, float] = (8, 4.5),
    dpi: int = 200,
    nan_color: str | None = None,
) -> None:
    da = _reduce_for_temporal_mean_map(data)
    da = _sort_lon_for_plot(da)
    plot_extent = extent if extent is not None else _extent_from_da(da)
    center_lon = _center_lon_from_extent(plot_extent)
    data_crs = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=center_lon)},
    )
    fig.subplots_adjust(right=0.88, top=0.88 if subtitle else 0.94)
    norm = None
    if vmin is not None and vmax is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0.0 < vmax else Normalize(vmin=vmin, vmax=vmax)
    plot_cmap = plt.get_cmap(cmap)
    if nan_color is not None:
        plot_cmap = plot_cmap.copy()
        plot_cmap.set_bad(color=nan_color)

    mappable = da.plot.pcolormesh(
        ax=ax,
        transform=data_crs,
        cmap=plot_cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        add_colorbar=False,
    )
    if plot_extent is None:
        ax.set_global()
    else:
        ax.set_extent(plot_extent, crs=data_crs)
    ax.coastlines(linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.35, linestyle="--", alpha=0.4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if lon_tick_step > 0:
        gl.xlocator = mticker.MultipleLocator(lon_tick_step)
    if lat_tick_step > 0:
        gl.ylocator = mticker.MultipleLocator(lat_tick_step)
    if title and subtitle:
        fig.suptitle(title, fontsize=10)
        ax.set_title(subtitle, fontsize=9, color="0.35", pad=8)
    elif title:
        ax.set_title(title, fontsize=10)
    elif subtitle:
        ax.set_title(subtitle, fontsize=9, color="0.35", pad=8)

    ax_pos = ax.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.02, ax_pos.y0, 0.02, ax_pos.height])
    cbar = fig.colorbar(mappable, cax=cax, orientation="vertical")
    cbar.set_label(cbar_label)

    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
