from typing import List, Optional, Sequence, Any
from numbers import Number
from pathlib import Path
from itertools import product

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

from .config import build_plot_config, get_var_plot_cmap, get_var_plot_limits
from ..logging import get_logger


logger = get_logger(__name__)


DEFAULT_PLOT_CONFIG = build_plot_config()


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


def _extent_from_da(da: xr.DataArray, lon_name="lon", lat_name="lat", pad_deg=2.0):
    lon = np.asarray(da[lon_name].values).ravel()
    lat = np.asarray(da[lat_name].values).ravel()

    lon = lon[np.isfinite(lon)]
    lat = lat[np.isfinite(lat)]

    if lon.size == 0 or lat.size == 0:
        raise ValueError("Cannot infer extent from empty lon/lat coordinates.")

    lon_cont = _continuous_lon_values(lon)
    lon_min = float(lon_cont.min())
    lon_max = float(lon_cont.max())

    lat_min, lat_max = float(lat.min()), float(lat.max())

    lon_min -= pad_deg
    lon_max += pad_deg
    lat_min -= pad_deg
    lat_max += pad_deg

    return (lon_min, lon_max, lat_min, lat_max)


def _create_plot_grid(
    nrows: int,
    ncols: int,
    figsize_per_cell=(10, 4),
    data_projection=ccrs.PlateCarree(),
    plt_fontsize: int = 12,
    plt_projection=ccrs.PlateCarree(),  # Robinson()
    plt_coastlines: bool = True,
    plt_global_extent: bool = False,
    plt_reg_extent: Sequence[float] | None = None,  # lon_min, lon_max, lat_min, lat_max
    gridlabel_mode: str = "outer",  # "outer" or "all"
) -> dict:
    if gridlabel_mode not in {"outer", "all"}:
        raise ValueError(
            f"Invalid gridlabel_mode={gridlabel_mode!r}. Use 'outer' or 'all'."
        )

    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": plt_projection},
        squeeze=False,
    )

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]

            if plt_coastlines:
                ax.coastlines(linewidth=0.6)

            if plt_global_extent:
                ax.set_global()
            elif plt_reg_extent is not None:
                ax.set_extent(plt_reg_extent, crs=data_projection)

            gl = ax.gridlines(
                crs=data_projection,
                draw_labels=False,
                linewidth=0.35,
                linestyle="--",
                alpha=0.4,
            )
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

            show_labels = (
                gridlabel_mode == "all"
                or c == 0
                or r == nrows - 1
            )

            if show_labels:
                gl2 = ax.gridlines(
                    crs=data_projection,
                    draw_labels=True,
                    linewidth=0,
                )
                gl2.xformatter = LONGITUDE_FORMATTER
                gl2.yformatter = LATITUDE_FORMATTER
                gl2.top_labels = False
                gl2.right_labels = False

                if gridlabel_mode == "all":
                    gl2.left_labels = True
                    gl2.bottom_labels = True
                else:
                    gl2.left_labels = (c == 0)
                    gl2.bottom_labels = (r == nrows - 1)

                gl2.xlabel_style = {"size": plt_fontsize}
                gl2.ylabel_style = {"size": plt_fontsize}

    plt.close(fig)

    return fig, axes


def _create_data_panel(
    ds: xr.Dataset | xr.DataArray | List[xr.Dataset] | List[xr.DataArray],
    col_index: str | Sequence[str], # if str, look in Dataset, can't be both Sequences
    row_index: str | Sequence[str],
    var_sel: Optional[str] = None, # ignored for xr.DataArray
    extra_dim_sequence: Optional[Sequence[str]] = None, # optional sequence of names for the extra dimension (useful if ds is list)
    vars_to_be_removed: Optional[str] = "_has_var",
) -> np.ndarray:

    if col_index == "variable" and row_index == "variable":
        raise ValueError("col_index and row_index cannot be both 'variable'")
    if isinstance(ds, xr.DataArray) and (col_index == "variable" or row_index == "variable"):
        raise ValueError("Variable index not supported for xr.DataArray")

    if isinstance(ds, list):
        # Guards
        if len(ds) == 0:
            raise ValueError("Empty ds list.")
        if not all(isinstance(x, type(ds[0])) for x in ds):
            raise ValueError("All elements in ds list must be the same type.")

        if isinstance(ds[0], xr.Dataset):
            ds = xr.merge(ds) # assume different variables in datasets in list
        elif isinstance(ds[0], xr.DataArray):
            # At least one index must be "variable"
            if not (isinstance(col_index, str) and isinstance(row_index, str)):
                raise ValueError("For sequence of xr.DataArray, col_index and row_index must be strings (coord names and 'variable').")
            if col_index != "variable" and row_index != "variable":
                raise ValueError("For sequence of xr.DataArray, one of col_index and row_index must be 'variable'")
            data_dict = {(extra_dim_sequence[i] or f"var_{i}"): da for i, da in enumerate(ds)}
            ds = xr.Dataset(data_dict)
        else:
            raise ValueError(f"Unsupported dataset type in sequence: {type(ds[0])}")

    panels: list[xr.DataArray] = []
    if isinstance(ds, xr.Dataset):
        # Guards
        if isinstance(row_index, str) and row_index != "variable" and row_index not in ds.coords:
            raise KeyError(f"row_index {row_index!r} not found in Dataset coords.")
        if isinstance(col_index, str) and col_index != "variable" and col_index not in ds.coords:
            raise KeyError(f"col_index {col_index!r} not found in Dataset coords.")

        if col_index == "variable":
            if not isinstance(row_index, str) or row_index == "variable":
                raise ValueError("row_index must be a coord name (string) when col_index='variable'")
            cols = [v for v in ds.data_vars if v != vars_to_be_removed]
            rows = list(ds.coords[row_index].values)
            for r in rows:
                for v in cols:
                    panels.append(ds[v].sel({row_index: r}))

        elif row_index == "variable":
            if not isinstance(col_index, str) or col_index == "variable":
                raise ValueError("col_index must be a coord name (string) when row_index='variable'")
            cols = list(ds.coords[col_index].values)
            rows = [v for v in ds.data_vars if v != vars_to_be_removed]
            for v in rows:
                for c in cols:
                    panels.append(ds[v].sel({col_index: c}))

        else:
            # both are coord names (strings) OR you can decide to support explicit label sequences later
            if not isinstance(col_index, str) or not isinstance(row_index, str):
                raise ValueError("For Dataset, non-'variable' indices must be coord names (strings).")
            cols = list(ds.coords[col_index].values)
            rows = list(ds.coords[row_index].values)
            # Choose which variable to plot elsewhere (or require a `var` argument)
            if len(ds.data_vars) != 1 and var_sel is None:
                raise ValueError("Dataset has multiple variables; specify which one to plot with var_sel.")
            if var_sel is None:
                v = list(ds.data_vars)[0]
            else:
                if var_sel in ds.data_vars:
                    v = var_sel
                else:
                    raise ValueError(f"Variable {var_sel} not in dataset.")
            for r in rows:
                for c in cols:
                    panels.append(ds[v].sel({row_index: r, col_index: c}))

    elif isinstance(ds, xr.DataArray):
        if not (isinstance(col_index, str) and isinstance(row_index, str)):
            raise ValueError("For xr.DataArray, col_index and row_index must be coord names (strings).")
        if col_index not in ds.coords or row_index not in ds.coords:
            raise KeyError("col_index/row_index not found in DataArray coords.")

        cols = list(ds.coords[col_index].values)
        rows = list(ds.coords[row_index].values)

        for r in rows:
            for c in cols:
                panels.append(ds.sel({row_index: r, col_index: c}))

    else:
        raise ValueError(f"Unsupported dataset type: {type(ds)}")

    nrows, ncols = len(rows), len(cols)
    if ncols == 0 or nrows == 0:
        raise ValueError("No axes found in dataset.")

    # Create panel grid
    if len(panels) != nrows * ncols:
        raise ValueError(f"Expected {nrows*ncols} panels, got {len(panels)}")

    return panels, rows, cols


def _detect_limits_from_panels(
    panels: Sequence | np.ndarray,
    mode: Optional[str | Sequence] = "quantile",  # "quantile" or "minmax"
    quantile: float = 0.02,
) -> tuple[float, float]:
    panels = np.array(panels)
    flat_panels = panels.ravel()

    mins: list[float] = []
    maxs: list[float] = []

    per_panel_limits = None
    global_limits = None

    for i, v in enumerate(flat_panels):
        vv = v
        if hasattr(vv.data, "persist"):
            vv = vv.persist()

        arr = np.asarray(vv.data)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue

        if per_panel_limits is not None:
            lo, hi = per_panel_limits[i]
            mins.append(float(lo)); maxs.append(float(hi))
        elif global_limits is not None:
            lo, hi = global_limits
            mins.append(float(lo)); maxs.append(float(hi))
        else:
            if mode == "quantile":
                mins.append(float(np.quantile(arr, quantile)))
                maxs.append(float(np.quantile(arr, 1 - quantile)))
            elif mode == "minmax":
                mins.append(float(arr.min()))
                maxs.append(float(arr.max()))
            else:
                raise ValueError(f"Limits mode not supported: {mode!r}")

    if not mins or not maxs:
        raise ValueError("All selected panels contain only NaNs/non-finite values.")

    mins, maxs = np.array(mins), np.array(maxs)
    mins, maxs = np.array(mins), np.array(maxs)
    return mins.reshape(panels.shape), maxs.reshape(panels.shape)

def _create_nparray_with_compatible_data(data, inner_data_type, rows, cols):
    if isinstance(data, Sequence) and not isinstance(data, str):
        if len(data)==len(rows):
            data_np = []
            for data_per_col in data:
                if isinstance(data_per_col, Sequence):
                    data_col = []
                    if len(data_per_col)==len(cols):
                        for data_per_row in data_per_col:
                            data_per_row = (None, None) if data_per_row is None else data_per_row # to avoid inhomogeneous array
                            data_col.append(data_per_row)
                        data_np.append(data_col)
                    else:
                        raise ValueError(f"Num of data per row {len(data_per_col)} != num panel cols {len(cols)}")
                else:
                    data_per_col = (None, None) if data_per_col is None else data_per_col # to avoid inhomogeneous array
                    data_np.append(len(rows)*[data_per_col])
        else:
            raise ValueError(f"Num of data per col {len(data)} != num panel rows {len(rows)}")
    elif isinstance(data, inner_data_type):
        data_np = len(rows)*[len(cols)*[data]]
    else:
        raise ValueError(f"Unsupported type {inner_data_type}")

    return np.array(data_np)

def _make_growing_norm(limits: Sequence[Number] | None):
    if (
        limits is None
        or (
            (isinstance(limits, Sequence) or isinstance(limits, np.ndarray))
            and len(limits)==2
            and limits[0] is None
            and limits[1] is None
        )
    ):
        return None

    if len(limits) != 2:
        raise ValueError(f"Limits length must be 2.")

    vmin, vmax = list(limits)
    if vmin < 0 < vmax:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        return Normalize(vmin=vmin, vmax=vmax)

def _crop_da_to_extent(da: xr.DataArray, extent, data_projection=None) -> xr.DataArray:
    lon_name = da.earthml.guessed_dims.longitude
    lat_name = da.earthml.guessed_dims.latitude

    lon_min, lon_max, lat_min, lat_max = extent

    # Convert extent from shifted PlateCarree coords back to regular lon coords
    lon0 = 0.0
    if data_projection is not None:
        try:
            lon0 = float(data_projection.proj4_params.get("lon_0", 0.0))
        except Exception:
            lon0 = 0.0

    # extent is given in the projection coordinates, convert back to real lon
    lon_min = ((lon_min + lon0 + 180) % 360) - 180
    lon_max = ((lon_max + lon0 + 180) % 360) - 180

    lon = da[lon_name]
    lat = da[lat_name]

    # normalize data lon to [-180, 180)
    lon180 = ((lon + 180) % 360) - 180

    lat_cond = (lat >= min(lat_min, lat_max)) & (lat <= max(lat_min, lat_max))

    if lon_min <= lon_max:
        lon_cond = (lon180 >= lon_min) & (lon180 <= lon_max)
    else:
        # dateline crossing
        lon_cond = (lon180 >= lon_min) | (lon180 <= lon_max)

    da = da.where(lat_cond, drop=True)
    da = da.where(lon_cond, drop=True)

    if da.size == 0:
        raise ValueError(
            f"Cropping produced empty array. "
            f"Requested extent={extent}, converted_lon_extent=({lon_min}, {lon_max}), "
            f"lon range=({float(lon180.min())}, {float(lon180.max())}), "
            f"lat range=({float(lat.min())}, {float(lat.max())})"
        )

    return da

def _sort_lon_for_plot(da: xr.DataArray) -> xr.DataArray:
    lon_name = da.earthml.guessed_dims.longitude
    lon = da[lon_name]

    lon_plot = _continuous_lon_values(lon.values)
    da = da.assign_coords({lon_name: lon_plot})
    da = da.sortby(lon_name)

    return da


def create_panel_from_data(
    ds: xr.Dataset | xr.DataArray | List[xr.Dataset] | List[xr.DataArray],
    col_index: str | Sequence[str],  # if str, look in Dataset, can't be both Sequences
    row_index: str | Sequence[str],
    var_sel: Optional[str] = None,  # ignored for xr.DataArray
    extra_dim_sequence: Optional[Sequence[str]] = None,  # optional sequence of names for the extra dimension (useful if ds is list)
    limits: Optional[str | dict | Sequence | Sequence[Sequence] | None] = None,  # "quantile" or "minmax" or dict with "rows" or "cols" keys or explicit limits
    limits_quantile: float = 0.02,
    save_path: str | Path | None = None,
    figsize_per_cell=(10, 4),
    cmap: str | Sequence[str] | Sequence[Sequence[str]] = "RdBu_r", # index 1: cols, index 2: rows
    cbar_orientation: str = "vertical",
    cbar_label: str = "",
    cbar_mode: str = "row",  # "row" or "subplot"
    gridlabel_mode: str = "outer",  # "outer" or "all"
    data_projection=ccrs.PlateCarree(),
    plt_col_titles: str | Sequence[str] | None = None,
    plt_row_titles: str | Sequence[str] | None = None,
    plt_ax_title_pre: str = "",
    plt_ax_title_suf: str = "",
    plt_fontsize: int = 12,
    plt_title_fontsize: int = 14,
    plt_projection=ccrs.PlateCarree(),  # Robinson()
    plt_coastlines: bool = True,
    plt_global_extent: bool = False,
    plt_regional_extent: Sequence[float] | None = None,  # lon_min, lon_max, lat_min, lat_max
    plt_pad_extent_deg: int = 0,
):
    if cbar_mode not in {"row", "subplot"}:
        raise ValueError(f"Invalid cbar_mode={cbar_mode!r}. Use 'row' or 'subplot'.")

    panels, rows, cols = _create_data_panel(ds, col_index, row_index, var_sel, extra_dim_sequence)
    nrows, ncols = len(rows), len(cols)

    if plt_regional_extent is None:
        ds_ref = panels[0]
        lon_name, lat_name = ds_ref.earthml.guessed_dims.longitude, ds_ref.earthml.guessed_dims.latitude
        if isinstance(ds_ref, xr.Dataset):
            vars_ref = [v for v in ds_ref.data_vars if v != "_has_var"]
            if len(vars_ref) != 0:
                da_ref = ds_ref[vars_ref[0]]
            else:
                raise ValueError("Cannot select valid reference DataArray. Check input")
        elif isinstance(ds_ref, xr.DataArray):
            da_ref = ds_ref
        else:
            raise ValueError(f"Reference Dataset / DataArray has invalid type {type(ds_ref)}")

        extent = _extent_from_da(
            da_ref,
            lon_name=lon_name,
            lat_name=lat_name,
            pad_deg=plt_pad_extent_deg,
        )
    else:
        extent = list(plt_regional_extent)
        # lon_min, lon_max, lat_min, lat_max = plt_regional_extent
        # if lon_max < lon_min:
        #     lon_max += 360
        # if lon_min < 0:
        #     lon_min += 360
        # if lon_max < 0:
        #     lon_max += 360
        # extent = [lon_min, lon_max, lat_min, lat_max]

    fig, axes = _create_plot_grid(
        nrows,
        ncols,
        figsize_per_cell,
        data_projection,
        plt_fontsize,
        plt_projection,
        plt_coastlines,
        plt_global_extent,
        extent,
        gridlabel_mode=gridlabel_mode,
    )

    # Cmaps
    cmaps = _create_nparray_with_compatible_data(cmap, str, rows, cols)

    # Limits
    norms = np.array(len(rows)*[len(cols)*[None]]) # default auto limits
    if isinstance(limits, str): # quantile or maxmin
        norm_flatten = norms.ravel()
        mins, maxs = _detect_limits_from_panels(panels, limits, limits_quantile)
        for i,_ in enumerate(norm_flatten):
            norm_flatten[i] = _make_growing_norm((mins[i], maxs[i]))
        norms = norm_flatten.reshape(norms.shape)

    elif isinstance(limits, Sequence):
        # Could be (vmin, vmax) or list of (vmin, vmax) of length = num of cols or list of lists of length = num of rows
        if len(limits) == 2 and isinstance(limits[0], Number) and isinstance(limits[1], Number):
            norm = _make_growing_norm(limits)
            norms = np.array(len(rows)*[len(cols)*[norm]])
        else:
            limits_array = _create_nparray_with_compatible_data(limits, tuple, rows, cols)
            for r,c in product(range(nrows), range(ncols)):
                norms[r,c] = _make_growing_norm(limits_array[r,c])

    elif isinstance(limits, dict): # dict with rows or cols
        if (
            "cols" in limits
            and isinstance(limits["cols"], Sequence)
            and len(limits["cols"]) == ncols
        ):
            for r,c in product(range(nrows), range(ncols)):
                norms[r,c] = _make_growing_norm(limits["cols"][c])

        elif (
            "rows" in limits
            and isinstance(limits["rows"], Sequence)
            and len(limits["rows"]) == nrows
        ):
            for r,c in product(range(nrows), range(ncols)):
                norms[r,c] = _make_growing_norm(limits["rows"][r]) 

    elif limits is None:
        pass

    else:
        raise ValueError("Unsupported limit type: limits must be a string ('minmax'/'quantile'), a (vmin, vmax) pair, or a list (of lists) of (vmin, vmax), same dims as panel.")

    mappables, cbar_axes = [], []

    logger.info("cmaps %s", cmaps)
    logger.info("limits %s", norms)

    for r in range(nrows):
        row_mappable = None

        for c in range(ncols):
            ax = axes[r, c]
            p_idx = r * ncols + c

            data = panels[p_idx]

            if plt_regional_extent is not None:
                data = _crop_da_to_extent(data, plt_regional_extent, data_projection=data_projection)

            data = _sort_lon_for_plot(data)

            # --- resolve row title ---
            if plt_row_titles is None:
                if extra_dim_sequence and len(extra_dim_sequence) == nrows:
                    base_row_title = f"{plt_ax_title_pre}{extra_dim_sequence[r]}{plt_ax_title_suf}"
                else:
                    base_row_title = f"{plt_ax_title_pre}{plt_ax_title_suf}"
            elif isinstance(plt_row_titles, str):
                base_row_title = plt_row_titles
            elif isinstance(plt_row_titles, Sequence):
                if len(plt_row_titles) != nrows:
                    raise ValueError("plt_row_titles must have length nrows")
                base_row_title = str(plt_row_titles[r])
            else:
                raise ValueError(
                    f"plt_row_titles must be str, sequence of str, or None, not {type(plt_row_titles)}"
                )

            # --- resolve column title ---
            if plt_col_titles is None:
                col_val = data.coords[col_index].values if isinstance(col_index, str) else col_index[c]
                col_title = f"{col_index}: {col_val}"
            elif isinstance(plt_col_titles, str):
                col_title = plt_col_titles
            elif isinstance(plt_col_titles, Sequence):
                if len(plt_col_titles) != ncols:
                    raise ValueError("plt_col_titles must have length ncols")
                col_title = str(plt_col_titles[c])
            else:
                raise ValueError(
                    f"plt_col_titles must be str, sequence of str, or None, not {type(plt_col_titles)}"
                )

            # combine both
            if base_row_title and col_title:
                ax_title = f"{base_row_title} - {col_title}"
            else:
                ax_title = base_row_title or col_title

            try:
                im = data.plot.pcolormesh(
                    ax=ax,
                    add_colorbar=False,
                    cmap=cmaps[r,c],
                    transform=data_projection,
                    norm=norms[r,c],
                )
            except Exception as e:
                ax.set_title("ERROR")
                ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.axis("off")
                im = None

            mappables.append(im)
            ax.set_title(ax_title, fontsize=plt_title_fontsize)

            if im is not None:
                row_mappable = im

            # One colorbar per subplot
            if cbar_mode == "subplot" and im is not None:
                divider = make_axes_locatable(ax)
                if cbar_orientation == "horizontal":
                    logger.warning("Colorbar horizontal orientation not yet supported")
                else:
                    cax = divider.append_axes(
                        "right",
                        size="2.5%",
                        pad=0.04,
                        axes_class=plt.Axes,
                    )
                    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
                    cbar.ax.tick_params(labelsize=plt_fontsize)
                    cbar.set_label(cbar_label, fontsize=plt_fontsize)
                    cbar_axes.append(cax)

        # One colorbar per row
        if cbar_mode == "row" and row_mappable is not None:
            last_ax = axes[r, ncols - 1]
            divider = make_axes_locatable(last_ax)
            if cbar_orientation == "horizontal":
                logger.warning("Colorbar horizontal orientation not yet supported")
            else:
                cax = divider.append_axes(
                    "right",
                    size="2.5%",
                    pad=0.04,
                    axes_class=plt.Axes,
                )
                cbar = fig.colorbar(row_mappable, cax=cax, orientation="vertical")
                cbar.ax.tick_params(labelsize=plt_fontsize)
                cbar.set_label(cbar_label, fontsize=plt_fontsize)
                cbar_axes.append(cax)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    plt.close(fig)

    return {
        "fig": fig,
        "axes": axes,
        "cbar_axes": cbar_axes,
        "mappable": mappables,
        "data": panels,
    }


def _reduce_realization_for_plot(
    ds: xr.Dataset | None,
    metric_category: str,
    realization: int | str | None,
) -> tuple[xr.Dataset | None, int | str]:
    """
    Reduce the realization dimension for plotting.

    Rules
    -----
    - deterministic:
        * int realization -> select that member
        * otherwise -> mean over realization if present
    - ensemble/probabilistic:
        * leave untouched
    """
    if ds is None:
        return None, realization if realization is not None else "all"

    if metric_category != "deterministic":
        return ds.compute(), realization if realization is not None else "all"

    ds = ds.compute()

    if "realization" not in ds.dims:
        return ds, realization if realization is not None else "all"

    if isinstance(realization, int):
        return ds.isel(realization=realization), realization

    return ds.mean(dim="realization"), "all"


def _resolve_plot_titles_and_suffixes(
    metric_name: str,
    metric_category: str,
    include_pr: bool,
    include_diff: bool,
    realization_label: int | str,
) -> tuple[list[str], list[str]]:
    metric_label = metric_name.upper()
    metric_suffix = " ENS" if metric_category == "ensemble" else ""

    titles = [f"{metric_label}{metric_suffix} FCvsAN"]
    suffixes = [f"FCvsAN [R={realization_label}]"]

    if include_pr:
        titles.append(f"{metric_label}{metric_suffix} MLFCvsAN")
        suffixes.append(f"MLFCvsAN [R={realization_label}]")

    if include_pr and include_diff:
        titles.append(f"{metric_label}{metric_suffix} MLFCvsAN - FCvsAN")
        suffixes.append(f"MLFCvsAN - FCvsAN [R={realization_label}]")

    return titles, suffixes


def _build_panel_dataset_for_variable(
    var_name: str,
    ds_fc: xr.Dataset,
    ds_pr: xr.Dataset | None,
    metric_name: str,
    metric_category: str,
    plot_diff: bool,
    realization_label: int | str,
) -> tuple[xr.Dataset, list[str], list[str]]:
    """
    Build one concatenated plotting dataset for a single variable.
    """
    include_pr = ds_pr is not None
    titles, suffixes = _resolve_plot_titles_and_suffixes(
        metric_name=metric_name,
        metric_category=metric_category,
        include_pr=include_pr,
        include_diff=plot_diff,
        realization_label=realization_label,
    )

    das = [ds_fc[var_name]]

    if include_pr:
        da_pr = ds_pr[var_name]
        das.append(da_pr)

        if plot_diff:
            das.append((da_pr - ds_fc[var_name]).assign_coords(model="pr-fc"))

    plot_dim = metric_name
    ds_concat = xr.concat(das, dim=plot_dim).assign_coords({plot_dim: suffixes})

    return ds_concat, titles, suffixes


def _resolve_cmap_list(
    var_names: Sequence[str],
    metric_name: str,
    plot_config: dict[str, dict[str, Any]],
    n_base_cols: int,
    include_diff: bool,
    diff_cmap: str = "RdBu_r",
) -> list[list[str]]:
    out = []
    for var in var_names:
        cmap = get_var_plot_cmap(plot_config, var, metric_name)
        cmaps = [cmap] * n_base_cols
        if include_diff:
            cmaps.append(diff_cmap)
        out.append(cmaps)
    return out


def _resolve_limits_list(
    var_names: Sequence[str],
    metric_name: str,
    plot_config: dict[str, dict[str, Any]],
    n_base_cols: int,
    include_diff: bool,
) -> list[list[Any]]:
    out = []
    for var in var_names:
        metric_limits = get_var_plot_limits(plot_config, var, metric_name)
        limits = [metric_limits] * n_base_cols
        if include_diff:
            diff_limits = get_var_plot_limits(plot_config, var, f"diff_{metric_name}")
            limits.append(diff_limits)
        out.append(limits)
    return out

def _resolve_variable_titles(
    var_names: Sequence[str],
    plot_config: dict[str, dict[str, Any]],
) -> list[str]:
    titles = []
    for var in var_names:
        cfg = plot_config.get(var, {})
        titles.append(cfg.get("title", cfg.get("long_name", var)))
    return titles

def plot_metric_map_panel(
    metrics: dict[str, dict[str, xr.Dataset]],
    var_plot_list: Sequence[str],
    var_plot_config: dict[str, dict[str, Any]] | None = None,
    *,
    metric_category: str = "ensemble",
    metric_dimension: str = "map",
    metric_name: str = "rmse",
    leadtime: Any,
    train_period: Any,
    loss: Any | None = None,
    model_fc: str = "fc",
    model_pr: str = "pr",
    plot_pr: bool = True,
    plot_diff: bool = True,
    realization: int | str | None = "all",
    center_lon: float = 0,
    plt_regional_extent: tuple[float, float, float, float] | None = None,
    plt_global_extent: bool = False,
    figsize_per_cell: tuple[float, float] = (12, 8),
    gridlabel_mode: str = "all",
    cbar_mode: str = "subplot",
    plt_fontsize: int = 16,
    plt_title_fontsize: int = 30,
    plt_ax_title_pre: str = "",
    plt_projection = None,
    data_projection = None,
    variable_titles: Sequence[str] | None = None,
):
    """
    Plot a panel of metric maps from the nested metrics output returned by
    get_runs_and_metrics(...).

    Parameters
    ----------
    metrics
        Nested metrics dict: metrics[metric_category][metric_dimension].
    var_plot_list
        Variables to plot, in row order.
    var_plot_config
        Plot metadata/config per variable.
    metric_category
        One of "deterministic", "ensemble", "probabilistic".
    metric_dimension
        One of "scalar", "map", "timeseries". This function expects "map".
    metric_name
        Metric name to select from the metric dimension, e.g. "rmse".
    leadtime, train_period, loss
        Selection coordinates.
    plot_pr
        Whether to include PR/MLFC maps.
    plot_diff
        Whether to include PR-FC difference maps. Only used if plot_pr=True.
    realization
        Deterministic-only realization selection. Use int for one member, or
        anything else/"all" to average over realization.

    Returns
    -------
    A dict containing "fig".
    """
    if metric_dimension != "map":
        raise ValueError("plot_metric_map_panel currently expects metric_dimension='map'")

    ds_metric = metrics.get(metric_category, {}).get(metric_dimension)
    if ds_metric is None or not ds_metric.data_vars:
        raise ValueError(f"No metrics found for metrics[{metric_category!r}][{metric_dimension!r}]")

    sel_kwargs = {
        "metric": metric_name,
        "leadtime": leadtime,
        "train_period": train_period,
    }
    if loss is not None and "loss" in ds_metric.coords:
        sel_kwargs["loss"] = loss

    if var_plot_config is None:
        var_plot_config = DEFAULT_PLOT_CONFIG

    ds_fc = ds_metric.sel(**sel_kwargs, model=model_fc)
    ds_pr = ds_metric.sel(**sel_kwargs, model=model_pr) if plot_pr and model_pr in ds_metric.model.values else None

    ds_fc, realization_label = _reduce_realization_for_plot(
        ds=ds_fc,
        metric_category=metric_category,
        realization=realization,
    )
    ds_pr, _ = _reduce_realization_for_plot(
        ds=ds_pr,
        metric_category=metric_category,
        realization=realization,
    )

    include_pr = ds_pr is not None
    include_diff = include_pr and plot_diff
    n_base_cols = 2 if include_pr else 1

    datasets = []
    titles = None
    suffixes = None

    for var in var_plot_list:
        ds_concat, titles_this_var, suffixes_this_var = _build_panel_dataset_for_variable(
            var_name=var,
            ds_fc=ds_fc,
            ds_pr=ds_pr,
            metric_name=metric_name,
            metric_category=metric_category,
            plot_diff=include_diff,
            realization_label=realization_label,
        )
        datasets.append(ds_concat)

        if titles is None:
            titles = titles_this_var
            suffixes = suffixes_this_var

    cmap_list = _resolve_cmap_list(
        var_names=var_plot_list,
        metric_name=metric_name,
        plot_config=var_plot_config,
        n_base_cols=n_base_cols,
        include_diff=include_diff,
    )

    limits_list = _resolve_limits_list(
        var_names=var_plot_list,
        metric_name=metric_name,
        plot_config=var_plot_config,
        n_base_cols=n_base_cols,
        include_diff=include_diff,
    )

    extra_dim_sequence = [v for v in ds_fc.data_vars if v != "_has_var"]

    if variable_titles is None:
        variable_titles = _resolve_variable_titles(var_plot_list, var_plot_config)

    panels = create_panel_from_data(
        datasets,
        col_index=metric_name,
        row_index="variable",
        extra_dim_sequence=extra_dim_sequence,
        cmap=cmap_list,
        plt_projection=plt_projection or ccrs.PlateCarree(central_longitude=center_lon),
        data_projection=data_projection or ccrs.PlateCarree(),
        plt_global_extent=plt_global_extent,
        plt_regional_extent=plt_regional_extent,
        limits=limits_list,
        figsize_per_cell=figsize_per_cell,
        cbar_mode=cbar_mode,
        gridlabel_mode=gridlabel_mode,
        plt_col_titles=titles,
        plt_row_titles=variable_titles,   # NEW
        plt_fontsize=plt_fontsize,
        plt_title_fontsize=plt_title_fontsize,
        plt_ax_title_pre=plt_ax_title_pre,
    )
    return panels
