from pathlib import Path

import cf_xarray
import xarray as xr

import matplotlib.pyplot as plt


def quickplot(
    ds: xr.Dataset,
    varname: str = "",
    folder: str | Path = "./",
    filename: str = "rolled.png",
    t_idx: int = 0
):
    """Quick diagnostic plot of a 2D field in ds."""
    da = ds[varname] if isinstance(ds, xr.Dataset) else ds

    # Select time slice
    while da.ndim > 2:
        da = da.isel({da.dims[0]: t_idx})

    # Get lon/lat
    lon_coord, lat_coord = ds.earthml.guessed_coords.longitude, ds.earthml.guessed_coords.latitude
    lon = ds[lon_coord]
    lat = ds[lat_coord]

    plt.figure(figsize=(10,5))
    plt.pcolormesh(lon, lat, da, shading="auto")
    plt.colorbar(label=varname)
    plt.title(f"{varname}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    name, _, ext = filename.rpartition('.')
    plt.savefig(f"{folder}/{name}_{t_idx}.{ext}", dpi=150)
    plt.close()
