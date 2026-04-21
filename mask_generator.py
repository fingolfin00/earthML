from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from earthml.experiments.mlbc.catalog import make_catalog

try:
    from global_land_mask import globe
except ImportError as exc:  # pragma: no cover - depends on local env
    raise SystemExit(
        "mask_generator.py requires the optional package 'global-land-mask'. "
        "Install it with: pip install global-land-mask"
    ) from exc


USER_KNOBS = {
    # Any key from earthml.experiments.mlbc.catalog.make_catalog().region
    "region": "med",
    # "land" keeps land cells True, "ocean" keeps ocean cells True.
    "mask_kind": "ocean",
    # Grid resolution in degrees. You can also pass a pair like (0.5, 1.0).
    "resolution": 0.04,
    # Override the catalog bounds if needed.
    "lon_bounds": None,
    "lat_bounds": None,
    # Output settings.
    "output_path": None,
    "output_format": "netcdf",  # "netcdf" or "zarr"
    "save_plot": True,
    "plot_path": None,
    "plot_dpi": 200,
    "variable_name": "mask",
    # Coordinate names expected by the metrics pipeline.
    "latitude_name": "latitude",
    "longitude_name": "longitude",
}


def _available_regions() -> dict[str, object]:
    catalog = make_catalog()
    return vars(catalog.region)


def _parse_resolution(value: str | float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Resolution tuples must have exactly two values: (lat_res, lon_res)")
        return float(value[0]), float(value[1])

    if isinstance(value, (int, float)):
        res = float(value)
        return res, res

    text = str(value).strip()
    if "," in text:
        left, right = text.split(",", maxsplit=1)
        return float(left), float(right)

    res = float(text)
    return res, res


def _build_axis(lower: float, upper: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Resolution must be strictly positive")

    start = min(lower, upper)
    stop = max(lower, upper)
    # Include the upper edge when it falls on the grid.
    n = int(np.floor((stop - start) / step + 1e-9))
    values = start + np.arange(n + 1, dtype=np.float64) * step
    if values[-1] < stop - 1e-9:
        values = np.append(values, stop)
    return values


def _default_output_path(region_key: str, mask_kind: str, resolution: tuple[float, float], output_format: str) -> Path:
    lat_res, lon_res = resolution
    res_tag = f"{lat_res:g}x{lon_res:g}".replace(".", "p")
    suffix = ".zarr" if output_format == "zarr" else ".nc"
    return Path.cwd() / f"{region_key}_{mask_kind}_mask_{res_tag}deg{suffix}"


def _default_plot_path(output_path: Path) -> Path:
    if output_path.suffix == ".zarr":
        return output_path.with_suffix(".png")
    return output_path.with_suffix(".png")


def build_mask_dataset(
    *,
    region_key: str,
    mask_kind: str,
    resolution: float | str | tuple[float, float],
    variable_name: str = "mask",
    latitude_name: str = "latitude",
    longitude_name: str = "longitude",
    lon_bounds: tuple[float, float] | None = None,
    lat_bounds: tuple[float, float] | None = None,
) -> xr.Dataset:
    regions = _available_regions()
    if region_key not in regions:
        raise ValueError(
            f"Unknown region {region_key!r}. Available regions: {sorted(regions)}"
        )

    region = regions[region_key]
    lat_res, lon_res = _parse_resolution(resolution)

    lon_lo, lon_hi = lon_bounds if lon_bounds is not None else region.lon
    lat_lo, lat_hi = lat_bounds if lat_bounds is not None else region.lat

    lats = _build_axis(lat_lo, lat_hi, lat_res)
    lons = _build_axis(lon_lo, lon_hi, lon_res)

    lon2d, lat2d = np.meshgrid(lons, lats)

    # global_land_mask expects longitudes on the conventional [-180, 180) range.
    lon_eval = ((lon2d + 180.0) % 360.0) - 180.0
    land_mask = globe.is_land(lat2d, lon_eval)

    mask_kind_norm = mask_kind.strip().lower()
    if mask_kind_norm == "land":
        mask = land_mask
    elif mask_kind_norm == "ocean":
        mask = ~land_mask
    else:
        raise ValueError("mask_kind must be either 'land' or 'ocean'")

    ds = xr.Dataset(
        {
            variable_name: ((latitude_name, longitude_name), mask.astype(bool)),
        },
        coords={
            latitude_name: lats,
            longitude_name: lons,
        },
        attrs={
            "region_key": region_key,
            "region_name": region.name,
            "mask_kind": mask_kind_norm,
            "lat_resolution_deg": lat_res,
            "lon_resolution_deg": lon_res,
            "lon_bounds": tuple(float(v) for v in (lon_lo, lon_hi)),
            "lat_bounds": tuple(float(v) for v in (lat_lo, lat_hi)),
            "generator": "earthML/mask_generator.py",
        },
    )

    return ds


def save_mask_dataset(ds: xr.Dataset, output_path: str | Path, output_format: str) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "zarr":
        ds.to_zarr(output_path, mode="w")
    elif output_format == "netcdf":
        ds.to_netcdf(output_path)
    else:
        raise ValueError("output_format must be either 'netcdf' or 'zarr'")

    return output_path


def save_mask_plot(
    ds: xr.Dataset,
    *,
    variable_name: str,
    plot_path: str | Path,
    dpi: int = 200,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_path = Path(plot_path).expanduser()
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    mask = ds[variable_name].astype(int)
    lon_name = mask.dims[1]
    lat_name = mask.dims[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    mask.plot(
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=1,
        add_colorbar=True,
        cbar_kwargs={"ticks": [0, 1], "label": "Mask"},
    )
    ax.set_title(
        f"{ds.attrs.get('region_name', ds.attrs.get('region_key', 'region'))} "
        f"{ds.attrs.get('mask_kind', 'mask')} mask"
    )
    ax.set_xlabel(lon_name)
    ax.set_ylabel(lat_name)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a land/ocean mask on a regular lat/lon grid using catalog.py regions.",
    )
    parser.add_argument("--region", default=USER_KNOBS["region"], help="Catalog region key, e.g. conus, europe, natlantic")
    parser.add_argument("--mask-kind", default=USER_KNOBS["mask_kind"], choices=("land", "ocean"))
    parser.add_argument(
        "--resolution",
        default=USER_KNOBS["resolution"],
        help="Grid resolution in degrees. Use one value like 1.0 or a pair like 0.5,1.0 for lat,lon.",
    )
    parser.add_argument("--lon-bounds", default=None, help="Optional lon bounds override as min,max")
    parser.add_argument("--lat-bounds", default=None, help="Optional lat bounds override as min,max")
    parser.add_argument("--output-path", default=USER_KNOBS["output_path"])
    parser.add_argument("--output-format", default=USER_KNOBS["output_format"], choices=("netcdf", "zarr"))
    parser.add_argument("--save-plot", action=argparse.BooleanOptionalAction, default=USER_KNOBS["save_plot"])
    parser.add_argument("--plot-path", default=USER_KNOBS["plot_path"])
    parser.add_argument("--plot-dpi", type=int, default=USER_KNOBS["plot_dpi"])
    parser.add_argument("--variable-name", default=USER_KNOBS["variable_name"])
    parser.add_argument("--latitude-name", default=USER_KNOBS["latitude_name"])
    parser.add_argument("--longitude-name", default=USER_KNOBS["longitude_name"])
    parser.add_argument("--list-regions", action="store_true", help="Print available catalog regions and exit")
    return parser.parse_args()


def _parse_bounds(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    left, right = str(text).split(",", maxsplit=1)
    return float(left), float(right)


def main() -> None:
    args = parse_args()
    regions = _available_regions()

    if args.list_regions:
        print("Available regions:")
        for key, region in sorted(regions.items()):
            print(f"  {key}: {region.name} lon={region.lon} lat={region.lat}")
        return

    resolution = _parse_resolution(args.resolution)
    output_path = (
        Path(args.output_path).expanduser()
        if args.output_path is not None
        else _default_output_path(args.region, args.mask_kind, resolution, args.output_format)
    )

    ds = build_mask_dataset(
        region_key=args.region,
        mask_kind=args.mask_kind,
        resolution=resolution,
        variable_name=args.variable_name,
        latitude_name=args.latitude_name,
        longitude_name=args.longitude_name,
        lon_bounds=_parse_bounds(args.lon_bounds),
        lat_bounds=_parse_bounds(args.lat_bounds),
    )
    saved_path = save_mask_dataset(ds, output_path=output_path, output_format=args.output_format)
    plot_path = None
    if args.save_plot:
        plot_path = save_mask_plot(
            ds,
            variable_name=args.variable_name,
            plot_path=args.plot_path if args.plot_path is not None else _default_plot_path(saved_path),
            dpi=args.plot_dpi,
        )

    region = regions[args.region]
    print(
        f"Saved {args.mask_kind} mask for region={args.region} ({region.name}) "
        f"to {saved_path}"
    )
    if plot_path is not None:
        print(f"Saved mask preview plot to {plot_path}")
    print(
        f"Variable={args.variable_name}, dims={dict(ds.sizes)}, "
        f"resolution=({ds.attrs['lat_resolution_deg']}, {ds.attrs['lon_resolution_deg']}) deg"
    )


if __name__ == "__main__":
    main()
