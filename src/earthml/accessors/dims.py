from typing import Collection
import warnings

import xarray as xr

from ..base import Dims

class EarthMLDims:
    def remove_dims_and_coords(
        self,
        allowed_dims: Collection[str] | Dims,
        *,
        drop_non_dim_coords: bool = False,
        attrs_key: str = "removed_metadata",
    ) -> xr.Dataset:
        ds = self._obj
        ds_out = ds.copy()
        allowed_dims = set(allowed_dims)

        removed_metadata: dict[str, dict[str, object]] = {}

        # Remove unwanted dimensions safely (only if size == 1)
        dims_to_remove = [d for d in ds_out.dims if d not in allowed_dims]

        for dim in dims_to_remove:
            if ds_out.sizes[dim] == 1:
                dim_info: dict[str, object] = {
                    "kind": "dim",
                    "size": int(ds_out.sizes[dim]),
                }

                if dim in ds_out.coords:
                    coord = ds_out.coords[dim]
                    try:
                        dim_info["value"] = coord.values.tolist()
                    except Exception:
                        dim_info["value"] = str(coord.values)

                    if coord.attrs:
                        dim_info["attrs"] = dict(coord.attrs)

                removed_metadata[dim] = dim_info
                ds_out = ds_out.squeeze(dim=dim, drop=True)
            else:
                warnings.warn(
                    f"Cannot remove dimension '{dim}' because its size is "
                    f"{ds_out.sizes[dim]} (>1). Dimension left unchanged."
                )

        # Drop coords that still depend on removed/disallowed dims
        coords_to_drop = [
            cname
            for cname, coord in ds_out.coords.items()
            if not set(coord.dims).issubset(allowed_dims)
        ]

        for cname in coords_to_drop:
            coord = ds_out.coords[cname]
            coord_info: dict[str, object] = {
                "kind": "coord",
                "dims": tuple(coord.dims),
            }

            try:
                coord_info["value"] = coord.values.tolist()
            except Exception:
                coord_info["value"] = str(coord.values)

            if coord.attrs:
                coord_info["attrs"] = dict(coord.attrs)

            removed_metadata[cname] = coord_info

        if coords_to_drop:
            ds_out = ds_out.drop_vars(coords_to_drop)

        # Optionally drop all remaining non-dimension coordinates
        if drop_non_dim_coords:
            non_dim_coords = [cname for cname in ds_out.coords if cname not in ds_out.dims]

            for cname in non_dim_coords:
                coord = ds_out.coords[cname]
                coord_info: dict[str, object] = {
                    "kind": "coord",
                    "dims": tuple(coord.dims),
                }

                try:
                    coord_info["value"] = coord.values.tolist()
                except Exception:
                    coord_info["value"] = str(coord.values)

                if coord.attrs:
                    coord_info["attrs"] = dict(coord.attrs)

                removed_metadata[cname] = coord_info

            if non_dim_coords:
                ds_out = ds_out.drop_vars(non_dim_coords)

        if removed_metadata:
            ds_out.attrs = dict(ds_out.attrs)
            ds_out.attrs[attrs_key] = removed_metadata

        return ds_out


    def rename_dim_and_coord(self, ds: xr.Dataset, src: str, target: str):
        rename_map = {}

        if src in ds.dims:
            rename_map[src] = target
        if src in ds.coords or src in ds.data_vars:
            rename_map[src] = target

        if rename_map:
            ds = ds.rename(rename_map)

        return ds

    def normalize_dims_and_coords(self):
        """
        Normalize the input dataset by renaming guessed dimensions to standard names.
        """
        ds = self._obj
        for src_dim, target_dim in (
            (ds.earthml.guessed_dims.time, "time"),
            (ds.earthml.guessed_dims.latitude, "latitude"),
            (ds.earthml.guessed_dims.longitude, "longitude"),
            (ds.earthml.guessed_dims.realization, "realization"),
            (ds.earthml.guessed_dims.leadtime, "leadtime"),
        ):
            if src_dim is not None and src_dim != target_dim:
                ds = self.rename_dim_and_coord(ds, src_dim, target_dim)

        return ds
