from typing import Collection
import warnings

class EarthMLDims:
    def remove_dims_and_coords(
        self,
        allowed_dims: Collection[str],
    ):
        ds = self._obj
        ds_out = ds.copy()

        # Remove unwanted dimensions safely (only if size == 1)
        dims_to_remove = [d for d in ds_out.dims if d not in allowed_dims]

        for dim in dims_to_remove:
            if ds_out.sizes[dim] == 1:
                ds_out = ds_out.squeeze(dim=dim, drop=True)
            else:
                warnings.warn(
                    f"Cannot remove dimension '{dim}' because its size is "
                    f"{ds_out.sizes[dim]} (>1). Dimension left unchanged."
                )

        # Drop coordinates that depend on unwanted dimensions
        coords_to_drop = [
            cname for cname, coord in ds_out.coords.items()
            if not set(coord.dims).issubset(allowed_dims)
        ]

        if coords_to_drop:
            ds_out = ds_out.drop_vars(coords_to_drop)

        return ds_out
