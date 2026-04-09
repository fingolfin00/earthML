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

    def rename_dim_and_coord(
        self,
        src: str,
        target: str
    ):
        """
        Rename a dimension and any coordinate/variable with the same name,
        and clean up leftover old-name coords/vars.

        Behavior:
        1. If `src` is a dimension, rename it to `target`.
        2. If a coord or data_var named `src` exists, attempt to rename it to `target`.
        3. After step 1-2, if an object named `src` still exists:
            - If `target` does NOT exist (in coords or data_vars), rename `src` -> `target`.
            - Else (target already exists) drop the leftover `src` to avoid duplicate names.

        Notes:
        - This function prefers dropping an old-named variable if the target name is already present
            (to avoid collisions / inconsistent shapes).
        """
        ds = self._obj

        # Rename dimension first (xarray's rename_dims)
        if src in ds.dims:
            ds = ds.rename_dims({src: target})

        # If there's a variable or coordinate with the source name, try renaming it.
        # Use rename_vars which handles both coords and data_vars; rename_vars will fail if
        # target already exists, so we guard against that.
        if (src in ds.coords) or (src in ds.data_vars):
            if target not in ds.coords and target not in ds.data_vars:
                # safe to rename directly
                ds = ds.rename_vars({src: target})
            else:
                # target already exists: decide policy — drop the old src to avoid duplicates
                # But if shapes are identical and you want to keep src -> merge, modify here.
                # We'll prefer dropping the leftover src variable/coord.
                if src in ds.coords:
                    ds = ds.drop_vars(src)
                if src in ds.data_vars:
                    # drop_vars will also drop data_vars; coords already handled above.
                    ds = ds.drop_vars(src)

        # Defensive pass: if something named `src` still exists (e.g., created by xarray internals),
        # attempt final resolution: if target free -> rename, else drop.
        if (src in ds.coords) or (src in ds.data_vars):
            if (target not in ds.coords) and (target not in ds.data_vars):
                ds = ds.rename_vars({src: target})
            else:
                ds = ds.drop_vars(src)

        return ds

    def normalize_dims_and_coords(self):
        """
        Normalize the input dataset by renaming guessed dimensions to standard names.
        """
        ds = self._obj
        for guessed_dim in (
            (ds.earthml.guessed_dims.time, "time"),
            (ds.earthml.guessed_dims.latitude, "latitude"),
            (ds.earthml.guessed_dims.longitude, "longitude"),
            (ds.earthml.guessed_dims.realization, "realization"),
            (ds.earthml.guessed_dims.leadtime, "leadtime")
        ):
            src_dim, target_dim = guessed_dim
            if src_dim is not None and src_dim != target_dim:
                ds = self.rename_dim_and_coord(src_dim, target_dim)

        return ds
