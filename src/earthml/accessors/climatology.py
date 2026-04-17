import numpy as np
import xarray as xr


class EarthMLClimatology:
    def climatology(
        self,
        groupby: str = "month",
        time_dim: str | None = None,
        start: str | None = None,
        end: str | None = None,
        keep_attrs: bool = True,
    ) -> xr.Dataset:
        ds = self._obj

        if time_dim is None:
            time_dim = ds.earthml.guessed_dims.time

        if start is not None or end is not None:
            ds = ds.sel({time_dim: slice(start, end)})

        time = ds[time_dim]
        if time.size == 0:
            raise ValueError(f"{time_dim!r} is empty; cannot compute climatology")

        if not hasattr(time, "dt"):
            raise TypeError(f"{time_dim!r} must be datetime-like")

        group = getattr(time.dt, groupby)
        ds = ds.assign_coords({groupby: (time_dim, group.data)})

        # Cast floating variables before grouped reduction is built to avoid
        # mixed dtypes, but leave bookkeeping vars like `_has_var` untouched.
        float_vars = [
            v
            for v in ds.data_vars
            if np.issubdtype(ds[v].dtype, np.floating)
        ]
        if float_vars:
            ds = ds.assign({
                v: ds[v].astype(np.float32)
                for v in float_vars
            })

        return ds.groupby(groupby).mean(
            dim=time_dim,
            keep_attrs=keep_attrs,
            engine="numpy", # safest option
        )
