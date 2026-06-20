import itertools
import numpy as np
import xarray as xr


class EarthMLClimatology:
    def climatology(
        self,
        groupby: str = "month",
        time_dim: str | None = None,
        reduce_dim: str | None = None,
        start: str | None = None,
        end: str | None = None,
        keep_attrs: bool = True,
    ) -> xr.Dataset:
        ds = self._obj

        if time_dim is None:
            time_dim = ds.earthml.guessed_dims.time

        if time_dim not in ds.coords and time_dim not in ds:
            raise ValueError(f"{time_dim!r} is not a coordinate or variable")

        ref_time = ds[time_dim]

        if ref_time.size == 0:
            raise ValueError(f"{time_dim!r} is empty; cannot compute climatology")

        if not hasattr(ref_time, "dt"):
            raise TypeError(f"{time_dim!r} must be datetime-like")

        # Cast floating variables before reduction.
        float_vars = [
            v for v in ds.data_vars
            if np.issubdtype(ds[v].dtype, np.floating)
        ]
        if float_vars:
            ds = ds.assign({
                v: ds[v].astype(np.float32)
                for v in float_vars
            })

        # Case 1: normal 1D time coordinate.
        if ref_time.ndim == 1:
            actual_time_dim = ref_time.dims[0]

            if reduce_dim is None:
                reduce_dim = actual_time_dim

            if start is not None or end is not None:
                if actual_time_dim in ds.dims:
                    mask = xr.ones_like(ref_time, dtype=bool)
                    if start is not None:
                        mask = mask & (ref_time >= np.datetime64(start))
                    if end is not None:
                        mask = mask & (ref_time <= np.datetime64(end))
                    ds = ds.where(mask, drop=True)
                    ref_time = ds[time_dim]

            group = getattr(ref_time.dt, groupby).compute()
            ds = ds.assign_coords({groupby: (actual_time_dim, group.values)})

            return ds.groupby(groupby).mean(
                dim=reduce_dim,
                keep_attrs=keep_attrs,
                engine="numpy",
            )

        # Case 2: multidimensional reference time, e.g.
        # climatology_time(leadtime, time).
        if reduce_dim is None:
            if "time" in ref_time.dims:
                reduce_dim = "time"
            else:
                raise ValueError(
                    f"{time_dim!r} is multidimensional with dims {ref_time.dims}. "
                    "Please provide reduce_dim explicitly."
                )

        if reduce_dim not in ref_time.dims:
            raise ValueError(
                f"reduce_dim={reduce_dim!r} is not one of {time_dim!r} dims "
                f"{ref_time.dims}"
            )

        preserve_dims = [d for d in ref_time.dims if d != reduce_dim]

        pieces = []

        for indices in itertools.product(*[range(ds.sizes[d]) for d in preserve_dims]):
            indexer = dict(zip(preserve_dims, indices))

            sub = ds.isel(indexer)
            sub_ref_time = ref_time.isel(indexer)

            if start is not None or end is not None:
                mask = xr.ones_like(sub_ref_time, dtype=bool)
                if start is not None:
                    mask = mask & (sub_ref_time >= np.datetime64(start))
                if end is not None:
                    mask = mask & (sub_ref_time <= np.datetime64(end))

                sub = sub.where(mask, drop=True)
                sub_ref_time = sub_ref_time.where(mask, drop=True)

            group = getattr(sub_ref_time.dt, groupby).compute()

            sub = sub.assign_coords({
                groupby: (reduce_dim, group.values)
            })

            clim = sub.groupby(groupby).mean(
                dim=reduce_dim,
                keep_attrs=keep_attrs,
                engine="numpy",
            )

            for dim, i in indexer.items():
                clim = clim.expand_dims({
                    dim: ds[dim].isel({dim: i}).values.reshape(1)
                })

            pieces.append(clim)

        if len(preserve_dims) == 1:
            dim = preserve_dims[0]
            out = xr.concat(pieces, dim=ds[dim])
        else:
            out = xr.combine_by_coords(pieces)

        # Keep dimension order predictable.
        ordered_dims = [
            d for d in ds.dims if d != reduce_dim and d in out.dims
        ]
        if groupby in out.dims:
            ordered_dims.append(groupby)

        remaining = [d for d in out.dims if d not in ordered_dims]
        out = out.transpose(*ordered_dims, *remaining)

        return out
