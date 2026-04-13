import xarray as xr


class EarthMLInpaintLonLat:
    def inpaint_nan_lonlat_bilinear(self) -> xr.Dataset:
        ds = self._obj
        lat_dim = ds.earthml.guessed_dims.latitude
        lon_dim = ds.earthml.guessed_dims.longitude

        # interpolate_na can become very memory-hungry when upstream datasets are
        # stored as one giant chunk across time/realization. Split the non-spatial
        # axes into smaller chunks first while keeping full lat/lon slices.
        if ds.chunks is not None:
            chunk_map: dict[str, int] = {lat_dim: -1, lon_dim: -1}
            for dim in ds.dims:
                if dim in chunk_map:
                    continue
                if dim in ("time", "valid_time", "source_time", "time_counter", "t"):
                    chunk_map[dim] = 12
                elif dim in ("realization", "number"):
                    chunk_map[dim] = 1
                else:
                    chunk_map[dim] = 1
            ds = ds.chunk(chunk_map)

            # Rechunking does not always update inherited Zarr chunk metadata on
            # coords such as valid_time/source_time. Clear stale chunk encodings
            # so later to_zarr() calls can validate against the live Dask layout.
            for name in ds.variables:
                ds[name].encoding.pop("chunks", None)
                ds[name].encoding.pop("preferred_chunks", None)

        # Bilinear inpainting on the rectilinear grid using index positions
        return (
            ds
            .interpolate_na(dim=lat_dim, method="linear", use_coordinate=False)
            .interpolate_na(dim=lon_dim, method="linear", use_coordinate=False)
        )
