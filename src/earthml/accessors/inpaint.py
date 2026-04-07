import xarray as xr


class EarthMLInpaintLonLat:
    def inpaint_nan_lonlat_bilinear(self) -> xr.Dataset:
        ds = self._obj
        # Bilinear inpainting on the rectilinear grid using index positions
        return (
            ds
            .interpolate_na(dim=ds.earthml.guessed_dims.latitude, method="linear", use_coordinate=False)
            .interpolate_na(dim=ds.earthml.guessed_dims.longitude, method="linear", use_coordinate=False)
        )
