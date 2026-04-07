import xarray as xr


class EarthMLMask:
    def mask(self) -> xr.DataArray:
        """
        Returns a boolean mask that is True where all variables in the dataset
        are non-null, after broadcasting across shared dims.
        """
        ds = self._obj

        if not ds.data_vars:
            raise ValueError("Cannot build mask from an empty dataset.")

        masks = []
        for var_name, da in ds.data_vars.items():
            # True where this variable is valid
            masks.append(da.notnull())

        valid_mask = masks[0]
        for m in masks[1:]:
            valid_mask = valid_mask & m

        valid_mask.name = "valid_mask"
        return valid_mask