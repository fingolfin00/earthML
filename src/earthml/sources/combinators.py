import xarray as xr
from rich import print

from ..utils import _guess_dim_name
from .base import BaseSource

class SumSource (BaseSource):
    def __init__(self, left: BaseSource, right: BaseSource):
        # Compatibility checks that don't require loading data
        # TODO add support for concat different regions
        if left.data_selection.region != right.data_selection.region:
            raise ValueError("Cannot add sources with different region")

        # Build a synthetic DataSource for the combined source
        combined_datasource = left.datasource + right.datasource
        super().__init__(combined_datasource)

        self._left = left
        self._right = right

    def _get_data(self) -> xr.Dataset:
        # This is where we finally touch the underlying data
        ds_left = self._left.load()
        ds_right = self._right.load()

        # Decide concat dimension
        time_dim = _guess_dim_name(
            ds_left, "time", ["valid_time", "time_counter"]
        )
        if time_dim is None:
            raise ValueError("Could not infer time dimension for concatenation")

        # Intersect data variables
        left_vars = set(ds_left.data_vars)
        right_vars = set(ds_right.data_vars)

        common_vars = sorted(left_vars & right_vars)
        if not common_vars:
            raise ValueError(
                "No common variables to concatenate between sources. "
                f"left={sorted(left_vars)}, right={sorted(right_vars)}"
            )

        extra_left = sorted(left_vars - right_vars)
        extra_right = sorted(right_vars - left_vars)
        if extra_left or extra_right:
            print(
                "[yellow]Warning:[/yellow] dropping non-common variables when adding sources:\n"
                f"  only in left:  {extra_left}\n"
                f"  only in right: {extra_right}"
            )

        ds_left_sel = ds_left[common_vars]
        ds_right_sel = ds_right[common_vars]

        # Intersect coords (by name)
        left_coords = set(ds_left.coords)
        right_coords = set(ds_right.coords)

        common_coords = left_coords & right_coords
        only_left_coords = left_coords - right_coords
        only_right_coords = right_coords - left_coords

        if only_left_coords or only_right_coords:
            print(
                "[yellow]Warning:[/yellow] dropping non-common coordinates when adding sources:\n"
                f"  only in left:  {sorted(only_left_coords)}\n"
                f"  only in right: {sorted(only_right_coords)}"
            )

        # Drop coords that are not shared
        ds_left_sel = ds_left_sel.drop_vars(list(only_left_coords), errors="ignore")
        ds_right_sel = ds_right_sel.drop_vars(list(only_right_coords), errors="ignore")

        # Concatenate lazily (xarray + dask)
        ds_combined = xr.concat([ds_left_sel, ds_right_sel], dim=time_dim)

        return ds_combined
