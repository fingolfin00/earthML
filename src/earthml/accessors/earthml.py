from typing import cast

import xarray as xr

from .geo import EarthMLGeoWeight
from .guess import EarthMLGuess
from .subset import EarthMLSubset
from .regrid import EarthMLRegrid
from .resolution import EarthMLResolution
from .convert import EarthMLConvert
from .inpaint import EarthMLInpaintLonLat
from .climatology import EarthMLClimatology
from .mask import EarthMLMask
from .dims import EarthMLDims

from ..base import Dims


@xr.register_dataset_accessor("earthml")
@xr.register_dataarray_accessor("earthml")
class EarthMLAccessor(
    EarthMLGeoWeight,
    EarthMLGuess,
    EarthMLSubset,
    EarthMLRegrid,
    EarthMLResolution,
    EarthMLConvert,
    EarthMLInpaintLonLat,
    EarthMLClimatology,
    EarthMLMask,
    EarthMLDims,
):
    _obj: xr.Dataset | xr.DataArray

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # For annotations only
    @property
    def guessed_dims(self) -> Dims:
        return cast(Dims, EarthMLGuess.guessed_dims.fget(self))

    @property
    def guessed_coords(self) -> Dims:
        return cast(Dims, EarthMLGuess.guessed_coords.fget(self))
