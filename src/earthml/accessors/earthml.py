import xarray as xr

from .geo import EarthMLGeoWeight
from .guess import EarthMLGuess
from .subset import EarthMLSubset
from .regrid import EarthMLRegrid
from .resolution import EarthMLResolution
from .convert import EarthMLConvert


@xr.register_dataset_accessor("earthml")
@xr.register_dataarray_accessor("earthml")
class EarthMLAccessor(
    EarthMLGeoWeight,
    EarthMLGuess,
    EarthMLSubset,
    EarthMLRegrid,
    EarthMLResolution,
    EarthMLConvert,
):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
