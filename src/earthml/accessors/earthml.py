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
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
