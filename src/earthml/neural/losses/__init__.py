from .registry import get_loss_class, build_loss, list_losses
from .crps import EmpiricalCRPSLoss
from .mse import (
    MaskedMSELoss,
    GeoMSELoss,
    GeoMaskedMSELoss,
    GeoMaskedMSELowFreqLoss,
    VarNormMaskMSELoss,
    HeteroBiasCorrectionLoss,
)
from .nll import GaussianNLLFromLogits

__all__ = [
    "get_loss_class",
    "build_loss",
    "list_losses",
    "EmpiricalCRPSLoss",
    "MaskedMSELoss",
    "GeoMSELoss",
    "GeoMaskedMSELoss",
    "GeoMaskedMSELowFreqLoss",
    "VarNormMaskMSELoss",
    "HeteroBiasCorrectionLoss",
    "GaussianNLLFromLogits",
]
