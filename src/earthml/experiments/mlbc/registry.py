from enum import StrEnum

class MLBCRunMode(StrEnum):
    TRAIN               = "train"
    TEST                = "test"
    TRAIN_TEST          = "train_test"
    TRAIN_TEST_ON_TRAIN = "train_test_on_train"
    PREPARE             = "prepare"


class MLBCExperimentDatasetRole(StrEnum):
    INPUT   = "input"
    TARGET  = "target"


class MLBCExperimentType(StrEnum):
    WEATHER     = "weather"
    SEASONAL    = "seasonal"

# Experiment preset names, naming convention is <input-source_target-source>
class MLBCExperimentName(StrEnum):
    # ocean seasonal
    JUNO_CMCC__JUNO_CMCC    = "juno-cmcc_juno-cmcc"
    CDS_CMCC__ORAS5         = "cds-cmcc_oras5"
    JUNO_CMCC__ORAS5        = "juno-cmcc_oras5"
    # atmo weather
    JUNO_ECMWF__JUNO_ECMWF  = "juno-ecmwf_juno-ecmwf"
    JUNO_ECMWF__ERA5        = "juno-ecmwf_era5"
    # TODO missing ocean weatter, atmo seasonal and atmo weather with ERA5


def available_exps() -> tuple[str, ...]:
    return tuple(sorted(e.value for e in MLBCExperimentName))

def available_runmodes() -> tuple[str, ...]:
    return tuple(sorted(m.value for m in MLBCRunMode))

def available_roles() -> tuple[str, ...]:
    return tuple(sorted(r.value for r in MLBCExperimentDatasetRole))

def available_exp_types() -> tuple[str, ...]:
    return tuple(sorted(t.value for t in MLBCExperimentType))
