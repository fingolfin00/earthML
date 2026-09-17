from typing import Literal


PlotMode = Literal[
    "maps",
    "profiles",
    "timeseries",
    "histograms"
    "scalar_diff_scatter",
    "all",
]

FieldModel = Literal[
    "an",
    "fc",
    "clim-fc",
    "mlfc",
]
