from copy import deepcopy


DEFAULT_METRIC_LIMITS = {
    "nrmse": (0, 1),
    "ncrmse": (0, 1),
    "nmae": (0, 1),
    "nbias": (-1, 1),
    "r2": (-1, 1),
    "corr": (-1, 1),
    "clim_acc": (-1, 1),
    "spatial_acc": (-1, 1),
}

DEFAULT_METRIC_CMAPS = {
    "field": "jet",
    "rmse": "jet",
    "crmse": "Reds",
    "mae": "Reds",
    "bias": "RdBu_r",
    "nrmse": "Reds",
    "ncrmse": "Reds",
    "nmae": "Reds",
    "nbias": "RdBu_r",
    "r2": "RdBu",
    "corr": "RdBu",
    "clim_acc": "RdBu",
    "spatial_acc": "RdBu",
}

VAR_PLOT_CONFIG_BASE = {
    "sst": {
        "fullname": "sea surface temperature",
        "plot": {
            "limits": {
                "field": (290, 305),
                "rmse": (0, 2),
                "diff_rmse": (-1.5, 1.5),
                "crmse": (0, 2.5),
                "mae": (0, 1.5),
                "diff_mae": (-1, 1),
                "bias": (-1.5, 1.5),
                "diff_bias": (-1.5, 1.5),
                "r2": (-20, 1),
            },
            "cmap": {
                "field": "jet",
            },
        },
        "units": "K",
    },
    "ssh": {
        "fullname": "sea surface height",
        "plot": {
            "limits": {
                "field": None,
                "rmse": (0, 0.15),
                "crmse": None,
                "mae": None,
                "bias": None,
                "r2": (-80, 1),
            },
            "cmap": {
                "field": "jet",
            },
        },
        "units": "m",
    },
    "sss": {
        "fullname": "sea surface salinity",
        "plot": {
            "limits": {
                "field": (33, 37),
                "rmse": (0, 2.5),
                "crmse": (0, 2.5),
                "mae": (0, 2.5),
                "bias": (-1.5, 1.5),
                "r2": (-20, 1),
            },
            "cmap": {
                "field": "viridis",
            },
        },
        "units": "psu",
    },
    "t14d": {
        "fullname": "14°C isotherm depth",
        "plot": {
            "limits": {
                "field": (0, 400),
                "rmse": (0, 2.5),
                "crmse": (0, 2.5),
                "mae": (0, 2.5),
                "bias": (-1.5, 1.5),
            },
            "cmap": {
                "field": "Blues",
            },
        },
        "units": "m",
    },
    "mslp": {
        "fullname": "mean sea level pressure",
        "plot": {
            "limits": {
                "field": None,
                "rmse": None,
                "crmse": None,
                "mae": None,
                "bias": None,
            },
            "cmap": {
                "field": "jet",
            },
        },
        "units": "Pa",
    },
    "t2m": {
        "fullname": "2 meter temperature",
        "plot": {
            "limits": {
                "field": None,
                "rmse": None,
                "crmse": None,
                "mae": None,
                "bias": None,
            },
            "cmap": {
                "field": "jet",
            },
        },
        "units": "K",
    },
}


def build_plot_config(
    base_config: dict | None = None,
    default_limits: dict | None = None,
    default_cmaps: dict | None = None,
) -> dict:
    """
    Build the final plotting config by merging defaults with
    variable-specific overrides.
    """
    base_config = VAR_PLOT_CONFIG_BASE if base_config is None else base_config
    default_limits = DEFAULT_METRIC_LIMITS if default_limits is None else default_limits
    default_cmaps = DEFAULT_METRIC_CMAPS if default_cmaps is None else default_cmaps

    out = deepcopy(base_config)

    for var_name, cfg in out.items():
        plot_cfg = cfg.setdefault("plot", {})

        plot_cfg["limits"] = {
            **default_limits,
            **plot_cfg.get("limits", {}),
        }

        plot_cfg["cmap"] = {
            **default_cmaps,
            **plot_cfg.get("cmap", {}),
        }

    return out


# --------------------------
# Access helpers
# --------------------------

def get_var_plot_limits(plot_config, var_name, metric_name):
    if var_name not in plot_config:
        raise KeyError(f"Unknown variable: {var_name}")
    return plot_config[var_name]["plot"]["limits"].get(metric_name)


def get_var_plot_cmap(plot_config: dict, var_name: str, metric_name: str):
    return plot_config[var_name]["plot"]["cmap"].get(metric_name)


def get_var_units(plot_config: dict, var_name: str) -> str | None:
    return plot_config[var_name].get("units")


def get_var_fullname(plot_config: dict, var_name: str) -> str:
    return plot_config[var_name].get("fullname", var_name)
