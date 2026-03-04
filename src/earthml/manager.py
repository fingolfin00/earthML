from dataclasses import dataclass
from typing import Any
import datetime

# Local imports
from .dataclasses import TimeRange
from .utils import halved_windows_split_by_cutoff, half_train_periods_days
from earthml.conversion import celsius_to_kelvin

@dataclass(frozen=True)
class ExperimentMonthlySpec:
    experiment_type: str
    month_start_flag: bool

    var_keys: list[tuple[str, str]]
    input_provider: str
    target_provider: str

    train_periods_input: list[Any]   # TimeRange or list[TimeRange]
    train_periods_target: list[Any]

    provider_kwargs_common: dict[str, Any]
    input_provider_kwargs: dict[str, Any]

    target_provider_kwargs_common: dict[str, Any]
    target_provider_kwargs_oras5_consolidated: dict[str, Any]
    target_provider_kwargs_oras5_operational: dict[str, Any]

    full_leadtimes: tuple
    full_leadtime_multiple: tuple
    leadtime_var_an_value: int
    leadtime_var_unit: str
    leadtime_unit: str

# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------
def build_loss (loss_cfg: dict):
    """Return (loss_name, loss_params, loss_suffix) given a loss config."""
    loss_sel = loss_cfg["loss_sel"]

    if loss_sel == "MSELoss":
        return "MSELoss", {}, "mseloss"

    loss_name = f"earthml.losses.{loss_sel}"
    loss_params = {}
    loss_suf = loss_sel.lower()

    if loss_sel == "VarianceNormalizedMSELoss":
        variance_type = loss_cfg.get("variance_type", "spatial")
        latitudes = loss_cfg.get("latitudes", True)
        loss_params = dict(loss=dict(variance_type=variance_type, latitudes=latitudes))
        loss_suf += f"_{variance_type}"

    elif loss_sel == "HeteroBiasCorrectionLoss":
        use_first_input = loss_cfg.get("use_first_input", True)
        lambda_identity = loss_cfg.get("lambda_identity", True)
        bias_scale      = loss_cfg.get("bias_scale", True)
        use_first_input = loss_cfg.get("use_first_input", True)
        loss_params = dict(
            net=dict(use_first_input=use_first_input),
            loss=dict(lambda_identity=lambda_identity, bias_scale=bias_scale, eps=1e-12),
        )
        # Encode variant choice in name
        if use_first_input:
            loss_suf += "_usefirst"

    # GaussianNLLFromLogits or others: keep defaults unless you add params
    return loss_name, loss_params, loss_suf

def period_bounds (p):
    # p can be TimeRange or list[TimeRange]
    if isinstance(p, list):
        return p[0].start, p[-1].end
    return p.start, p.end

def oras5_train_kwargs (train_p_tar, cutoff, kw_consol, kw_oper):
    start, end = period_bounds(train_p_tar)
    # spans cutoff => use both (consolidated then operational)
    if start <= cutoff < end:
        return [kw_consol, kw_oper]
    # fully before cutoff => consolidated
    if end <= cutoff:
        return kw_consol
    # fully after cutoff => operational
    return kw_oper

def build_monthly_experiment_spec (
    *,
    var_exp: str,
    host_machine: str,
    only_longest_train_period: bool,
    train_period: TimeRange,
    months_train: int,
    cutoff_oras5_consolidated: datetime,
    earthkit_cache_dir: str,
    vars_cloud_oras5: tuple[str, ...],
    vars_cloud_cds_atmo: tuple[str, ...],
    full_leadtimes_days: tuple,
    full_leadtimes_atmo_days: tuple,
    full_leadtimes_months: tuple,
):
    # 1) experiment type
    if host_machine == "juno":
        # experiment_type = "juno-cmcc_juno-cmcc"  # kept disabled as in original
        experiment_type = "cds-cmcc_oras5" if var_exp in vars_cloud_oras5 else "juno-cmcc_oras5"
    else:
        experiment_type = "cds-cmcc_oras5"

    # 2) month start
    month_start_flag = not (var_exp == "sst" and experiment_type == "juno-cmcc_oras5")  # SST is 6-hourly in Juno

    # 3) train periods input
    train_periods_input = (
        [train_period]
        if only_longest_train_period
        else half_train_periods_days(train_period, min_months=12, anchor="end", month_start=month_start_flag)
    )

    # 4) per experiment_type settings
    if experiment_type == "juno-cmcc_oras5":
        var_keys = [(f"{var_exp}_juno_fc", f"{var_exp}_oras5_an")]
        input_provider = "ocean.juno.cmcc.hindcast"
        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

        train_periods_target = halved_windows_split_by_cutoff(
            train_period,
            cutoff_oras5_consolidated,
            min_months=(months_train if only_longest_train_period else 12),
            anchor="end",
            month_start=month_start_flag,
        )

        regrid_resolution = 1  # WRONG but tolerable (analysis res should be used)

    elif experiment_type == "juno-cmcc_juno-cmcc":
        var_keys = [(f"{var_exp}_juno_fc", f"{var_exp}_juno_an")]
        input_provider = "ocean.juno.cmcc.hindcast"
        target_provider = "ocean.juno.cmcc.hindcast"

        train_periods_target = train_periods_input
        regrid_resolution = 1  # forecast model resolution

    elif experiment_type == "cds-cmcc_oras5":
        var_keys = [(f"{var_exp}_cds_fc", f"{var_exp}_oras5_an")]
        input_provider = (
            "atmo.earthkit.cmcc.hindcast.monthly"
            if var_exp in vars_cloud_cds_atmo
            else "ocean.earthkit.cmcc.hindcast.monthly"
        )
        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

        train_periods_target = halved_windows_split_by_cutoff(
            train_period,
            cutoff_oras5_consolidated,
            min_months=(months_train if only_longest_train_period else 12),
            anchor="end",
            month_start=month_start_flag,
        )

        regrid_resolution = 1  # WRONG but tolerable (analysis res should be used)

    else:
        raise ValueError(f"Experiment type {experiment_type} not supported.")

    # 5) common provider kwargs
    provider_kwargs_common = dict(regrid_resolution=regrid_resolution)

    # 6) leadtime conventions + target provider kwargs common
    # if experiment_type == "juno-cmcc_juno-cmcc":
    #     target_provider_kwargs_common = provider_kwargs_common | dict(
    #         file_path_var_prefix="00_ocean_6hr_surface_",  # WRONG
    #     )

    #     full_leadtimes = full_leadtime_hours_sst
    #     full_leadtime_multiple = full_leadtime_hours_sst
    #     leadtime_var_an_value = 0
    #     leadtime_var_unit = "hours"
    #     leadtime_unit = "hours"
    # else:
    target_provider_kwargs_common = provider_kwargs_common

    full_leadtimes = full_leadtimes_atmo_days if var_exp in vars_cloud_cds_atmo else full_leadtimes_days
    full_leadtime_multiple = full_leadtimes_months
    leadtime_var_an_value = 15
    leadtime_var_unit = "days"
    leadtime_unit = "months"

    if len(full_leadtimes) != len(full_leadtime_multiple):
        raise ValueError(f"Leadtime arrays mismatch: {len(full_leadtimes)} vs {len(full_leadtime_multiple)}")

    # 7) ORAS5 kwargs
    target_provider_kwargs_oras5_consolidated = target_provider_kwargs_common | dict(
        earthkit_cache_dir=earthkit_cache_dir,
        request_extra_args=dict(product_type="consolidated", vertical_resolution="single_level"),
        convert_unit={"sst": (celsius_to_kelvin, "K")} if var_exp == "sst" else None,
    )
    target_provider_kwargs_oras5_operational = target_provider_kwargs_common | dict(
        earthkit_cache_dir=earthkit_cache_dir,
        request_extra_args=dict(product_type="operational", vertical_resolution="single_level"),
        convert_unit={"sst": (celsius_to_kelvin, "K")} if var_exp == "sst" else None,
    )

    # 8) input provider kwargs
    input_provider_kwargs = (
        provider_kwargs_common | dict(earthkit_cache_dir=earthkit_cache_dir, split_month=12)
        if experiment_type == "cds-cmcc_oras5"
        else provider_kwargs_common
    )

    # 9) sanity
    if len(train_periods_input) != len(train_periods_target):
        raise ValueError(f"train_periods_input ({len(train_periods_input)}) != train_periods_target ({len(train_periods_target)})")

    return ExperimentMonthlySpec(
        experiment_type=experiment_type,
        month_start_flag=month_start_flag,
        var_keys=var_keys,
        input_provider=input_provider,
        target_provider=target_provider,
        train_periods_input=train_periods_input,
        train_periods_target=train_periods_target,
        provider_kwargs_common=provider_kwargs_common,
        input_provider_kwargs=input_provider_kwargs,
        target_provider_kwargs_common=target_provider_kwargs_common,
        target_provider_kwargs_oras5_consolidated=target_provider_kwargs_oras5_consolidated,
        target_provider_kwargs_oras5_operational=target_provider_kwargs_oras5_operational,
        full_leadtimes=full_leadtimes,
        full_leadtime_multiple=full_leadtime_multiple,
        leadtime_var_an_value=leadtime_var_an_value,
        leadtime_var_unit=leadtime_var_unit,
        leadtime_unit=leadtime_unit,
    )
