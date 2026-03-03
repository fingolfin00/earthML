from copy import deepcopy
from datetime import datetime
from dateutil.relativedelta import relativedelta
from earthml.dataclasses import TimeRange
from earthml.launchers.mlfc import MLFCScenario, MLFCRunner
from earthml.utils import halved_windows_split_by_cutoff, half_train_periods_days
from earthml.conversion import celsius_to_kelvin

if __name__ == "__main__":

    # -------------------------
    # User params
    # -------------------------
    max_retries = 10
    var_exp = "sst"                  # sst (atmo), sss, t14d, t17d, ssh
    target_realization_avg = False   # average over realizations for target variable
    realization_as_channel = False   # use realization a channel dim
    n_input_realizations = 30        # used only if realization_as_channel = True
    only_longest_train_period = True # only train on the largest train period (if False, train on all periods, which can be much slower but allows to see variability across train periods)
    loss_sel = "MaskedMSELoss"       # MSELoss, MaskedMSELoss, GeoWeightedMSELoss, VarianceNormalizedMSELoss, HeteroBiasCorrectionLoss, GaussianNLLFromLogits
    # For HeteroBiasCorrectionLoss only
    use_first_input = True
    # For GeoWeightedMSELoss and VarianceNormalizedMSELoss (only for variance_type: geochannel, geotemporal)
    latitudes =  False               # if latitudes=True latitudes are extracted from input test dataset in experiment (mlfc.py) initialization and passed as loss_params
    # For VarianceNormalizedMSELoss only
    variance_type = "spatial"        # channel, geochannel, spatial, temporal, geotemporal
    # -------------------------

    vars_cloud_oras5 = ("sst", "ssh")
    vars_cloud_cds_atmo = ("sst",)
    vars_cloud_cds_ocean = ("ssh",)

    full_leadtimes_days = (15, 45, 75, 105, 135, 165)
    full_leadtimes_atmo_days = (30, 60, 90, 120, 150, 180) # atmo seasonal forecast has end of month leadtimes
    full_leadtimes_months = (1, 2, 3, 4, 5, 6)
    full_leadtime_hours_sst = (12, 24, 48, 72, 96, 120, 144, 168)

    start_train_date = datetime(1993, 7, 1)
    end_train_date = datetime(2020, 12, 31)
    # Short exp for debug
    # full_leadtimes_days = (165,)
    # full_leadtimes_atmo_days = (180,)
    # full_leadtimes_months = (6,)
    # start_train_date = datetime(1993, 7, 1)
    # end_train_date = datetime(1993, 12, 31)

    # experiment_type = "juno-cmcc_juno-cmcc" # analysis in forecast dataset (lt=15d) WRONG!
    experiment_type = "cds-cmcc_oras5" if var_exp in vars_cloud_oras5 else "juno-cmcc_oras5"
    # experiment_type = "cds-cmcc_oras5"

    train_period = TimeRange(start=start_train_date, end=end_train_date, freq='MS')
    test_period = TimeRange(start=datetime(2021, 1, 1), end=datetime(2022, 12, 31), freq='MS')

    month_start_flag = False if var_exp=="sst" and experiment_type=="juno-cmcc_oras5" else True # SST is 6-hourly in Juno

    # input has no cutoff date (only ORAS5 for consolidate vs operational datasets)
    train_periods_input = [train_period] if only_longest_train_period else half_train_periods_days(train_period, min_months=12, anchor="end", month_start=month_start_flag)

    delta_train = relativedelta(end_train_date, start_train_date)
    months_train = delta_train.years * 12 + delta_train.months
    cutoff_oras5_consolidated = datetime(2014, 12, 31) # cutoff date between consolidated and operational ORAS5 datasets
    if experiment_type == "juno-cmcc_oras5":
        var_keys = [(f"{var_exp}_juno_fc", f"{var_exp}_oras5_an")]

        input_provider = "ocean.juno.cmcc.hindcast"
        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

        if only_longest_train_period:
            train_periods_target = halved_windows_split_by_cutoff(train_period, cutoff_oras5_consolidated, min_months=months_train, anchor="end", month_start=month_start_flag)
        else:
            train_periods_target = halved_windows_split_by_cutoff(train_period, cutoff_oras5_consolidated, min_months=12, anchor="end", month_start=month_start_flag)

        regrid_resolution = 1 # WRONG but tolerable (analysis res should be used)

    elif experiment_type == "juno-cmcc_juno-cmcc":
        var_keys = [(f"{var_exp}_juno_fc", f"{var_exp}_juno_an")] # _an has fixed leadtime selection, see in Scenario options

        input_provider = "ocean.juno.cmcc.hindcast"
        target_provider = "ocean.juno.cmcc.hindcast"

        train_periods_target = train_periods_input
        regrid_resolution = 1 # forecast model resolution

    elif experiment_type == "cds-cmcc_oras5":
        var_keys = [(f"{var_exp}_cds_fc", f"{var_exp}_oras5_an")]

        input_provider = "atmo.earthkit.cmcc.hindcast.monthly" if var_exp in vars_cloud_cds_atmo else "ocean.earthkit.cmcc.hindcast.monthly"
        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

        if only_longest_train_period:
            train_periods_target = halved_windows_split_by_cutoff(train_period, cutoff_oras5_consolidated, min_months=months_train, anchor="end", month_start=month_start_flag)
        else:
            train_periods_target = halved_windows_split_by_cutoff(train_period, cutoff_oras5_consolidated, min_months=12, anchor="end", month_start=month_start_flag)

        regrid_resolution = 1 # WRONG but tolerable (analysis res should be used)

    else:
        raise ValueError(f"Experiment type {experiment_type} not supported.")

    provider_kwargs_common = dict(regrid_resolution=regrid_resolution)
    # Print train periods
    for i, p in enumerate(train_periods_target, 1):
        if isinstance(p, list):
            p_start = p[0].start
            p_end = p[-1].end
        else:
            p_start = p.start
            p_end = p.end
        rd = relativedelta(p_end, p_start)
        months = rd.years * 12 + rd.months
        print(i, p_start.date(), "->", p_end.date(), "months:", months)

    if experiment_type == "juno-cmcc_juno-cmcc":
        target_provider_kwargs_common =  provider_kwargs_common | dict(
            file_path_var_prefix="00_ocean_6hr_surface_", # WRONG
        )

        full_leadtimes, full_leadtime_multiple = full_leadtime_hours_sst, full_leadtime_hours_sst
        leadtime_var_an_value = 0 # 0 hour forecast is analysis
        leadtime_var_unit = "hours"
        leadtime_unit = "hours"
    else:
        target_provider_kwargs_common = provider_kwargs_common

        full_leadtimes = full_leadtimes_atmo_days if var_exp in vars_cloud_cds_atmo else full_leadtimes_days
        full_leadtime_multiple = full_leadtimes_months
        leadtime_var_an_value = 15 # 15 days forecast is analysis
        leadtime_var_unit = "days"
        leadtime_unit = "months"

    target_provider_kwargs_earthkit_oras5_consolidated = target_provider_kwargs_common | dict(
        earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
        request_extra_args=dict(
            product_type="consolidated",
            vertical_resolution="single_level"
        ),
        convert_unit={"sst": (celsius_to_kelvin, "K")} if var_exp=="sst" else None, # ORAS5 has SST in °C, while seasonal models has SST in K (only earthkit source supports on the fly conersion)
    )
    target_provider_kwargs_earthkit_oras5_operational = target_provider_kwargs_common | dict(
        earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
        request_extra_args=dict(
            product_type="operational",
            vertical_resolution="single_level"
        ),
        convert_unit={"sst": (celsius_to_kelvin, "K")} if var_exp=="sst" else None,
    )

    input_provider_kwargs = provider_kwargs_common | dict(earthkit_cache_dir="/Users/jacopodallaglio/ML/.earthkit-cache", split_month=12) if experiment_type == "cds-cmcc_oras5" else provider_kwargs_common

    # Loss selection
    if loss_sel == "MSELoss":
        loss_name, loss_params, loss_suf = "MSELoss" , {}, loss_sel.lower()
    else:
        loss_name, loss_params, loss_suf = f"earthml.losses.{loss_sel}", {}, loss_sel.lower()
        if loss_sel == "VarianceNormalizedMSELoss":
            loss_params = dict(loss=dict(variance_type=variance_type, latitudes=latitudes))
            loss_suf += f"_{variance_type}"
        elif loss_sel == "HeteroBiasCorrectionLoss":
            loss_params = dict(net=dict(use_first_input=use_first_input), loss=dict(reduction="mean", lambda_identity=0.1, bias_scale=0.5, eps=1e-12))

    for var_fc_key, var_an_key in var_keys:
        for leadtime_var_fc_value, leadtime_mult in zip(full_leadtimes, full_leadtime_multiple):
            for train_p_in, train_p_tar in zip(train_periods_input, train_periods_target):
                # print(train_p_in, train_p_tar)
                # train_p_in_copy = deepcopy(train_p_in)
                # train_p_tar_copy = deepcopy(train_p_tar)

                if experiment_type in ("juno-cmcc_oras5", "cds-cmcc_oras5"):
                    if len(train_p_tar) != 2:
                        if train_p_tar[0].end < cutoff_oras5_consolidated:
                            oras5_train_target_provider_kwargs = target_provider_kwargs_earthkit_oras5_consolidated
                        else:
                            oras5_train_target_provider_kwargs = target_provider_kwargs_earthkit_oras5_operational
                    target_provider_kwargs = dict(
                        train=[target_provider_kwargs_earthkit_oras5_consolidated, target_provider_kwargs_earthkit_oras5_operational] if len(train_p_tar) == 2 else oras5_train_target_provider_kwargs,
                        test=target_provider_kwargs_earthkit_oras5_operational, # test always operational (if test period after 2014)
                    )
                else:
                    target_provider_kwargs = dict(
                        train=target_provider_kwargs_common,
                        test=target_provider_kwargs_common,
                    )

                ocean_scenario = MLFCScenario(
                    name="ocean",
                    leadtime_var_fc_name="step" if (experiment_type=="cds-cmcc_oras5" and var_exp in vars_cloud_cds_atmo) else "leadtime", # CDS monthly season forecast leadtime variable is "step" instead of "leadtime"
                    leadtime_var_an_name="leadtime",
                    leadtime_var_fc_value=leadtime_var_fc_value,
                    leadtime_var_an_value=leadtime_var_an_value, # fixed leadtime for analysis, ignored if no leadtime in analysis dataset
                    leadtime_var_unit=leadtime_var_unit, # SST -> hours, other vars -> days
                    leadtime_value=leadtime_mult,
                    leadtime_unit=leadtime_unit, # SST -> hours, other vars -> months
                    var_fc_key=var_fc_key,
                    var_an_key=var_an_key,
                    region_key="pacific",
                    train_period=dict(
                        input=train_p_in,
                        target=train_p_tar,
                    ),
                    test_period=dict(
                        input=test_period,
                        target=test_period,
                    ),
                    input_provider=input_provider,
                    target_provider=target_provider,
                    input_provider_kwargs=input_provider_kwargs,
                    target_provider_kwargs=target_provider_kwargs,
                    save_train=True,
                    save_test=True,
                    torch_preprocess_fn=None,
                    target_realization_avg=target_realization_avg,
                    realization_as_channel=realization_as_channel,
                )

                batch_size = int(32*n_input_realizations) if realization_as_channel else 32 # an attempt to adapt batch size to sample size
                runner = MLFCRunner(
                    scenario=ocean_scenario,
                    exp_root_folder="/Users/jacopodallaglio/ML/experiments_earthML_ocean/",
                    exp_suffix=f"_{str(batch_size)}bs_50epoch_{loss_suf}"+("_taravg" if target_realization_avg else "")+("_rasc" if realization_as_channel else ""),
                    # ML options
                    n_channels=n_input_realizations, # ignored if realization_as_channel is false
                    n_classes=1,                     # ignored if realization_as_channel is false
                    learning_rate=1e-3,
                    batch_size=batch_size,
                    epochs=50,
                    loss=loss_name,
                    loss_params=loss_params,
                    accumulate_grad_batches=2,
                    earlystopping_patience=30,
                )

                success = False

                for _ in range(max_retries):
                    try:
                        runner.run(mode="dryrun") # train_test
                        success = True
                        break
                    except (RuntimeError, OSError) as e:
                        print(e)
                        pass

                if not success:
                    continue
