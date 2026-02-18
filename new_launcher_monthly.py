from copy import deepcopy
from datetime import datetime
from dateutil.relativedelta import relativedelta
from earthml.dataclasses import TimeRange
from earthml.launchers.mlfc import MLFCScenario, MLFCRunner
from earthml.utils import halved_windows_split_by_cutoff, half_train_periods_days

if __name__ == "__main__":
    max_retries = 4

    var_exp = "sst" # sst (atmo), sss, t14d, t17d
    target_realization_avg = False # average over realizations for target variable
    only_longest_train_period = True # only train on the largest train period (if False, train on all periods, which can be much slower but allows to see variability across train periods)

    vars_atmo = ("sst",)

    full_leadtimes_days = (15, 45, 75, 105, 135, 165)
    full_leadtimes_atmo_days = (30, 60, 90, 120, 150, 180) # atmo seasonal forecast has end of month leadtimes
    full_leadtimes_months = (1, 2, 3, 4, 5, 6)
    full_leadtime_hours_sst = (12, 24, 48, 72, 96, 120, 144, 168)

    start_train_date = datetime(1993, 6, 1)
    end_train_date = datetime(2020, 12, 31)
    # Short exp for debug
    # full_leadtimes_days = (45,)
    # full_leadtimes_months = (1,)
    # start_train_date = datetime(1993, 6, 1)
    # end_train_date = datetime(1994, 12, 31)

    # experiment_type = "juno-cmcc_juno-cmcc" # analysis in forecast dataset (lt=15d) WRONG!
    experiment_type = "cds-cmcc_oras5" if var_exp in vars_atmo else "juno-cmcc_oras5"
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

        input_provider = "atmo.earthkit.cmcc.hindcast.monthly" if var_exp in vars_atmo else "ocean.earthkit.cmcc.hindcast.monthly"
        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

        if only_longest_train_period:
            train_periods_target = halved_windows_split_by_cutoff(train_period, cutoff_oras5_consolidated, min_months=months_train, anchor="end", month_start=month_start_flag)
        else:
            train_periods_target = halved_windows_split_by_cutoff(train_period, cutoff_oras5_consolidated, min_months=12, anchor="end", month_start=month_start_flag)

        regrid_resolution = 1 # WRONG but tolerable (analysis res should be used)

    else:
        raise ValueError(f"Experiment type {experiment_type} not supported.")

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

    if experiment_type == "juno-cmcc_oras5" or experiment_type == "juno-cmcc_juno-cmcc":
        # SST is not monthly in Juno
        if var_exp == "sst":
            target_provider_kwargs_common = dict(
                file_path_var_prefix="00_ocean_6hr_surface_",
                regrid_resolution=regrid_resolution,
            )

            full_leadtimes, full_leadtime_multiple = full_leadtime_hours_sst, full_leadtime_hours_sst
            leadtime_var_an_value = 0 # 0 hour forecast is analysis
            leadtime_var_unit = "hours"
            leadtime_unit = "hours"
        else:
            target_provider_kwargs_common = dict(
                # file_path_var_prefix="00_ocean_mon_ocean2d_", # default
                regrid_resolution=regrid_resolution,
            )

            full_leadtimes, full_leadtime_multiple = full_leadtimes_atmo_days, full_leadtimes_months if var_exp in vars_atmo else full_leadtimes_days, full_leadtimes_months
            leadtime_var_an_value = 15 # 15 days forecast is analysis
            leadtime_var_unit = "days"
            leadtime_unit = "months"
    else:
        target_provider_kwargs_common = dict(
                regrid_resolution=regrid_resolution,
            )

        full_leadtimes, full_leadtime_multiple = full_leadtimes_atmo_days if var_exp in vars_atmo else full_leadtimes_days, full_leadtimes_months
        leadtime_var_an_value = 15 # 15 days forecast is analysis
        leadtime_var_unit = "days"
        leadtime_unit = "months"

    target_provider_kwargs_earthkit_oras5_consolidated = target_provider_kwargs_common | dict(
        earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
        request_extra_args=dict(
            product_type="consolidated",
            vertical_resolution="single_level"
        )
    )
    target_provider_kwargs_earthkit_oras5_operational = target_provider_kwargs_common | dict(
        earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
        request_extra_args=dict(
            product_type="operational",
            vertical_resolution="single_level"
        )
    )

    input_provider_kwargs=target_provider_kwargs_common

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
                    leadtime_var_fc_name="leadtime" if experiment_type!="cds-cmcc_oras5" else "step", # ORAS5 leadtime variable is "step" instead of "leadtime"
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
                )

                runner = MLFCRunner(
                    scenario=ocean_scenario,
                    exp_root_folder="/work/cmcc/jd19424/test-ML/experiments_earthML_ocean/",
                    exp_suffix="_32bs_50epoch_mse"+("_taravg" if target_realization_avg else ""),
                    # ML options
                    learning_rate=1e-3,
                    batch_size=32,
                    epochs=50,
                    loss="MSELoss",
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
