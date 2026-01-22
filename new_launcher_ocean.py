from datetime import datetime
from dateutil.relativedelta import relativedelta
from earthml.dataclasses import TimeRange
from earthml.launchers.mlfc import MLFCScenario, MLFCRunner
from earthml.utils import halved_windows_split_by_cutoff, half_train_periods_days

if __name__ == "__main__":
    max_retries = 4

    var_exp = "sss"

    full_leadtimes_days = (45, 75, 105, 135, 165)
    full_leadtimes_months = (1, 2, 3, 4, 5)
    start_train_date = datetime(1993, 6, 1)
    end_train_date = datetime(2020, 12, 31)
    # Short exp for debug
    # full_leadtimes_days = (45,)
    # full_leadtimes_months = (1,)
    # start_train_date = datetime(1993, 6, 1)
    # end_train_date = datetime(1994, 12, 31)

    experiment_type = "juno-cmcc_juno-cmcc" # analysis in forecast dataset (lt=15d)
    # experiment_type = "juno-cmcc_oras5"
    # experiment_type = "cds-cmcc_oras5"


    full_train_period_target = TimeRange(start=start_train_date, end=end_train_date, freq='MS')
    full_train_period_input = TimeRange(start=start_train_date, end=end_train_date, freq='MS')
    test_period = TimeRange(start=datetime(2021, 1, 1), end=datetime(2022, 12, 31), freq='MS')

    # input has no cutoff date (only ORAS5 for consolidate vs operational datasets)
    train_periods_input = half_train_periods_days(full_train_period_input, min_months=12, anchor="end")

    cutoff_oras5_consolidated = datetime(2014, 12, 31) # cutoff date between consolidated and operational ORAS5 datasets
    if experiment_type == "juno-cmcc_oras5":
        var_keys = [(f"{var_exp}_juno_fc", f"{var_exp}_oras5_an")]

        input_provider = "ocean.juno.cmcc.hindcast.monthly"
        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

        train_periods_target = halved_windows_split_by_cutoff(full_train_period_target, cutoff_oras5_consolidated, min_months=12, anchor="end")
        regrid_resolution = 0.25

    elif experiment_type == "juno-cmcc_juno-cmcc":
        var_keys = [(f"{var_exp}_juno_fc", f"{var_exp}_juno_an")] # _an has fixed leadtime selection, see in Scenario options

        input_provider = "ocean.juno.cmcc.hindcast.monthly"
        target_provider = "ocean.juno.cmcc.hindcast.monthly"

        train_periods_target = half_train_periods_days(full_train_period_target, min_months=12, anchor="end")
        regrid_resolution = 1 # forecast model resolution

    elif experiment_type == "cds-cmcc_oras5":
        var_keys = [(f"{var_exp}_cds_fc", f"{var_exp}_oras5_an")]

        input_provider = "ocean.earthkit.cmcc.hindcast.monthly"
        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"

        train_periods_target = halved_windows_split_by_cutoff(full_train_period_target, cutoff_oras5_consolidated, min_months=12, anchor="end")
        regrid_resolution = 0.25

    else:
        raise ValueError(f"Experiment type {experiment_type} not supported.")

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

    target_provider_kwargs_common = dict(
        regrid_resolution=regrid_resolution,
    )

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

    for var_fc_key, var_an_key in var_keys:
        for leadtime_days, leadtime_months in zip(full_leadtimes_days, full_leadtimes_months):
            for train_p_in, train_p_tar in zip(train_periods_input, train_periods_target):

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
                    leadtime_var_name="leadtime",
                    leadtime_var_fc_value=leadtime_days,
                    leadtime_var_an_value=15, # fixed leadtime for analysis, ignored if no leadtime in analysis dataset
                    leadtime_var_unit="days",
                    leadtime_value=leadtime_months,
                    leadtime_unit="months",
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
                    input_provider_kwargs=dict(regrid_resolution=regrid_resolution),
                    target_provider_kwargs=target_provider_kwargs,
                    save_train=True,
                    save_test=True,
                    torch_preprocess_fn=None,
                )

                runner = MLFCRunner(
                    scenario=ocean_scenario,
                    exp_root_folder="/work/cmcc/jd19424/test-ML/experiments_earthML_ocean/",
                    exp_suffix="_32bs_50epoch_mse",
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
                        runner.run(mode="dryrun")
                        success = True
                        break
                    except (RuntimeError, OSError) as e:
                        print(e)
                        pass

                if not success:
                    continue
