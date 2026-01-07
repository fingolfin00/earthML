from datetime import datetime
from dateutil.relativedelta import relativedelta
from earthml.dataclasses import TimeRange
from earthml.launchers.mlfc import MLFCScenario, MLFCRunner
from earthml.utils import halved_windows_split_by_cutoff, half_train_periods_days

if __name__ == "__main__":

    max_retries = 4

    full_leadtimes_days = (15, 45, 75, 105, 135, 165)
    full_leadtimes_months = (0, 1, 2, 3, 4, 5)
    # start_train_date = datetime(1993, 1, 1)
    # end_train_date = datetime(2020, 12, 31)
    # Short exp for debug
    start_train_date = datetime(1993, 7, 1)
    end_train_date = datetime(1994, 12, 31)

    full_train_period_target = TimeRange(start=start_train_date, end=end_train_date, freq='MS')
    cutoff_consolidated = datetime(2014, 12, 31) # cutoff date between consolidated and operational ORAS5 datasets
    train_periods_target = halved_windows_split_by_cutoff(full_train_period_target, cutoff_consolidated, min_months=12, anchor="end")

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

    earthkit_consolidated = dict(
        earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
        request_extra_args=dict(
            product_type="consolidated",
            vertical_resolution="single_level"
        )
    )
    earthkit_operational = dict(
        earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
        request_extra_args=dict(
            product_type="operational",
            vertical_resolution="single_level"
        )
    )

    for leadtime_days, leadtime_months in zip(full_leadtimes_days, full_leadtimes_months):

        full_train_period_input = TimeRange(start=start_train_date, end=end_train_date, freq='MS')
        train_periods_input = half_train_periods_days(full_train_period_input, min_months=12, anchor="end")

        for train_p_in, train_p_tar in zip(train_periods_input, train_periods_target):
            ocean_scenario = MLFCScenario(
                name="ocean",
                leadtime_var_name="leadtime",
                leadtime_var_value=leadtime_days,
                leadtime_var_unit="days",
                leadtime_value=leadtime_months,
                leadtime_unit="months",
                var_fc_key="sss_juno_fc",
                var_an_key="sss_oras5_an",
                region_key="pacific",
                train_period=dict(
                    input=train_p_in,
                    target=train_p_tar,
                ),
                test_period=dict(
                    input=TimeRange(start=datetime(2021, 1, 1), end=datetime(2022, 12, 31), freq='MS'),
                    target=TimeRange(start=datetime(2021, 1, 1), end=datetime(2022, 12, 31), freq='MS'),
                ),
                input_provider= "ocean.juno.cmcc.hindcast.monthly",
                target_provider="ocean.earthkit.oras5.reanalysis.monthly",
                input_provider_kwargs=dict(),
                target_provider_kwargs=dict(
                    train=[earthkit_consolidated, earthkit_operational] if len(train_p_tar) == 2 else earthkit_consolidated,
                    test=earthkit_operational,
                ),
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
