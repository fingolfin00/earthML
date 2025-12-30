from datetime import datetime
from earthml.dataclasses import TimeRange
from earthml.launchers.mlfc import MLFCScenario, MLFCRunner
from earthml.utils import half_train_periods_days

if __name__ == "__main__":

    full_leadtimes = (.5,1,2,3)

    full_train_period = TimeRange(start=datetime(2019, 10, 11), end=datetime(2024, 12, 31), freq="12h")
    train_periods = half_train_periods_days(full_train_period, min_months=3, anchor="end")
    # Short train period for debug
    # full_train_period = TimeRange(start=datetime(2021, 8, 30), end=datetime(2021, 9, 2), freq='12h')
    # train_periods = [full_train_period]
    for i, p in enumerate(train_periods, 1):
        print(i, p.start.date(), "->", p.end.date(), "days:", (p.end - p.start).days)

    for leadtime_days in full_leadtimes:
        leadtime_hours = int(leadtime_days * 24)
        for train_p in train_periods:
            weather_scenario = MLFCScenario(
                name="weather",
                leadtime_value=leadtime_hours,
                leadtime_unit="hours",
                var_fc_key="t2m_juno",
                var_an_key="t2m_juno",
                region_key="conus",
                train_period=train_p,
                # train_period=TimeRange(start=datetime(2021, 8, 30), end=datetime(2021, 9, 2), freq='12h'),
                test_period= TimeRange(start=datetime(2025, 1, 1), end=datetime(2025, 10, 31), freq='12h'),
                input_provider= "atmo.juno.ecmwf.forecast.hourly",
                target_provider="atmo.juno.ecmwf.analysis.6hourly",
                input_provider_kwargs=dict(),
                target_provider_kwargs=dict(),
                save_train=True,
                save_test=True,
                torch_preprocess_fn=None,
            )

            runner = MLFCRunner(
                scenario=weather_scenario,
                exp_root_folder="/work/cmcc/jd19424/test-ML/experiments_earthML_weather/",
                exp_suffix="_32bs_50epoch_mse",
                # ML options
                learning_rate=1e-3,
                batch_size=32,
                epochs=50,
                loss="MSELoss",
                accumulate_grad_batches=2,
                earlystopping_patience=30,
            )

            max_retries = 4
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
