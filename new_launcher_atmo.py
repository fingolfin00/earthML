from datetime import datetime
from earthml.dataclasses import TimeRange
from earthml.launchers.mlfc import MLFCScenario, MLFCRunner
from earthml.utils import half_train_periods_days

if __name__ == "__main__":
    max_retries = 4

    full_leadtimes = (.5,1,2,3)
    var_keys = ["msl_juno"]

    full_train_period = TimeRange(start=datetime(2019, 10, 11), end=datetime(2024, 12, 31), freq="12h")
    test_period       = TimeRange(start=datetime(2025, 1, 1), end=datetime(2025, 10, 14), freq='12h')
    test_period_new   = TimeRange(start=datetime(2025, 10, 15), end=datetime(2025, 12, 31), freq='12h')
    train_periods     = half_train_periods_days(full_train_period, min_months=3, anchor="end")
    # Shorter train/test periods for debug
    # full_train_period = TimeRange(start=datetime(2020, 2, 10), end=datetime(2020, 2, 20), freq='12h')
    # test_period = TimeRange(start=datetime(2025, 1, 1), end=datetime(2025, 1, 2), freq='12h')
    # train_periods = [full_train_period]

    root_path_forecast     = "/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/1.0forecast/1h/grib/"
    root_path_forecast_new = "/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010_new/1.0forecast/1h/grib/"
    root_path_analysis     = "/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010/analysis/6h/grib/"
    root_path_analysis_new = "/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010_new/analysis/6h/grib/"

    for i, p in enumerate(train_periods, 1):
        print(i, p.start.date(), "->", p.end.date(), "days:", (p.end - p.start).days)

    for var_key in var_keys:
        for leadtime_days in full_leadtimes:
            leadtime_hours = int(leadtime_days * 24)
            for train_p in train_periods:
                weather_scenario = MLFCScenario(
                    name="weather",
                    leadtime_var_fc_name="leadtime",
                    leadtime_var_an_name="leadtime",
                    leadtime_var_fc_value=leadtime_hours,
                    leadtime_var_an_value=leadtime_hours,
                    leadtime_var_unit="hours",
                    leadtime_value=leadtime_hours,
                    leadtime_unit="hours",
                    var_fc_key=var_key,
                    var_an_key=var_key,
                    region_key="conus",
                    train_period=train_p,
                    test_period=dict(
                        input= [test_period, test_period_new],
                        target=[test_period, test_period_new],
                    ),
                    input_provider= "atmo.juno.ecmwf.forecast.hourly",
                    target_provider="atmo.juno.ecmwf.analysis.6hourly",
                    input_provider_kwargs=dict(
                        train=dict(root_path=root_path_forecast),
                        test=[dict(root_path=root_path_forecast), dict(root_path=root_path_forecast_new, file_header="CMS")],
                    ),
                    target_provider_kwargs=dict(
                        train=dict(root_path=root_path_analysis),
                        test=[dict(root_path=root_path_analysis), dict(root_path=root_path_analysis_new, file_header="CMD")],
                    ),
                    save_train=True,
                    save_test=True,
                    torch_preprocess_fn=None,
                )

                runner = MLFCRunner(
                    scenario=weather_scenario,
                    exp_root_folder="/work/cmcc/jd19424/test-ML/experiments_earthML_weather/",
                    exp_suffix="_32bs_50epoch_mse",
                    # exp_suffix="_32bs_50epoch_gnll",
                    # exp_suffix="_32bs_50epoch_het",
                    # ML options
                    learning_rate=1e-3,
                    batch_size=32,
                    epochs=50,
                    loss="MSELoss",
                    # loss="earthml.losses.GaussianNLLFromLogits",
                    # loss="earthml.losses.HeteroBiasCorrectionLoss",
                    # loss_params=dict(
                    #     # net=dict(use_first_input=False), # gnll
                    #     net=dict(use_first_input=True),
                    #     loss=dict(
                    #         # reduction="mean", # gnll
                    #         lambda_identity=0.1,
                    #         bias_scale=0.5,
                    #         eps=1e-6)
                    # ),
                    accumulate_grad_batches=2,
                    earlystopping_patience=30,
                )

                success = False

                for _ in range(max_retries):
                    try:
                        runner.run(mode="train_test")
                        success = True
                        break
                    except (RuntimeError, OSError) as e:
                        print(e)
                        pass

                if not success:
                    continue
