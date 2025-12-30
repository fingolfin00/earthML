from datetime import datetime
from earthml.dataclasses import TimeRange
from earthml.launchers.mlfc import MLFCScenario, MLFCRunner

if __name__ == "__main__":

    for leadtime_days in (15, 45, 75, 105, 135, 165):
    # leadtime_days = 45
        ocean_scenario = MLFCScenario(
            name="ocean",
            leadtime_value=leadtime_days,
            leadtime_unit="days",
            var_fc_key="sss_juno_fc",
            var_an_key="sss_oras5_an",
            region_key="pacific",
            train_period=dict(
                input=TimeRange(start=datetime(1993, 1, 1), end=datetime(2020, 12, 31), freq='MS', shifted=dict(days=leadtime_days)),
                target=[
                    TimeRange(start=datetime(1993, 1, 1), end=datetime(2014, 12, 31), freq='MS'),
                    TimeRange(start=datetime(2015, 1, 1), end=datetime(2020, 12, 31), freq='MS'),
                ],
            ),
            test_period=dict(
                input=TimeRange(start=datetime(2021, 1, 1), end=datetime(2024, 12, 31), freq='MS', shifted=dict(days=leadtime_days)),
                target=TimeRange(start=datetime(2021, 1, 1), end=datetime(2024, 12, 31), freq='MS'),
            ),
            input_provider= "ocean.juno.cmcc.hindcast.monthly",
            target_provider="ocean.earthkit.oras5.reanalysis.monthly",
            input_provider_kwargs=dict(),
            target_provider_kwargs=dict(
                train=[
                    dict(earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
                        request_extra_args=dict(
                            product_type="consolidated",
                            vertical_resolution="single_level"
                    )),
                    dict(earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
                        request_extra_args=dict(
                            product_type="operational",
                            vertical_resolution="single_level"
                    )),
                ],
                test=dict(earthkit_cache_dir="/work/cmcc/jd19424/.earthkit-cache",
                        request_extra_args=dict(
                            product_type="operational",
                            vertical_resolution="single_level"
                        )),
            ),
            save_train=True,
            save_test=True,
            torch_preprocess_fn=None,
        )

        runner = MLFCRunner(
            scenario=ocean_scenario,
            exp_root_folder="/work/cmcc/jd19424/test-ML/experiments_earthML/",
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
            except (RuntimeError, OSError):
                pass

        if not success:
            continue
