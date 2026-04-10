from datetime import datetime

from rich import print
from dateutil.relativedelta import relativedelta

from earthml import *

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------
    # User params
    # ----------------------------------------------------------------------------------
    max_retries                 = 4
    experiment_type             = "weather"                 # seasonal, weather
    experiment_name             = "cmems_cmems"             # cmems_cmems, juno-ecmwf_juno-ecmwf

    run_mode                    = "prepare"     # train_test_on_train, train_test, prepare, train, test
    experiment_mode             = "debug"                   # full, short, debug
    only_longest_train_period   = True                      # only train on the largest train period (if False, train on all periods, which can be much slower but allows to see variability across train periods)
    # add_hyper_exp_name_suffix   = False                     # if True use also batch_size, max_epochs, initial_learning_rate in exp name (and resulting folder) automatically extending exp_suffix
    extra_exp_suffix            = ""                # additional custom suffix to add to exp name (and resulting folder), e.g. "_debug" or "_try1"

    exp_root_folder             = "/Users/jacopodallaglio/ML/experiments_earthML_ocean_weather"
    earthkit_cache_dir          = "/Users/jacopodallaglio/ML/.earthkit-cache"       # if using earthkit datasource

    if experiment_name == "juno-ecmwf_juno-ecmwf":
        variables               = ["msl"]                   # e.g. ["msl", "t2m", "u10", "v10", "d2m", "tcc"]
        regions                 = ["conus"]                 # e.g. ["conus", "westconus", "europe", "westeurope"]
    if experiment_name == "cmems_cmems":
        variables               = ["sst", "ssh", "sss"]     # e.g. ["sst", "ssh", "sss"]
        regions                 = ["pacific"]               # e.g. ["pacific", "natlantic", "satlantic", "indian"]
    leadtimes                   = (.5, 1, 2, 3)

    inpaint_nan                 = True                      # whether to inpaint nan values in input and target datasets (after loading, before torch dataset generation)
    anomaly                     = False                     # if True, predict anomaly (i.e. remove climatology from target variable), otherwise predict absolute values
    per_epoch_resplit           = False                     # if True, split test and validation randomly per epoch with a different seed, if False only one initial split for all epochs (fixed seed)

    if experiment_name == "juno-ecmwf_juno-ecmwf":
        start_train_date            = datetime(2019, 10, 11)
        end_train_date              = datetime(2024, 12, 31)
        start_test_date             = datetime(2025, 1, 1)
        end_test_date               = datetime(2025, 12, 31)
        freq                        = "12h"
    if experiment_name == "cmems_cmems":
        start_train_date            = datetime(2022, 6, 4)
        end_train_date              = datetime(2025, 12, 31)
        start_test_date             = datetime(2026, 1, 1)
        end_test_date               = datetime(2026, 3, 31)
        freq                        = "12h"

    target_realization_avg      = False                     # average over realizations for target variable (if True, add _taravg suffix to exp_suffix)
    realization_as_channel      = False                     # use realization a channel dim (if True, add _rasc suffix to exp_suffix)
    n_input_realizations        = 30                        # used only if realization_as_channel = True

    # Hyperparams
    batch_size                  = 32
    max_epochs                  = 50*n_input_realizations if realization_as_channel else 50
    init_learning_rate          = 1e-3
    accumulate_grad_batches     = 2
    earlystopping_patience      = 100 if realization_as_channel else 30
    train_percent               = 0.9

    losses = [
        {"MSELoss": dict(loss={}, net={})},
        {"GeoMSELoss": dict(loss={"latitudes": True}, net={})},
        {"MaskedMSELoss": dict(loss={}, net={})},
        {"VarNormMaskMSELoss": {"variance_type": "spatial", "latitudes": False}}, # channel, geochannel, spatial, temporal, geotemporal
        {"GeoMaskedMSELoss": dict(loss={"latitudes": True}, net={})},
        {"VarNormMaskMSELoss": dict( # channel, geochannel, spatial, temporal, geotemporal
            loss={"variance_type": "spatial", "latitudes": False},
            net={},
        )},
        {"HeteroBiasCorrectionLoss": dict( # channel, geochannel, spatial, temporal, geotemporal
            loss={"variance_type": "spatial", "latitudes": False, "lambda_identity": 0.1, "bias_scale": 0.5},
            net={"use_first_input": True},
        )},
        # {"GaussianNLLFromLogits": dict(loss={}, net={})},
    ]
    if realization_as_channel:
        losses.append(
            {"EmpiricalCRPSLoss": dict(
                loss={"num_realizations": n_input_realizations, "fair": True, "packed_dim": 1, "variance_type": "spatial", "latitudes": False},
                net={},
            )}
        )


    # ----------------------------------------------------------------------------------
    # Deeper settings
    # ----------------------------------------------------------------------------------
    if experiment_mode in ("short", "debug"):
        # Short exp
        leadtimes       = (3,)
        losses          = [{"MSELoss": dict(loss={}, net={})}]
        if experiment_mode == "debug":
            # Very short periods for debug
            if experiment_name == "juno-ecmwf_juno-ecmwf":
                start_train_date        = datetime(2019, 10, 11)
                # start_train_date        = datetime(2020, 2, 10)
                end_train_date          = datetime(2019, 11, 2)
                start_test_date         = datetime(2025, 10, 1)
                end_test_date           = datetime(2025, 11, 1)
            if experiment_name == "cmems_cmems":
                start_train_date        = datetime(2022, 6, 4)
                end_train_date          = datetime(2022, 6, 9)
                start_test_date         = datetime(2025, 1, 1)
                end_test_date           = datetime(2025, 1, 2)

    # Set test periods and data providers
    juno_location_cutoff = datetime(2025, 10, 14)

    # Local Juno data paths
    root_path_forecast          = "/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/1.0forecast/1h/grib/"
    root_path_forecast_new      = "/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010_new/1.0forecast/1h/grib/"
    root_path_analysis          = "/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010/analysis/6h/grib/"
    root_path_analysis_new      = "/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010_new/analysis/6h/grib/"

    cmems_credential = dict(
        username="jacopo.dallaglio2@unibo.it",
        password="ovUqR&5Q$q9%",
    )

    # Default test periods and kwargs
    test_periods = TimeRange(start=start_test_date, end=end_test_date, freq=freq)
    input_test_provider_kwargs = None
    target_test_provider_kwargs = None

    if experiment_name == "juno-ecmwf_juno-ecmwf":
        if start_test_date <= juno_location_cutoff and end_test_date > juno_location_cutoff:
            test_periods = [
                TimeRange(start=start_test_date, end=juno_location_cutoff, freq=freq),
                TimeRange(start=juno_location_cutoff + relativedelta(days=1), end=end_test_date, freq=freq), # TODO use freq?
            ]
            input_test_provider_kwargs = dict(
                train=dict(root_path=root_path_forecast),
                # test partly in old folder and partly in new folder
                test=[
                    dict(root_path=root_path_forecast),
                    dict(root_path=root_path_forecast_new, file_header="CMS"),
                ],
            )
            target_test_provider_kwargs = dict(
                train=dict(root_path=root_path_analysis),
                test=[
                    dict(root_path=root_path_analysis),
                    dict(root_path=root_path_analysis_new, file_header="CMD"),
                ],
            )
        else:
            if start_test_date >= juno_location_cutoff and end_test_date > juno_location_cutoff:
                input_test_provider_kwargs = dict(
                    train=dict(root_path=root_path_forecast),
                    # test fully in new folder
                    test=dict(root_path=root_path_forecast_new, file_header="CMS"),
                )
                target_test_provider_kwargs = dict(
                    train=dict(root_path=root_path_analysis),
                    # test fully in new folder
                    test=dict(root_path=root_path_analysis_new, file_header="CMS"),
            )

    if experiment_name == "cmems_cmems":
        input_test_provider_kwargs = dict(
            train=cmems_credential,
            test=cmems_credential,
        )
        target_test_provider_kwargs = input_test_provider_kwargs

    # Train period
    train_periods = TimeRange(start=start_train_date, end=end_train_date, freq=freq)


    # ----------------------------------------------------------------------------------
    # Mainloop
    # ----------------------------------------------------------------------------------
    for loss_d in losses:
        loss_name, loss_params = next(iter(loss_d.keys())), next(iter(loss_d.values()))

        launcher_cfg = MLBCExperimentLauncherConfig(
            type=experiment_type,
            name=experiment_name,
            root_path=exp_root_folder,
            suffix=f"_{loss_name.lower()}{extra_exp_suffix}",
        )

        net = MLBCNeuralNet(
            # Reproducibility
            seed=42,
            # Net name
            name="SmaAt_UNet",
            # Net config
            supervised=True,
            n_channels=n_input_realizations if realization_as_channel else 1, # NN input C
            n_classes=n_input_realizations if realization_as_channel else 1, # NN output C
            extra_net_args={},
            # Net hyperparams
            learning_rate=init_learning_rate,
            batch_size=batch_size,
            epochs=max_epochs,
            # Loss
            loss=loss_name,
            loss_params=loss_params,
            # Other
            norm_strategy="BatchNorm2d",
            train_percent=train_percent,
            earlystopping_patience=earlystopping_patience,
            accumulate_grad_batches=accumulate_grad_batches,
        )

        leadtimes_lt = [
            Leadtime(name="leadtime", unit="days", value=lt)
            for lt in leadtimes
        ]
        launcher = MLBCExperimentLauncher(
            experiment=launcher_cfg,
            run_mode=run_mode,
            max_retries=max_retries,
            variables_input=variables,
            variables_target=variables,
            leadtimes=leadtimes_lt,
            regions=regions,
            train_periods=train_periods,
            test_periods=test_periods,
            net=net,
            # Optional
            dask_workers=None,
            only_longest_train_period=only_longest_train_period,
            # Providers args
            earthkit_cache_dir=earthkit_cache_dir,
            regrid_resolution=0.1,
            providers_kwargs={
                "input": input_test_provider_kwargs,
                "target": target_test_provider_kwargs,
            },
            # Experiment
            inpaint_nan=inpaint_nan,
            anomaly=anomaly,
            per_epoch_resplit=per_epoch_resplit,
            torch_preprocess_fn=None,
            target_realization_avg=target_realization_avg,  # whether to average over target realizations when loading target data to torch
            realization_as_channel=realization_as_channel,  # whether to use realization a channel dimension
        )

        launcher.run()
