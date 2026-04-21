import os
from datetime import datetime

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")

from rich import print
from dateutil.relativedelta import relativedelta

from earthml import *

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------
    # User params
    # ----------------------------------------------------------------------------------
    max_retries                 = 4
    force_retrain               = False
    force_rebuild_dataset       = False # False, "all", "train", "test", "input", "target", "train_input", "train_target", "test_input", "test_target"
    dataset_cache_enabled       = True
    log_level                   = "info" # debug, info
    weights_filename            = None # e.g. "/abs/path/to/custom_weights.ckpt"; if set, testing uses this file and training is skipped
    juno_file_open_workers      = 1 # 1: good for grib (atmo weather), 8: good for netcdf (ocean weather)

    experiment_type             = "weather" # seasonal, weather
    experiment_name             = "juno-ecmwf_juno-ecmwf" # cmems_cmems, juno-ecmwf_juno-ecmwf, juno-cmcc_cmems, juno-medfs_cmems

    run_mode                    = "train_test_on_train" # train_test_on_train, train_test, prepare, train, test
    experiment_mode             = "full" # full, short, debug
    only_longest_train_period   = True # only train on the largest train period (if False, train on all periods, which can be much slower but allows to see variability across train periods)
    extra_exp_suffix            = "" # additional custom suffix to add to exp name (and resulting folder), e.g. "_debug" or "_try1"

    exp_root_folder             = "/Users/jacopodallaglio/ML/experiments_earthML_ocean_weather"
    dataset_cache_root          = "/Users/jacopodallaglio/ML/.earthml-dataset-cache"
    earthkit_cache_dir          = "/Users/jacopodallaglio/ML/.earthkit-cache"       # if using earthkit datasource
    juno_per_sample_regrid      = False
    external_mask_path          = None # e.g. "/abs/path/to/med_ocean_mask_0p04.nc"
    external_mask_variable      = None # e.g. "mask"

    if experiment_name == "juno-ecmwf_juno-ecmwf":
        variables               = ["msl", "t2m", "d2m", "u10", "v10", "tcc"] # e.g. ["msl", "t2m", "u10", "v10", "d2m", "tcc"]
        regions                 = ["westconus", "westeurope"] # e.g. ["conus", "westconus", "europe", "westeurope"]
        leadtimes               = (3,) # (.5, 1, 2, 3)
    if experiment_name in ("cmems_cmems", "juno-cmcc_cmems", "juno-medfs_cmems", "juno-medfs_juno-medfs"):
        variables               = ["sst", "ssh", "sss"] # e.g. ["sst", "ssh", "sss"]
        regions                 = ["med"] # e.g. ["pacific", "natlantic", "satlantic", "indian", "atlanticbox"]
        leadtimes               = (1, 2, 3, 4, 5, 6, 7, 8, 9) # (1, 2, 3, 4, 5, 6, 7, 8, 9)

    inpaint_nan                 = True # whether to inpaint nan values in input and target datasets (after loading, before torch dataset generation)
    anomaly                     = False # if True, predict anomaly (i.e. remove climatology from target variable), otherwise predict absolute values
    per_epoch_resplit           = False # if True, split test and validation randomly per epoch with a different seed, if False only one initial split for all epochs (fixed seed)
    skip_train_test_plots       = False # whether to skip train/test diagnostic plots and Lightning curve exports

    if experiment_name == "juno-ecmwf_juno-ecmwf":
        start_train_date            = datetime(2019, 10, 11)
        end_train_date              = datetime(2024, 12, 31)
        start_test_date             = datetime(2025, 1, 1)
        end_test_date               = datetime(2025, 12, 31)
        freq                        = "12h"
        regrid_resolution           = 0.1
    if experiment_name == "cmems_cmems":
        start_train_date            = datetime(2022, 6, 4)
        end_train_date              = datetime(2025, 12, 31)
        start_test_date             = datetime(2026, 1, 1)
        end_test_date               = datetime(2026, 3, 31)
        freq                        = "12h"
        regrid_resolution           = 0.08
    if experiment_name in ("juno-cmcc_cmems", "juno-medfs_cmems"):
        start_train_date            = datetime(2024, 4, 23) # this is the earliest available
        end_train_date              = datetime(2025, 12, 31)
        start_test_date             = datetime(2026, 1, 1)
        end_test_date               = datetime(2026, 4, 11)
        freq                        = "1d"
        regrid_resolution           = 0.08
    if experiment_name == "juno-medfs_juno-medfs":
        start_train_date            = datetime(2022, 1, 10)
        end_train_date              = datetime(2025, 12, 31)
        start_test_date             = datetime(2026, 1, 1)
        end_test_date               = datetime(2026, 4, 11)
        freq                        = "1d"
        regrid_resolution           = 0.04

    target_realization_avg      = False # average over realizations for target variable (if True, add _taravg suffix to exp_suffix)
    realization_as_channel      = False # use realization a channel dim (if True, add _rasc suffix to exp_suffix)
    n_input_realizations        = 30 # used only if realization_as_channel = True

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

    grib_index_path             = "/tmp/.grib_index"
    # grib_index_path             = "/work/cmcc/jd19424/test-ML/.grib_index"

    cmems_credential = dict(
        username="jacopo.dallaglio2@unibo.it",
        password="ovUqR&5Q$q9%",
    )

    # Default test periods and kwargs
    test_periods = TimeRange(start=start_test_date, end=end_test_date, freq=freq)
    input_test_provider_kwargs = None
    target_test_provider_kwargs = None

    if experiment_name == "juno-ecmwf_juno-ecmwf":
        grib_indexing = dict(cfgrib_idx_path=grib_index_path)
        if start_test_date <= juno_location_cutoff and end_test_date > juno_location_cutoff:
            test_periods = [
                TimeRange(start=start_test_date, end=juno_location_cutoff, freq=freq),
                TimeRange(start=juno_location_cutoff + relativedelta(days=1), end=end_test_date, freq=freq),
            ]
            input_test_provider_kwargs = dict(
                train=dict(root_path=root_path_forecast) | grib_indexing,
                # test partly in old folder and partly in new folder
                test=[
                    dict(root_path=root_path_forecast) | grib_indexing,
                    dict(root_path=root_path_forecast_new, file_header="CMS") | grib_indexing,
                ],
            )
            target_test_provider_kwargs = dict(
                train=dict(root_path=root_path_analysis) | grib_indexing,
                test=[
                    dict(root_path=root_path_analysis) | grib_indexing,
                    dict(root_path=root_path_analysis_new, file_header="CMD") | grib_indexing,
                ],
            )
        else:
            if start_test_date >= juno_location_cutoff and end_test_date > juno_location_cutoff:
                input_test_provider_kwargs = dict(
                    train=dict(root_path=root_path_forecast) | grib_indexing,
                    # test fully in new folder
                    test=dict(root_path=root_path_forecast_new, file_header="CMS") | grib_indexing,
                )
                target_test_provider_kwargs = dict(
                    train=dict(root_path=root_path_analysis) | grib_indexing,
                    # test fully in new folder
                    test=dict(root_path=root_path_analysis_new, file_header="CMS") | grib_indexing,
            )
    if experiment_name in ("cmems_cmems", "juno-cmcc_cmems", "juno-medfs_cmems"):
        target_test_provider_kwargs = dict(
            train=cmems_credential,
            test=cmems_credential,
        )
    if experiment_name == "cmems_cmems":
        input_test_provider_kwargs = target_test_provider_kwargs

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

        launcher = MLBCExperimentLauncher(
            experiment=launcher_cfg,
            run_mode=run_mode,
            max_retries=max_retries,
            log_level=log_level,
            variables_input=variables,
            variables_target=variables,
            leadtimes=leadtimes,
            leadtime_unit="days",
            regions=regions,
            train_periods=train_periods,
            test_periods=test_periods,
            net=net,
            # Optional
            dask_workers=24,
            memory_limit="16GB",
            juno_file_open_workers=juno_file_open_workers,
            external_mask_path=external_mask_path,
            external_mask_variable=external_mask_variable,
            only_longest_train_period=only_longest_train_period,
            force_retrain=force_retrain,
            force_rebuild_dataset=force_rebuild_dataset,
            dataset_cache_enabled=dataset_cache_enabled,
            dataset_cache_root=dataset_cache_root,
            weights_filename=weights_filename,
            # Providers args
            earthkit_cache_dir=earthkit_cache_dir,
            regrid_resolution=regrid_resolution,
            juno_per_sample_regrid=juno_per_sample_regrid,
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
            skip_train_test_plots=skip_train_test_plots,
        )

        launcher.run()
