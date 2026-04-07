from datetime import datetime

from rich import print

from earthml import *

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------
    # User params
    # ----------------------------------------------------------------------------------
    experiment_type             = "seasonal"                # seasonal, weather
    experiment_name             = "cds-cmcc_oras5"          # cds-cmcc_oras5

    run_mode                    = "prepare"     # train_test_on_train, train_test, prepare, train, test
    experiment_mode             = "debug"                   # full, short, debug
    only_longest_train_period   = True                      # only train on the largest train period (if False, train on all periods, which can be much slower but allows to see variability across train periods)
    # add_hyper_exp_name_suffix   = False                     # if True use also batch_size, max_epochs, initial_learning_rate in exp name (and resulting folder) automatically extending exp_suffix
    extra_exp_suffix            = ""                # additional custom suffix to add to exp name (and resulting folder), e.g. "_debug" or "_try1"

    exp_root_folder             = "/Users/jacopodallaglio/ML/experiments_earthML_ocean_refactor_test"
    earthkit_cache_dir          = "/Users/jacopodallaglio/ML/.earthkit-cache"       # if using earthkit datasource

    variables                   = ["sst"]             # e.g. ["sst", "ssh", "sss", "t14d", "t17d"]
    regions                     = ["pacific"]                               # e.g. ["pacific", "natlantic", "indian"]
    leadtimes                   = (1, 2, 3, 4, 5, 6)

    inpaint_nan                 = True                      # whether to inpaint nan values in input and target datasets (after loading, before torch dataset generation)
    anomaly                     = False                      # if True, predict anomaly (i.e. remove climatology from target variable), otherwise predict absolute values
    per_epoch_resplit           = False                     # if True, split test and validation randomly per epoch with a different seed, if False only one initial split for all epochs (fixed seed)

    start_train_date            = datetime(1993, 6, 1)
    end_train_date              = datetime(2020, 12, 1)
    start_test_date             = datetime(2021, 1, 1)
    end_test_date               = datetime(2022, 12, 1)

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
            net={"use_full_batch_input_as_baseline": True},
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
        leadtimes       = (6,)
        if experiment_mode == "debug":
            # Very short periods for debug
            start_train_date        = datetime(1993, 6, 1)
            # end_train_date          = datetime(1994, 12, 31)
            end_train_date          = datetime(1994, 12, 31)
            start_test_date         = datetime(2021, 1, 1)
            end_test_date           = datetime(2021, 12, 31)        

    # Set periods and data providers
    train_period    = TimeRange(start=start_train_date, end=end_train_date, freq='MS')
    test_period     = TimeRange(start=start_test_date,  end=end_test_date,  freq='MS')


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
            Leadtime(name="leadtime", unit="months", value=lt)
            for lt in leadtimes
        ]
        launcher = MLBCExperimentLauncher(
            experiment=launcher_cfg,
            run_mode=run_mode,
            max_retries=10,
            variables_input=variables,
            variables_target=variables,
            leadtimes=leadtimes_lt,
            regions=regions,
            train_period=train_period,
            test_period=test_period,
            net=net,
            # Optional
            dask_workers=None,
            only_longest_train_period=only_longest_train_period,
            # Providers args
            earthkit_cache_dir=earthkit_cache_dir,
            regrid_resolution=1,
            # Experiment
            inpaint_nan=inpaint_nan,
            anomaly=anomaly,
            per_epoch_resplit=per_epoch_resplit,
            torch_preprocess_fn=None,
            target_realization_avg=target_realization_avg,  # whether to average over target realizations when loading target data to torch
            realization_as_channel=realization_as_channel,  # whether to use realization a channel dimension
        )

        launcher.run()

