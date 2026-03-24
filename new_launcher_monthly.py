import traceback
from itertools import product
from datetime import datetime
from dateutil.relativedelta import relativedelta

from earthml.dataclasses import TimeRange
from earthml.launchers.mlfc import MLFCScenario, MLFCRunner
from earthml.manager import build_loss, period_bounds, oras5_train_kwargs, build_monthly_experiment_spec
from earthml.logging import RunKey, log_event

if __name__ == "__main__":

    # ----------------------------------------------------------------------------------
    # User params
    # ----------------------------------------------------------------------------------
    max_retries                 = 10                # n attempts in case of errors in inner loop
    run_mode                    = "train_test"      # train_test, dryrun, train, test
    host_machine                = "local"           # local, juno
    experiment_mode             = "full"            # full, short, debug
    only_longest_train_period   = True              # only train on the largest train period (if False, train on all periods, which can be much slower but allows to see variability across train periods)
    add_hyper_exp_name_suffix   = False             # if True use also batch_size, max_epochs, initial_learning_rate in exp name (and resulting folder) automatically extending exp_suffix
    inpaint_nan                 = True              # if True, inpaint NaN values in input and target data with bilinear interpolation
    extra_exp_suffix            = ""                # additional custom suffix to add to exp name (and resulting folder), e.g. "_debug" or "_try1"

    exp_root_folder             = "/Users/jacopodallaglio/ML/experiments_earthML_ocean/"
    earthkit_cache_dir          = "/Users/jacopodallaglio/ML/.earthkit-cache"   # if using earthkit datasource

    vars_sweep                  = ["sst", "ssh", "sss", "t14d", "t17d"] # e.g. ["sst", "ssh", "sss", "t14d", "t17d"]
    regions_sweep               = ["pacific"]                           # e.g. ["pacific", "atlantic", "indian"]

    inpaint_nan                 = False                     # whether to inpaint nan values in input and target datasets (after loading, before torch dataset generation)
    anomaly                     = False                     # TODO: if True, predict anomaly (i.e. remove climatology from target variable), otherwise predict absolute values

    start_train_date            = datetime(1993, 7, 1)
    end_train_date              = datetime(2020, 12, 31)
    start_test_date             = datetime(2021, 1, 1)
    end_test_date               = datetime(2022, 12, 31)

    target_realization_avg      = False             # average over realizations for target variable (if True, add _taravg suffix to exp_suffix)
    realization_as_channel      = False             # use realization a channel dim (if True, add _rasc suffix to exp_suffix)
    n_input_realizations        = 30                # used only if realization_as_channel = True

    # Hyperparams
    batch_size                  = 32
    max_epochs                  = 50*n_input_realizations if realization_as_channel else 50
    init_learning_rate          = 1e-3
    accumulate_grad_batches     = 2
    earlystopping_patience      = 30

    loss_sweep = [
        {"loss_sel": "MSELoss"},
        {"loss_sel": "MaskedMSELoss"},
        {"loss_sel": "VarianceNormalizedMSELoss", "variance_type": "spatial", "latitudes": False}, # channel, geochannel, spatial, temporal, geotemporal
        {"loss_sel": "VarianceNormalizedMSELoss", "variance_type": "geochannel", "latitudes": True},
        {"loss_sel": "HeteroBiasCorrectionLoss", "use_first_input": True, "variance_type": "spatial", "latitudes": False, "lambda_identity": 0.1, "bias_scale": 0.5},
        # {"loss_sel": "GaussianNLLFromLogits"}, # TODO fix
    ]

    # ----------------------------------------------------------------------------------
    # Deeper settings
    # ----------------------------------------------------------------------------------
    if experiment_mode in ("short", "debug"):
        # Short exp
        full_leadtimes_days         = (165,)
        full_leadtimes_atmo_days    = (180,)
        full_leadtimes_months       = (6,)
        if experiment_mode == "debug":
            # Very short periods for debug
            start_train_date        = datetime(1993, 7, 1)
            end_train_date          = datetime(1994, 12, 31)
            start_test_date         = datetime(2021, 1, 1)
            end_test_date           = datetime(2021, 12, 31)
    else:
        # Full exp
        full_leadtimes_days         = (15, 45, 75, 105, 135, 165)
        full_leadtimes_atmo_days    = (30, 60, 90, 120, 150, 180) # atmo seasonal forecast has end of month leadtimes
        full_leadtimes_months       = (1, 2, 3, 4, 5, 6)

    # Set periods and data providers
    train_period    = TimeRange(start=start_train_date, end=end_train_date, freq='MS')
    test_period     = TimeRange(start=start_test_date,  end=end_test_date,  freq='MS')

    # Set var categories
    vars_cloud_oras5        = ("sst", "ssh")
    vars_cloud_cds_atmo     = ("sst",)
    vars_cloud_cds_ocean    = ("ssh",)

    delta_train = relativedelta(end_train_date, start_train_date)
    months_train = delta_train.years * 12 + delta_train.months
    cutoff_oras5_consolidated = datetime(2014, 12, 31) # cutoff date between consolidated and operational ORAS5 datasets

    failed_runs: list[tuple[RunKey, str, str]] = []

    # ----------------------------------------------------------------------------------
    # Mainloop
    # ----------------------------------------------------------------------------------
    for var_exp, region_sel, loss_cfg in product(vars_sweep, regions_sweep, loss_sweep):

        loss_name, loss_params, loss_suf = build_loss(loss_cfg)

        try:
            spec = build_monthly_experiment_spec(
                var_exp=var_exp,
                host_machine=host_machine,
                only_longest_train_period=only_longest_train_period,
                train_period=train_period,
                months_train=months_train,
                cutoff_oras5_consolidated=cutoff_oras5_consolidated,
                earthkit_cache_dir=earthkit_cache_dir,
                vars_cloud_oras5=vars_cloud_oras5,
                vars_cloud_cds_atmo=vars_cloud_cds_atmo,
                full_leadtimes_days=full_leadtimes_days,
                full_leadtimes_atmo_days=full_leadtimes_atmo_days,
                full_leadtimes_months=full_leadtimes_months,
            )
        except Exception as e:
            log_event("SKIP", msg=f"spec build failed var={var_exp} region={region_sel} loss={loss_name}: {e}")
            traceback.print_exc()
            continue

        log_event("EXP", msg=f"var={var_exp} region={region_sel} loss={loss_name} exp={spec.experiment_type}")

        for i, p in enumerate(spec.train_periods_target, 1):
            p_start, p_end = period_bounds(p)
            rd = relativedelta(p_end, p_start)
            months = rd.years * 12 + rd.months
            log_event(
                "PERIOD",
                msg=f"{i}: {p_start:%Y-%m-%d} -> {p_end:%Y-%m-%d}",
                months=months,
                exp_type=spec.experiment_type,
                var=var_exp,
                region=region_sel,
            )

        # Inner loop
        for var_fc_key, var_an_key in spec.var_keys:
            for leadtime_var_fc_value, leadtime_mult in zip(spec.full_leadtimes, spec.full_leadtime_multiple):
                for train_p_in, train_p_tar in zip(spec.train_periods_input, spec.train_periods_target):

                    # Create unique run key
                    train_in_s, train_in_e = period_bounds(train_p_in)
                    train_tar_s, train_tar_e = period_bounds(train_p_tar)

                    run_key = RunKey(
                        var_exp=var_exp,
                        region=region_sel,
                        experiment_type=spec.experiment_type,
                        var_fc_key=var_fc_key,
                        var_an_key=var_an_key,
                        loss_suf=loss_suf,
                        lt_fc_value=leadtime_var_fc_value,
                        lt_fc_unit=spec.leadtime_var_unit,
                        lt_value=leadtime_mult,
                        lt_unit=spec.leadtime_unit,
                        train_in_start=train_in_s,
                        train_in_end=train_in_e,
                        train_tar_start=train_tar_s,
                        train_tar_end=train_tar_e,
                    )

                    # print(train_p_in, train_p_tar)
                    if spec.experiment_type in ("juno-cmcc_oras5", "cds-cmcc_oras5"):
                        target_provider_kwargs = dict(
                            train=oras5_train_kwargs(
                                train_p_tar,
                                cutoff_oras5_consolidated,
                                spec.target_provider_kwargs_oras5_consolidated,
                                spec.target_provider_kwargs_oras5_operational,
                            ),
                            test=oras5_train_kwargs(
                                test_period,
                                cutoff_oras5_consolidated,
                                spec.target_provider_kwargs_oras5_consolidated,
                                spec.target_provider_kwargs_oras5_operational,
                            ),
                        )
                    else:
                        target_provider_kwargs = dict(
                            train=spec.target_provider_kwargs_common,
                            test=spec.target_provider_kwargs_common,
                        )

                    ocean_scenario = MLFCScenario(
                        name="ocean",
                        leadtime_var_fc_name="step" if (spec.experiment_type=="cds-cmcc_oras5" and var_exp in vars_cloud_cds_atmo) else "leadtime", # CDS monthly season forecast leadtime variable is "step" instead of "leadtime"
                        leadtime_var_an_name="leadtime",
                        leadtime_var_fc_value=leadtime_var_fc_value,
                        leadtime_var_an_value=spec.leadtime_var_an_value, # fixed leadtime for analysis, ignored if no leadtime in analysis dataset
                        leadtime_var_unit=spec.leadtime_var_unit, # SST -> hours, other vars -> days
                        leadtime_value=leadtime_mult,
                        leadtime_unit=spec.leadtime_unit, # SST -> hours, other vars -> months
                        var_fc_key=var_fc_key,
                        var_an_key=var_an_key,
                        region_key=region_sel,
                        train_period=dict(
                            input=train_p_in,
                            target=train_p_tar,
                        ),
                        test_period=dict(
                            input=test_period,
                            target=test_period,
                        ),
                        input_provider=spec.input_provider,
                        target_provider=spec.target_provider,
                        input_provider_kwargs=spec.input_provider_kwargs,
                        target_provider_kwargs=target_provider_kwargs,
                        save_train=True,
                        save_test=True,
                        anomaly=anomaly,
                        inpaint_nan=inpaint_nan,
                        torch_preprocess_fn=None,
                        target_realization_avg=target_realization_avg,
                        realization_as_channel=realization_as_channel,
                    )

                    # Generate exp_suffix
                    if add_hyper_exp_name_suffix:
                        exp_suffix = f"_{str(batch_size)}bs_{str(max_epochs)}epoch_{str(init_learning_rate)}lr_{loss_suf}"
                    else:
                        exp_suffix = f"_{loss_suf}"
                    exp_suffix += ("_taravg" if target_realization_avg else "") + ("_rasc" if realization_as_channel else "")
                    exp_suffix += extra_exp_suffix

                    runner = MLFCRunner(
                        scenario=ocean_scenario,
                        exp_root_folder=exp_root_folder,
                        exp_suffix=exp_suffix,
                        # ML options
                        n_channels=n_input_realizations, # ignored if realization_as_channel is false
                        n_classes=n_input_realizations,  # ignored if realization_as_channel is false
                        learning_rate=init_learning_rate,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        loss=loss_name,
                        loss_params=loss_params,
                        accumulate_grad_batches=accumulate_grad_batches,
                        earlystopping_patience=earlystopping_patience,
                    )

                    success = False

                    log_event("RUN", run_key, msg=f"mode={run_mode}", bs=batch_size, epochs=max_epochs, lr=init_learning_rate)

                    e_last = None
                    for attempt in range(max_retries):
                        try:
                            runner.run(mode=run_mode)
                            success = True
                            log_event("OK", run_key)
                            break
                        except (RuntimeError, OSError) as e:
                            e_last = e
                            log_event("RETRY", run_key, msg=str(e), attempt=attempt+1, max_retries=max_retries)
                            if attempt + 1 == max_retries: # print traceback on last retry only
                                traceback.print_exc()

                    if not success:
                        if e_last is None:
                            failed_runs.append((run_key, "UnknownError", "No exception captured"))
                        else:
                            failed_runs.append((run_key, type(e_last).__name__, repr(e_last)))
                        log_event("FAIL", run_key, msg="giving up")
                        continue

    log_event("SUMMARY", msg=f"failed={len(failed_runs)}")
    for rk, exc_type, err in failed_runs:
        log_event("FAIL", rk, msg=f"{exc_type}: {err}")
