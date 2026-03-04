from pathlib import Path
from datetime import datetime, timedelta
from rich import print

import numpy as np
import xarray as xr

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Dask.*")
warnings.filterwarnings("ignore", message=".*distributed.scheduler.*")
import logging
logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)
logging.getLogger("distributed.worker").setLevel(logging.ERROR)
logging.getLogger("distributed.nanny").setLevel(logging.ERROR)

# from earthml.manager import Launcher
# from earthml.logging import Logger
from earthml.utils import Dask, load_exp, add_ke_to_runs, make_exp_folder_path, guess_time_dim, guess_lon_dim, guess_lat_dim
from earthml.metrics import get_runs_and_metrics, save_metrics #, PowerSpectrum

if __name__ == "__main__":

    exp_type = "ocean"
    # exp_type = "weather"

    dask_earthml = Dask(n_workers=5, memory_limit="100GiB") # memory_limit: "25GiB" or float (fraction of tot mem per worker) or int (byte per worker)
    dask_earthml.start()
    client, cluster = dask_earthml.client, dask_earthml.cluster
    base_url = "http://localhost:"
    print("Dask dashboard:", client.dashboard_link.replace("http://127.0.0.1:", base_url))

    exp_root_path = "/work/cmcc/jd19424/test-ML/"

    if exp_type == "weather":
        exp_root_folder = exp_root_path+"experiments_earthML_weather/"
        analysis_folder = exp_root_path+"analysis_earthML_weather/"

        vars_dict = {
            # 'msl_mse': {'exp_var': {'fc': 'msl', 'an': 'msl'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_mse'},
            # 'u10_mse': {'exp_var': {'fc': 'u10', 'an': 'u10'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_mse'},
            # 'v10_mse': {'exp_var': {'fc': 'v10', 'an': 'v10'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_mse'},
            # 'd2m_mse': {'exp_var': {'fc': 'd2m', 'an': 'd2m'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_mse'},
            # 'tcc_mse': {'exp_var': {'fc': 'tcc', 'an': 'tcc'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_mse'},
            # 't2m_mse': {'exp_var': {'fc': 't2m', 'an': 't2m'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_mse'},
            # 't2m_mse_smoothed_R25.0_s0.8':  {'exp_var': {'fc': 't2m_smoothed_R25.0_s0.8', 'an': 't2m_smoothed_R25.0_s0.8'},  'region': 'ConUS', 'exp_suffix': '_32bs_smoothed'},
            # 't2m_mse_smoothed_R50.0_s1.5':  {'exp_var': {'fc': 't2m_smoothed_R50.0_s1.5', 'an': 't2m_smoothed_R50.0_s1.5'},  'region': 'ConUS', 'exp_suffix': '_32bs_smoothed'},
            # 't2m_mse_smoothed_R100.0_s3.0': {'exp_var': {'fc': 't2m_smoothed_R100.0_s3.0', 'an': 't2m_smoothed_R100.0_s3.0'}, 'region': 'ConUS', 'exp_suffix': '_32bs_smoothed'},
            # 't2m_mse_smoothed_R200.0_s6.0': {'exp_var': {'fc': 't2m_smoothed_R200.0_s6.0', 'an': 't2m_smoothed_R200.0_s6.0'}, 'region': 'ConUS', 'exp_suffix': '_32bs_smoothed'},
            'msl_het': {'exp_var': {'fc': 'msl', 'an': 'msl'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_het'},
            # 'u10_het': {'exp_var': {'fc': 'u10', 'an': 'u10'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_het'},
            # 'v10_het': {'exp_var': {'fc': 'v10', 'an': 'v10'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_het'},
            # 'd2m_het': {'exp_var': {'fc': 'd2m', 'an': 'd2m'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_het'},
            # 'tcc_het': {'exp_var': {'fc': 'tcc', 'an': 'tcc'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_het'},
            't2m_het': {'exp_var': {'fc': 't2m', 'an': 't2m'}, 'region': 'ConUS', 'exp_suffix': '32bs_50epoch_het'},
        }

        experiments = {
            "name": "weather",
            "input_provider": "atmo.juno.ecmwf.forecast.hourly",
            "target_provider": "atmo.juno.ecmwf.analysis.6hourly",
            "vars": vars_dict,
            "leadtimes": (12, 24, 48, 72),
            "leadtime_unit": "hours",
            "train_periods": ("20191011-20241231", "20220522-20241231", "20230911-20241231", "20240507-20241231", "20240903-20241231"),
            "test_period": "20250101-20251031",
            "root": exp_root_folder,
        }

        runs, _ = load_exp(
            exp_root=exp_root_folder,
            exp_cfg=experiments,
            type_data="test",
            # type_data="train",
        )

    elif exp_type == "ocean":
        exp_root_folder = exp_root_path+"experiments_earthML_ocean/"
        analysis_folder = exp_root_path+"analysis_earthML_ocean/"

        type_data = "test"
        var_suffix = ("_mseloss", "_maskedmseloss", "_maskedmseloss_rasc", "_variancenormalizedmseloss_spatial", "_variancenormalizedmseloss_geochannel", "_heterobiascorrectionloss")
        # vars_d = [{"an": "sosaline", "fc": "sos"}, {"an": "so14chgt", "fc": "t14d"}, {"an": "sosstsst", "fc": "sst"}]
        # vars_d, input_provider = {"an": "sosaline", "fc": "sos"}, "ocean.juno.cmcc.hindcast"
        # var_plot, var_plot_nicename, var_plot_unit = "sos_mse", "sos ORAS5 target", "psu"
        # var_cmap, var_limits = "viridis", (33, 37)
        # vars_d, input_provider = {"an": "so14chgt", "fc": "t14d"}, "ocean.juno.cmcc.hindcast" # ORAS5
        # var_plot, var_plot_nicename, var_plot_unit = "t14d_mse", "t14d ORAS5 target", "m"
        # var_cmap, var_limits = "Blues", (0, 400)
        vars_d, input_provider = {"an": "sosstsst", "fc": "sst"}, "atmo.earthkit.cmcc.hindcast.monthly"
        # var_plot, var_plot_nicename, var_plot_unit = "sst_maskedmseloss_rasc", "sst maske MSE R as C loss", "K"
        var_plot, var_plot_nicename, var_plot_unit = tuple("sst"+item for item in var_suffix), ("sst MSE", "sst masked MSE", "sst masked MSE R as C", "sst vMSE spatial", "sst vMSE geochannel", "sst hetMSE"), "K"
        # var_plot, var_plot_nicename, var_plot_unit = ("sst_maskedmse", "sst_varspatialmse"), ("sst masked MSE loss", "sst masked varCMSE loss"), "K"
        var_cmap, var_limits_field = "jet", (290, 305)
        # var_plot, var_plot_nicename = "t14d_mse_taravg", "t14d averaged target"
        # var_plot, var_plot_nicename = "t14d_mse", "t14d realization target"
        target_provider = "ocean.earthkit.oras5.reanalysis.monthly"
        leadtime_plot = 6
        leadtimes, leadtime_unit = (6,), "months"
        # leadtimes, leadtime_unit = (1, 2, 3, 4, 5, 6), "months"
        vars_mse = 'sss_mse'
        # vars_mse = ['sss_mse', 't14d_mse', 'sss_mse_taravg', 't14d_mse_taravg']
        vars_het = []
        center_lon = 180
        plt_regional_extent = (-20, 60, -30, 30) # referenced to the central longitude
        # Scoreboard  setup
        lt_tick_labels_sb = ["1M", "2M", "3M", "4M", "5M"]
        var_tick_labels_sb = ["sss", 't14d']
        tp_index_name_sb = "years"
        tp_value_sb = 27 # max train period in years
        # vars_dict_cmcc = [{'fc': 't14d', 'an': 't14d'},  {'fc': 'sos', 'an': 'sos'}, {'fc': 't17d', 'an': 't17d'}]
        # vars_dict_oras5 = [{'fc': 't14d', 'an': 'so14chgt'}, {'fc': 'sos', 'an': 'sosaline'}]

        runs, metrics = get_runs_and_metrics(
            var_names=vars_d,
            # var_suffix="_maskedmseloss_rasc",
            # var_suffix=("_maskedmseloss", "_maskedmseloss_rasc"),
            var_suffix=var_suffix,
            region="CentralPacific",
            # exp_suffix_root="_32bs_50epoch",
            exp_suffix_root="",
            exp_name="ocean",
            exp_root=exp_root_folder,
            input_provider=input_provider,
            target_provider=target_provider,
            leadtimes=leadtimes,
            leadtime_unit="months",
            # train_periods=("19930701-20201231", "20070401-20201231", "20140201-20201231", "20170801-20201231", "20190401-20201231"),
            train_periods=("19930701-20201231",),
            test_period="20210101-20221231",
            type_data=type_data,
            models=("an", "fc", "pr"),
        )
    # runs_cmcc, _ = load_exp(
    #     exp_root=exp_root_folder,
    #     exp_cfg=experiments_juno_cmcc,
    #     type_data="test",
    #     # type_data="train",
    # )
    # runs_oras5, _ = load_exp(
    #     exp_root=exp_root_folder,
    #     exp_cfg=experiments_cds_cmcc_oras5,
    #     type_data="test",
    #     # type_data="train",
    # )

    # Calculate train period sizes
    # _, tp_sizes_cmcc = load_exp(
    #     exp_root=exp_root_folder,
    #     exp_cfg=experiments_juno_cmcc,
    #     type_data="train",
    #     only_sizes=True,
    # )
    # _, tp_sizes_oras5 = load_exp(
    #     exp_root=exp_root_folder,
    #     exp_cfg=experiments_cds_cmcc_oras5,
    #     type_data="train",
    #     only_sizes=True,
    # )

    # Combine runs
    # runs = {}

    # for tp in runs_cmcc:
    #     runs[tp] = {}
    #     for ds_name in runs_cmcc[tp]:
    #         ds_cmcc = runs_cmcc[tp][ds_name]
    #         ds_oras5 = runs_oras5[tp][ds_name]

    #         runs[tp][ds_name] = xr.merge([ds_cmcc, ds_oras5])

    # runs = runs_oras5

    vars_from_fc = [v for v in runs[next(iter(runs))]['fc'].data_vars if v != '_has_var']

    print(f"Processed vars: {vars_from_fc}")

    # 'mae', 'bias', 'rmse', 'stderr', 'std_of_errs', 'mape', 'smape', 'nrmse_global', 'nrmse_map_mean', 'r2_global'
    # metric_names = ["mae", "bias", "rmse", "mape", "smape", "nrmse_global", "nmae_global", "nbias_global", "abs_nbias_global", "r2_global", "corr_global", "corr_flat"]
    metric_names = None # all available metrics

    metrics_df = save_metrics(
        metrics=metrics,
        base_folder=analysis_folder,
        vars_list=vars_from_fc,
        metric_names=metric_names,
        models_diff=("fc", "pr"),
        partition_cols=("train_period", "model"), # no leadtime
    )

    df_nodiff, path_nodiff = metrics_df["no"]
    df_delta, path_delta = metrics_df["delta"]
    df_ratio, path_ratio = metrics_df["ratio"]

    print("Saved nodiff to:", path_nodiff)
    print("Saved diff delta to:", path_delta)
    print("Saved diff ratio to:", path_ratio)


    # df.to_parquet(f"{analysis_folder_weather}/metrics_{str(metric_names)}_{str(vars_weather_from_fc)}.parquet", engine="pyarrow", compression="snappy")
    # pd.read_parquet(parquet_filename, filters=[("model", "==", "fc")])


    # joblib.dump(metrics, f"/work/cmcc/jd19424/test-ML/dataML/metrics_ConUS-20210323-20231231_ConUS-20240101-20240531_{exp_suffix}.mtr")

    # ps_fc = PowerSpectrum(fc.isel(valid_time=0), dx_deg=0.1, dy_deg=0.1)
    # ps_pr = PowerSpectrum(pr, dx_deg=0.1, dy_deg=0.1)
    # ps_fc.plot_all(savepath="/work/cmcc/jd19424/test-ML/earthML/ps_fc0.png", show=False)
    # ps_pr.plot_all(savepath="/work/cmcc/jd19424/test-ML/earthML/ps_pr.png", show=False)
    
