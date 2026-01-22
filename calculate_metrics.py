from pathlib import Path
from datetime import datetime, timedelta
from rich import print
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
from earthml.utils import Dask, load_exp, add_ke_to_runs, make_exp_folder_path
from earthml.metrics import Metrics, save_metrics_parquets #, PowerSpectrum

if __name__ == "__main__":
    dask_earthml = Dask(n_workers=32)
    dask_earthml.start()
    client, cluster = dask_earthml.client, dask_earthml.cluster
    base_url = "http://localhost:"
    print("Dask dashboard:", client.dashboard_link.replace("127.0.0.1:", base_url))

    exp_root_folder_weather = "/work/cmcc/jd19424/test-ML/experiments_earthML_weather/"
    analysis_folder_weather = "/work/cmcc/jd19424/test-ML/analysis_earthML_weather/"

    region = "ConUS"
    # region = "Europe"
    # train_period = "20140101-20151231"
    train_period_ocean = "19950101-20201231"
    test_period_ocean = "20210101-20241231"
    suffixes = ['']
    # suffixes = ['_mse', '_het']

    # vars = ['msl', 't2m', 'tcc']
    vars = {}
    vars_weather = {
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

    vars_ocean = {
        'sss_month': {'exp_var': {'fc': 'sos', 'an': 'sosaline'}, 'region': "CentralPacific", 'exp_suffix': '_32bs_mse_allrel_ocean_cmcc_oras5'},
        't14d_month': {'exp_var': {'fc': 't14d', 'an': 'so14chgt'}, 'region': "CentralPacific", 'exp_suffix': '_32bs_mse_allrel_ocean_cmcc_oras5'},
        't17d_month': {'exp_var': {'fc': 't14d', 'an': 'so14chgt'}, 'region': "CentralPacific", 'exp_suffix': '_32bs_mse_allrel_ocean_cmcc_oras5'},
    }

    experiments_weather = {
        "name": "weather",
        "input_provider": "atmo.juno.ecmwf.forecast.hourly",
        "target_provider": "atmo.juno.ecmwf.analysis.6hourly",
        "vars": vars_weather,
        "leadtimes": (12, 24, 48, 72),
        "leadtime_unit": "hours",
        "train_periods": ("20191011-20241231", "20220522-20241231", "20230911-20241231", "20240507-20241231", "20240903-20241231"),
        "test_period": "20250101-20251031",
        "root": exp_root_folder_weather,
    }

    # experiments_ocean = {
    #     "name": "ocean",
    #     "input_provider": "...",
    #     "target_provider": "...",
    #     "vars": vars_ocean,
    #     "leadtimes": (12, 24, 48, 72),
    #     "leadtime_unit": "hours",
    #     "train_period": "...",
    #     "test_period": "...",
    #     "root": exp_root_folder_ocean,
    # }

    runs_weather, _ = load_exp(
        exp_root=exp_root_folder_weather,
        exp_cfg=experiments_weather,
        type_data="test",
        # type_data="train",
    )

    # Calculate train period sizes
    _, tp_sizes = load_exp(
        exp_root=exp_root_folder_weather,
        exp_cfg=experiments_weather,
        type_data="train",
        only_sizes=True,
    )

    runs_weather = add_ke_to_runs(runs_weather, suffixes=("_mse", "_het"))

    metrics = {}
    for name, run in runs_weather.items():
        print(f"Run {name}")
        m = Metrics(truth=run['an'], data=[run['fc'], run['pr']], truth_name="an", data_name=["fc", "pr"])
        metrics[name] = m.compute_all_metrics(geo_weighted=True) #, eps=1e-12)

    vars_weather_from_fc = [v for v in runs_weather[next(iter(runs_weather))]['fc'].data_vars if v != '_has_var']

    print(f"Processed vars: {vars_weather_from_fc}")

    # Plot maps in grid
    # m = "nrmse_map"
    # for var_name in vars_weather_from_fc:
    #     save_path = Path(exp_root_folder_weather) / Path("plots") / Path(
    #         f"{m}_{experiments_weather['name']}_{var_name}_{experiments_weather['test_period']}" \
    #         f"_{experiments_weather['input_provider']}_{experiments_weather['target_provider']}.png"
    #     )

    #     plot_map_metric_grid(
    #         metrics,
    #         metric=m,
    #         variable=var_name,
    #         diff=True,
    #         model_a="fc",
    #         model_b="pr",
    #         cmap="RdBu_r",
    #         save_path=save_path,
    #     )
    #     print(f"Saved {m} to {save_path}")

    # 'mae', 'bias', 'rmse', 'stderr', 'std_of_errs', 'mape', 'smape', 'nrmse_global', 'nrmse_map_mean', 'r2_global'
    # metric_names = ["mae", "bias", "rmse", "mape", "smape", "nrmse_global", "nmae_global", "nbias_global", "abs_nbias_global", "r2_global", "corr_global", "corr_flat"]
    metric_names = None # all available metrics

    metrics_df = save_metrics_parquets(
        metrics=metrics,
        base_folder=analysis_folder_weather,
        vars_list=vars_weather_from_fc,
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
    
