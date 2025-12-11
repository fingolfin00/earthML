# from earthml.manager import Launcher
# from earthml.logging import Logger
from earthml.utils import Dask, XarrayDataset
from earthml.metrics import Metrics, PowerSpectrum
from pathlib import Path
from datetime import datetime, timedelta
import joblib
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

if __name__ == "__main__":
    dask_earthml = Dask(n_workers=32)
    client, cluster = dask_earthml.client, dask_earthml.cluster
    print("Dask dashboard:", client.dashboard_link)
    exp_root_folder = "/work/cmcc/jd19424/test-ML/experiments_earthML/"
    pr, fc, an = None, None, None
    # for v in {'t2m_original', 't2m_smoothed_R25.0_s0.8', 't2m_smoothed_R50.0_s1.5', 't2m_smoothed_R100.0_s3.0', 't2m_smoothed_R200.0_s6.0'}:
    for v in {'msl', 'u10', 'd2m', 't2m', 'v10', 'tcc'}:
        exp_name = f"exp_{v}-ConUS-20210323-20231231_{v}-ConUS-20240101-20240531"
        # exp_suffix = "_32bs_smoothed"
        exp_suffix = "_32bs_juno"
        # exp_suffix = "_32bs_era5"
        exp_path = Path(exp_root_folder).joinpath(exp_name+exp_suffix+"/experiment.cfg")
        print(f"Experiment path: {exp_path}")
        experiment = joblib.load(exp_path)
        # print(experiment)
        pr_v = experiment['test_data']['prediction'].load()
        fc_v = experiment['test_data']['input'].load()
        an_v = XarrayDataset.regrid(fc_v, experiment['test_data']['target'].load())
        pr = xr.merge([pr, pr_v], compat='no_conflicts') if pr else pr_v
        fc = xr.merge([fc, fc_v], compat='no_conflicts') if fc else fc_v
        an = xr.merge([an, an_v], compat='no_conflicts') if an else an_v
    
    m = Metrics(an, [fc, pr], "Analysis", ["Forecast", "Prediction"])
    metrics = m.compute_spatial_metrics(fss_threshold=1)
    # joblib.dump(metrics, f"/work/cmcc/jd19424/test-ML/dataML/metrics_ConUS-20210323-20231231_ConUS-20240101-20240531_{exp_suffix}.mtr")
    m.print_aggregated_metrics(metrics)

    # ps_fc = PowerSpectrum(fc.isel(valid_time=0), dx_deg=0.1, dy_deg=0.1)
    # ps_pr = PowerSpectrum(pr, dx_deg=0.1, dy_deg=0.1)
    # ps_fc.plot_all(savepath="/work/cmcc/jd19424/test-ML/earthML/ps_fc0.png", show=False)
    # ps_pr.plot_all(savepath="/work/cmcc/jd19424/test-ML/earthML/ps_pr.png", show=False)
    
