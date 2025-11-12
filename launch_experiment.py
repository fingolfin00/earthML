# from earthml.manager import Launcher
# from earthml.logging import Logger
from earthml.utils import Dask
from earthml.experiment import ExperimentMLFC
from earthml.dataclasses import Region, Variable, TimeRange, DataSelection, DataSource, ExperimentConfig
from pathlib import Path
import joblib, torch
from datetime import datetime, timedelta
import dask
from zarr.codecs import BloscCodec
# from zarr.storage import ZipStore

if __name__ == "__main__":
    dask_earthml = Dask()
    client, cluster = dask_earthml.client, dask_earthml.cluster
    print("Dask dashboard:", client.dashboard_link)
    # launcher = Launcher("earthml.toml")
    # logger = Logger(
    #         Path("./").joinpath("combo.log"),
    #         log_level="info"
    # ).logger
    t2m = Variable(name="t2m", unit="K")
    msl = Variable(name="msl", unit="Pa")
    u10 = Variable(name="u10", unit="m/s")
    v10 = Variable(name="v10", unit="m/s")
    d2m = Variable(name="d2m", unit="K")
    tcc = Variable(name="tcc", unit="[0-1]")
    gh = Variable(name="gh", unit="gpm", levhpa=850) # 3d variable

    north_atl = Region(name="NorthAtlantic", lon=(-80, -20), lat=(60, 20))
    conus = Region(name="ConUS", lon=(-130, -90), lat=(45, 30))
    europe = Region(name="Europe", lon=(0, 36), lat=(50, 40))
    italy = Region(name="ItalianPeninsula", lon=(5, 23.5), lat=(49, 25.5))
    pacific = Region(name="CentralPacific", lon=(-100, -120), lat=(40, 0))

    train_period = TimeRange(start=datetime(2021, 3, 23), end=datetime(2023, 12, 31), freq='12h')
    # train_period = TimeRange(start=datetime(2021, 3, 23), end=datetime(2021, 4, 8), freq='12h')
    test_period = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 5, 31), freq='12h')
    # test_period = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 1, 2), freq='12h')

    # var = [t2m, msl, u10]
    var = t2m
    datasel_train = DataSelection(variable=var, region=conus, period=train_period)
    datasel_test = DataSelection(variable=var, region=conus, period=test_period)
    train_src = DataSource(source="juno-local", data_selection=datasel_train)
    test_src = DataSource(source="juno-local", data_selection=datasel_test)

    exp_root_folder = "/work/cmcc/jd19424/test-ML/experiments_earthML/"
    exp_train_var =  ''.join([var.name for var in datasel_train.variable] if isinstance(datasel_train.variable, list) else [datasel_train.variable.name])
    exp_test_var =  ''.join([var.name for var in datasel_test.variable] if isinstance(datasel_test.variable, list) else [datasel_test.variable.name])
    exp_name = f"exp_{exp_train_var}-{datasel_train.region.name}-{datasel_train.period.start.strftime('%Y%m%d')}-{datasel_train.period.end.strftime('%Y%m%d')}" \
               f"_{exp_test_var}-{datasel_test.region.name}-{datasel_test.period.start.strftime('%Y%m%d')}-{datasel_test.period.end.strftime('%Y%m%d')}"
    exp_suffix = "_32bs"
    exp_path = Path(exp_root_folder).joinpath(exp_name+exp_suffix)
    print(f"Experiment path: {exp_path}")
    experiment_cfg = ExperimentConfig(
        name=exp_name,
        work_path=exp_path,
        seed=42,
        net="SmaAt_UNet",
        extra_net_args=dict(n_channels=len(var), n_classes=len(var)) if isinstance(var, list) else dict(n_channels=1, n_classes=1),
        # Hyperparameters
        learning_rate=1e-3,
        batch_size=32,
        epochs=50,
        loss="MSELoss",
        norm_strategy="BatchNorm2d", # not used by SmaAt_UNet (batchnorm hardcoded)
        supervised=True,
        train_percent=0.9,
        earlystopping_patience=30,
        accumulate_grad_batches=2,
        # Dataset Parameters
        lead_time="72h",
        train=train_src,
        test=test_src
    )

    experiment = ExperimentMLFC(
        config=experiment_cfg,
        source_input_args={
            "root_path": "/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/1.0forecast/1h/grib/",
            "engine": "cfgrib",
            "file_path_date_format": "%Y%m%d",
            "file_header": "JLS",
            "file_suffx": "*",
            "file_date_format": "%m%d%H%M",
            "lead_time": timedelta(hours=72),
            "minus_timedelta": timedelta(hours=1),
            "plus_timedelta": timedelta(hours=1)
        },
        source_target_args={
            "root_path": "/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010/analysis/6h/grib/",
            "engine": "cfgrib",
            "file_path_date_format": "%Y/%m",
            "file_header": "JLD",
            "file_suffx": "*",
            "file_date_format": "%m%d%H%M",
            "lead_time": timedelta(hours=0),
            "minus_timedelta": timedelta(hours=1),
            "plus_timedelta": timedelta(hours=1)
        }
    )
    # experiment.train()
    experiment.test()
    # torch.save(experiment, exp_path.joinpath("experiment.pt"))
    # exp_path.joinpath("data").mkdir(parents=True, exist_ok=True)
    # store = ZipStore('test_pred.zarr.zip', mode='w')
    store = exp_path.joinpath("data/test_pred.zarr")
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
    encoding_zarr = (
        {v.name: {"compressors": compressor} for v in var}
        if isinstance(var, list)
        else {var.name: {"compressors": compressor}}
    )
    # experiment.test_pred_ds.to_zarr(
    #     store,
    #     mode='w',
    #     consolidated=False,
    #     compute=True,
    #     encoding=encoding_zarr
    # )
    # with dask.config.set(scheduler='single-threaded'):
    experiment.test_pred_ds.to_zarr(store, encoding=encoding_zarr, mode='w', consolidated=False)
    # experiment.test_pred_ds.to_netcdf(exp_path.joinpath("data/test_pred.nc"))
