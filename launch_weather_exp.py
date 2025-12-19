from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from rich import print

import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)

# from earthml.manager import Launcher
# from earthml.logging import Logger
from earthml.utils import Dask
from earthml.experiment import ExperimentMLFC
from earthml.dataclasses import TimeRange, DataSelection, DataSource, ExperimentDataset, ExperimentConfig
import earthml.catalog as catalog

if __name__ == "__main__":

    # launcher = Launcher("earthml.toml")

    # --------------------+
    # Experiment settings |
    # --------------------+

    exp_root_folder = "/work/cmcc/jd19424/test-ML/experiments_earthML/"
    # exp_root_folder = "/data/cmcc/jd19424/ML/experiments_earthML/"
    # exp_suffix = "_32bs_heteroloss_weather_juno"
    exp_suffix = "_32bs_mse_weather_ecmwf_era5"
    leadtime_hours = 10*24
    # input_src, target_src = 'juno-local', 'juno-local'
    input_src, target_src = 'earthkit', 'earthkit'

    # Full experiment
    # train_period = TimeRange(start=datetime(2021, 3, 23), end=datetime(2023, 12, 31), freq='12h')
    # test_period = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 5, 31), freq='12h')

    # Short test experiment
    train_period = TimeRange(start=datetime(2020, 1, 1), end=datetime(2020, 1, 3), freq='6h')
    test_period = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 1, 2), freq='6h')

    cat = catalog.make_catalog(leadtime=leadtime_hours, leadtime_unit='hours')
    # var = [t2m, msl, u10, v10, d2m, tcc]
    var_fc = cat.var.t2m_era5
    var_an = cat.var.t2m_era5
    region = cat.region.conus

    # ----------------------

    datasel_train_fc = DataSelection(variable=var_fc, region=region, period=train_period)
    datasel_train_an = DataSelection(variable=var_an, region=region, period=train_period)
    datasel_test_fc = DataSelection(variable=var_fc, region=region, period=test_period)
    datasel_test_an = DataSelection(variable=var_an, region=region, period=test_period)

    datasource_train_fc = DataSource(source=input_src, data_selection=datasel_train_fc)
    datasource_train_an = DataSource(source=target_src, data_selection=datasel_train_an)
    datasource_test_fc = DataSource(source=input_src, data_selection=datasel_test_fc)
    datasource_test_an = DataSource(source=target_src, data_selection=datasel_test_an)

    source_params_fc, source_params_an = {}, {}
    source_params_fc['juno-local'] = {
        "root_path": "/data/inputs/METOCEAN/rolling/model/atmos/ECMWF/IFS_010/1.0forecast/1h/grib/",
        "engine": "cfgrib",
        "file_path_date_format": "%Y%m%d",
        "file_header": "JLS",
        "file_suffix": "*",
        "file_date_format": "%m%d%H%M",
        "both_data_and_previous_date_in_file": True,
        "lead_time": relativedelta(hours=72),
        "minus_timedelta": relativedelta(hours=1),
        "plus_timedelta": relativedelta(hours=1),
        "concat_dim": "valid_time",
    }

    source_params_an['juno-local'] = {
        "root_path": "/data/inputs/METOCEAN/historical/model/atmos/ECMWF/IFS_010/analysis/6h/grib/",
        "engine": "cfgrib",
        "file_path_date_format": "%Y/%m",
        "file_header": "JLD",
        "file_suffix": "*",
        "file_date_format": "%m%d%H%M",
        "both_data_and_previous_date_in_file": True,
        "lead_time": relativedelta(hours=0),
        "minus_timedelta": relativedelta(hours=1),
        "plus_timedelta": relativedelta(hours=1),
        "concat_dim": "valid_time",
    }

    source_params_fc['earthkit'] = {
        # "provider": "mars",
        "provider": "ecmwf-open-data",
        "split_request": True,
        "select_area_after_request": True,
        "request_extra_args": dict(request=dict(
            param=var_fc.name,
            levtype="sfc",
            # area=[50, -10, 40, 10],
            # grid=[2, 2],
            # date="2023-05-10"
        )),
        "to_xarray_args": dict(
            time_dim_mode="valid_time",
            chunks={"valid_time": 1},
            add_earthkit_attrs=False,
            # backend_kwargs={
            #     "allow_holes": True,
            #     "squeeze": False,
            #     "drop_dims": ["number"],  # Drop the number dimension
            # }
        )
    }

    source_params_an['earthkit'] = {
        "provider": "cds",
        "dataset": "reanalysis-era5-single-levels",
        "split_request": True,
        "request_extra_args": dict(
            product_type="reanalysis",
            # grid=[.1, .1],
            # format="netcdf"
        ),
        "to_xarray_args": dict(
            time_dim_mode="valid_time",
            chunks={"valid_time": 1},
            add_earthkit_attrs=False,
            # backend_kwargs={
            #     "allow_holes": True,
            #     "squeeze": False,
            #     "drop_dims": ["number"],  # Drop the number dimension
            # }
        )
    }

    dataset_train_input = ExperimentDataset(
        role='input',
        save=True,
        datasource=datasource_train_fc,
        source_params=source_params_fc[input_src]
    )
    dataset_train_target = ExperimentDataset(
        role='target',
        save=True,
        datasource=datasource_train_an,
        source_params=source_params_an[target_src],
    )
    dataset_test_input = ExperimentDataset(
        role='input',
        save=True,
        datasource=datasource_test_fc,
        source_params=source_params_fc[input_src]
    )
    dataset_test_target = ExperimentDataset(
        role='target',
        save=True,
        datasource=datasource_test_an,
        source_params=source_params_an[target_src]
    )

    # logger = Logger(
    #         Path("./").joinpath("combo.log"),
    #         log_level="info"
    # ).logger

    # Use forecast for experiment naming
    exp_train_var =  ''.join([var.name for var in datasel_train_fc.variable] if isinstance(datasel_train_fc.variable, list) else [datasel_train_fc.variable.name])
    exp_test_var =  ''.join([var.name for var in datasel_test_fc.variable] if isinstance(datasel_test_fc.variable, list) else [datasel_test_fc.variable.name])
    exp_name = f"exp_{exp_train_var}-{datasel_train_fc.region.name}-{datasel_train_fc.period.start.strftime('%Y%m%d')}-{datasel_train_fc.period.end.strftime('%Y%m%d')}" \
               f"_{exp_test_var}-{datasel_test_fc.region.name}-{datasel_test_fc.period.start.strftime('%Y%m%d')}-{datasel_test_fc.period.end.strftime('%Y%m%d')}"
    exp_path = Path(exp_root_folder).joinpath(exp_name+exp_suffix)
    print(f"Experiment path: {exp_path}")

    experiment_cfg = ExperimentConfig(
        name=exp_name,
        work_path=exp_path,
        seed=42,
        net="SmaAt_UNet",
        extra_net_args=dict(n_channels=len(var_fc), n_classes=len(var_fc)) if isinstance(var_fc, list) else dict(n_channels=1, n_classes=1),
        # Hyperparameters
        learning_rate=1e-3,
        batch_size=32,
        epochs=50,
        # loss="earthml.lightning.HeteroBiasCorrectionLoss", # MSELoss
        # loss_params=dict(
        #     {'net': dict(use_first_input=True),
        #      'loss': dict(
        #         lambda_identity=0.1,
        #         bias_scale=0.5,
        #         eps = 1e-6)
        #     },
        # ),
        loss="MSELoss",
        loss_params={'net': {}, 'loss': {}},
        norm_strategy="BatchNorm2d", # not used by SmaAt_UNet (batchnorm hardcoded)
        supervised=True,
        train_percent=0.9,
        earlystopping_patience=30,
        accumulate_grad_batches=2,
        # Dataset Parameters
        train=[dataset_train_input, dataset_train_target],
        test=[dataset_test_input, dataset_test_target],
    )

    dask_earthml = Dask(n_workers=None)
    client, cluster = dask_earthml.client, dask_earthml.cluster
    print("Dask dashboard:", client.dashboard_link)

    experiment = ExperimentMLFC(experiment_cfg)
    experiment.train()
    experiment.test()
