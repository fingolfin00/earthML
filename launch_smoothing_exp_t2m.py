# from earthml.manager import Launcher
# from earthml.logging import Logger
from earthml.utils import Dask
from earthml.experiment import ExperimentMLFC
from earthml.dataclasses import Region, Variable, TimeRange, DataSelection, DataSource, ExperimentDataset, ExperimentConfig
from pathlib import Path
from datetime import datetime, timedelta
from rich import print

if __name__ == "__main__":
    dask_earthml = Dask()
    client, cluster = dask_earthml.client, dask_earthml.cluster
    print("Dask dashboard:", client.dashboard_link)

    t2m = Variable(name="t2m_original", unit="K")
    t2mR25 = Variable(name="t2m_smoothed_R25.0_s0.8", unit="K")
    t2mR50 = Variable(name="t2m_smoothed_R50.0_s1.5", unit="K")
    t2mR100 = Variable(name="t2m_smoothed_R100.0_s3.0", unit="K")
    t2mR200 = Variable(name="t2m_smoothed_R200.0_s6.0", unit="K")
    var_list = [t2m, t2mR25, t2mR50, t2mR100, t2mR200]

    # north_atl = Region(name="NorthAtlantic", lon=(-80, -20), lat=(60, 20))
    conus = Region(name="ConUS", lon=(-130, -90), lat=(45, 30))
    # europe = Region(name="Europe", lon=(0, 36), lat=(50, 40))
    # italy = Region(name="ItalianPeninsula", lon=(5, 23.5), lat=(49, 25.5))
    # pacific = Region(name="CentralPacific", lon=(-100, -120), lat=(40, 0))

    train_period = TimeRange(start=datetime(2021, 3, 23), end=datetime(2023, 12, 31), freq='12h')
    # train_period = TimeRange(start=datetime(2021, 3, 23), end=datetime(2021, 4, 8), freq='12h')
    test_period = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 5, 31), freq='12h')
    # test_period = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 1, 2), freq='12h')

    var = t2mR200
    
    datasel_train = DataSelection(variable=var, region=conus, period=train_period)
    datasel_test = DataSelection(variable=var, region=conus, period=test_period)

    source_params_zarr_train_input = {
        "root_path": "/work/cmcc/jd19424/test-ML/dataML/filtered_data/smoothed_t2m_train_input.zarr",
        'xarray_args': {'drop_variables': [v.name for v in var_list if v is not var]}
    }
    source_params_zarr_train_target = {
        "root_path": "/work/cmcc/jd19424/test-ML/dataML/filtered_data/smoothed_t2m_train_target.zarr",
        'xarray_args': {'drop_variables': [v.name for v in var_list if v is not var]}
    }
    source_params_zarr_test_input = {
        "root_path": "/work/cmcc/jd19424/test-ML/dataML/filtered_data/smoothed_t2m_test_input.zarr",
        'xarray_args': {'drop_variables': [v.name for v in var_list if v is not var]}
    }
    source_params_zarr_test_target = {
        "root_path": "/work/cmcc/jd19424/test-ML/dataML/filtered_data/smoothed_t2m_test_target.zarr",
        'xarray_args': {'drop_variables': [v.name for v in var_list if v is not var]}
    }

    dataset_train_input = ExperimentDataset(
        role='input',
        datasource=DataSource(source="xarray-local", data_selection=datasel_train),
        source_params=source_params_zarr_train_input
    )
    dataset_train_target = ExperimentDataset(
        role='target',
        datasource=DataSource(source="xarray-local", data_selection=datasel_train),
        source_params=source_params_zarr_train_target
    )
    dataset_test_input = ExperimentDataset(
        role='input',
        datasource=DataSource(source="xarray-local", data_selection=datasel_test),
        source_params=source_params_zarr_test_input
    )
    dataset_test_target = ExperimentDataset(
        role='target',
        datasource=DataSource(source="xarray-local", data_selection=datasel_test),
        source_params=source_params_zarr_test_target
    )
    exp_root_folder = "/work/cmcc/jd19424/test-ML/experiments_earthML/"
    # exp_root_folder = "/data/cmcc/jd19424/ML/experiments_earthML/"
    exp_train_var =  ''.join([var.name for var in datasel_train.variable] if isinstance(datasel_train.variable, list) else [datasel_train.variable.name])
    exp_test_var =  ''.join([var.name for var in datasel_test.variable] if isinstance(datasel_test.variable, list) else [datasel_test.variable.name])
    exp_name = f"exp_{exp_train_var}-{datasel_train.region.name}-{datasel_train.period.start.strftime('%Y%m%d')}-{datasel_train.period.end.strftime('%Y%m%d')}" \
               f"_{exp_test_var}-{datasel_test.region.name}-{datasel_test.period.start.strftime('%Y%m%d')}-{datasel_test.period.end.strftime('%Y%m%d')}"
    exp_suffix = "_32bs_smoothed"
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
        train=[dataset_train_input, dataset_train_target],
        test=[dataset_test_input, dataset_test_target]
    )

    experiment = ExperimentMLFC(experiment_cfg)
    experiment.train()
    experiment.test()
