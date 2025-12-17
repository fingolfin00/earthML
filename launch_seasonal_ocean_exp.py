# from earthml.manager import Launcher
# from earthml.logging import Logger
from earthml.utils import Dask
from earthml.experiment import ExperimentMLFC
from earthml.dataclasses import Region, Leadtime, Variable, TimeRange, DataSelection, DataSource, ExperimentDataset, ExperimentConfig
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from rich import print
import warnings, logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("distributed.scheduler").setLevel(logging.ERROR)

if __name__ == "__main__":
    dask_earthml = Dask(n_workers=None)
    client, cluster = dask_earthml.client, dask_earthml.cluster
    print("Dask dashboard:", client.dashboard_link)

    leadtime_days = 15

    mld00_1 = Variable(name='mixed_layer_depth_0_01', unit='m')
    sss_oras5_an = Variable(longname='sea_surface_salinity', name='sosaline')
    sos_juno_fc = Variable(name='sos', levm=0, leadtime=Leadtime('leadtime', 'days', leadtime_days))
    sos_juno_an = Variable(name='sss_m', levm=0)
    t14d_oras5_an = Variable(longname='depth_of_14_c_isotherm', name='so14chgt', unit='m')
    t14d_juno_fc = Variable(name='t14d', leadtime=Leadtime('leadtime', 'days', leadtime_days))
    ssh_oras5_an = Variable(name= 'ssh', longname='sea_surface_heigth_above_geoid', unit='m')
    ssh_juno_fc = Variable(name='ssh', levm=0, leadtime=Leadtime('leadtime', 'days', leadtime_days))

    north_atl = Region(name="NorthAtlantic", lon=(-80, -20), lat=(60, 20))
    conus = Region(name="ConUS", lon=(-130, -90), lat=(45, 30))
    europe = Region(name="Europe", lon=(-10, 36), lat=(55, 35))
    italy = Region(name="ItalianPeninsula", lon=(5, 23.5), lat=(49, 25.5))
    pacific = Region(name="CentralPacific", lon=(-200, -120), lat=(30, -30))

    # Full experiment
    train_period = TimeRange(start=datetime(1995, 1, 1), end=datetime(2020, 12, 31), freq='MS', shifted=dict(days=leadtime_days))
    train_period_until2014 = TimeRange(start=datetime(1995, 1, 1), end=datetime(2014, 12, 31), freq='MS', shifted=dict(days=leadtime_days))
    train_period_after2014 = TimeRange(start=datetime(2015, 1, 1), end=datetime(2020, 12, 31), freq='MS', shifted=dict(days=leadtime_days))
    test_period = TimeRange(start=datetime(2021, 1, 1), end=datetime(2024, 12, 31), freq='MS', shifted=dict(days=leadtime_days))

    # Fast test experiment
    # train_period = TimeRange(start=datetime(2014, 1, 1), end=datetime(2015, 12, 31), freq='MS', shifted=dict(days=leadtime_days))
    # train_period_until2014 = TimeRange(start=datetime(2014, 1, 1), end=datetime(2014, 12, 31), freq='MS', shifted=dict(days=leadtime_days))
    # train_period_after2014 = TimeRange(start=datetime(2015, 1, 1), end=datetime(2015, 12, 31), freq='MS', shifted=dict(days=leadtime_days))
    # test_period = TimeRange(start=datetime(2024, 1, 1), end=datetime(2024, 12, 31), freq='MS', shifted=dict(days=leadtime_days))

    # var = [t2m, msl, u10, v10, d2m, tcc]
    var_fc = sos_juno_fc
    # var_fc = sss_cds_hc
    # var_an = sos_juno_an
    var_an = sss_oras5_an
    region = pacific
    datasel_train_fc = DataSelection(variable=var_fc, region=region, period=train_period)
    datasel_train_an_until2014 = DataSelection(variable=var_an, region=region, period=train_period_until2014)
    datasel_train_an_after2014 = DataSelection(variable=var_an, region=region, period=train_period_after2014)
    datasel_test_fc = DataSelection(variable=var_fc, region=region, period=test_period)
    datasel_test_an = DataSelection(variable=var_an, region=region, period=test_period)

    realization = "*" # ensemble member
    source_params_junolocal_ocean_input = {
        "root_path": "/work/cmcc/cp1/CMCC-CM/archive/C3S/",
        # "engine": "netcdf4",
        "engine": "h5netcdf",
        "file_path_date_format": "%Y%m",
        "file_header": "cmcc_CMCC-CM3-v20231101_hindcast_S",
        "file_suffix": f"*ocean_mon_ocean2d_{var_fc.name}_r{realization}i00p00.nc",
        "file_date_format": "%Y%m%d",
        "both_data_and_previous_date_in_file": False,
        # "realizations": 7,
        "realizations": "all",
        "lead_time": relativedelta(days=leadtime_days),
        "regrid_resolution": 0.25,
        # "minus_timedelta": timedelta(hours=1),
        # "plus_timedelta": timedelta(hours=1),
        "concat_dim": "time",
    }

    realization = '*'
    source_params_junolocal_ocean_target = {
        # "root_path": "/work/cmcc/cp1/CMCC-CM/archive/C3S/",
        "root_path": "/data/cmcc/cp1/archive/IC/NEMO_CPS1",
        "engine": "netcdf4",
        "file_path_date_format": "%m",
        "file_header": "CPS1.nemo.r.",
        "file_suffix": f"-00000.{realization}.nc",
        "file_date_format": "%Y-%m-%d",
        "both_data_and_previous_date_in_file": False,
        "lead_time": timedelta(days=0),
        "regrid_resolution": 0.25,
        # "minus_timedelta": timedelta(hours=1),
        # "plus_timedelta": timedelta(hours=1)
        "concat_dim": "time",
    }

    source_params_cds_ocean_input = {
        "provider": "cds",
        "dataset": "seasonal-monthly-ocean",
        "regrid_resolution": 0.25,
        "split_request": True,
        "split_month": 2,
        # "split_month_jump": ['03'],
        "request_type": "seasonal",
        "request_extra_args": dict(
            forecast_type="hindcast",
            originating_centre="meteo_france",
            system="9",
            # grid=[.25, .25],
            # month=["01", "02"]
            # format="netcdf"
        ),
        "xarray_args": dict(
            # engine="h5netcdf",
            decode_timedelta=True, # maybe dropped in the future
            data_vars='all',
            # combine="by_coords",
            # preprocess=ignore_leadtime,
            coords='minimal',
            # chunks={"valid_time": 1},
            # combine_attrs='override',
            # join='override',
            compat='override',
            concat_dim='leadtime',
            combine='nested'
        ),
        "xarray_concat_dim": "leadtime",
        "xarray_concat_extra_args": dict(
            # combine_attrs='no_conflicts',
            # join='override',
            coords='minimal',
            compat='override',
        )
    }
    source_params_cds_ocean_target_consolidated = {
        "provider": "cds",
        "dataset": "reanalysis-oras5",
        "split_request": True,
        "select_area_after_request": True,
        "regrid_resolution": 0.25,
        "request_type": "seasonal",
        "request_extra_args": dict(
            product_type="consolidated",
            vertical_resolution="single_level",
            # grid=[1.0, 1.0],
            # format="netcdf"
        ),
        "xarray_args": dict(
            decode_timedelta=True,
            data_vars='all',
            combine="by_coords",
            # preprocess=ignore_leadtime,
            coords='minimal',
            # chunks={"valid_time": 1},
            # combine_attrs='override',
            # join='override',
            compat='override',
            # concat_dim='leadtime',
            # combine='nested'
        ),
        "xarray_concat_dim": None,
        "xarray_concat_extra_args": dict(
            # combine_attrs='no_conflicts',
            # join='override',
            coords='minimal',
            compat='override',
        )
    }
    from copy import deepcopy
    source_params_cds_ocean_target_operational = deepcopy(source_params_cds_ocean_target_consolidated)
    source_params_cds_ocean_target_operational["request_extra_args"]["product_type"] = "operational"

    dataset_train_input = ExperimentDataset(
        role='input',
        datasource=DataSource(source="juno-local", data_selection=datasel_train_fc),
        source_params=source_params_junolocal_ocean_input,
        # datasource=DataSource(source="earthkit", data_selection=datasel_train_fc),
        # source_params=source_params_cds_ocean_input,
        save=True,
    )
    dataset_train_target = ExperimentDataset(
        role='target',
        datasource=[
            DataSource(source="earthkit", data_selection=datasel_train_an_until2014),
            DataSource(source="earthkit", data_selection=datasel_train_an_after2014),
        ],
        source_params=[
            source_params_cds_ocean_target_consolidated,
            source_params_cds_ocean_target_operational
        ],
        save=True,
        # datasource=DataSource(source="juno-local", data_selection=datasel_train_an),
        # source_params=source_params_junolocal_ocean_target,
    )
    dataset_test_input = ExperimentDataset(
        role='input',
        datasource=DataSource(source="juno-local", data_selection=datasel_test_fc),
        source_params=source_params_junolocal_ocean_input,
        save=True,
        # datasource=DataSource(source="earthkit", data_selection=datasel_test_fc),
        # source_params=source_params_cds_ocean_input,
    )
    dataset_test_target = ExperimentDataset(
        role='target',
        datasource=DataSource(source="earthkit", data_selection=datasel_test_an),
        source_params=source_params_cds_ocean_target_operational,
        save=True,
        # datasource=DataSource(source="juno-local", data_selection=datasel_test_an),
        # source_params=source_params_junolocal_ocean_target,
    )
    exp_root_folder = "/work/cmcc/jd19424/test-ML/experiments_earthML/"
    # exp_root_folder = "/data/cmcc/jd19424/ML/experiments_earthML/"
    # Use forecast for experiment naming
    exp_train_var =  ''.join([var.name for var in datasel_train_fc.variable] if isinstance(datasel_train_fc.variable, list) else [datasel_train_fc.variable.name])
    exp_test_var =  ''.join([var.name for var in datasel_test_fc.variable] if isinstance(datasel_test_fc.variable, list) else [datasel_test_fc.variable.name])
    exp_name = f"exp_{exp_train_var}-{datasel_train_fc.region.name}-{datasel_train_fc.period.start.strftime('%Y%m%d')}-{datasel_train_fc.period.end.strftime('%Y%m%d')}" \
               f"_{exp_test_var}-{datasel_test_fc.region.name}-{datasel_test_fc.period.start.strftime('%Y%m%d')}-{datasel_test_fc.period.end.strftime('%Y%m%d')}"
    # exp_suffix = "_32bs_juno_heteroloss"
    exp_suffix = "_32bs_mse_allrel_ocean_cmcc_oras5"
    exp_path = Path(exp_root_folder).joinpath(exp_name+exp_suffix)
    print(f"Experiment path: {exp_path}")

    def preprocess_fix_salinity_analysis (input_ds, target_ds):
        # Custom fix for wrong salinity units in analysis
        for (name_target, da_target), (name_input, da_input) in zip(target_ds.data_vars.items(), input_ds.data_vars.items()):
            if name_target == "sss_m" and name_input == "sos":
                mean_target = float(da_target.mean().values)
                mean_input = float(da_input.mean().values)
                if mean_target > 1.8 * mean_input:
                    print(f"Target {name_target} ~ input {mean_target / mean_input:.1f}x {name_input}. Assume target is 2x input")
                    target_ds[name_target] = target_ds[name_target] / 2.0
        return input_ds, target_ds

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
        lead_time="15d",
        train=[dataset_train_input, dataset_train_target],
        test=[dataset_test_input, dataset_test_target],
        # torch_preprocess_fn=preprocess_fix_salinity_analysis,
    )

    experiment = ExperimentMLFC(experiment_cfg)
    experiment.train()
    experiment.test()
