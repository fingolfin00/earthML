from types import SimpleNamespace
from ...base.dataclasses import Variable, Region, Leadtime

def make_var(
    leadtime_input: Leadtime | None = None,
    leadtime_target: Leadtime | None = None,
):
    return SimpleNamespace(
        # Ocean weather
        sss_gopaf_fc=Variable(longname="sea surface salinity", name="so", levm=0, leadtime=leadtime_input),
        sss_gopaf_an=Variable(longname="sea surface salinity", name="so", levm=0),

        sst_gopaf_fc=Variable(longname="sea_surface_temperature", name="thetao", levm=0, leadtime=leadtime_input),
        sst_gopaf_an=Variable(longname="sea_surface_temperature", name="thetao", levm=0),

        ssh_gopaf_fc=Variable(longname="Sea surface height", name="zos", leadtime=leadtime_input),
        ssh_gopaf_an=Variable(longname="Sea surface height", name="zos"), # only surface variable

        # Atmo weather
        # local forecast
        t2m_juno_fc=Variable(name="t2m", unit="K"),
        msl_juno_fc=Variable(name="msl", unit="Pa"),
        u10_juno_fc=Variable(name="u10", unit="m/s"),
        v10_juno_fc=Variable(name="v10", unit="m/s"),
        d2m_juno_fc=Variable(name="d2m", unit="K"),
        tcc_juno_fc=Variable(name="tcc", unit="[0-1]"),
        gh850_juno_fc=Variable(name="gh", unit="gpm", levhpa=850),
        # local analysis
        t2m_juno_an=Variable(name="t2m", unit="K"),
        msl_juno_an=Variable(name="msl", unit="Pa"),
        u10_juno_an=Variable(name="u10", unit="m/s"),
        v10_juno_an=Variable(name="v10", unit="m/s"),
        d2m_juno_an=Variable(name="d2m", unit="K"),
        tcc_juno_an=Variable(name="tcc", unit="[0-1]"),
        gh850_juno_an=Variable(name="gh", unit="gpm", levhpa=850),

        # ERA5 analysis
        t2m_era5_an=Variable(name="2t", unit="K"),

        # Ocean seasonal
        # mld00_1=Variable(name="mixed_layer_depth_0_01", unit="m"),

        sss_cds_fc=Variable(longname="sea_surface_salinity", name="sos", leadtime=leadtime_input),
        sss_juno_fc=Variable(name="sos", levm=0, leadtime=leadtime_input), # leadtime in var only if multimple leadtime in same file
        sss_oras5_an=Variable(longname="sea_surface_salinity", name="sosaline"),
        sss_juno_an=Variable(name="sos", levm=0, leadtime=leadtime_target), # analysis leadtime in dataset is 15 days ??
        # sss_juno_an=Variable(name="sss_m", levm=0),

        sst_cds_fc=Variable(longname="sea_surface_temperature", name="sst", leadtime=leadtime_input), # in CDS atmo product
        sst_juno_fc=Variable(name="tso", leadtime=leadtime_input), # SST, surface variable
        sst_oras5_an=Variable(longname="sea_surface_temperature", name="sosstsst", unit="K"),
        sst_juno_an=Variable(name="tso", leadtime=leadtime_target),

        t14d_cds_fc=Variable(longname="depth_of_14_c_isotherm", name="t14d", leadtime=leadtime_input),
        t14d_juno_fc=Variable(name="t14d", levm=0, leadtime=leadtime_input),
        t14d_oras5_an=Variable(longname="depth_of_14_c_isotherm", name="so14chgt", unit="m"),
        t14d_juno_an=Variable(name="t14d", levm=0, leadtime=leadtime_target),

        t17d_cds_fc=Variable(longname="depth_of_17_c_isotherm", name="t17d", leadtime=leadtime_input),
        t17d_juno_fc=Variable(name="t17d", levm=0, leadtime=leadtime_input),
        t17d_oras5_an=Variable(longname="depth_of_17_c_isotherm", name="so17chgt", unit="m"),
        t17d_juno_an=Variable(name="t17d", levm=0, leadtime=leadtime_target),

        t20d_cds_fc=Variable(longname="depth_of_20_c_isotherm", name="t20d", leadtime=leadtime_input),
        t20d_juno_fc=Variable(name="t20d", levm=0, leadtime=leadtime_input),
        t20d_oras5_an=Variable(longname="depth_of_20_c_isotherm", name="so20chgt", unit="m"),
        t20d_juno_an=Variable(name="t20d", levm=0, leadtime=leadtime_target),

        ssh_cds_fc=Variable(longname="sea_surface_height_above_geoid", name="zos", unit="m", leadtime=leadtime_input),
        ssh_oras5_an=Variable(longname="sea_surface_height", name="sossheig", unit="m"),

        # Atmo seasonal
        t2m_cds_fc=Variable(longname="2m_temperature", name="t2m", unit="K", leadtime=leadtime_input),
        t2m_era5_seasonal_an=Variable(longname="2m_temperature", name="t2m", unit="K"),

        d2m_cds_fc=Variable(longname="2m_dewpoint_temperature", name="d2m", unit="K", leadtime=leadtime_input),
        d2m_era5_seasonal_an=Variable(longname="2m_dewpoint_temperature", name="d2m", unit="K"),

        msl_cds_fc=Variable(longname="mean_sea_level_pressure", name="msl", unit="Pa", leadtime=leadtime_input),
        msl_era5_seasonal_an=Variable(longname="mean_sea_level_pressure", name="msl", unit="Pa"),


        # TODO not comparable, need conversion
        tp_cds_fc=Variable(longname="total_precipitation", name="tprate", unit="", leadtime=leadtime_input),
        tp_era5_seasonal_an=Variable(longname="total_precipitation", name="tp", unit="m/s"),

        u10_cds_fc=Variable(longname="10m_u_component_of_wind", name="u10", unit="m/s", leadtime=leadtime_input),
        u10_era5_seasonal_an=Variable(longname="10m_u_component_of_wind", name="u10", unit="m/s"),

        v10_cds_fc=Variable(longname="10m_v_component_of_wind", name="v10", unit="m/s", leadtime=leadtime_input),
        v10_era5_seasonal_an=Variable(longname="10m_v_component_of_wind", name="v10", unit="m/s"),

        sst_era5_seasonal_an=Variable(longname="sea_surface_temperature", name="sst", unit="K"),
    )

def make_region ():
    return SimpleNamespace(
        westconus=Region(name="WestConUS", lon=(-130, -90), lat=(45, 30)),
        conus=Region(name="ConUS", lon=(-130, -60), lat=(50, 25)),
        westeurope=Region(name="WestEurope", lon=(-10, 36), lat=(55, 35)),
        europe=Region(name="Europe", lon=(-30, 60), lat=(80, 30)),
        italy=Region(name="ItalianPeninsula", lon=(5, 23.5), lat=(49, 25.5)),
        pacific=Region(name="CentralPacific", lon=(-200, -120), lat=(30, -30)),
        natlantic=Region(name="NorthAtlantic", lon=(-100, 40), lat=(80, 0)),
        atlanticbox=Region(name="AtlanticBox", lon=(-18, 1), lat=(46, 30)),
        satlantic=Region(name="SouthAtlantic", lon=(-70, 30), lat=(0, -80)),
        indian=Region(name="Indian", lon=(20, 140), lat=(30, -60)),
    )

def make_catalog (
    *, # force explicit keyword args
    leadtime_input: Leadtime | None = None,
    leadtime_target: Leadtime | None = None,
):
    return SimpleNamespace(
        var=make_var(
            leadtime_input=leadtime_input,
            leadtime_target=leadtime_target,
        ),
        region=make_region(),
    )
