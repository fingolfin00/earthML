from types import SimpleNamespace
from ...base.dataclasses import Variable, Region, Leadtime

def make_var(
    leadtime_input: Leadtime | None = None,
    leadtime_target: Leadtime | None = None,
):
    return SimpleNamespace(
        # Atmo
        t2m_juno=Variable(name="t2m", unit="K"),
        msl_juno=Variable(name="msl", unit="Pa"),
        u10_juno=Variable(name="u10", unit="m/s"),
        v10_juno=Variable(name="v10", unit="m/s"),
        d2m_juno=Variable(name="d2m", unit="K"),
        tcc_juno=Variable(name="tcc", unit="[0-1]"),
        gh850_juno=Variable(name="gh", unit="gpm", levhpa=850),

        t2m_era5=Variable(name="2t", unit="K"),

        # Ocean
        mld00_1=Variable(name="mixed_layer_depth_0_01", unit="m"),

        sss_cds_fc=Variable(longname="sea_surface_salinity", name="sos", leadtime=leadtime_input),
        sss_juno_fc=Variable(name="sos", levm=0, leadtime=leadtime_input), # leadtime in var only if multimple leadtime in same file
        sss_oras5_an=Variable(longname="sea_surface_salinity", name="sosaline"),
        sss_juno_an=Variable(name="sos", levm=0, leadtime=leadtime_target), # analysis leadtime in dataset is 15 days ??
        # sss_juno_an=Variable(name="sss_m", levm=0),

        sst_cds_fc=Variable(longname="sea_surface_temperature", name="sst", leadtime=leadtime_input),
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
    )

def make_region ():
    return SimpleNamespace(
        conus=Region(name="ConUS", lon=(-130, -90), lat=(45, 30)),
        europe=Region(name="Europe", lon=(-10, 36), lat=(55, 35)),
        italy=Region(name="ItalianPeninsula", lon=(5, 23.5), lat=(49, 25.5)),
        pacific=Region(name="CentralPacific", lon=(-200, -120), lat=(30, -30)),
        natlantic=Region(name="NorthAtlantic", lon=(-80, 20), lat=(60, 0)),
        satlantic=Region(name="SouthAtlantic", lon=(-80, 20), lat=(0, -60)),
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
