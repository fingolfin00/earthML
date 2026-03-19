from types import SimpleNamespace
from .dataclasses import Variable, Region, Leadtime

def make_var (
        leadtime_fc_var: str = "leadtime",
        leadtime_an_var: str = "leadtime",
        leadtime_fc: int | None = None,
        leadtime_an: int | None = None,
        leadtime_unit: str | None = None
):
    lt_fc = None if leadtime_fc is None and leadtime_unit is None else Leadtime(leadtime_fc_var, leadtime_unit, leadtime_fc)
    lt_an = None if leadtime_an is None and leadtime_unit is None else Leadtime(leadtime_an_var, leadtime_unit, leadtime_an)

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

        sss_cds_fc=Variable(longname="sea_surface_salinity", name="sos", leadtime=lt_fc),
        sss_juno_fc=Variable(name="sos", levm=0, leadtime=lt_fc), # leadtime in var only if multimple leadtime in same file
        sss_oras5_an=Variable(longname="sea_surface_salinity", name="sosaline"),
        sss_juno_an=Variable(name="sos", levm=0, leadtime=lt_an), # analysis leadtime in dataset is 15 days ??
        # sss_juno_an=Variable(name="sss_m", levm=0),

        sst_cds_fc=Variable(longname="sea_surface_temperature", name="sst", leadtime=lt_fc),
        sst_juno_fc=Variable(name="tso", leadtime=lt_fc), # SST, surface variable
        sst_oras5_an=Variable(longname="sea_surface_temperature", name="sosstsst", unit="K"),
        sst_juno_an=Variable(name="tso", leadtime=lt_an),

        t14d_cds_fc=Variable(longname="depth_of_14_c_isotherm", name="t14d", leadtime=lt_fc),
        t14d_juno_fc=Variable(name="t14d", levm=0, leadtime=lt_fc),
        t14d_oras5_an=Variable(longname="depth_of_14_c_isotherm", name="so14chgt", unit="m"),
        t14d_juno_an=Variable(name="t14d", levm=0, leadtime=lt_an),

        t17d_cds_fc=Variable(longname="depth_of_17_c_isotherm", name="t17d", leadtime=lt_fc),
        t17d_juno_fc=Variable(name="t17d", levm=0, leadtime=lt_fc),
        t17d_oras5_an=Variable(longname="depth_of_17_c_isotherm", name="so17chgt", unit="m"),
        t17d_juno_an=Variable(name="t17d", levm=0, leadtime=lt_an),

        t20d_cds_fc=Variable(longname="depth_of_20_c_isotherm", name="t20d", leadtime=lt_fc),
        t20d_juno_fc=Variable(name="t20d", levm=0, leadtime=lt_fc),
        t20d_oras5_an=Variable(longname="depth_of_20_c_isotherm", name="so20chgt", unit="m"),
        t20d_juno_an=Variable(name="t20d", levm=0, leadtime=lt_an),

        ssh_cds_fc=Variable(longname="sea_surface_height_above_geoid", name="zos", unit="m", leadtime=lt_fc),
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
        *,
        leadtime_fc_var: str = "leadtime",
        leadtime_an_var: str = "leadtime",
        leadtime_fc: int | None = None,
        leadtime_an: int | None = None,
        leadtime_unit: str | None = None
):
    return SimpleNamespace(
        var=make_var(leadtime_fc_var=leadtime_fc_var, leadtime_an_var=leadtime_an_var, leadtime_fc=leadtime_fc, leadtime_an=leadtime_an, leadtime_unit=leadtime_unit),
        region=make_region(),
    )
