from types import SimpleNamespace
from .dataclasses import Variable, Region, Leadtime

def make_var (
        leadtime: int | None = None,
        leadtime_unit: str | None = None
):
    lt = None if leadtime is None and leadtime_unit is None else Leadtime("leadtime", leadtime_unit, leadtime)

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

        sss_juno_fc=Variable(name="sos", levm=0, leadtime=lt), # leadtime in var only if multimple leadtime in same file
        sss_oras5_an=Variable(longname="sea_surface_salinity", name="sosaline"),
        sss_juno_an=Variable(name="sss_m", levm=0),

        t14d_juno_fc=Variable(name="t14d", leadtime=lt),
        t14d_oras5_an=Variable(longname="depth_of_14_c_isotherm", name="so14chgt", unit="m"),

        t17d_juno_fc=Variable(name="t17d", leadtime=lt),
        t17d_oras5_an=Variable(longname="depth_of_17_c_isotherm", name="so17chgt", unit="m"),

        t20d_juno_fc=Variable(name="t20d", leadtime=lt),
        t20d_oras5_an=Variable(longname="depth_of_20_c_isotherm", name="so20chgt", unit="m"),

        ssh_cds_fc=Variable(name="ssh", longname="sea_surface_height_above_geoid", unit="m", leadtime=lt),
        ssh_oras5_an=Variable(name="sossheig", longname="sea_surface_height", unit="m"),
    )

def make_region ():
    return SimpleNamespace(
    north_atl=Region(name="NorthAtlantic", lon=(-80, -20), lat=(60, 20)),
    conus=Region(name="ConUS", lon=(-130, -90), lat=(45, 30)),
    europe=Region(name="Europe", lon=(-10, 36), lat=(55, 35)),
    italy=Region(name="ItalianPeninsula", lon=(5, 23.5), lat=(49, 25.5)),
    pacific=Region(name="CentralPacific", lon=(-200, -120), lat=(30, -30)),
)

def make_catalog (
        *,
        leadtime: int | None = None,
        leadtime_unit: str | None = None
):
    return SimpleNamespace(
        var=make_var(leadtime=leadtime, leadtime_unit=leadtime_unit),
        region=make_region(),
    )
