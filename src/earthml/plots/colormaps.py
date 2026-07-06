from matplotlib.colors import ListedColormap


from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


PiBRdY = LinearSegmentedColormap.from_list(
    "PiBRdY",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:F84CFF-16.7:5A00B3-25:4477D9-33.3:8BC8F8-50:FFFFFF-66.7:EB8B90-75:B00C1C-83.3:ECAE04-100:F6EC01
        (0.000, (0.973, 0.298, 1.000)),
        (0.167, (0.353, 0.000, 0.702)),
        (0.250, (0.267, 0.467, 0.851)),
        (0.333, (0.545, 0.784, 0.973)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.667, (0.922, 0.545, 0.565)),
        (0.750, (0.690, 0.047, 0.110)),
        (0.833, (0.925, 0.682, 0.016)),
        (1.000, (0.965, 0.925, 0.004)),
    )
)

WRdY = LinearSegmentedColormap.from_list(
    'WRdY',
    (
        # Edit this gradient at https://eltos.github.io/gradient/#0:FFFFFF-25:EB8B90-49.9:B00C1C-75:ECAE04-100:F6EC01
        (0.000, (1.000, 1.000, 1.000)),
        (0.250, (0.922, 0.545, 0.565)),
        (0.499, (0.690, 0.047, 0.110)),
        (0.750, (0.925, 0.682, 0.016)),
        (1.000, (0.965, 0.925, 0.004))
    )
)


SeqWRdY = ListedColormap(
    [
        "#FFFFFF", "#F8B4B4", "#F08080", "#E85D5D", "#D73027",
        "#C81D25", "#E66101", "#F18F01", "#FDB863", "#F6EC01",
    ],
    name="SeqWRdY",
)

SeqPiBRdY = ListedColormap(
    [
        "#F707D3", "#AF30E1", "#4B08F4", "#3A0EEC", "#3164EF",
        "#546FC0", "#6AABF5", "#7BD0F7", "#ADDAF0",
        "#E0FAF7", "#FCEBEB",
        "#F8B4B4", "#F08080", "#E85D5D", "#D73027", "#C81D25",
        "#E66101", "#F18F01", "#FDB863", "#F6EC01",
    ],
    name="SeqPiBRdY",
)

SeqBPi = ListedColormap(
    [
        "#08306B", "#08519C", "#2171B5", "#4292C6", "#6BAED6",
        "#9ECAE1", "#D6EFFA",
        "#FFFFFF",
        "#F7EB08", "#EDB61F", "#FA8900", "#F32B2B",
        "#972312", "#580000", "#EB09E8",
    ],
    name="SeqBPi",
)

SeqBYRd = ListedColormap(
    [
        # negative side (blue → lightblue)
        "#08306B",
        "#2171B5",
        "#53ACEB",
        "#7CDAFC",
        "#7DF5DB",
        # white
        "#FFFFFF",
        # positive side (yellow → orange → red → purple)
        "#F0FA7D",
        "#EDB61F",
        "#FA8900",
        "#F32B2B",
        "#972312",
    ],
    name="SeqBYRd"
)

def register_colormaps() -> None:
    try:
        colormaps.register(PiBRdY)
        colormaps.register(WRdY)
        colormaps.register(SeqWRdY)
        colormaps.register(SeqPiBRdY)
        colormaps.register(SeqBPi)
        colormaps.register(SeqBYRd)
    except ValueError:
        # Safe on repeated imports if the colormap is already registered.
        pass


register_colormaps()
