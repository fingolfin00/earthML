from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap


PiBRdY = LinearSegmentedColormap.from_list(
    "PiBRdY",
    (
        # Edit this gradient at https://eltos.github.io/gradient/#F84CFF-0025B3-FFFFFF-C7030D-F6C401
        (0.000, (0.973, 0.298, 1.000)),
        (0.250, (0.000, 0.145, 0.702)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.750, (0.780, 0.012, 0.051)),
        (1.000, (0.965, 0.769, 0.004)),
    ),
)


def register_colormaps() -> None:
    try:
        colormaps.register(PiBRdY)
    except ValueError:
        # Safe on repeated imports if the colormap is already registered.
        pass


register_colormaps()

__all__ = ["PiBRdY", "register_colormaps"]
