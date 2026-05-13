from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap


PiBRdY = LinearSegmentedColormap.from_list(
    "PiBRdY",
    (
    # Edit this gradient at https://eltos.github.io/gradient/#0:F84CFF-16.7:5A00B3-33.3:0032F1-50:FFFFFF-66.7:C7030D-83.3:ECAE04-100:F6EC01
        (0.000, (0.973, 0.298, 1.000)),
        (0.167, (0.353, 0.000, 0.702)),
        (0.333, (0.000, 0.196, 0.945)),
        (0.500, (1.000, 1.000, 1.000)),
        (0.667, (0.780, 0.012, 0.051)),
        (0.833, (0.925, 0.682, 0.016)),
        (1.000, (0.965, 0.925, 0.004)),
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
