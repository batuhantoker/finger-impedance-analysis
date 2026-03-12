"""Visualization modules: interactive and publication-quality plots."""

__all__: list = []

try:
    from finger_impedance.visualization.interactive_plot import (
        auto_canny,
        image_features,
        intensity_max,
        local_maximum_pos,
        mean_activation,
    )
    __all__ += [
        "auto_canny",
        "image_features",
        "intensity_max",
        "local_maximum_pos",
        "mean_activation",
    ]
except ImportError:
    pass
