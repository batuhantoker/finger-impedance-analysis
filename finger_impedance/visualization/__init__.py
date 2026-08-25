"""Visualization modules: interactive and publication-quality plots."""

from importlib import import_module
from importlib.util import find_spec

__all__ = [
    "PlotData",
    "load_plot_data",
    "plot_activation_metrics",
    "save_figure",
]

_IMAGE_EXPORTS = [
    "auto_canny",
    "image_features",
    "intensity_max",
    "local_maximum_pos",
    "mean_activation",
]
if find_spec("cv2") is not None and find_spec("skimage") is not None:
    __all__ += _IMAGE_EXPORTS

_EXPORT_MODULES = {
    "PlotData": "finger_impedance.visualization.figure_plots",
    "load_plot_data": "finger_impedance.visualization.figure_plots",
    "plot_activation_metrics": "finger_impedance.visualization.figure_plots",
    "save_figure": "finger_impedance.visualization.figure_plots",
    "auto_canny": "finger_impedance.visualization.interactive_plot",
    "image_features": "finger_impedance.visualization.interactive_plot",
    "intensity_max": "finger_impedance.visualization.interactive_plot",
    "local_maximum_pos": "finger_impedance.visualization.interactive_plot",
    "mean_activation": "finger_impedance.visualization.interactive_plot",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        value = getattr(import_module(module_name), name)
    except ModuleNotFoundError as error:
        if module_name.endswith("interactive_plot") and error.name in {"cv2", "skimage"}:
            raise ModuleNotFoundError(
                "image visualization requires 'finger-impedance-analysis[image]'"
            ) from error
        raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
