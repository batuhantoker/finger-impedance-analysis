"""Leakage-safe EMG and image-feature classification helpers."""

from importlib import import_module

__all__ = [
    "evaluate_emg_models",
    "evaluate_image_models",
    "flatten_feature_arrays",
    "load_emg_feature_files",
    "load_image_feature_files",
]

_EXPORT_MODULES = {
    "evaluate_emg_models": ("finger_impedance.classification.emg_classifier", "evaluate_models"),
    "load_emg_feature_files": (
        "finger_impedance.classification.emg_classifier",
        "load_feature_files",
    ),
    "evaluate_image_models": (
        "finger_impedance.classification.image_classifier",
        "evaluate_models",
    ),
    "flatten_feature_arrays": (
        "finger_impedance.classification.image_classifier",
        "flatten_feature_arrays",
    ),
    "load_image_feature_files": (
        "finger_impedance.classification.image_classifier",
        "load_feature_files",
    ),
}


def __getattr__(name: str) -> object:
    target = _EXPORT_MODULES.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
