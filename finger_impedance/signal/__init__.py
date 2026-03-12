"""Signal processing: HD-sEMG feature extraction and activation maps."""

from finger_impedance.signal.signal_features import activation_map
from finger_impedance.signal.signal_features import class_map as signal_class_map

__all__ = [
    "activation_map",
    "signal_class_map",
]
