"""Classification modules: EMG-based gesture recognition pipelines."""

from finger_impedance.classification.emg_classifier import (
    classification_report_with_accuracy_score,
)
from finger_impedance.classification.image_classifier import (
    data_preprocess,
)

__all__ = [
    "classification_report_with_accuracy_score",
    "data_preprocess",
]
