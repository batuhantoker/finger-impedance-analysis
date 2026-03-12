"""Finger Impedance Analysis Framework.

Computational framework for analyzing human finger impedance modulation,
stiffness estimation, and force capabilities using HD-sEMG and force sensors.
Developed as part of an MS thesis in Mechatronics Engineering at Sabanci University.
"""

__version__ = "0.1.0"

from finger_impedance.core.functions import (
    butter_bandpass,
    butter_lowpass,
    butter_lowpass_filter,
    class_map,
    data_preprocess,
    evaluate_regression_metrics,
    feature_extraction,
    force_mean,
    force_stiffness,
    moving_average,
    r_square,
    rmse,
    rolling_rms,
    running_mean,
    vaf,
    zero_lag_filter,
)
from finger_impedance.core.tfestimate import tfest

__all__ = [
    "__version__",
    "tfest",
    "butter_lowpass",
    "butter_lowpass_filter",
    "butter_bandpass",
    "zero_lag_filter",
    "data_preprocess",
    "rolling_rms",
    "feature_extraction",
    "force_stiffness",
    "force_mean",
    "class_map",
    "evaluate_regression_metrics",
    "moving_average",
    "running_mean",
    "rmse",
    "r_square",
    "vaf",
]
