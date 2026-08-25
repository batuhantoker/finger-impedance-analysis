"""Finger Impedance Analysis Framework.

Computational framework for analyzing human finger activation and force, plus
stiffness estimation when synchronized displacement measurements are available.
Developed as part of an MS thesis in Mechatronics Engineering at Sabanci University.
"""

__version__ = "0.2.0"

from finger_impedance.core.functions import (
    butter_bandpass,
    butter_lowpass,
    butter_lowpass_filter,
    class_map,
    co_contraction_index,
    data_preprocess,
    estimate_stiffness,
    evaluate_regression_metrics,
    feature_extraction,
    force_mean,
    moving_average,
    r_square,
    rmse,
    rolling_rms,
    running_mean,
    stiffness_proxy,
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
    "estimate_stiffness",
    "co_contraction_index",
    "stiffness_proxy",
    "force_mean",
    "class_map",
    "evaluate_regression_metrics",
    "moving_average",
    "running_mean",
    "rmse",
    "r_square",
    "vaf",
]
