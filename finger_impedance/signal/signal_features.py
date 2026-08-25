"""HD-sEMG activation map feature extraction.

Computes RMS-based activation maps and epoch-wise class labels from
8x8 HD-sEMG grid data. Also provides Mean Shift clustering features.
"""

from typing import Tuple

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

from finger_impedance.core.functions import class_map as core_class_map
from finger_impedance.core.functions import feature_extraction


def activation_map(
    data: np.ndarray, epoch: int, fs: float
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Extract time-domain and frequency-domain features from multi-channel EMG.

    Args:
        data: 2D array of shape (n_samples, n_channels).
        epoch: Number of samples per epoch window.
        fs: Sampling frequency in Hz.

    Returns:
        Tuple of 10 feature arrays (RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM),
        each of shape (n_segments, n_channels).
    """
    return feature_extraction(data, epoch, fs)


def class_map(data: np.ndarray, epoch: int) -> np.ndarray:
    """Return one label per pure epoch and NaN for transition epochs.

    Args:
        data: 1D label/class signal array.
        epoch: Number of samples per epoch window.

    Returns:
        Array of shape (n_segments,) with one class per pure epoch.
    """
    return core_class_map(data, epoch)


def mean_shift_feature(data_emg: np.ndarray) -> np.ndarray:
    """Compute Mean Shift clustering labels for 8x8 EMG activation maps.

    Args:
        data_emg: 2D array of shape (n_samples, 64) representing flattened 8x8 grids.

    Returns:
        Cluster label array of shape (n_samples, 64), dtype object.
    """
    labels = np.zeros((len(data_emg), 8, 8))
    for i in range(len(data_emg)):
        data = data_emg[i, :]
        data = np.reshape(data, (8, 8))
        flat_image = np.reshape(data, [-1, 1])
        if np.ptp(flat_image) == 0:
            continue
        bandwidth2 = estimate_bandwidth(flat_image, quantile=0.1, n_samples=2500)
        ms = MeanShift(bandwidth=bandwidth2 if bandwidth2 > 0 else None)
        ms.fit(flat_image)
        labels[i, :, :] = np.reshape(ms.labels_, [8, 8])
    return np.reshape(labels, (len(data_emg), 64)).astype(int)


def data_reshape(data: np.ndarray) -> np.ndarray:
    """Reshape flat EMG data to (n_samples, 64) float64 format.

    Args:
        data: Input array of any shape with total elements divisible by 64.

    Returns:
        Array of shape (n_samples, 64) as float64.
    """
    data = np.reshape(data, (len(data), 64))
    data = data.astype(np.float64)
    return data
