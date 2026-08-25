"""Core utility functions for EMG/force signal processing and analysis.

Provides Butterworth filters, EMG preprocessing, feature extraction
(time and frequency domain), stiffness estimation, regression metrics,
and ML classification helpers.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def butter_lowpass(cutoff: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Design a lowpass Butterworth filter and return (b, a) coefficients."""
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 5,
    axis: int = 0,
) -> np.ndarray:
    """Apply a lowpass Butterworth filter to data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=axis)
    return y


def butter_bandpass(
    lowcut: float, highcut: float, fs: float, order: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a bandpass Butterworth filter and return (b, a) coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def zero_lag_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 5,
    axis: int = 0,
) -> np.ndarray:
    """Apply a zero-lag (forward-backward) bandpass Butterworth filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data, axis=axis)
    return y


def data_preprocess(
    emg_data: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
) -> np.ndarray:
    """Bandpass-filter EMG along its sample axis without rectification.

    Args:
        emg_data: Raw EMG signal array.
        fs: Sampling frequency in Hz.
        lowcut: Lower cutoff frequency for bandpass filter.
        highcut: Upper cutoff frequency for bandpass filter.
    Returns:
        Signed, zero-phase filtered EMG signal.
    """
    return zero_lag_filter(np.asarray(emg_data, dtype=float), lowcut, highcut, fs, order=4)


def rolling_rms(x: np.ndarray, window_size: int = 150) -> np.ndarray:
    """Compute rolling RMS of a signal using cumulative sum method."""
    x = np.asarray(x, dtype=float)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > len(x):
        raise ValueError("window_size cannot exceed the signal length")
    xc = np.cumsum(np.insert(np.abs(x) ** 2, 0, 0.0))
    return np.sqrt((xc[window_size:] - xc[:-window_size]) / window_size)


def mape(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    return float(mean_absolute_percentage_error(real, estimate))


def rmse(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(real, estimate)))


def nrmse1(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute NRMSE normalized by the observed range."""
    value_range = np.ptp(real)
    if value_range == 0:
        raise ValueError("nRMSE is undefined for a constant reference signal")
    return rmse(real, estimate) / value_range


def nrmse2(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute NRMSE normalized by maximum absolute value."""
    maximum = np.max(np.abs(real))
    if maximum == 0:
        raise ValueError("nRMSE is undefined for an all-zero reference signal")
    return rmse(real, estimate) / maximum


def rmspe(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Root Mean Square Prediction Error."""
    real = np.asarray(real)
    estimate = np.asarray(estimate)
    return float(np.linalg.norm(estimate - real) / np.sqrt(len(real)))


def vaf(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Variance Accounted For (VAF) as a percentage."""
    reference_variance = np.var(real)
    if reference_variance == 0:
        raise ValueError("VAF is undefined for a constant reference signal")
    return float(100 * (1 - np.var(real - estimate) / reference_variance))


def r_square(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute coefficient of determination (R-squared)."""
    return float(r2_score(real, estimate))


def force_mean(data: np.ndarray, epoch: int) -> np.ndarray:
    """Compute epoch-wise mean of multi-channel force data.

    Args:
        data: 2D array of shape (n_samples, n_channels).
        epoch: Number of samples per epoch window.

    Returns:
        Array of shape (n_segments, n_channels) with mean values per epoch.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must have shape (samples, channels)")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    number_of_segments = len(data) // epoch
    if number_of_segments == 0:
        raise ValueError("data must contain at least one complete epoch")
    segmented = data[: number_of_segments * epoch].reshape(number_of_segments, epoch, data.shape[1])
    return np.mean(segmented, axis=1)


def feature_extraction(
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

    Time-domain features: RMS, MAV, IAV, VAR, WL.
    Frequency-domain features: MF (mean freq), PF (peak freq), MP (mean power),
    TP (total power), SM (spectral moment).

    Args:
        data: 2D array of shape (n_samples, n_channels).
        epoch: Number of samples per epoch window.
        fs: Sampling frequency in Hz.

    Returns:
        Tuple of 10 feature arrays, each (n_segments, n_channels).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must have shape (samples, channels)")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")
    number_of_segments = len(data) // epoch
    if number_of_segments == 0:
        raise ValueError("data must contain at least one complete epoch")

    segments = data[: number_of_segments * epoch].reshape(number_of_segments, epoch, data.shape[1])
    rms = np.sqrt(np.mean(np.square(segments), axis=1))
    mav = np.mean(np.abs(segments), axis=1)
    iav = np.sum(np.abs(segments), axis=1)
    variance = np.var(segments, axis=1)
    waveform_length = np.sum(np.abs(np.diff(segments, axis=1)), axis=1)

    frequencies, power = signal.periodogram(segments, fs=fs, axis=1)
    frequency_weighted_power = power * frequencies[np.newaxis, :, np.newaxis]
    power_sum = np.sum(power, axis=1)
    mean_frequency = np.divide(
        np.sum(frequency_weighted_power, axis=1),
        power_sum,
        out=np.zeros_like(power_sum),
        where=power_sum > 0,
    )
    peak_frequency = frequencies[np.argmax(power, axis=1)]
    mean_power = np.mean(power, axis=1)
    frequency_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0.0
    total_power = power_sum * frequency_resolution
    spectral_moment = np.sum(frequency_weighted_power, axis=1) * frequency_resolution
    return (
        rms,
        mav,
        iav,
        variance,
        waveform_length,
        mean_frequency,
        peak_frequency,
        mean_power,
        total_power,
        spectral_moment,
    )


def class_map(data: np.ndarray, epoch: int) -> np.ndarray:
    """Return one label per pure epoch and NaN for transition epochs."""
    data = np.asarray(data)
    if data.ndim == 2 and 1 in data.shape:
        data = data.reshape(-1)
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional or a singleton row/column")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    number_of_segments = len(data) // epoch
    if number_of_segments == 0:
        raise ValueError("data must contain at least one complete epoch")
    segments = data[: number_of_segments * epoch].reshape(number_of_segments, epoch)
    labels = segments[:, 0].astype(float, copy=True)
    labels[~np.all(segments == segments[:, :1], axis=1)] = np.nan
    return labels


def estimate_stiffness(
    displacement: np.ndarray,
    force: np.ndarray,
    epoch: int,
    *,
    min_displacement_range: float = 0.0,
) -> np.ndarray:
    """Estimate quasi-static stiffness from synchronized displacement and force.

    A least-squares slope is fitted independently for each epoch and channel.
    The result has force/displacement units and is undefined without measurable
    displacement variation. ``min_displacement_range`` can be set to the
    calibrated resolution of the displacement sensor.
    """
    displacement = np.asarray(displacement, dtype=float)
    force = np.asarray(force, dtype=float)
    if displacement.ndim == 1:
        displacement = displacement[:, np.newaxis]
    if force.ndim == 1:
        force = force[:, np.newaxis]
    if displacement.shape != force.shape:
        raise ValueError("displacement and force must have identical shapes")
    if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(force)):
        raise ValueError("displacement and force must contain only finite values")
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if not np.isfinite(min_displacement_range) or min_displacement_range < 0:
        raise ValueError("min_displacement_range must be finite and non-negative")

    number_of_segments = len(force) // epoch
    if number_of_segments == 0:
        raise ValueError("signals must contain at least one complete epoch")
    displacement_segments = displacement[: number_of_segments * epoch].reshape(
        number_of_segments, epoch, displacement.shape[1]
    )
    force_segments = force[: number_of_segments * epoch].reshape(
        number_of_segments, epoch, force.shape[1]
    )
    centered_displacement = displacement_segments - np.mean(
        displacement_segments, axis=1, keepdims=True
    )
    centered_force = force_segments - np.mean(force_segments, axis=1, keepdims=True)
    denominator = np.sum(centered_displacement**2, axis=1)
    displacement_range = np.ptp(displacement_segments, axis=1)
    numeric_resolution = np.spacing(np.max(np.abs(displacement_segments), axis=1))
    required_range = np.maximum(min_displacement_range, numeric_resolution)
    if np.any(displacement_range <= required_range):
        raise ValueError("stiffness requires measurable displacement variation in every epoch")
    return np.sum(centered_displacement * centered_force, axis=1) / denominator


def co_contraction_index(
    flexor_activation: np.ndarray,
    extensor_activation: np.ndarray,
) -> np.ndarray:
    """Compute a dimensionless antagonist co-contraction index in [0, 1]."""
    flexor = np.asarray(flexor_activation, dtype=float)
    extensor = np.asarray(extensor_activation, dtype=float)
    if flexor.shape != extensor.shape:
        raise ValueError("flexor and extensor activation must have identical shapes")
    if not np.all(np.isfinite(flexor)) or not np.all(np.isfinite(extensor)):
        raise ValueError("activation values must be finite")
    if np.any(flexor < 0) or np.any(extensor < 0):
        raise ValueError("activation values must be non-negative")

    total = flexor + extensor
    return np.divide(
        2 * np.minimum(flexor, extensor),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )


def stiffness_proxy(
    flexor_activation: np.ndarray,
    extensor_activation: np.ndarray,
) -> np.ndarray:
    """Return total antagonist activation as a non-physical stiffness proxy.

    Inputs should be normalized to comparable reference contractions before
    values are compared across muscles, sessions, or subjects.
    """
    flexor = np.asarray(flexor_activation, dtype=float)
    extensor = np.asarray(extensor_activation, dtype=float)
    if flexor.shape != extensor.shape:
        raise ValueError("flexor and extensor activation must have identical shapes")
    if not np.all(np.isfinite(flexor)) or not np.all(np.isfinite(extensor)):
        raise ValueError("activation values must be finite")
    if np.any(flexor < 0) or np.any(extensor < 0):
        raise ValueError("activation values must be non-negative")
    return flexor + extensor


def moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    """Compute a forward moving average with edge padding."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError("a must be one-dimensional")
    if n <= 0:
        raise ValueError("n must be positive")
    if len(a) == 0:
        return a.copy()
    padded = np.pad(a, (0, n - 1), mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage change relative to the first row."""
    baseline = df.iloc[0].replace(0, np.nan)
    return df.subtract(df.iloc[0]).divide(baseline)


def running_mean(x: np.ndarray, N: int) -> np.ndarray:
    """Compute running mean using cumulative sum method."""
    x = np.asarray(x, dtype=float)
    if N <= 0:
        raise ValueError("N must be positive")
    if N > len(x):
        raise ValueError("N cannot exceed the signal length")
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def evaluate_regression_metrics(
    *, y_true: np.ndarray, y_pred: np.ndarray, index: str
) -> pd.DataFrame:
    """Evaluate a comprehensive set of regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        index: Label for the resulting DataFrame row.

    Returns:
        Single-row DataFrame with R2, MAE, MSE, RMSEP, VAF, RMSE, nRMSE1, nRMSE2.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmsep = np.sqrt(mse)
    vaf_value = vaf(y_true, y_pred)
    nrmse_value1 = nrmse1(y_true, y_pred)
    nrmse_value2 = nrmse2(y_true, y_pred)

    metrics = {
        "R2": r2 * 100,
        "MAE": mae,
        "MSE": mse,
        "RMSEP": rmsep,
        "vaf": vaf_value,
        "RMSE": rmsep,
        "nRMSE1": nrmse_value1 * 100,
        "nRMSE2": nrmse_value2 * 100,
    }

    metrics_df = pd.DataFrame(metrics, index=[index])
    return metrics_df
