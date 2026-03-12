"""Core utility functions for EMG/force signal processing and analysis.

Provides Butterworth filters, EMG preprocessing, feature extraction
(time and frequency domain), stiffness estimation, regression metrics,
and ML classification helpers.
"""

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.svm import SVC

from finger_impedance.core.tfestimate import tfest

np.seterr(divide="ignore")


def butter_lowpass(cutoff: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Design a lowpass Butterworth filter and return (b, a) coefficients."""
    return butter(order, cutoff, fs=fs, btype="low", analog=False)


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply a lowpass Butterworth filter to data."""
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Design a bandpass Butterworth filter and return (b, a) coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def zero_lag_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """Apply a zero-lag (forward-backward) bandpass Butterworth filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def data_preprocess(emg_data: np.ndarray, fs: float, lowcut: float, highcut: float, cutoff: float) -> np.ndarray:
    """Preprocess EMG data: bandpass filter and full-wave rectification.

    Args:
        emg_data: Raw EMG signal array.
        fs: Sampling frequency in Hz.
        lowcut: Lower cutoff frequency for bandpass filter.
        highcut: Upper cutoff frequency for bandpass filter.
        cutoff: Lowpass cutoff (currently unused, kept for API compatibility).

    Returns:
        Filtered and rectified EMG signal.
    """
    print("Data filtering...")
    emg_ = zero_lag_filter(emg_data, lowcut, highcut, fs, order=4)
    emg_ = abs(emg_)
    return emg_


def rolling_rms(x: np.ndarray, window_size: int = 150) -> np.ndarray:
    """Compute rolling RMS of a signal using cumulative sum method."""
    xc = np.cumsum(abs(x) ** 2)
    return np.sqrt((xc[window_size:] - xc[:-window_size]) / window_size)


def mape(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error."""
    return mean_absolute_percentage_error(real, estimate)


def rmse(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Root Mean Squared Error."""
    return mean_squared_error(real, estimate, squared=False)


def nrmse1(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute NRMSE normalized by peak-to-peak range of absolute values."""
    return rmse(real, estimate) / (np.max(np.abs(real)) - np.min(np.abs(real)))


def nrmse2(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute NRMSE normalized by maximum absolute value."""
    return rmse(real, estimate) / (np.max(np.abs(real)))


def rmspe(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Root Mean Square Prediction Error."""
    return np.linalg.norm(estimate - real) / np.sqrt(len(real))


def vaf(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute Variance Accounted For (VAF) as a percentage."""
    return 100 * (1 - (np.var(real - estimate) / np.var(estimate)))


def r_square(real: np.ndarray, estimate: np.ndarray) -> float:
    """Compute coefficient of determination (R-squared)."""
    residuals = real - estimate
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((real - np.mean(real)) ** 2)
    return 1 - (ss_res / ss_tot)


def force_mean(data: np.ndarray, epoch: int) -> np.ndarray:
    """Compute epoch-wise mean of multi-channel force data.

    Args:
        data: 2D array of shape (n_samples, n_channels).
        epoch: Number of samples per epoch window.

    Returns:
        Array of shape (n_segments, n_channels) with mean values per epoch.
    """
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(
        data[0 : number_of_segments * epoch, :], number_of_segments
    )
    result = np.empty([number_of_segments, data.shape[1]])
    for i in range(number_of_segments):
        result[i, :] = np.mean(splitted_data[i], axis=0)
    return result


def force_window(splitted_data: List[np.ndarray]) -> List[np.ndarray]:
    """Create overlapping windows by concatenating adjacent segments.

    Each segment is combined with its neighbors to form a wider analysis window.
    Boundary segments reuse nearby segments to maintain consistent window size.
    """
    new_data = []

    for i in range(len(splitted_data)):
        if i == 0:
            new_data.append(
                np.concatenate(
                    (splitted_data[i], splitted_data[i + 1], splitted_data[i + 2])
                )
            )
        elif i < len(splitted_data) - 2:
            new_data.append(
                np.concatenate(
                    (splitted_data[i - 1], splitted_data[i], splitted_data[i + 1])
                )
            )
        else:
            new_data.append(
                np.concatenate(
                    (splitted_data[i], splitted_data[i - 1], splitted_data[i - 2])
                )
            )
    return new_data


def stiffness(freq: np.ndarray, K: float) -> np.ndarray:
    """Compute stiffness magnitude response: |K / (j*freq)|."""
    return np.absolute(-K * 1j / (freq))


def bode_plot(w: np.ndarray, mag: np.ndarray) -> None:
    """Plot a Bode magnitude diagram."""
    plt.title("Bode magnitude plot")
    plt.semilogx(w, mag, "x")
    plt.grid()


def feature_extraction(data: np.ndarray, epoch: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Extract time-domain and frequency-domain features from multi-channel EMG.

    Time-domain features: RMS, MAV, IAV, VAR, WL.
    Frequency-domain features: MF (mean freq), PF (peak freq), MP (mean power),
    TP (total power), SM (spectral moment).

    Args:
        data: 2D array of shape (n_samples, n_channels).
        epoch: Number of samples per epoch window.

    Returns:
        Tuple of 10 feature arrays, each (n_segments, n_channels).
    """
    print("Feature extraction...")
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(
        data[0 : number_of_segments * epoch, :], number_of_segments
    )
    RMS = np.empty([number_of_segments, data.shape[1]])
    MAV = np.empty([number_of_segments, data.shape[1]])
    IAV = np.empty([number_of_segments, data.shape[1]])
    VAR = np.empty([number_of_segments, data.shape[1]])
    WL = np.empty([number_of_segments, data.shape[1]])
    MF = np.empty([number_of_segments, data.shape[1]])
    PF = np.empty([number_of_segments, data.shape[1]])
    MP = np.empty([number_of_segments, data.shape[1]])
    TP = np.empty([number_of_segments, data.shape[1]])
    SM = np.empty([number_of_segments, data.shape[1]])

    for i in range(number_of_segments):
        RMS[i, :] = np.sqrt(np.mean(np.square(splitted_data[i]), axis=0))
        MAV[i, :] = np.mean(np.abs(splitted_data[i]), axis=0)
        IAV[i, :] = np.sum(np.abs(splitted_data[i]), axis=0)
        VAR[i, :] = np.var(splitted_data[i], axis=0)
        WL[i, :] = np.sum(np.diff(splitted_data[i], prepend=0), axis=0)
        freq, power = signal.periodogram(splitted_data[i], axis=0)
        fp = np.empty([len(freq), power.shape[1]])
        for k in range(len(freq)):
            fp[k] = power[k, :] * freq[k]
        MF[i, :] = np.sum(fp, axis=0) / np.sum(power, axis=0)  # Mean frequency
        PF[i, :] = freq[np.argmax(power, axis=0)]  # Peak frequency
        MP[i, :] = np.mean(power, axis=0)  # Mean power
        TP[i, :] = np.sum(power, axis=0)  # Total power
        SM[i, :] = np.sum(fp, axis=0)  # Spectral moment
    return RMS, MAV, IAV, VAR, WL, MF, PF, MP, TP, SM


def class_map(data: np.ndarray, epoch: int) -> np.ndarray:
    """Compute epoch-wise mean for class/label signals."""
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(data[0 : number_of_segments * epoch], number_of_segments)
    class_value = np.empty([number_of_segments])
    for i in range(number_of_segments):
        class_value[i] = np.mean(splitted_data[i])
    return class_value


def classification_report_with_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Scoring function returning accuracy for use with cross_val_score."""
    return accuracy_score(y_true, y_pred)


def classifier(features: np.ndarray, labels: np.ndarray, k_fold: int) -> None:
    """Run k-fold cross-validation with multiple classifiers and plot results.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Class labels of shape (n_samples,).
        k_fold: Number of cross-validation folds.
    """
    Y = labels
    X = features
    random_seed = 42
    outcome = []
    model_names = []

    models = [
        ("LogReg", LogisticRegression()),
        ("SVM", SVC()),
        # ('DecTree', DecisionTreeClassifier()),
        # ('KNN', KNeighborsClassifier(n_neighbors=15)),
        # ('LinDisc', LinearDiscriminantAnalysis()),
        # ('GaussianNB', GaussianNB()),
        # ('MLPC', MLPClassifier(activation='relu', solver='adam', max_iter=500)),
        # ('RFC',RandomForestClassifier()),
        # ('ABC', AdaBoostClassifier())
    ]

    for model_name, model in models:
        k_fold_validation = model_selection.KFold(
            n_splits=k_fold, random_state=random_seed, shuffle=True
        )
        results = model_selection.cross_val_score(
            model,
            X,
            Y,
            cv=k_fold_validation,
            scoring=make_scorer(classification_report_with_accuracy_score),
        )
        outcome.append(results)
        model_names.append(model_name)
        output_message = "%s| Mean=%f STD=%f" % (
            model_name,
            results.mean(),
            results.std(),
        )
        print(output_message)

    fig = plt.figure()
    fig.suptitle("Machine Learning Model Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(outcome)
    plt.ylabel("Accuracy [%]")
    ax.set_xticklabels(model_names)
    plt.show()


def force_stiffness(data: np.ndarray, epoch: int) -> np.ndarray:
    """Estimate stiffness from force data using FFT-based transfer function estimation.

    For each epoch and channel, fits a transfer function to estimate
    the DC stiffness magnitude (lowest frequency component).

    Args:
        data: 2D force array of shape (n_samples, n_channels).
        epoch: Number of samples per epoch window.

    Returns:
        Stiffness estimates of shape (n_segments, n_channels).
    """
    print("Stiffness estimation...")
    number_of_segments = math.trunc(len(data) / epoch)
    splitted_data = np.split(
        data[0 : number_of_segments * epoch, :], number_of_segments
    )
    stiffness_estimation = np.empty([number_of_segments, data.shape[1]])
    sr = 2048

    for i in range(number_of_segments):
        k = []
        for j in range(data.shape[1]):
            x = splitted_data[i][:, j]
            y = np.ones(len(x))
            tf = tfest(y, x)
            tf.estimate(0, 0, sr, method="fft")
            w1, mag1 = tf.bode_estimate()
            k.append(mag1[0])
        stiffness_estimation[i, :] = k
    return stiffness_estimation


def moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    """Compute moving average with padding to maintain output length."""
    ret = np.cumsum(a, dtype=float)
    ret = np.append(ret, [ret[-1] * (np.ones(n - 1))])
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage change relative to the first row."""
    pct = 1 - df.iloc[0] / df
    return pct


def running_mean(x: np.ndarray, N: int) -> np.ndarray:
    """Compute running mean using cumulative sum method."""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def evaluate_regression_metrics(y_pred: np.ndarray, y_true: np.ndarray, index: str) -> pd.DataFrame:
    """Evaluate a comprehensive set of regression metrics.

    Args:
        y_pred: Predicted values.
        y_true: Ground truth values.
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
