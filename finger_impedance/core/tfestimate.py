"""Frequency-response and parametric transfer-function estimation."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import minimize


class tfest:
    """Estimate a SISO transfer function from synchronized one-dimensional signals."""

    def __init__(self, u, y):
        self.u = self._as_signal(u, "u")
        self.y = self._as_signal(y, "y")
        if self.u.shape != self.y.shape:
            raise ValueError("u and y must have the same length")
        if len(self.u) < 2:
            raise ValueError("u and y must contain at least two samples")
        self.res = None
        self.frequency = None
        self.H = None
        self.npoles = 0
        self.nzeros = 0
        self.numerator_count = 0

    @staticmethod
    def _as_signal(values, name):
        signal_values = np.asarray(values, dtype=float)
        if signal_values.ndim == 2 and 1 in signal_values.shape:
            signal_values = signal_values.reshape(-1)
        if signal_values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional signal")
        if not np.all(np.isfinite(signal_values)):
            raise ValueError(f"{name} must contain only finite values")
        return signal_values

    @staticmethod
    def _coefficients(x, numerator_count):
        numerator = x[:numerator_count]
        denominator = np.concatenate(([1.0], x[numerator_count:]))
        return numerator, denominator

    def loss(self, x, numerator_count, freq, H, regularization=0):
        """Return frequency-response fit error for a monic denominator model."""
        numerator, denominator = self._coefficients(x, numerator_count)
        angular_frequency = 2j * np.pi * freq
        response = np.polyval(numerator, angular_frequency) / np.polyval(
            denominator, angular_frequency
        )
        penalty = regularization * np.linalg.norm(x)
        return float(np.sum(np.abs(response - H)) + penalty)

    def transfer_function_H(self, fs, method="h1", nperseg=None):
        """Return the estimated response and corresponding frequencies in Hz."""
        if fs <= 0:
            raise ValueError("fs must be positive")
        if method == "fft":
            u_f = np.fft.rfft(self.u)
            y_f = np.fft.rfft(self.y)
            frequency = np.fft.rfftfreq(len(self.u), d=1 / fs)
            peak_input = float(np.max(np.abs(u_f)))
            valid = np.abs(u_f) > np.sqrt(np.finfo(float).eps) * peak_input
            frequency = frequency[valid]
            H = y_f[valid] / u_f[valid]
        elif method == "h1":
            nperseg = self._spectral_segment_length(nperseg)
            frequency, cross_sd = signal.csd(self.u, self.y, fs=fs, nperseg=nperseg)
            _, power_sd = signal.welch(self.u, fs=fs, nperseg=nperseg)
            peak_power = float(np.max(power_sd))
            valid = power_sd > np.finfo(float).eps * peak_power
            frequency = frequency[valid]
            H = cross_sd[valid] / power_sd[valid]
        elif method == "h2":
            nperseg = self._spectral_segment_length(nperseg)
            frequency, cross_sd = signal.csd(self.y, self.u, fs=fs, nperseg=nperseg)
            _, power_sd = signal.welch(self.y, fs=fs, nperseg=nperseg)
            peak_cross_power = float(np.max(np.abs(cross_sd)))
            valid = np.abs(cross_sd) > np.finfo(float).eps * peak_cross_power
            frequency = frequency[valid]
            H = power_sd[valid] / cross_sd[valid]
        else:
            raise ValueError(f"unknown transfer-function method: {method}")
        if len(frequency) == 0:
            raise ValueError("input signal has no measurable frequency content")
        self.frequency = frequency
        self.H = H
        return self.H, frequency

    def _spectral_segment_length(self, nperseg):
        if len(self.u) < 4:
            raise ValueError("H1/H2 estimation requires at least four samples")
        if nperseg is None:
            return min(256, len(self.u) // 2)
        if not isinstance(nperseg, int) or not 2 <= nperseg <= len(self.u) // 2:
            raise ValueError("nperseg must be an integer between 2 and half the signal length")
        return nperseg

    def estimate(
        self,
        nzeros,
        npoles,
        fs,
        init_value=1,
        options=None,
        method="h1",
        regularization=0,
        nperseg=None,
    ):
        """Fit numerator and monic-denominator coefficients to the response."""
        if not isinstance(nzeros, int) or nzeros < 0:
            raise ValueError("nzeros must be a non-negative integer")
        if not isinstance(npoles, int) or npoles < 0:
            raise ValueError("npoles must be a non-negative integer")
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and non-negative")
        numerator_count = nzeros + 1
        self.npoles = npoles
        self.nzeros = nzeros
        self.numerator_count = numerator_count

        x0 = np.full(numerator_count + npoles, init_value, dtype=float)
        H, frequency = self.transfer_function_H(fs, method=method, nperseg=nperseg)

        def pass_to_loss(x):
            return self.loss(x, numerator_count, frequency, H, regularization)

        self.res = minimize(
            pass_to_loss,
            x0,
            method="nelder-mead",
            options={"xatol": 1e-3, "disp": False} if options is None else options,
        )
        return self.res

    def get_transfer_function(self):
        """Return the fitted SciPy continuous-time transfer function."""
        if self.res is None:
            raise RuntimeError("run estimate() before requesting the transfer function")
        numerator, denominator = self._coefficients(self.res.x, self.numerator_count)
        return signal.lti(numerator, denominator)

    def bode_estimate(self):
        """Return the fitted Bode frequency and magnitude arrays."""
        if self.res is None:
            raise RuntimeError("run estimate() before requesting a Bode response")
        tf = self.get_transfer_function()
        w, mag, _ = tf.bode()
        return w, mag

    def plot_bode(self):
        """Plot the fitted Bode magnitude and phase."""
        if self.res is None:
            raise RuntimeError("run estimate() before plotting")
        tf = self.get_transfer_function()
        w, mag, phase = tf.bode()
        plt.figure()
        plt.title("Bode magnitude plot")
        plt.semilogx(w, mag)
        plt.grid()
        plt.figure()
        plt.title("Bode phase plot")
        plt.semilogx(w, phase)
        plt.grid()

    def plot(self):
        """Plot measured and fitted frequency-response magnitudes."""
        if self.res is None:
            raise RuntimeError("run estimate() before plotting")
        numerator, denominator = self._coefficients(self.res.x, self.numerator_count)
        angular_frequency = 2j * np.pi * self.frequency
        response = np.polyval(numerator, angular_frequency) / np.polyval(
            denominator, angular_frequency
        )
        plt.plot(self.frequency, 20 * np.log10(np.abs(response)), label="estimation")
        plt.plot(self.frequency, 20 * np.log10(np.abs(self.H)), label="train data")
        plt.legend(loc="upper right")
        plt.grid()
