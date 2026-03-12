"""Tests for Butterworth filter design and application."""

import numpy as np
import pytest

from finger_impedance.core.functions import (
    butter_bandpass,
    butter_lowpass,
    butter_lowpass_filter,
    zero_lag_filter,
)


FS = 2048
N = 4096


def _sine(freq: float, n: int = N, fs: float = FS) -> np.ndarray:
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t)


class TestButterLowpass:
    def test_returns_ba_tuple(self):
        b, a = butter_lowpass(100.0, FS)
        assert len(b) > 0
        assert len(a) > 0

    def test_coefficients_are_finite(self):
        b, a = butter_lowpass(100.0, FS, order=4)
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(a))

    def test_output_length_matches_input(self):
        x = _sine(10.0)
        y = butter_lowpass_filter(x, 100.0, FS, order=4)
        assert y.shape == x.shape

    def test_attenuates_high_frequencies(self):
        # A 500 Hz component should be strongly attenuated by a 100 Hz lowpass
        x_low = _sine(10.0)
        x_high = _sine(500.0)
        y_low = butter_lowpass_filter(x_low, 100.0, FS, order=4)
        y_high = butter_lowpass_filter(x_high, 100.0, FS, order=4)
        assert np.std(y_low) > np.std(y_high) * 10


class TestButterBandpass:
    def test_returns_ba_tuple(self):
        b, a = butter_bandpass(15.0, 350.0, FS)
        assert len(b) > 0
        assert len(a) > 0

    def test_coefficients_are_finite(self):
        b, a = butter_bandpass(15.0, 350.0, FS)
        assert np.all(np.isfinite(b))
        assert np.all(np.isfinite(a))


class TestZeroLagFilter:
    def test_output_length_matches_input(self):
        x = _sine(100.0)
        y = zero_lag_filter(x, 15.0, 350.0, FS, order=4)
        assert y.shape == x.shape

    def test_passband_signal_preserved(self):
        # A 100 Hz component is in the 15–350 Hz passband; energy should be retained
        x = _sine(100.0)
        y = zero_lag_filter(x, 15.0, 350.0, FS, order=4)
        assert np.std(y) > 0.1

    def test_stopband_signal_attenuated(self):
        # A 1 Hz component is below the 15 Hz lowcut and should be suppressed
        x = _sine(1.0)
        y = zero_lag_filter(x, 15.0, 350.0, FS, order=4)
        assert np.std(y) < 0.1
