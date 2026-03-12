"""Tests for the tfest transfer function estimator."""

import numpy as np
import pytest

from finger_impedance.core.tfestimate import tfest


FS = 512
N = 512
rng = np.random.default_rng(7)


def _signals():
    """Return a pair (u, y) of synthetic signals."""
    t = np.arange(N) / FS
    u = np.sin(2 * np.pi * 10 * t) + 0.1 * rng.standard_normal(N)
    y = np.sin(2 * np.pi * 10 * t + np.pi / 4) + 0.1 * rng.standard_normal(N)
    return u, y


class TestTfestInstantiation:
    def test_can_create(self):
        u, y = _signals()
        tf = tfest(u, y)
        assert tf is not None

    def test_attributes_initialised(self):
        u, y = _signals()
        tf = tfest(u, y)
        assert tf.res is None
        assert tf.H is None
        assert tf.frequency is None


class TestTransferFunctionH:
    def test_fft_method_returns_arrays(self):
        u, y = _signals()
        tf = tfest(u, y)
        H, freq = tf.transfer_function_H(FS, method="fft", time=N / FS)
        assert H is not None
        assert freq is not None
        assert len(H) == len(freq)

    def test_h1_method_returns_arrays(self):
        u, y = _signals()
        tf = tfest(u, y)
        H, freq = tf.transfer_function_H(FS, method="h1", time=N / FS)
        assert len(H) == len(freq)

    def test_h2_method_returns_arrays(self):
        u, y = _signals()
        tf = tfest(u, y)
        H, freq = tf.transfer_function_H(FS, method="h2", time=N / FS)
        assert len(H) == len(freq)

    def test_unknown_method_raises(self):
        u, y = _signals()
        tf = tfest(u, y)
        with pytest.raises(Exception, match="Unknown method"):
            tf.transfer_function_H(FS, method="bogus")


class TestEstimate:
    def test_estimate_fft_returns_result(self):
        u, y = _signals()
        tf = tfest(u, y)
        result = tf.estimate(
            0, 0, FS,
            method="fft",
            time=N / FS,
            options={"xatol": 1e-2, "disp": False},
        )
        assert result is not None
        assert tf.res is not None

    def test_bode_estimate_after_fit(self):
        u, y = _signals()
        tf = tfest(u, y)
        tf.estimate(
            0, 0, FS,
            method="fft",
            time=N / FS,
            options={"xatol": 1e-2, "disp": False},
        )
        w, mag = tf.bode_estimate()
        assert len(w) > 0
        assert len(mag) > 0

    def test_bode_estimate_before_fit_raises(self):
        u, y = _signals()
        tf = tfest(u, y)
        with pytest.raises(Exception):
            tf.bode_estimate()
