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

    def test_rejects_multichannel_arrays(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            tfest(np.ones((10, 2)), np.ones((10, 2)))


class TestTransferFunctionH:
    def test_fft_method_returns_arrays(self):
        u, y = _signals()
        tf = tfest(u, y)
        H, freq = tf.transfer_function_H(FS, method="fft")
        assert H is not None
        assert freq is not None
        assert len(H) == len(freq)

    def test_h1_method_returns_arrays(self):
        u, y = _signals()
        tf = tfest(u, y)
        H, freq = tf.transfer_function_H(FS, method="h1")
        assert len(H) == len(freq)

    def test_h2_method_returns_arrays(self):
        u, y = _signals()
        tf = tfest(u, y)
        H, freq = tf.transfer_function_H(FS, method="h2")
        assert len(H) == len(freq)

    def test_unknown_method_raises(self):
        u, y = _signals()
        tf = tfest(u, y)
        with pytest.raises(ValueError, match="unknown transfer-function method"):
            tf.transfer_function_H(FS, method="bogus")

    def test_zero_input_has_no_transfer_function(self):
        tf = tfest(np.zeros(N), np.ones(N))

        with pytest.raises(ValueError, match="no measurable frequency content"):
            tf.transfer_function_H(FS, method="fft")

    def test_fft_frequency_and_gain_are_correct(self):
        t = np.arange(N) / FS
        u = np.sin(2 * np.pi * 10 * t)
        H, freq = tfest(u, 2 * u).transfer_function_H(FS, method="fft")
        ten_hz = np.flatnonzero(np.isclose(freq, 10.0))

        assert len(ten_hz) == 1
        np.testing.assert_allclose(H[ten_hz[0]], 2.0)

    @pytest.mark.parametrize("method", ["h1", "h2"])
    def test_spectral_estimators_recover_known_gain(self, method):
        t = np.arange(N) / FS
        u = np.sin(2 * np.pi * 10 * t)
        H, freq = tfest(u, 2 * u).transfer_function_H(FS, method=method)
        index = np.argmin(np.abs(freq - 10.0))

        assert freq[index] == pytest.approx(10.0)
        assert H[index] == pytest.approx(2.0)

    def test_h1_and_h2_are_distinct_with_output_noise(self):
        local_rng = np.random.default_rng(11)
        u = local_rng.standard_normal(2048)
        y = 2 * u + local_rng.standard_normal(2048)

        h1, _ = tfest(u, y).transfer_function_H(FS, method="h1", nperseg=256)
        h2, _ = tfest(u, y).transfer_function_H(FS, method="h2", nperseg=256)

        assert not np.allclose(h1, h2)


class TestEstimate:
    def test_estimate_fft_returns_result(self):
        u = rng.standard_normal(N)
        y = 2 * u
        tf = tfest(u, y)
        result = tf.estimate(
            0,
            0,
            FS,
            method="fft",
            options={"xatol": 1e-2, "disp": False},
        )
        assert result.success
        np.testing.assert_allclose(result.x, [2.0], atol=1e-2)

    def test_bode_estimate_after_fit(self):
        u = rng.standard_normal(N)
        y = 2 * u
        tf = tfest(u, y)
        tf.estimate(
            0,
            0,
            FS,
            method="fft",
            options={"xatol": 1e-2, "disp": False},
        )
        w, mag = tf.bode_estimate()
        assert len(w) > 0
        np.testing.assert_allclose(mag, 20 * np.log10(2), atol=0.1)

    def test_bode_estimate_before_fit_raises(self):
        u, y = _signals()
        tf = tfest(u, y)
        with pytest.raises(RuntimeError):
            tf.bode_estimate()

    def test_negative_regularization_is_rejected(self):
        u, y = _signals()

        with pytest.raises(ValueError, match="non-negative"):
            tfest(u, y).estimate(0, 0, FS, regularization=-1)
