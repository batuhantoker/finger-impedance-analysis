"""Tests for EMG feature extraction and related epoch utilities."""

import numpy as np

from finger_impedance.core.functions import (
    class_map,
    feature_extraction,
    force_mean,
    moving_average,
    rolling_rms,
    running_mean,
)
from finger_impedance.signal.signal_features import activation_map, mean_shift_feature

FS = 2048
EPOCH = 256
N_CHANNELS = 16
N_EPOCHS = 5
N_SAMPLES = EPOCH * N_EPOCHS
rng = np.random.default_rng(0)


def _emg(n_samples: int = N_SAMPLES, n_ch: int = N_CHANNELS) -> np.ndarray:
    return rng.standard_normal((n_samples, n_ch)).astype(np.float64)


def _force(n_samples: int = N_SAMPLES, n_ch: int = 2) -> np.ndarray:
    return np.abs(rng.standard_normal((n_samples, n_ch)))


class TestFeatureExtraction:
    def test_returns_ten_arrays(self):
        out = feature_extraction(_emg(), EPOCH, FS)
        assert len(out) == 10

    def test_output_shape(self):
        data = _emg()
        out = feature_extraction(data, EPOCH, FS)
        expected_segments = N_SAMPLES // EPOCH
        for feat in out:
            assert feat.shape == (expected_segments, N_CHANNELS), f"shape mismatch: {feat.shape}"

    def test_rms_nonnegative(self):
        RMS, *_ = feature_extraction(_emg(), EPOCH, FS)
        assert np.all(RMS >= 0)

    def test_mav_nonnegative(self):
        _, MAV, *_ = feature_extraction(_emg(), EPOCH, FS)
        assert np.all(MAV >= 0)

    def test_iav_nonnegative(self):
        _, _, IAV, *_ = feature_extraction(_emg(), EPOCH, FS)
        assert np.all(IAV >= 0)

    def test_var_nonnegative(self):
        _, _, _, VAR, *_ = feature_extraction(_emg(), EPOCH, FS)
        assert np.all(VAR >= 0)

    def test_all_finite(self):
        data = _emg()
        for feat in feature_extraction(data, EPOCH, FS):
            assert np.all(np.isfinite(feat)), "non-finite values in feature array"

    def test_waveform_length_uses_temporal_absolute_differences(self):
        data = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 4.0]])
        *_, waveform_length, _, _, _, _, _ = feature_extraction(data, 3, 3)

        np.testing.assert_allclose(waveform_length[0], [2.0, 4.0])

    def test_peak_frequency_is_reported_in_hz(self):
        frequency = 64.0
        data = np.sin(2 * np.pi * frequency * np.arange(FS) / FS)[:, None]
        *_, peak_frequency, _, _, _ = feature_extraction(data, FS, FS)

        np.testing.assert_allclose(peak_frequency, [[frequency]])

    def test_integer_input_is_converted_before_squaring(self):
        data = np.full((8, 2), 30_000, dtype=np.int16)
        rms, *_ = feature_extraction(data, 8, FS)

        np.testing.assert_allclose(rms, 30_000)


class TestClassMap:
    def test_output_shape(self):
        labels = np.repeat(np.arange(5), N_SAMPLES // 5).astype(float)
        out = class_map(labels, EPOCH)
        assert out.shape == (N_SAMPLES // EPOCH,)

    def test_constant_label_preserved(self):
        # All ones should map to ones
        labels = np.ones(N_SAMPLES)
        out = class_map(labels, EPOCH)
        np.testing.assert_allclose(out, 1.0)

    def test_transition_epoch_is_rejected(self):
        labels = np.array([1, 1, 3, 3], dtype=float)
        out = class_map(labels, 4)

        assert np.isnan(out[0])

    def test_multicolumn_labels_are_rejected(self):
        labels = np.ones((4, 2))

        with np.testing.assert_raises_regex(ValueError, "one-dimensional"):
            class_map(labels, 2)


class TestForceMean:
    def test_output_shape(self):
        force = _force()
        out = force_mean(force, EPOCH)
        expected = N_SAMPLES // EPOCH
        assert out.shape == (expected, 2)

    def test_constant_signal_preserved(self):
        force = np.ones((N_SAMPLES, 2)) * 3.0
        out = force_mean(force, EPOCH)
        np.testing.assert_allclose(out, 3.0)


class TestRollingRms:
    def test_output_shorter_than_input(self):
        x = rng.standard_normal(1000)
        y = rolling_rms(x, window_size=150)
        assert len(y) == len(x) - 150 + 1

    def test_constant_signal(self):
        x = np.ones(500) * 2.0
        y = rolling_rms(x, window_size=50)
        np.testing.assert_allclose(y, 2.0)


class TestMovingAverage:
    def test_output_length(self):
        a = np.arange(100, dtype=float)
        out = moving_average(a, n=5)
        assert len(out) == len(a)

    def test_constant_signal(self):
        a = np.ones(50) * 7.0
        out = moving_average(a, n=3)
        np.testing.assert_allclose(out, 7.0)


class TestRunningMean:
    def test_output_length(self):
        # running_mean(x, N) produces len(x) - N + 1 elements
        x = np.arange(100, dtype=float)
        out = running_mean(x, 10)
        assert len(out) == len(x) - 10 + 1

    def test_constant_signal(self):
        x = np.ones(100) * 4.0
        out = running_mean(x, 10)
        np.testing.assert_allclose(out, 4.0)


def test_signal_activation_map_uses_core_feature_definitions():
    data = _emg()

    expected = feature_extraction(data, EPOCH, FS)
    actual = activation_map(data, EPOCH, FS)

    for actual_feature, expected_feature in zip(actual, expected):
        np.testing.assert_allclose(actual_feature, expected_feature)


def test_mean_shift_feature_handles_constant_maps():
    result = mean_shift_feature(np.ones((2, 64)))

    assert result.shape == (2, 64)
    assert np.issubdtype(result.dtype, np.integer)
    np.testing.assert_array_equal(result, 0)
