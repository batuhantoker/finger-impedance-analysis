"""Tests for regression and smoothing metrics."""

import numpy as np
import pandas as pd

from finger_impedance.core.functions import (
    evaluate_regression_metrics,
    moving_average,
    nrmse1,
    nrmse2,
    pct_change,
    r_square,
    rmse,
    vaf,
)


def test_rmse_supports_current_scikit_learn():
    assert rmse(np.array([0.0, 1.0]), np.array([0.0, 2.0])) == np.sqrt(0.5)


def test_vaf_uses_reference_variance():
    real = np.array([1.0, 2.0, 3.0])
    estimate = np.array([2.0, 4.0, 6.0])

    assert vaf(real, estimate) == 0.0


def test_pct_change_uses_initial_value_as_denominator():
    values = pd.DataFrame({"force": [10.0, 20.0]})

    np.testing.assert_allclose(pct_change(values)["force"], [0.0, 1.0])


def test_moving_average_preserves_a_constant_tail():
    np.testing.assert_allclose(moving_average(np.full(5, 7.0), 3), 7.0)


def test_normalized_errors_reject_undefined_reference_scales():
    constant = np.ones(3)

    with np.testing.assert_raises(ValueError):
        nrmse1(constant, constant)
    with np.testing.assert_raises(ValueError):
        nrmse2(np.zeros(3), np.zeros(3))
    with np.testing.assert_raises(ValueError):
        vaf(constant, constant)


def test_regression_metrics_are_exact_for_a_perfect_prediction():
    values = np.array([1.0, 2.0, 3.0])

    metrics = evaluate_regression_metrics(y_true=values, y_pred=values, index="force")

    assert metrics.loc["force", "R2"] == 100.0
    assert metrics.loc["force", "RMSE"] == 0.0
    assert metrics.loc["force", "vaf"] == 100.0
    assert r_square(values, values) == 1.0


def test_regression_metrics_use_y_true_as_the_reference():
    y_true = np.array([1.0, 2.0, 4.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    metrics = evaluate_regression_metrics(y_true=y_true, y_pred=y_pred, index="force")

    assert metrics.loc["force", "R2"] < 0
    assert metrics.loc["force", "vaf"] < 100
