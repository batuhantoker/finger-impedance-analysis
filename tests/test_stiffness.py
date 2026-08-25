"""Tests for measured stiffness and activation-based proxy metrics."""

import numpy as np
import pytest

from finger_impedance import co_contraction_index, estimate_stiffness, stiffness_proxy


def test_estimate_stiffness_recovers_known_slopes():
    displacement = np.column_stack((np.linspace(-0.01, 0.01, 100), np.linspace(-0.02, 0.02, 100)))
    force = displacement * np.array([250.0, 100.0]) + np.array([4.0, -2.0])

    result = estimate_stiffness(displacement, force, epoch=50)

    np.testing.assert_allclose(result, [[250.0, 100.0], [250.0, 100.0]])


def test_estimate_stiffness_rejects_isometric_data():
    displacement = np.zeros((100, 2))
    force = np.ones((100, 2))

    with pytest.raises(ValueError, match="measurable displacement"):
        estimate_stiffness(displacement, force, epoch=50)


def test_estimate_stiffness_is_invariant_to_coordinate_offset():
    displacement = np.linspace(-1.0, 1.0, 100) + 1e9
    force = 250 * displacement - 4

    result = estimate_stiffness(displacement, force, epoch=50)

    np.testing.assert_allclose(result, [[250.0], [250.0]], rtol=1e-7)


def test_estimate_stiffness_honors_measurement_resolution():
    displacement = np.linspace(0.0, 1e-6, 100)
    force = 100 * displacement

    with pytest.raises(ValueError, match="measurable displacement"):
        estimate_stiffness(
            displacement,
            force,
            epoch=50,
            min_displacement_range=1e-5,
        )


def test_co_contraction_index_reports_balance():
    flexor = np.array([0.0, 1.0, 1.0, 0.5])
    extensor = np.array([0.0, 0.0, 1.0, 1.0])

    result = co_contraction_index(flexor, extensor)

    np.testing.assert_allclose(result, [0.0, 0.0, 1.0, 2.0 / 3.0])


def test_stiffness_proxy_is_total_normalized_activation():
    flexor = np.array([0.2, 0.8])
    extensor = np.array([0.4, 0.5])

    np.testing.assert_allclose(stiffness_proxy(flexor, extensor), [0.6, 1.3])


@pytest.mark.parametrize("function", [co_contraction_index, stiffness_proxy])
def test_activation_metrics_reject_negative_values(function):
    with pytest.raises(ValueError, match="non-negative"):
        function(np.array([-0.1]), np.array([0.2]))
