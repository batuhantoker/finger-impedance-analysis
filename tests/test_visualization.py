"""Tests for schema-v2 plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from finger_impedance.visualization.figure_plots import (
    load_plot_data,
    plot_activation_metrics,
    save_figure,
)


def test_plot_data_load_and_save(tmp_path: Path):
    input_path = tmp_path / "data.npz"
    output_path = tmp_path / "plots" / "overview.png"
    np.savez_compressed(
        input_path,
        schema_version=2,
        labels=np.array([1, 1, 2]),
        co_contraction_index=np.array([0.1, 0.2, 0.4]),
        stiffness_proxy=np.array([0.4, 0.5, 0.9]),
        force=np.ones((3, 2)),
        force_names=np.array(["index", "middle"]),
        force_units=np.array(["N", "N"]),
    )

    data = load_plot_data(input_path)
    figure, axes = plot_activation_metrics(data)
    try:
        save_figure(figure, output_path)
        assert output_path.is_file()
        with pytest.raises(FileExistsError):
            save_figure(figure, output_path)
    finally:
        plt.close(figure)

    assert len(axes) == 4


def test_figure_output_requires_explicit_extension(tmp_path: Path):
    data_path = tmp_path / "data.npz"
    np.savez_compressed(
        data_path,
        schema_version=2,
        labels=np.array([1]),
        co_contraction_index=np.array([0.5]),
        stiffness_proxy=np.array([1.0]),
    )
    figure, _ = plot_activation_metrics(load_plot_data(data_path))
    try:
        with pytest.raises(ValueError, match="file extension"):
            save_figure(figure, tmp_path / "plot")
    finally:
        plt.close(figure)


def test_constant_activation_image_has_stable_features():
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")
    from finger_impedance.visualization.interactive_plot import image_features

    _, _, _, resized, clusters, center, _ = image_features(np.ones((8, 8)))

    np.testing.assert_array_equal(resized, 0)
    np.testing.assert_array_equal(clusters, 0)
    np.testing.assert_allclose(center, [15.5, 15.5])
