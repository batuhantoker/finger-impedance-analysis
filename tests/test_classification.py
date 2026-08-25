"""Tests for safe feature loading and grouped model evaluation."""

from pathlib import Path

import numpy as np
import pytest

from finger_impedance.classification import emg_classifier, image_classifier


def test_emg_loader_supports_hyser_schema_and_record_groups(tmp_path: Path):
    path = tmp_path / "hyser.npz"
    labels = np.tile([1, 2], 12)
    groups = np.repeat(np.arange(6), 4)
    features = np.column_stack((labels, labels**2)).astype(float)
    np.savez_compressed(
        path,
        schema_version=2,
        label_space="hyser_1dof_finger_v1",
        window_duration_seconds=0.25,
        emg_sampling_frequency=2048.0,
        emg_epoch_samples=512,
        subject_id=1,
        labels=labels,
        features=features,
        feature_names=np.array(["rms:ED-1", "rms:FD-1"]),
        group_ids=groups,
    )

    loaded_features, loaded_labels, loaded_groups = emg_classifier.load_feature_files(
        [path], group_by="record"
    )

    np.testing.assert_array_equal(loaded_features, features)
    np.testing.assert_array_equal(loaded_labels, labels)
    assert np.unique(loaded_groups).size == 6

    _, _, subject_groups = emg_classifier.load_feature_files([path])
    assert np.all(subject_groups == 1)


def test_emg_evaluation_uses_grouped_folds():
    labels = np.tile([0, 1], 20)
    groups = np.repeat(np.arange(10), 4)
    features = np.column_stack((labels, labels + 0.1, np.arange(labels.size) % 3))

    scores = emg_classifier.evaluate_models(features, labels, groups, n_splits=5)

    assert set(scores) == {"LogReg", "SVM"}
    assert all(values.shape == (5,) for values in scores.values())


def test_grouped_evaluation_rejects_a_single_recording():
    labels = np.tile([0, 1], 4)
    features = labels[:, None].astype(float)

    with pytest.raises(ValueError, match="at least two independent groups"):
        emg_classifier.evaluate_models(features, labels, np.zeros(labels.size))


def test_emg_loader_rejects_object_arrays(tmp_path: Path):
    path = tmp_path / "unsafe.npz"
    np.savez(path, schema_version=2, labels=np.array([1], dtype=object), features=[[1.0]])

    with pytest.raises(ValueError):
        emg_classifier.load_feature_files([path])


def test_emg_loader_rejects_incompatible_window_metadata(tmp_path: Path):
    paths = []
    for index, duration in enumerate((0.25, 0.5)):
        path = tmp_path / f"hyser-{index}.npz"
        np.savez_compressed(
            path,
            schema_version=2,
            label_space="hyser_1dof_finger_v1",
            window_duration_seconds=duration,
            emg_sampling_frequency=2048.0,
            emg_epoch_samples=round(duration * 2048),
            subject_id=index + 1,
            labels=np.array([1, 2]),
            features=np.ones((2, 2)),
            feature_names=np.array(["rms:ED-1", "rms:FD-1"]),
        )
        paths.append(path)

    with pytest.raises(ValueError, match="feature metadata signature"):
        emg_classifier.load_feature_files(paths)


def test_image_loader_requires_fixed_shape_schema(tmp_path: Path):
    path = tmp_path / "images.npz"
    arrays = {key: np.ones((8, 4, 4), dtype=float) for key in image_classifier.FEATURE_KEYS}
    np.savez_compressed(
        path,
        schema_version=2,
        label_space="test_movements_v1",
        image_feature_signature="canny-harris-v1",
        movement_id=np.tile([0, 1], 4),
        group_ids=np.repeat([0, 1], 4),
        **arrays,
    )

    features, labels, groups = image_classifier.load_feature_files([path])

    assert features.shape == (8, 64)
    assert labels.shape == groups.shape == (8,)


def test_image_loader_rejects_missing_schema_version(tmp_path: Path):
    path = tmp_path / "images.npz"
    arrays = {key: np.ones((4, 2, 2), dtype=float) for key in image_classifier.FEATURE_KEYS}
    np.savez_compressed(path, movement_id=np.arange(4), **arrays)

    with pytest.raises(ValueError, match="schema_version"):
        image_classifier.load_feature_files([path])


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_image_evaluation_uses_valid_knn_size_for_small_folds():
    rng = np.random.default_rng(7)
    labels = np.tile([0, 1], 8)
    groups = np.repeat(np.arange(4), 4)
    features = rng.normal(size=(labels.size, 3)) + labels[:, None]

    scores = image_classifier.evaluate_models(features, labels, groups, n_splits=2)

    assert all(np.all(np.isfinite(values)) for values in scores.values())
