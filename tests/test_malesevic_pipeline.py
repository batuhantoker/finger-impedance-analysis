"""Tests for the Malesevic schema-v2 converter."""

from pathlib import Path

import mat73
import numpy as np

from finger_impedance.classification.emg_classifier import load_feature_files
from scripts import main as malesevic


def test_process_subject_preserves_timeline_and_writes_activation_metrics(
    tmp_path: Path,
    monkeypatch,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "s1.mat").touch()

    rng = np.random.default_rng(4)
    sample_count = 1024
    labels = np.repeat([0, 1, 2, 3], sample_count // 4)
    source = {
        "class": labels,
        "emg_extensors": rng.normal(size=(sample_count, 8, 8)),
        "emg_flexors": rng.normal(size=(sample_count, 8, 8)),
        "force": rng.normal(size=(sample_count, 2)),
    }
    monkeypatch.setattr(mat73, "loadmat", lambda _: source)

    path = malesevic.process_subject(
        input_dir,
        output_dir,
        1,
        fs=512,
        epoch=128,
        lowcut=15,
        highcut=200,
    )

    with np.load(path, allow_pickle=False) as data:
        assert data["schema_version"].item() == 2
        assert data["movement_id"].shape == (8,)
        assert data["rms_ext"].shape == (8, 64)
        assert data["force"].shape == (8, 2)
        np.testing.assert_array_equal(data["force_units"], ["raw", "raw"])
        assert "stiffness" not in data
        assert data["stiffness_proxy"].shape == (8,)
        assert np.all((0 <= data["co_contraction_index"]) & (data["co_contraction_index"] <= 1))

    features, labels, groups = load_feature_files([path])
    assert features.shape == (8, 512)
    assert labels.shape == groups.shape == (8,)


def test_discover_subjects_uses_available_files(tmp_path: Path):
    for name in ("s12.mat", "s2.mat", "subject3.mat"):
        (tmp_path / name).touch()

    assert malesevic.discover_subjects(tmp_path) == [2, 12]


def test_process_subject_rejects_wrong_emg_sample_axis(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "s1.mat").touch()
    source = {
        "class": np.arange(4),
        "emg_extensors": np.ones((2, 2, 2)),
        "emg_flexors": np.ones((4, 2, 2)),
        "force": np.ones((4, 1)),
    }
    monkeypatch.setattr(mat73, "loadmat", lambda _: source)

    with np.testing.assert_raises_regex(ValueError, "share a sample axis"):
        malesevic.process_subject(
            input_dir,
            tmp_path / "output",
            1,
            fs=100,
            epoch=2,
            lowcut=5,
            highcut=40,
        )
