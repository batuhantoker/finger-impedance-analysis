"""Tests for processing paired Hyser WFDB records."""

from pathlib import Path

import numpy as np
import pytest

from finger_impedance.classification.emg_classifier import load_feature_files
from scripts import hyser


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "1dof_dataset" / "subject01_session1"
    session.mkdir(parents=True)
    for kind in ("preprocess", "force"):
        (session / f"1dof_{kind}_finger1_sample1.hea").touch()
    return session


def test_process_session_aligns_windows_and_writes_safe_schema(tmp_path: Path, monkeypatch):
    session = _session(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    time = np.arange(8) / 8
    emg = np.column_stack(
        (
            np.sin(2 * np.pi * time) + 2,
            np.sin(2 * np.pi * time) + 1,
            np.cos(2 * np.pi * time) + 2,
            np.cos(2 * np.pi * time) + 1,
        )
    )
    force = np.arange(4, dtype=float)[:, None]

    def fake_read(path: Path, *, require_signal_names: bool):
        if "preprocess" in path.name:
            return emg, 8.0, ("ED-1", "EP-1", "FD-1", "FP-1"), ("V",) * 4
        return force, 4.0, ("force",), ("N",)

    monkeypatch.setattr(hyser, "_read_record", fake_read)

    path = hyser.process_session(session, output, window_duration=0.5)

    with np.load(path, allow_pickle=False) as data:
        assert data["schema_version"].item() == 2
        assert data["features"].shape == (2, 40)
        assert data["force"].shape == (2, 1)
        np.testing.assert_array_equal(data["labels"], [1, 1])
        np.testing.assert_array_equal(data["force_units"], ["N"])
        assert np.all((0 <= data["co_contraction_index"]) & (data["co_contraction_index"] <= 1))

    features, labels, groups = load_feature_files([path])
    assert features.shape == (2, 40)
    assert labels.shape == groups.shape == (2,)


def test_process_session_rejects_mismatched_record_durations(tmp_path: Path, monkeypatch):
    session = _session(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    def fake_read(path: Path, *, require_signal_names: bool):
        if "preprocess" in path.name:
            return np.ones((8, 2)), 8.0, ("ED-1", "FD-1"), ("V", "V")
        return np.ones((8, 1)), 4.0, ("force",), ("N",)

    monkeypatch.setattr(hyser, "_read_record", fake_read)

    with pytest.raises(ValueError, match="mismatched EMG and force durations"):
        hyser.process_session(session, output, window_duration=0.5)


def test_process_session_rejects_non_commensurate_windows(tmp_path: Path, monkeypatch):
    session = _session(tmp_path)

    def fake_read(path: Path, *, require_signal_names: bool):
        if "preprocess" in path.name:
            return np.ones((8, 2)), 8.0, ("ED-1", "FD-1"), ("V", "V")
        return np.ones((5, 1)), 5.0, ("force",), ("N",)

    monkeypatch.setattr(hyser, "_read_record", fake_read)

    with pytest.raises(ValueError, match="cannot be represented equally"):
        hyser.process_session(session, tmp_path / "output", window_duration=0.3)


def test_discovery_rejects_unpaired_records(tmp_path: Path):
    session = tmp_path / "subject01_session1"
    session.mkdir()
    (session / "1dof_preprocess_finger1_sample1.hea").touch()

    with pytest.raises(FileNotFoundError, match="missing force"):
        hyser._discover_record_pairs(session)
