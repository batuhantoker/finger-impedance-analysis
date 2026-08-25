"""Tests for safe proxy summaries and measured-stiffness files."""

from pathlib import Path

import numpy as np
import pytest

from finger_impedance.analysis.force import estimate_stiffness_from_npz
from finger_impedance.analysis.stiffness import (
    load_proxy_file,
    summarize_proxy_data,
    write_proxy_summary_csv,
)
from finger_impedance.analysis.stiffness import (
    main as summary_main,
)


def _proxy_data() -> dict[str, np.ndarray]:
    return {
        "labels": np.array([1, 1, 2]),
        "co_contraction_index": np.array([0.2, 0.4, 0.8]),
        "stiffness_proxy": np.array([0.5, 0.7, 1.2]),
        "force": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "force_names": np.array(["index", "middle"]),
        "force_units": np.array(["N", "N"]),
    }


def test_proxy_summary_groups_by_movement():
    rows = summarize_proxy_data(_proxy_data(), source="subject01.npz")

    assert len(rows) == 2
    assert rows[0]["movement_label"] == 1
    assert rows[0]["sample_count"] == 2
    assert rows[0]["stiffness_proxy_mean"] == pytest.approx(0.6)
    assert rows[0]["force_channel_1_name"] == "middle"
    assert rows[0]["force_channel_1_unit"] == "N"
    assert rows[1]["force_channel_1_mean"] == pytest.approx(6.0)


def test_proxy_file_uses_safe_schema(tmp_path: Path):
    path = tmp_path / "proxy.npz"
    np.savez_compressed(path, schema_version=2, **_proxy_data())

    loaded = load_proxy_file(path)

    np.testing.assert_array_equal(loaded["labels"], [1, 1, 2])


def test_summary_csv_refuses_overwrite(tmp_path: Path):
    rows = summarize_proxy_data(_proxy_data())
    output = tmp_path / "summary.csv"
    write_proxy_summary_csv(rows, output)

    with pytest.raises(FileExistsError):
        write_proxy_summary_csv(rows, output)


def test_summary_csv_requires_explicit_extension(tmp_path: Path):
    with pytest.raises(ValueError, match="end in .csv"):
        write_proxy_summary_csv(summarize_proxy_data(_proxy_data()), tmp_path / "summary")


def test_summary_cli_validates_all_outputs_before_writing(tmp_path: Path):
    input_path = tmp_path / "proxy.npz"
    csv_path = tmp_path / "summary.csv"
    plot_path = tmp_path / "summary.png"
    np.savez_compressed(input_path, schema_version=2, **_proxy_data())
    plot_path.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        summary_main([str(input_path), "--output", str(csv_path), "--plot", str(plot_path)])

    assert not csv_path.exists()
    assert plot_path.read_bytes() == b"keep"


def test_stiffness_npz_requires_displacement_and_preserves_units(tmp_path: Path):
    displacement = np.linspace(-0.01, 0.01, 100)
    input_path = tmp_path / "measurement.npz"
    output_path = tmp_path / "stiffness.npz"
    np.savez_compressed(
        input_path,
        displacement=displacement,
        force=300 * displacement + 2,
        epoch=50,
        force_unit="N",
        displacement_unit="m",
    )

    result = estimate_stiffness_from_npz(input_path, output_path)

    np.testing.assert_allclose(result, [[300.0], [300.0]])
    with np.load(output_path, allow_pickle=False) as output:
        assert output["stiffness_unit"].item() == "N/m"


def test_stiffness_output_refuses_overwrite(tmp_path: Path):
    displacement = np.linspace(0, 1, 10)
    input_path = tmp_path / "measurement.npz"
    output_path = tmp_path / "existing.npz"
    np.savez_compressed(
        input_path,
        displacement=displacement,
        force=2 * displacement,
        epoch=5,
        force_unit="N",
        displacement_unit="m",
    )
    output_path.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        estimate_stiffness_from_npz(input_path, output_path)

    assert output_path.read_bytes() == b"keep"


def test_stiffness_output_requires_npz_extension(tmp_path: Path):
    displacement = np.linspace(0, 1, 10)
    input_path = tmp_path / "measurement.npz"
    np.savez_compressed(
        input_path,
        displacement=displacement,
        force=2 * displacement,
        epoch=5,
        force_unit="N",
        displacement_unit="m",
    )

    with pytest.raises(ValueError, match="end in .npz"):
        estimate_stiffness_from_npz(input_path, tmp_path / "result")
    with pytest.raises(ValueError, match="end in .npz"):
        estimate_stiffness_from_npz(input_path, tmp_path / "result.NPZ")
