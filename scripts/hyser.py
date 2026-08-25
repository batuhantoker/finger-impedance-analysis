"""Build windowed Hyser 1-DoF datasets from the PhysioNet WFDB records."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from finger_impedance.core.functions import (
    co_contraction_index,
    feature_extraction,
    force_mean,
    stiffness_proxy,
)

DEFAULT_WINDOW_DURATION = 0.25
FEATURE_NAMES = ("rms", "mav", "iav", "var", "wl", "mf", "pf", "mp", "tp", "sm")
SESSION_PATTERN = re.compile(r"^subject(?P<subject>\d{2})_session(?P<session>\d+)$")
RECORD_PATTERN = re.compile(
    r"^1dof_(?P<kind>preprocess|force)_finger(?P<finger>[1-5])_sample"
    r"(?P<sample>[1-9]\d*)\.hea$"
)


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _resolve_1dof_directory(dataset_root: Path) -> Path:
    dataset_root = dataset_root.expanduser()
    one_dof_directory = (
        dataset_root if dataset_root.name == "1dof_dataset" else dataset_root / "1dof_dataset"
    )
    if not one_dof_directory.is_dir():
        raise FileNotFoundError(
            f"Hyser 1-DoF directory not found: {one_dof_directory}. "
            "Pass the PhysioNet v1.0.0 root or its 1dof_dataset directory."
        )
    return one_dof_directory


def _discover_sessions(one_dof_directory: Path) -> list[Path]:
    sessions = [
        path
        for path in one_dof_directory.iterdir()
        if path.is_dir() and SESSION_PATTERN.fullmatch(path.name)
    ]
    sessions.sort(
        key=lambda path: tuple(
            int(value) for value in SESSION_PATTERN.fullmatch(path.name).groups()
        )
    )
    if not sessions:
        raise FileNotFoundError(
            f"No subjectNN_sessionN directories found under {one_dof_directory}"
        )
    return sessions


def _discover_record_pairs(session_directory: Path) -> list[tuple[int, int, Path, Path]]:
    records: dict[str, dict[tuple[int, int], Path]] = {
        "preprocess": {},
        "force": {},
    }
    malformed: list[str] = []

    for header in session_directory.glob("*.hea"):
        match = RECORD_PATTERN.fullmatch(header.name)
        if match is None:
            if header.name.startswith(("1dof_preprocess_", "1dof_force_")):
                malformed.append(header.name)
            continue

        key = (int(match.group("finger")), int(match.group("sample")))
        records[match.group("kind")][key] = header.with_suffix("")

    if malformed:
        names = ", ".join(sorted(malformed))
        raise ValueError(f"{session_directory.name}: malformed 1-DoF header name(s): {names}")

    emg_keys = set(records["preprocess"])
    force_keys = set(records["force"])
    if not emg_keys and not force_keys:
        raise FileNotFoundError(
            f"{session_directory.name}: no 1dof_preprocess/force WFDB headers found"
        )

    missing_force = sorted(emg_keys - force_keys)
    missing_emg = sorted(force_keys - emg_keys)
    if missing_force or missing_emg:
        details = []
        if missing_force:
            details.append(
                "missing force for "
                + ", ".join(f"finger{finger}/sample{sample}" for finger, sample in missing_force)
            )
        if missing_emg:
            details.append(
                "missing preprocess EMG for "
                + ", ".join(f"finger{finger}/sample{sample}" for finger, sample in missing_emg)
            )
        raise FileNotFoundError(f"{session_directory.name}: unpaired records: {'; '.join(details)}")

    return [
        (
            finger,
            sample,
            records["preprocess"][(finger, sample)],
            records["force"][(finger, sample)],
        )
        for finger, sample in sorted(emg_keys)
    ]


def _read_record(
    record_path: Path, *, require_signal_names: bool
) -> tuple[np.ndarray, float, tuple[str, ...], tuple[str, ...]]:
    try:
        import wfdb
    except ModuleNotFoundError as exc:
        if exc.name != "wfdb":
            raise
        raise RuntimeError(
            "The 'wfdb' package is required to read the PhysioNet Hyser records"
        ) from exc

    try:
        record: Any = wfdb.rdrecord(str(record_path))
    except Exception as exc:
        raise ValueError(f"Could not read WFDB record '{record_path}': {exc}") from exc

    physical_signal = getattr(record, "p_signal", None)
    if physical_signal is None:
        raise ValueError(f"Malformed WFDB record '{record_path}': p_signal is missing")
    signal = np.asarray(physical_signal, dtype=float)
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
        raise ValueError(f"Malformed WFDB record '{record_path}': expected a non-empty 2D p_signal")
    if not np.all(np.isfinite(signal)):
        raise ValueError(f"Malformed WFDB record '{record_path}': p_signal is not finite")

    try:
        sampling_frequency = float(record.fs)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed WFDB record '{record_path}': invalid fs") from exc
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError(f"Malformed WFDB record '{record_path}': fs must be positive and finite")

    raw_signal_names = getattr(record, "sig_name", None)
    if raw_signal_names is None:
        if require_signal_names:
            raise ValueError(f"Malformed WFDB record '{record_path}': sig_name is missing")
        signal_names: tuple[str, ...] = ()
    else:
        signal_names = tuple(str(name) for name in raw_signal_names)
        if len(signal_names) != signal.shape[1]:
            raise ValueError(
                f"Malformed WFDB record '{record_path}': sig_name count does not match p_signal"
            )

    raw_units = getattr(record, "units", None)
    units = tuple(str(unit) for unit in raw_units) if raw_units is not None else ()
    if units and len(units) != signal.shape[1]:
        raise ValueError(
            f"Malformed WFDB record '{record_path}': unit count does not match p_signal"
        )
    return signal, sampling_frequency, signal_names, units


def _epoch_length(window_duration: float, sampling_frequency: float, record_path: Path) -> int:
    epoch = round(window_duration * sampling_frequency)
    if epoch < 1:
        raise ValueError(
            f"Window duration {window_duration:g}s is shorter than one sample for "
            f"'{record_path}' at {sampling_frequency:g} Hz"
        )
    return epoch


def process_session(
    session_directory: Path,
    output_directory: Path,
    window_duration: float,
    *,
    overwrite: bool = False,
) -> Path:
    """Process every matched EMG/force record in one subject session."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{session_directory.name}.npz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    pairs = _discover_record_pairs(session_directory)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    force_chunks: list[np.ndarray] = []
    source_records: list[str] = []
    group_chunks: list[np.ndarray] = []
    extensor_chunks: list[np.ndarray] = []
    flexor_chunks: list[np.ndarray] = []
    expected_signal_names: tuple[str, ...] | None = None
    expected_force_channels: int | None = None
    expected_force_names: tuple[str, ...] | None = None
    expected_force_units: tuple[str, ...] | None = None
    expected_emg_fs: float | None = None
    expected_force_fs: float | None = None
    expected_emg_epoch: int | None = None
    expected_force_epoch: int | None = None

    for group_id, (finger, _sample, emg_path, force_path) in enumerate(pairs):
        emg, emg_fs, signal_names, _ = _read_record(emg_path, require_signal_names=True)
        force, force_fs, force_names, force_units = _read_record(
            force_path, require_signal_names=False
        )
        force_names = force_names or tuple(f"force_{index + 1}" for index in range(force.shape[1]))
        force_units = force_units or ("unknown",) * force.shape[1]

        if expected_signal_names is None:
            expected_signal_names = signal_names
        elif signal_names != expected_signal_names:
            raise ValueError(
                f"{session_directory.name}: EMG channel names/order differ in '{emg_path.name}'"
            )
        if expected_force_channels is None:
            expected_force_channels = force.shape[1]
            expected_force_names = force_names
            expected_force_units = force_units
        elif force.shape[1] != expected_force_channels:
            raise ValueError(
                f"{session_directory.name}: force channel count differs in '{force_path.name}'"
            )
        elif force_names != expected_force_names:
            raise ValueError(
                f"{session_directory.name}: force channel names differ in '{force_path.name}'"
            )
        elif force_units != expected_force_units:
            raise ValueError(f"{session_directory.name}: force units differ in '{force_path.name}'")

        extensor_channels = [
            index for index, name in enumerate(signal_names) if name.startswith("E")
        ]
        flexor_channels = [index for index, name in enumerate(signal_names) if name.startswith("F")]
        if not extensor_channels or not flexor_channels:
            raise ValueError(
                f"Malformed WFDB record '{emg_path}': signal names must include channels "
                "beginning with E and F"
            )

        emg_epoch = _epoch_length(window_duration, emg_fs, emg_path)
        force_epoch = _epoch_length(window_duration, force_fs, force_path)
        realized_emg_window = emg_epoch / emg_fs
        realized_force_window = force_epoch / force_fs
        if not np.isclose(realized_emg_window, realized_force_window, rtol=0, atol=1e-12):
            raise ValueError(
                f"Window duration {window_duration:g}s cannot be represented equally at "
                f"{emg_fs:g} Hz and {force_fs:g} Hz"
            )
        if expected_emg_fs is None:
            expected_emg_fs = emg_fs
            expected_force_fs = force_fs
            expected_emg_epoch = emg_epoch
            expected_force_epoch = force_epoch
        elif (
            emg_fs != expected_emg_fs
            or force_fs != expected_force_fs
            or emg_epoch != expected_emg_epoch
            or force_epoch != expected_force_epoch
        ):
            raise ValueError(f"{session_directory.name}: sampling metadata differs across records")
        duration_tolerance = max(1 / emg_fs, 1 / force_fs)
        if abs(len(emg) / emg_fs - len(force) / force_fs) > duration_tolerance:
            raise ValueError(f"Pair '{emg_path.name}' has mismatched EMG and force durations")
        try:
            extracted_features = feature_extraction(emg, emg_epoch, emg_fs)
            mean_force = force_mean(force, force_epoch)
        except ValueError as exc:
            raise ValueError(f"Could not window pair '{emg_path.name}': {exc}") from exc

        if len(extracted_features) != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature extraction for '{emg_path.name}' returned "
                f"{len(extracted_features)} arrays; expected {len(FEATURE_NAMES)}"
            )
        feature_shapes = {feature.shape for feature in extracted_features}
        if feature_shapes != {(extracted_features[0].shape)}:
            raise ValueError(f"Feature extraction returned mismatched shapes for '{emg_path.name}'")

        rms = extracted_features[0]
        features = np.concatenate(extracted_features, axis=1)
        if features.shape[0] != mean_force.shape[0]:
            raise ValueError(f"Pair '{emg_path.name}' produced mismatched window counts")
        segment_count = features.shape[0]

        feature_chunks.append(features[:segment_count])
        force_chunks.append(mean_force[:segment_count])
        label_chunks.append(np.full(segment_count, finger, dtype=np.int8))
        source_records.extend([emg_path.name] * segment_count)
        group_chunks.append(np.full(segment_count, group_id, dtype=np.int32))
        extensor_chunks.append(np.mean(rms[:segment_count, extensor_channels], axis=1))
        flexor_chunks.append(np.mean(rms[:segment_count, flexor_channels], axis=1))

    features = np.concatenate(feature_chunks)
    labels = np.concatenate(label_chunks)
    mean_force = np.concatenate(force_chunks)
    group_ids = np.concatenate(group_chunks)
    extensor_activation = np.concatenate(extensor_chunks)
    flexor_activation = np.concatenate(flexor_chunks)

    extensor_reference = float(np.percentile(extensor_activation, 95))
    flexor_reference = float(np.percentile(flexor_activation, 95))
    if not np.isfinite(extensor_reference) or extensor_reference <= 0:
        raise ValueError(
            f"{session_directory.name}: extensor 95th-percentile activation must be positive"
        )
    if not np.isfinite(flexor_reference) or flexor_reference <= 0:
        raise ValueError(
            f"{session_directory.name}: flexor 95th-percentile activation must be positive"
        )

    normalized_extensor = extensor_activation / extensor_reference
    normalized_flexor = flexor_activation / flexor_reference
    co_contraction = co_contraction_index(normalized_flexor, normalized_extensor)
    proxy = stiffness_proxy(normalized_flexor, normalized_extensor)

    session_match = SESSION_PATTERN.fullmatch(session_directory.name)
    if session_match is None:
        raise ValueError(f"Invalid subject/session directory name: {session_directory.name}")
    subject_id = int(session_match.group("subject"))
    session_id = int(session_match.group("session"))

    feature_names = np.asarray(
        [
            f"{feature_name}:{signal_name}"
            for feature_name in FEATURE_NAMES
            for signal_name in expected_signal_names or ()
        ]
    )
    if (
        expected_emg_fs is None
        or expected_force_fs is None
        or expected_emg_epoch is None
        or expected_force_epoch is None
    ):
        raise RuntimeError(f"{session_directory.name}: no sampling metadata was collected")
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(2, dtype=np.int64),
        label_space=np.asarray("hyser_1dof_finger_v1"),
        subject_id=np.asarray(subject_id, dtype=np.int16),
        session_id=np.asarray(session_id, dtype=np.int16),
        window_duration_seconds=np.asarray(expected_emg_epoch / expected_emg_fs),
        emg_sampling_frequency=np.asarray(expected_emg_fs, dtype=np.float64),
        force_sampling_frequency=np.asarray(expected_force_fs, dtype=np.float64),
        emg_epoch_samples=np.asarray(expected_emg_epoch, dtype=np.int64),
        force_epoch_samples=np.asarray(expected_force_epoch, dtype=np.int64),
        features=features,
        feature_names=feature_names,
        labels=labels,
        force=mean_force,
        force_names=np.asarray(expected_force_names or ()),
        force_units=np.asarray(expected_force_units or ()),
        source_record=np.asarray(source_records),
        group_ids=group_ids,
        extensor_activation=normalized_extensor,
        flexor_activation=normalized_flexor,
        co_contraction_index=co_contraction,
        stiffness_proxy=proxy,
        extensor_activation_reference=np.asarray(extensor_reference, dtype=np.float64),
        flexor_activation_reference=np.asarray(flexor_reference, dtype=np.float64),
    )
    return output_path


def process_dataset(
    dataset_root: Path,
    output_directory: Path,
    window_duration: float = DEFAULT_WINDOW_DURATION,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Process all discovered Hyser 1-DoF subject/session directories."""
    if not np.isfinite(window_duration) or window_duration <= 0:
        raise ValueError("window duration must be a positive finite number")

    one_dof_directory = _resolve_1dof_directory(dataset_root)
    sessions = _discover_sessions(one_dof_directory)
    output_directory = output_directory.expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        existing = [output_directory / f"{session.name}.npz" for session in sessions]
        existing = [path for path in existing if path.exists()]
        if existing:
            raise FileExistsError("Output file(s) already exist: " + ", ".join(map(str, existing)))

    output_paths = []
    for session_directory in sessions:
        print(f"Processing {session_directory.name}...")
        output_paths.append(
            process_session(
                session_directory,
                output_directory,
                window_duration,
                overwrite=overwrite,
            )
        )
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create safe, windowed NPZ files from the Hyser v1.0.0 1-DoF WFDB records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="PhysioNet Hyser v1.0.0 root, or its 1dof_dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hyser_processed"),
        help="Directory for one NPZ file per subject/session",
    )
    parser.add_argument(
        "--window-duration",
        type=_positive_float,
        default=DEFAULT_WINDOW_DURATION,
        metavar="SECONDS",
        help="Non-overlapping EMG and force window duration",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing NPZ outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_paths = process_dataset(
            args.dataset_root,
            args.output_dir,
            args.window_duration,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    for output_path in output_paths:
        print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
