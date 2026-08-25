"""Create per-subject EMG feature and activation-proxy datasets."""

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from finger_impedance.core.functions import (
    class_map,
    co_contraction_index,
    data_preprocess,
    feature_extraction,
    force_mean,
    stiffness_proxy,
)

FEATURE_NAMES = ("rms", "mav", "iav", "var", "wl", "mf", "pf", "mp", "tp", "sm")
SUBJECT_PATTERN = re.compile(r"^s(?P<subject>[1-9]|1[0-9]|20)\.mat$")
FS = 2048
LOWCUT = 15
HIGHCUT = 350
EPOCH = 256


def extract_features(
    data: np.ndarray,
    muscle: str,
    *,
    epoch: int,
    fs: float,
) -> dict[str, np.ndarray]:
    """Extract all epoch features and add the muscle suffix to their names."""
    values = feature_extraction(data, epoch, fs)
    return {f"{name}_{muscle}": value for name, value in zip(FEATURE_NAMES, values)}


def process_subject(
    input_dir: Path,
    output_dir: Path,
    subject: int,
    *,
    fs: float = FS,
    epoch: int = EPOCH,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    force_scale: float = 1.0,
    force_offset: float = 0.0,
    force_unit: str = "raw",
    overwrite: bool = False,
) -> Path:
    """Process one subject and return the generated archive path."""
    import mat73

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"s{subject}.mat"
    output_path = output_dir / f"data_s{subject}.npz"
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if not np.isfinite(fs) or fs <= 0 or epoch <= 0:
        raise ValueError("sampling frequency and epoch must be positive")
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("cutoffs must satisfy 0 < lowcut < highcut < Nyquist")
    if not np.isfinite(force_scale) or not np.isfinite(force_offset):
        raise ValueError("force scale and offset must be finite")
    if not force_unit.strip():
        raise ValueError("force unit must not be empty")

    data = mat73.loadmat(str(input_path))
    required_keys = {"class", "emg_extensors", "emg_flexors", "force"}
    missing = required_keys.difference(data)
    if missing:
        raise ValueError(f"{input_path}: missing fields: {', '.join(sorted(missing))}")

    labels = np.asarray(data["class"], dtype=float)
    if labels.ndim == 2 and 1 in labels.shape:
        labels = labels.reshape(-1)
    if labels.ndim != 1:
        raise ValueError(f"{input_path}: class labels must be a vector")
    extensor_grid = np.asarray(data["emg_extensors"], dtype=float)
    flexor_grid = np.asarray(data["emg_flexors"], dtype=float)
    force = np.asarray(data["force"], dtype=float)
    if extensor_grid.ndim != 3 or flexor_grid.ndim != 3:
        raise ValueError(f"{input_path}: EMG arrays must have shape (samples, rows, columns)")
    if extensor_grid.shape[0] != len(labels) or flexor_grid.shape[0] != len(labels):
        raise ValueError(f"{input_path}: EMG and class arrays must share a sample axis")
    extensor = extensor_grid.reshape(len(labels), -1)
    flexor = flexor_grid.reshape(len(labels), -1)
    if force.ndim == 1:
        force = force[:, np.newaxis]
    if force.ndim != 2:
        raise ValueError(f"{input_path}: force must have shape (samples, channels)")
    if not (len(extensor) == len(flexor) == len(force) == len(labels)):
        raise ValueError(f"{input_path}: EMG, force, and class arrays must share a sample axis")
    if not all(np.all(np.isfinite(values)) for values in (labels, extensor, flexor, force)):
        raise ValueError(f"{input_path}: input arrays must contain only finite values")

    extensor = data_preprocess(extensor, fs, lowcut, highcut)
    flexor = data_preprocess(flexor, fs, lowcut, highcut)
    force = force * force_scale + force_offset
    if not np.all(np.isfinite(force)):
        raise ValueError(f"{input_path}: calibrated force contains non-finite values")

    extensor_features = extract_features(extensor, "ext", epoch=epoch, fs=fs)
    flexor_features = extract_features(flexor, "flex", epoch=epoch, fs=fs)
    epoch_force = force_mean(force, epoch)
    epoch_labels = class_map(labels, epoch)
    valid_epochs = np.isfinite(epoch_labels) & np.isin(epoch_labels, np.arange(13))
    if not np.any(valid_epochs):
        raise ValueError(f"Subject {subject} has no pure-label epochs")

    epoch_count = epoch_labels.size
    epoch_arrays = (*extensor_features.values(), *flexor_features.values(), epoch_force)
    if any(array.shape[0] != epoch_count for array in epoch_arrays):
        raise ValueError(f"Subject {subject} has inconsistent epoch counts")

    extensor_activation = np.mean(extensor_features["rms_ext"], axis=1)
    flexor_activation = np.mean(flexor_features["rms_flex"], axis=1)
    extensor_reference = float(np.percentile(extensor_activation[valid_epochs], 95))
    flexor_reference = float(np.percentile(flexor_activation[valid_epochs], 95))
    if not np.isfinite(extensor_reference) or extensor_reference <= 0:
        raise ValueError(f"Subject {subject} has a non-positive extensor activation reference")
    if not np.isfinite(flexor_reference) or flexor_reference <= 0:
        raise ValueError(f"Subject {subject} has a non-positive flexor activation reference")

    normalized_extensor = extensor_activation[valid_epochs] / extensor_reference
    normalized_flexor = flexor_activation[valid_epochs] / flexor_reference
    output = {
        **{name: values[valid_epochs] for name, values in flexor_features.items()},
        **{name: values[valid_epochs] for name, values in extensor_features.items()},
        "movement_id": epoch_labels[valid_epochs],
        "force": epoch_force[valid_epochs],
        "co_contraction_index": co_contraction_index(normalized_flexor, normalized_extensor),
        "stiffness_proxy": stiffness_proxy(normalized_flexor, normalized_extensor),
        "extensor_activation_reference": np.float64(extensor_reference),
        "flexor_activation_reference": np.float64(flexor_reference),
        "schema_version": np.int64(2),
        "label_space": np.asarray("malesevic_movement_v1"),
        "emg_layout": np.asarray("separate_flexor_extensor_grids"),
        "extensor_grid_shape": np.asarray(extensor_grid.shape[1:], dtype=np.int64),
        "flexor_grid_shape": np.asarray(flexor_grid.shape[1:], dtype=np.int64),
        "filter_lowcut_hz": np.float64(lowcut),
        "filter_highcut_hz": np.float64(highcut),
        "subject_id": np.int64(subject),
        "emg_sampling_frequency": np.float64(fs),
        "force_sampling_frequency": np.float64(fs),
        "emg_epoch_samples": np.int64(epoch),
        "force_epoch_samples": np.int64(epoch),
        "window_duration_seconds": np.float64(epoch / fs),
        "force_scale": np.float64(force_scale),
        "force_offset": np.float64(force_offset),
        "force_names": np.asarray([f"force_{index + 1}" for index in range(force.shape[1])]),
        "force_units": np.full(force.shape[1], force_unit),
    }

    np.savez_compressed(output_path, **output)
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract Malesevic EMG features and dimensionless activation proxies."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing s<subject>.mat files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for data_s<subject>.npz files",
    )
    parser.add_argument(
        "--subject",
        action="append",
        type=int,
        choices=range(1, 21),
        help="subject number to process; may be repeated (default: discover available files)",
    )
    parser.add_argument("--sampling-frequency", type=float, default=FS, help="EMG sampling rate")
    parser.add_argument("--epoch", type=int, default=EPOCH, help="samples per feature epoch")
    parser.add_argument("--lowcut", type=float, default=LOWCUT, help="bandpass low cutoff in Hz")
    parser.add_argument("--highcut", type=float, default=HIGHCUT, help="bandpass high cutoff in Hz")
    parser.add_argument("--force-scale", type=float, default=1.0)
    parser.add_argument("--force-offset", type=float, default=0.0)
    parser.add_argument("--force-unit", default="raw", help="unit after force calibration")
    parser.add_argument("--overwrite", action="store_true", help="replace existing NPZ outputs")
    return parser.parse_args(argv)


def discover_subjects(input_dir: Path) -> list[int]:
    """Return subject IDs represented by valid s<number>.mat filenames."""
    subjects = [
        int(match.group("subject"))
        for path in input_dir.glob("s*.mat")
        if (match := SUBJECT_PATTERN.fullmatch(path.name)) is not None
    ]
    if not subjects:
        raise FileNotFoundError(f"No s<subject>.mat files found in {input_dir}")
    return sorted(set(subjects))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dataset conversion CLI."""
    args = parse_args(argv)
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")
    if not np.isfinite(args.sampling_frequency) or args.sampling_frequency <= 0 or args.epoch <= 0:
        raise ValueError("sampling frequency and epoch must be positive")
    if not 0 < args.lowcut < args.highcut < args.sampling_frequency / 2:
        raise ValueError("cutoffs must satisfy 0 < lowcut < highcut < Nyquist")
    if not args.force_unit.strip():
        raise ValueError("force unit must not be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subject or discover_subjects(args.input_dir)
    if not args.overwrite:
        existing = [args.output_dir / f"data_s{subject}.npz" for subject in subjects]
        existing = [path for path in existing if path.exists()]
        if existing:
            raise FileExistsError("Output file(s) already exist: " + ", ".join(map(str, existing)))

    for subject in subjects:
        print(f"Processing subject {subject}")
        output_path = process_subject(
            args.input_dir,
            args.output_dir,
            subject,
            fs=args.sampling_frequency,
            epoch=args.epoch,
            lowcut=args.lowcut,
            highcut=args.highcut,
            force_scale=args.force_scale,
            force_offset=args.force_offset,
            force_unit=args.force_unit,
            overwrite=args.overwrite,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
