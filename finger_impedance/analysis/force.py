"""Estimate quasi-static stiffness from synchronized displacement and force."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from finger_impedance.core import estimate_stiffness

SCHEMA_VERSION = 2
REQUIRED_KEYS = ("displacement", "force", "epoch", "force_unit", "displacement_unit")


def _load_unit(value: np.ndarray, key: str, path: Path) -> str:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "SU":
        raise ValueError(f"{path}: '{key}' must be a scalar string")
    unit_value = array.item()
    if isinstance(unit_value, bytes):
        try:
            unit_value = unit_value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path}: '{key}' must be valid UTF-8") from exc
    unit = str(unit_value).strip()
    if not unit:
        raise ValueError(f"{path}: '{key}' must not be empty")
    return unit


def _load_epoch(value: np.ndarray, path: Path) -> int:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{path}: 'epoch' must be a positive integer")
    epoch_value = float(array.item())
    if not np.isfinite(epoch_value) or not epoch_value.is_integer() or epoch_value <= 0:
        raise ValueError(f"{path}: 'epoch' must be a positive integer")
    return int(epoch_value)


def _load_signal(value: np.ndarray, key: str, path: Path) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (1, 2) or array.shape[0] == 0:
        raise ValueError(f"{path}: '{key}' must have shape (samples,) or (samples, channels)")
    if array.ndim == 2 and array.shape[1] == 0:
        raise ValueError(f"{path}: '{key}' must contain at least one channel")
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: '{key}' must contain finite numeric values")
    return array.astype(float, copy=False)


def load_force_displacement_file(path: str | Path) -> dict[str, np.ndarray | int | str]:
    """Load the synchronized signals, epoch, and explicit units from an NPZ file."""
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"{input_path}: missing required keys: {', '.join(missing)}")
        try:
            displacement = _load_signal(archive["displacement"], "displacement", input_path)
            force = _load_signal(archive["force"], "force", input_path)
            epoch = _load_epoch(archive["epoch"], input_path)
            force_unit = _load_unit(archive["force_unit"], "force_unit", input_path)
            displacement_unit = _load_unit(
                archive["displacement_unit"], "displacement_unit", input_path
            )
        except ValueError as exc:
            if str(exc).startswith(str(input_path)):
                raise
            raise ValueError(f"{input_path}: could not safely load required arrays: {exc}") from exc

    if displacement.shape != force.shape:
        raise ValueError(
            f"{input_path}: 'displacement' and 'force' must have identical synchronized shapes"
        )
    return {
        "displacement": displacement,
        "force": force,
        "epoch": epoch,
        "force_unit": force_unit,
        "displacement_unit": displacement_unit,
    }


def estimate_stiffness_from_npz(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    min_displacement_range: float = 0.0,
) -> np.ndarray:
    """Estimate true quasi-static stiffness and save a safe schema-v2 NPZ result."""
    inputs = load_force_displacement_file(input_path)
    displacement = inputs["displacement"]
    force = inputs["force"]
    epoch = inputs["epoch"]
    if not isinstance(displacement, np.ndarray) or not isinstance(force, np.ndarray):
        raise TypeError("validated displacement and force inputs must be arrays")
    if not isinstance(epoch, int):
        raise TypeError("validated epoch must be an integer")

    stiffness = estimate_stiffness(
        displacement,
        force,
        epoch,
        min_displacement_range=min_displacement_range,
    )
    if not np.all(np.isfinite(stiffness)):
        raise ValueError("stiffness estimation produced non-finite values")

    force_unit = str(inputs["force_unit"])
    displacement_unit = str(inputs["displacement_unit"])
    path = Path(output_path)
    if path.suffix != ".npz":
        raise ValueError("output path must end in .npz")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        stiffness=stiffness,
        epoch=np.asarray(epoch, dtype=np.int64),
        force_unit=np.asarray(force_unit),
        displacement_unit=np.asarray(displacement_unit),
        stiffness_unit=np.asarray(f"{force_unit}/{displacement_unit}"),
        min_displacement_range=np.asarray(min_displacement_range, dtype=np.float64),
    )
    return stiffness


def build_parser() -> argparse.ArgumentParser:
    """Build the measured quasi-static stiffness argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate quasi-static stiffness from synchronized displacement and force in an NPZ "
            "file. The input must include epoch, force_unit, and displacement_unit."
        )
    )
    parser.add_argument("input", type=Path, help="input force/displacement NPZ file")
    parser.add_argument("output", type=Path, help="output schema-v2 stiffness NPZ file")
    parser.add_argument(
        "--min-displacement-range",
        type=float,
        default=0.0,
        help="minimum resolvable displacement range in the declared displacement unit",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measured quasi-static stiffness CLI."""
    args = build_parser().parse_args(argv)
    stiffness = estimate_stiffness_from_npz(
        args.input,
        args.output,
        overwrite=args.overwrite,
        min_displacement_range=args.min_displacement_range,
    )
    print(f"Saved {stiffness.shape[0]} stiffness epochs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
