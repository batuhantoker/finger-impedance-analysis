"""Reusable plots for schema-v2 finger impedance datasets."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SCHEMA_VERSION = 2
SCHEMA_VERSION_KEY = "schema_version"
LABEL_KEYS = ("movement_id", "labels")
REQUIRED_KEYS = ("co_contraction_index", "stiffness_proxy")


@dataclass(frozen=True)
class PlotData:
    """Validated arrays used by the impedance overview plot."""

    labels: np.ndarray
    label_key: str
    co_contraction_index: np.ndarray
    stiffness_proxy: np.ndarray
    force: np.ndarray | None = None
    force_names: tuple[str, ...] | None = None
    force_units: tuple[str, ...] | None = None


def _validate_schema_version(value: np.ndarray, path: Path) -> None:
    version = np.asarray(value)
    if version.size != 1 or str(version.item()).lower().removeprefix("v") not in {"2", "2.0"}:
        raise ValueError(f"{path}: {SCHEMA_VERSION_KEY!r} must be {SCHEMA_VERSION}")


def _validate_labels(value: np.ndarray, key: str, path: Path) -> np.ndarray:
    labels = np.asarray(value)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{path}: {key!r} must be a non-empty 1D array")
    if labels.dtype.kind in "biuf":
        if not np.all(np.isfinite(labels)):
            raise ValueError(f"{path}: {key!r} must contain finite labels")
    elif labels.dtype.kind not in "US":
        raise ValueError(f"{path}: {key!r} must contain numeric or string labels")
    return labels


def _validate_vector(
    value: np.ndarray,
    key: str,
    sample_count: int,
    path: Path,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != sample_count:
        raise ValueError(f"{path}: {key!r} must have shape ({sample_count},)")
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{path}: {key!r} must contain finite numeric values")
    return array.astype(float, copy=False)


def _validate_force(value: np.ndarray, sample_count: int, path: Path) -> np.ndarray:
    force = np.asarray(value)
    if force.ndim not in (1, 2) or force.shape[0] != sample_count:
        raise ValueError(
            f"{path}: 'force' must have shape ({sample_count},) or ({sample_count}, n)"
        )
    if force.ndim == 2 and force.shape[1] == 0:
        raise ValueError(f"{path}: 'force' must contain at least one channel")
    if force.dtype.kind not in "biuf" or not np.all(np.isfinite(force)):
        raise ValueError(f"{path}: 'force' must contain finite numeric values")
    return force.astype(float, copy=False)


def _validate_force_labels(
    value: np.ndarray, key: str, channel_count: int, path: Path
) -> tuple[str, ...]:
    labels = np.asarray(value)
    if labels.ndim != 1 or len(labels) != channel_count or labels.dtype.kind not in "US":
        raise ValueError(f"{path}: '{key}' must contain one string per force channel")
    return tuple(labels.astype(str))


def load_plot_data(path: str | Path) -> PlotData:
    """Load and validate one schema-v2 NPZ file without enabling pickle."""
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        if SCHEMA_VERSION_KEY not in archive:
            raise ValueError(f"{input_path}: missing required key {SCHEMA_VERSION_KEY!r}")
        _validate_schema_version(archive[SCHEMA_VERSION_KEY], input_path)

        label_key = next((key for key in LABEL_KEYS if key in archive), None)
        if label_key is None:
            choices = " or ".join(repr(key) for key in LABEL_KEYS)
            raise ValueError(f"{input_path}: missing required label key {choices}")

        missing = [key for key in REQUIRED_KEYS if key not in archive]
        if missing:
            raise ValueError(f"{input_path}: missing required keys: {', '.join(missing)}")

        labels = _validate_labels(archive[label_key], label_key, input_path)
        sample_count = labels.size
        co_contraction = _validate_vector(
            archive["co_contraction_index"],
            "co_contraction_index",
            sample_count,
            input_path,
        )
        stiffness = _validate_vector(
            archive["stiffness_proxy"],
            "stiffness_proxy",
            sample_count,
            input_path,
        )
        force = (
            _validate_force(archive["force"], sample_count, input_path)
            if "force" in archive
            else None
        )
        force_units = None
        force_names = None
        if force is not None:
            missing_force_metadata = [
                key for key in ("force_names", "force_units") if key not in archive
            ]
            if missing_force_metadata:
                raise ValueError(
                    f"{input_path}: force data requires {', '.join(missing_force_metadata)}"
                )
            channel_count = 1 if force.ndim == 1 else force.shape[1]
            force_names = _validate_force_labels(
                archive["force_names"], "force_names", channel_count, input_path
            )
            force_units = _validate_force_labels(
                archive["force_units"], "force_units", channel_count, input_path
            )

    return PlotData(labels, label_key, co_contraction, stiffness, force, force_names, force_units)


def _plot_labels(axis: Axes, samples: np.ndarray, data: PlotData) -> None:
    labels = data.labels
    if labels.dtype.kind in "US":
        names, values = np.unique(labels.astype(str), return_inverse=True)
        axis.step(samples, values, where="post")
        axis.set_yticks(np.arange(names.size), names)
    else:
        axis.step(samples, labels, where="post")
    axis.set_ylabel("Movement ID" if data.label_key == "movement_id" else "Label")


def plot_activation_metrics(
    data: PlotData,
    *,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Create aligned plots for labels, co-contraction, proxy, and force."""
    include_force = data.force is not None
    panel_count = 4 if include_force else 3
    samples = np.arange(data.labels.size)

    with plt.style.context("bmh"):
        figure, axes = plt.subplots(
            panel_count,
            1,
            figsize=(12, 2.4 * panel_count),
            sharex=True,
            constrained_layout=True,
        )

    axes = np.asarray(axes)
    _plot_labels(axes[0], samples, data)
    axes[1].plot(samples, data.co_contraction_index, color="tab:blue")
    axes[1].set_ylabel("Co-contraction index")
    axes[2].plot(samples, data.stiffness_proxy, color="tab:orange")
    axes[2].set_ylabel("Stiffness proxy\n(dimensionless)")

    if data.force is not None:
        if data.force_names is None or data.force_units is None:
            raise ValueError("force names and units are required when force data is present")
        if data.force.ndim == 1:
            axes[3].plot(
                samples,
                data.force,
                color="tab:green",
                label=data.force_names[0],
            )
            force_label = data.force_names[0]
        else:
            for channel, values in enumerate(data.force.T, start=1):
                unit = data.force_units[channel - 1]
                name = data.force_names[channel - 1]
                axes[3].plot(samples, values, label=f"{name} [{unit}]")
            if data.force.shape[1] > 1:
                axes[3].legend(loc="best")
            force_label = "Force"
        unit_label = data.force_units[0] if len(set(data.force_units)) == 1 else "source units"
        axes[3].set_ylabel(f"{force_label} [{unit_label}]")

    axes[-1].set_xlabel("Epoch")
    if title:
        figure.suptitle(title)
    return figure, axes


def save_figure(
    figure: Figure,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a figure to an explicit path, refusing accidental replacement."""
    path = Path(output_path)
    if not path.suffix:
        raise ValueError("output path must include a file extension")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot movement and activation metrics from one schema-v2 NPZ file."
    )
    parser.add_argument("input", type=Path, help="input schema-v2 NPZ file")
    parser.add_argument("--output", "-o", required=True, type=Path, help="output figure path")
    parser.add_argument("--title", help="optional figure title")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output if it already exists",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the schema-v2 plotting command."""
    args = parse_args(argv)
    data = load_plot_data(args.input)
    figure, _ = plot_activation_metrics(data, title=args.title)
    try:
        save_figure(figure, args.output, overwrite=args.overwrite)
    finally:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
