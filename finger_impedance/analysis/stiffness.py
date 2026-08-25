"""Summarize dimensionless activation proxies from schema-v2 NPZ files."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
LABEL_KEYS = ("movement_id", "labels")
VALUE_KEYS = ("co_contraction_index", "stiffness_proxy", "force")
SUMMARY_FIELDS = (
    "file",
    "movement_label",
    "sample_count",
    "co_contraction_index_mean",
    "co_contraction_index_std",
    "stiffness_proxy_mean",
    "stiffness_proxy_std",
)

ProxyData = dict[str, np.ndarray]
SummaryValue = str | int | float
SummaryRow = dict[str, SummaryValue]


def _validate_schema_version(value: np.ndarray, path: Path) -> None:
    version = np.asarray(value)
    if version.size != 1 or str(version.item()).lower().removeprefix("v") not in {
        "2",
        "2.0",
    }:
        raise ValueError(f"{path}: 'schema_version' must be {SCHEMA_VERSION}")


def _validate_proxy_arrays(data: Mapping[str, np.ndarray], source: str) -> ProxyData:
    label_key = next((key for key in LABEL_KEYS if key in data), None)
    if label_key is None:
        raise ValueError(f"{source}: missing a label array ('movement_id' or 'labels')")

    labels = np.asarray(data[label_key])
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"{source}: '{label_key}' must be a non-empty 1D array")
    if labels.dtype.kind in "iuf":
        if not np.all(np.isfinite(labels)):
            raise ValueError(f"{source}: '{label_key}' must contain finite labels")
    elif labels.dtype.kind == "S":
        labels = labels.astype(str)
    elif labels.dtype.kind != "U":
        raise ValueError(f"{source}: '{label_key}' must contain numeric or string labels")

    arrays: ProxyData = {"labels": labels}
    for key in VALUE_KEYS:
        if key not in data:
            raise ValueError(f"{source}: missing required key: {key}")
        array = np.asarray(data[key])
        expected_dimensions = (1, 2) if key == "force" else (1,)
        if array.ndim not in expected_dimensions:
            shape_description = (
                "(samples,) or (samples, channels)" if key == "force" else "(samples,)"
            )
            raise ValueError(f"{source}: '{key}' must have shape {shape_description}")
        if array.shape[0] != labels.size:
            raise ValueError(
                f"{source}: '{key}' has {array.shape[0]} samples; expected {labels.size}"
            )
        if array.ndim == 2 and array.shape[1] == 0:
            raise ValueError(f"{source}: '{key}' must contain at least one channel")
        if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
            raise ValueError(f"{source}: '{key}' must contain finite numeric values")
        arrays[key] = array.astype(float, copy=False)

    missing_force_metadata = [key for key in ("force_names", "force_units") if key not in data]
    if missing_force_metadata:
        raise ValueError(f"{source}: missing required keys: {', '.join(missing_force_metadata)}")
    names = np.asarray(data["force_names"])
    units = np.asarray(data["force_units"])
    force_channels = 1 if arrays["force"].ndim == 1 else arrays["force"].shape[1]
    for key, values in (("force_names", names), ("force_units", units)):
        if values.ndim != 1 or len(values) != force_channels or values.dtype.kind not in "US":
            raise ValueError(f"{source}: '{key}' must contain one string per force channel")
    arrays["force_names"] = names.astype(str)
    arrays["force_units"] = units.astype(str)
    return arrays


def load_proxy_file(path: str | Path) -> ProxyData:
    """Load and validate one schema-v2 activation-proxy NPZ file.

    The returned mapping always uses ``labels`` regardless of whether the source
    archive stores ``movement_id`` or ``labels``.
    """
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        if "schema_version" not in archive:
            raise ValueError(f"{input_path}: missing required key: schema_version")
        try:
            _validate_schema_version(archive["schema_version"], input_path)
            label_key = next((key for key in LABEL_KEYS if key in archive), None)
            if label_key is None:
                raise ValueError(f"{input_path}: missing a label array ('movement_id' or 'labels')")
            raw_data = {
                label_key: archive[label_key],
                **{
                    key: archive[key]
                    for key in (*VALUE_KEYS, "force_names", "force_units")
                    if key in archive
                },
            }
        except ValueError as exc:
            if str(exc).startswith(str(input_path)):
                raise
            raise ValueError(f"{input_path}: could not safely load required arrays: {exc}") from exc
    return _validate_proxy_arrays(raw_data, str(input_path))


def summarize_proxy_data(
    data: Mapping[str, np.ndarray],
    *,
    source: str | Path = "<memory>",
) -> list[SummaryRow]:
    """Summarize one validated data set independently by movement label."""
    source_name = str(source)
    arrays = _validate_proxy_arrays(data, source_name)
    labels = arrays["labels"]
    force = arrays["force"]
    if force.ndim == 1:
        force = force[:, np.newaxis]

    rows: list[SummaryRow] = []
    for label in np.unique(labels):
        selected = labels == label
        co_contraction = arrays["co_contraction_index"][selected]
        proxy = arrays["stiffness_proxy"][selected]
        selected_force = force[selected]
        label_value = label.item() if isinstance(label, np.generic) else label
        row: SummaryRow = {
            "file": source_name,
            "movement_label": label_value,
            "sample_count": int(np.count_nonzero(selected)),
            "co_contraction_index_mean": float(np.mean(co_contraction)),
            "co_contraction_index_std": float(np.std(co_contraction)),
            "stiffness_proxy_mean": float(np.mean(proxy)),
            "stiffness_proxy_std": float(np.std(proxy)),
        }
        for channel in range(selected_force.shape[1]):
            channel_force = selected_force[:, channel]
            row[f"force_channel_{channel}_name"] = str(arrays["force_names"][channel])
            row[f"force_channel_{channel}_unit"] = str(arrays["force_units"][channel])
            row[f"force_channel_{channel}_mean"] = float(np.mean(channel_force))
            row[f"force_channel_{channel}_std"] = float(np.std(channel_force))
        rows.append(row)
    return rows


def summarize_proxy_file(path: str | Path) -> list[SummaryRow]:
    """Load one proxy file and return movement-level summary rows."""
    input_path = Path(path)
    return summarize_proxy_data(load_proxy_file(input_path), source=input_path)


def summarize_proxy_files(paths: Iterable[str | Path]) -> list[SummaryRow]:
    """Summarize files independently without aligning or truncating their samples."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    input_paths = [Path(path) for path in paths]
    if not input_paths:
        raise ValueError("at least one NPZ file is required")

    rows: list[SummaryRow] = []
    for path in input_paths:
        rows.extend(summarize_proxy_file(path))
    return rows


def _summary_fieldnames(rows: Sequence[Mapping[str, SummaryValue]]) -> list[str]:
    force_channels = {
        int(key.removeprefix("force_channel_").split("_", maxsplit=1)[0])
        for row in rows
        for key in row
        if key.startswith("force_channel_")
    }
    force_fields = [
        field
        for channel in sorted(force_channels)
        for field in (
            f"force_channel_{channel}_name",
            f"force_channel_{channel}_unit",
            f"force_channel_{channel}_mean",
            f"force_channel_{channel}_std",
        )
    ]
    return [*SUMMARY_FIELDS, *force_fields]


def _validate_output_path(
    path: Path,
    *,
    overwrite: bool,
    required_suffix: str | None = None,
) -> None:
    if required_suffix is not None and path.suffix != required_suffix:
        raise ValueError(f"output path must end in {required_suffix}")
    if required_suffix is None and not path.suffix:
        raise ValueError("plot output path must include a file extension")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {path}")


def write_proxy_summary_csv(
    rows: Sequence[Mapping[str, SummaryValue]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write movement-level proxy summaries to CSV."""
    if not rows:
        raise ValueError("summary must contain at least one row")
    path = Path(output_path)
    _validate_output_path(path, overwrite=overwrite, required_suffix=".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=_summary_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_proxy_summary_plot(
    rows: Sequence[Mapping[str, SummaryValue]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Plot the dimensionless proxy mean and spread for each file and movement."""
    if not rows:
        raise ValueError("summary must contain at least one row")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = list(dict.fromkeys(str(row["file"]) for row in rows))
    label_keys = list(
        dict.fromkeys(
            (type(row["movement_label"]).__name__, str(row["movement_label"])) for row in rows
        )
    )
    label_positions = {key: index for index, key in enumerate(label_keys)}
    offsets = np.linspace(-0.2, 0.2, len(sources)) if len(sources) > 1 else np.zeros(1)

    figure, axis = plt.subplots(figsize=(max(7.0, len(label_keys) * 0.8), 4.5))
    for source, offset in zip(sources, offsets):
        source_rows = [row for row in rows if str(row["file"]) == source]
        positions = [
            label_positions[(type(row["movement_label"]).__name__, str(row["movement_label"]))]
            + offset
            for row in source_rows
        ]
        axis.errorbar(
            positions,
            [float(row["stiffness_proxy_mean"]) for row in source_rows],
            yerr=[float(row["stiffness_proxy_std"]) for row in source_rows],
            fmt="o",
            capsize=3,
            label=Path(source).name,
        )

    axis.set_xticks(range(len(label_keys)), [label for _, label in label_keys])
    axis.set_xlabel("Movement label")
    axis.set_ylabel("Dimensionless stiffness proxy")
    axis.set_title("Activation-based proxy by movement label")
    if len(sources) > 1:
        axis.legend(title="Input file")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    path = Path(output_path)
    _validate_output_path(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def build_parser() -> argparse.ArgumentParser:
    """Build the activation-proxy summary argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize dimensionless activation-based stiffness proxies from schema-v2 NPZ files."
        )
    )
    parser.add_argument("files", nargs="+", type=Path, help="input schema-v2 NPZ files")
    parser.add_argument(
        "-o",
        "--output",
        "--csv",
        required=True,
        type=Path,
        help="output CSV path",
    )
    parser.add_argument("--plot", type=Path, help="optional proxy summary plot path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing CSV and plot outputs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the activation-proxy summary CLI."""
    args = build_parser().parse_args(argv)
    _validate_output_path(args.output, overwrite=args.overwrite, required_suffix=".csv")
    if args.plot is not None:
        _validate_output_path(args.plot, overwrite=args.overwrite)
    rows = summarize_proxy_files(args.files)
    csv_path = write_proxy_summary_csv(rows, args.output, overwrite=args.overwrite)
    print(f"Saved {len(rows)} movement summaries to {csv_path}")
    if args.plot is not None:
        plot_path = save_proxy_summary_plot(rows, args.plot, overwrite=args.overwrite)
        print(f"Saved proxy plot to {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
