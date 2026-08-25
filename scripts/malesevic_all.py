"""Summarize dimensionless activation proxies from Malesevic schema-v2 NPZ files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from finger_impedance.analysis.stiffness import (
    summarize_proxy_files,
    write_proxy_summary_csv,
)

DEFAULT_INPUT_GLOB = "malesevic_processed/data_s*.npz"
DEFAULT_OUTPUT = Path("malesevic_activation_proxy_summary.csv")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Write a CSV summary of Malesevic dimensionless activation proxies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help=f"schema-v2 NPZ files; defaults to files matching {DEFAULT_INPUT_GLOB!r}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output path",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing CSV")
    return parser


def _input_paths(files: Sequence[Path]) -> list[Path]:
    if files:
        return [path.expanduser() for path in files]
    return sorted(path for path in Path().glob(DEFAULT_INPUT_GLOB) if path.is_file())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Malesevic activation-proxy summary CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = _input_paths(args.files)
    if not paths:
        parser.error(
            f"no Malesevic schema-v2 NPZ files matched {DEFAULT_INPUT_GLOB!r}; "
            "pass input files explicitly"
        )

    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error("input file(s) not found: " + ", ".join(map(str, missing)))
    invalid = [path for path in paths if path.suffix.lower() != ".npz"]
    if invalid:
        parser.error("input file(s) must be NPZ archives: " + ", ".join(map(str, invalid)))

    rows = summarize_proxy_files(paths)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_proxy_summary_csv(rows, output, overwrite=args.overwrite)
    print(f"Saved {len(rows)} movement summaries from {len(paths)} NPZ file(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
