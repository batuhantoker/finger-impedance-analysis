"""Hyser entry point for the generic dimensionless proxy summary CLI."""

from collections.abc import Sequence

from finger_impedance.analysis.stiffness import main as proxy_summary_main


def main(argv: Sequence[str] | None = None) -> int:
    """Summarize schema-v2 Hyser proxy files by movement label."""
    return proxy_summary_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
