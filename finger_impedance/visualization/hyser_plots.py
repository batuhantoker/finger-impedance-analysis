"""Compatibility entry point for the generic schema-v2 plotting tools."""

from finger_impedance.visualization.figure_plots import (
    PlotData,
    load_plot_data,
    main,
    parse_args,
    plot_activation_metrics,
    save_figure,
)

__all__ = [
    "PlotData",
    "load_plot_data",
    "main",
    "parse_args",
    "plot_activation_metrics",
    "save_figure",
]


if __name__ == "__main__":
    raise SystemExit(main())
