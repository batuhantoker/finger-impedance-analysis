"""Safe NPZ-based proxy summaries and measured stiffness analysis."""

from importlib import import_module

__all__ = [
    "load_proxy_file",
    "summarize_proxy_data",
    "summarize_proxy_file",
    "summarize_proxy_files",
    "write_proxy_summary_csv",
    "save_proxy_summary_plot",
    "load_force_displacement_file",
    "estimate_stiffness_from_npz",
]

_EXPORT_MODULES = {
    "load_proxy_file": "finger_impedance.analysis.stiffness",
    "summarize_proxy_data": "finger_impedance.analysis.stiffness",
    "summarize_proxy_file": "finger_impedance.analysis.stiffness",
    "summarize_proxy_files": "finger_impedance.analysis.stiffness",
    "write_proxy_summary_csv": "finger_impedance.analysis.stiffness",
    "save_proxy_summary_plot": "finger_impedance.analysis.stiffness",
    "load_force_displacement_file": "finger_impedance.analysis.force",
    "estimate_stiffness_from_npz": "finger_impedance.analysis.force",
}


def __getattr__(name: str) -> object:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
