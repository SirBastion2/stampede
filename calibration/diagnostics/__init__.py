"""Calibration run diagnostics (M27). Writes artifacts under /calibration/diagnostics."""

from calibration.diagnostics.metrics import (
    BindRates,
    bind_rates_from_log,
    withdrawn_share_from_log,
)
from calibration.diagnostics.recorder import (
    DEFAULT_DIAGNOSTICS_ROOT,
    GenerationData,
    attach_on_generation,
    diagnostics_dir,
    record_generation,
    write_run_diagnostics,
)
from calibration.diagnostics.summary import format_summary, print_summary

__all__ = (
    "DEFAULT_DIAGNOSTICS_ROOT",
    "BindRates",
    "GenerationData",
    "attach_on_generation",
    "diagnostics_dir",
    "bind_rates_from_log",
    "format_summary",
    "print_summary",
    "record_generation",
    "withdrawn_share_from_log",
    "write_run_diagnostics",
)
