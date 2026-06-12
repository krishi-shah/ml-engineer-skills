"""mlcheck — the ML rigor toolkit.

Runnable diagnostics that catch the silent mistakes between "works in the
notebook" and "works in production": unbeaten baselines, broken validation
splits, data leakage, and metric malpractice.

Public API:
    from mlcheck import (
        Finding, Severity, format_report, worst_severity,
        evaluate_baseline, audit_split, scan_source, scan_data, advise_metric,
    )
"""
from __future__ import annotations

from .core import Finding, Severity, format_report, format_json, format_markdown, worst_severity
from .baselines import evaluate_baseline, baseline_score
from .splits import audit_split
from .leakage import scan_source, scan_data
from .metrics import advise_metric
from .detectors import (
    adversarial_validation,
    signal_is_real,
    target_leak_scan,
    can_overfit_small_sample,
)
from .repro import scan_reproducibility, data_fingerprint
from .impact import split_impact

__version__ = "0.4.0"

__all__ = [
    "Finding",
    "Severity",
    "format_report",
    "format_json",
    "format_markdown",
    "worst_severity",
    "split_impact",
    "evaluate_baseline",
    "baseline_score",
    "audit_split",
    "scan_source",
    "scan_data",
    "advise_metric",
    "adversarial_validation",
    "signal_is_real",
    "target_leak_scan",
    "can_overfit_small_sample",
    "scan_reproducibility",
    "data_fingerprint",
    "__version__",
]
