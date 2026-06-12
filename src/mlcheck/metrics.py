"""Metrics: check whether a chosen metric fits the labels. Backs the
`choosing-evaluation-metrics` skill.
"""
from __future__ import annotations

import numpy as np

from .core import Finding, Severity

SKILL = "choosing-evaluation-metrics"

IMBALANCE_RATIO = 0.80
# Excess kurtosis above this means RMSE/R² will be dominated by a few outliers.
HEAVY_TAIL_KURTOSIS = 6.0

_ACCURACY_METRICS = {"accuracy", "acc"}
_RANKING_ONLY_METRICS = {"roc_auc", "auc", "roc-auc"}
_OUTLIER_SENSITIVE = {"rmse", "mse", "r2", "r2_score"}
_IMBALANCE_SAFE = {
    "average_precision", "pr_auc", "pr-auc", "f1", "f_beta", "fbeta",
    "recall", "precision", "balanced_accuracy", "recall_at_precision",
}


def majority_class_share(y) -> float:
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


def _looks_continuous(y) -> bool:
    arr = np.asarray(y)
    return np.issubdtype(arr.dtype, np.floating) and len(np.unique(arr)) > 20


def advise_metric(y_true, metric_name: str, y_score=None) -> list[Finding]:
    """Return findings about whether `metric_name` fits `y_true`."""
    findings: list[Finding] = []
    metric = metric_name.lower().strip()
    arr = np.asarray(y_true)

    if not _looks_continuous(arr):
        share = majority_class_share(arr)
        imbalanced = share >= IMBALANCE_RATIO
        if metric in _ACCURACY_METRICS and imbalanced:
            findings.append(Finding(
                Severity.ERROR, "ACCURACY_ON_IMBALANCE",
                f"Accuracy on imbalanced labels (majority share {share:.0%}): predicting the "
                f"majority class alone scores {share:.0%} while catching none of the rare class.",
                fix="Use PR-AUC (average_precision) / recall / F-beta instead.",
                skill=SKILL,
            ))
        if metric in _RANKING_ONLY_METRICS and y_score is not None:
            findings.append(Finding(
                Severity.WARNING, "CALIBRATION_UNCHECKED",
                f"'{metric_name}' measures ranking only; thresholded decisions need calibration.",
                fix="Also report a reliability curve / Brier score, not just AUC.",
                skill=SKILL,
            ))
    else:
        if metric in _OUTLIER_SENSITIVE:
            arr_f = arr.astype(float)
            mean, std = arr_f.mean(), arr_f.std()
            if std > 0:
                kurt = float(np.mean(((arr_f - mean) / std) ** 4))
                if kurt >= HEAVY_TAIL_KURTOSIS:
                    findings.append(Finding(
                        Severity.WARNING, "OUTLIER_SENSITIVE_METRIC",
                        f"'{metric_name}' on a heavy-tailed target (kurtosis {kurt:.1f}) is "
                        f"dominated by a few extreme points.",
                        fix="Report MAE alongside; consider quantile loss if tails are asymmetric.",
                        skill=SKILL,
                    ))

    if not any(f.severity >= Severity.WARNING for f in findings):
        note = "appropriate for these labels"
        if metric in _IMBALANCE_SAFE:
            note = "appropriate (punishes errors on the rare class)"
        findings.append(Finding(
            Severity.INFO, "METRIC_OK", f"Metric '{metric_name}' is {note}.", skill=SKILL,
        ))
    return findings
