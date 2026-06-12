"""Baselines: compute the dumb baseline a model must beat, and flag a missing or
trivial lift. Backs the `establishing-baselines` skill.
"""
from __future__ import annotations

import numpy as np

from .core import Finding, Severity

SKILL = "establishing-baselines"

# Lift below this fraction of the available headroom counts as "negligible".
NEGLIGIBLE_LIFT_FRACTION = 0.02
# Lift this large on imbalanced data is more likely a leak than skill.
SUSPICIOUS_LIFT_FRACTION = 0.9
IMBALANCE_RATIO = 0.85

_LOWER_IS_BETTER = {"mae", "neg_mae", "rmse", "mse", "log_loss"}


def majority_class_share(y) -> float:
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    return counts.max() / counts.sum()


def baseline_score(X, y, task: str, metric: str) -> float:
    """Score a no-skill predictor on the SAME data and metric."""
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    metric = metric.lower()
    if task == "classification":
        if metric in ("roc_auc", "auc"):
            # A prior-rate predictor has AUC 0.5 by construction.
            return 0.5
        dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
        pred = dummy.predict(X)
        if metric in ("accuracy", "acc"):
            return float(accuracy_score(y, pred))
        if metric == "f1":
            return float(f1_score(y, pred, average="macro"))
        if metric in ("balanced_accuracy", "balanced_acc"):
            return float(balanced_accuracy_score(y, pred))
        raise ValueError(f"unsupported classification metric: {metric}")

    if task == "regression":
        dummy = DummyRegressor(strategy="mean").fit(X, y)
        pred = dummy.predict(X)
        if metric in ("r2", "r2_score"):
            return float(r2_score(y, pred))  # 0.0 for the mean predictor
        if metric in ("mae", "neg_mae"):
            return float(mean_absolute_error(y, pred))
        raise ValueError(f"unsupported regression metric: {metric}")

    raise ValueError(f"unsupported task: {task}")


def _higher_is_better(metric: str) -> bool:
    return metric.lower() not in _LOWER_IS_BETTER


def evaluate_baseline(X, y, model_score: float, task: str, metric: str = "accuracy") -> list[Finding]:
    """Compare a reported model score against the dumb baseline.

    No ERROR/WARNING (only the LIFT_OK note) means the lift is real and defensible.
    """
    findings: list[Finding] = []
    base = baseline_score(X, y, task, metric)

    if _higher_is_better(metric):
        lift = model_score - base
        headroom = max(1.0 - base, 1e-9)
        rel = lift / headroom
        if lift <= 0:
            findings.append(Finding(
                Severity.ERROR, "NO_LIFT",
                f"Model {metric}={model_score:.4f} does NOT beat the dumb baseline "
                f"({metric}={base:.4f}). The model has learned nothing useful.",
                fix="Investigate the model/features, or accept that the signal isn't there. "
                    "Report the lift over baseline, never the raw score.",
                skill=SKILL,
            ))
        elif rel < NEGLIGIBLE_LIFT_FRACTION:
            findings.append(Finding(
                Severity.WARNING, "NEGLIGIBLE_LIFT",
                f"Model {metric}={model_score:.4f} beats baseline {base:.4f} by only "
                f"{lift:.4f} ({rel:.1%} of the available headroom).",
                fix="Treat this as 'learned almost nothing'. Don't ship on this margin.",
                skill=SKILL,
            ))
        if task == "classification" and majority_class_share(y) >= IMBALANCE_RATIO and rel >= SUSPICIOUS_LIFT_FRACTION:
            findings.append(Finding(
                Severity.WARNING, "SUSPICIOUS_LIFT",
                f"Lift is {rel:.0%} of headroom on imbalanced data "
                f"(majority share {majority_class_share(y):.0%}). A jump this large is a leak signal.",
                fix="Run the leakage scan before trusting this result.",
                skill="detecting-data-leakage",
            ))
    else:  # lower is better
        if model_score >= base:
            findings.append(Finding(
                Severity.ERROR, "NO_LIFT",
                f"Model {metric}={model_score:.4f} is no better than the dumb baseline "
                f"({metric}={base:.4f}). The model has learned nothing useful.",
                fix="Report the gap to baseline, not the raw error.",
                skill=SKILL,
            ))

    if not findings:
        findings.append(Finding(
            Severity.INFO, "LIFT_OK",
            f"Model {metric}={model_score:.4f} beats baseline {base:.4f} by a defensible margin.",
            fix="Report the lift over baseline, not the raw score alone.",
            skill=SKILL,
        ))
    return findings
