"""Splits: audit an existing train/test split for time, group, and distribution
leakage. Backs the `designing-validation-splits` skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Finding, Severity

SKILL = "designing-validation-splits"

SKEW_TOLERANCE = 0.10        # absolute positive-rate drift that signals no stratification
SMALL_TEST_FRACTION = 0.05   # test set smaller than this is too small to trust


def _check_size(train: pd.DataFrame, test: pd.DataFrame) -> list[Finding]:
    total = len(train) + len(test)
    if total == 0:
        return []
    frac = len(test) / total
    if frac < SMALL_TEST_FRACTION:
        return [Finding(
            Severity.WARNING, "SMALL_TEST",
            f"Test set is only {frac:.1%} of the data ({len(test)} rows). "
            f"The metric will be noisy and unreliable.",
            fix="Use at least 10-20% for test, or k-fold cross-validation on small data.",
            skill=SKILL,
        )]
    return []


def _native(value):
    """Plain Python scalar for clean display (avoids `np.int64(...)` reprs)."""
    return value.item() if hasattr(value, "item") else value


def _check_time(train: pd.DataFrame, test: pd.DataFrame, time_col: str) -> list[Finding]:
    train_max = train[time_col].max()
    test_min = test[time_col].min()
    if pd.isna(train_max) or pd.isna(test_min):
        return []
    if test_min <= train_max:
        n_overlap = int((test[time_col] <= train_max).sum())
        return [Finding(
            Severity.ERROR, "TIME_LEAK",
            f"Test is not strictly after train on '{time_col}': train max={_native(train_max)} "
            f">= test min={_native(test_min)}. {n_overlap} test rows occur at/before the train "
            f"horizon - the model can train on the future.",
            fix="Sort by time and split chronologically (sklearn TimeSeriesSplit); never shuffle.",
            skill=SKILL, location=time_col,
        )]
    return []


def _check_group(train: pd.DataFrame, test: pd.DataFrame, group_col: str) -> list[Finding]:
    shared = set(train[group_col].unique()) & set(test[group_col].unique())
    if shared:
        sample = [_native(v) for v in sorted(shared, key=str)[:5]]
        return [Finding(
            Severity.ERROR, "GROUP_LEAK",
            f"{len(shared)} value(s) of '{group_col}' appear in BOTH train and test "
            f"(e.g. {sample}). The same entity spans the split.",
            fix=f"Split on the entity id with GroupKFold/GroupShuffleSplit('{group_col}').",
            skill=SKILL, location=group_col,
        )]
    return []


def _check_target(train: pd.DataFrame, test: pd.DataFrame, target: str) -> list[Finding]:
    findings: list[Finding] = []
    train_classes = set(train[target].unique())
    test_classes = set(test[target].unique())
    missing = train_classes - test_classes
    if missing:
        findings.append(Finding(
            Severity.ERROR, "MISSING_CLASS",
            f"Class(es) {sorted(missing, key=str)} present in train are absent from test.",
            fix=f"Use a stratified split on '{target}' (StratifiedKFold).",
            skill=SKILL, location=target,
        ))
    if len(train_classes | test_classes) == 2:
        pos = max(train_classes | test_classes, key=str)
        train_rate = float((train[target] == pos).mean())
        test_rate = float((test[target] == pos).mean())
        if abs(train_rate - test_rate) > SKEW_TOLERANCE:
            findings.append(Finding(
                Severity.WARNING, "DISTRIBUTION_SKEW",
                f"Positive rate differs between splits: train={train_rate:.2%} vs test={test_rate:.2%}.",
                fix=f"Stratify on '{target}' so both sides share the class balance.",
                skill=SKILL, location=target,
            ))
    return findings


def audit_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    time_col: str | None = None,
    group_col: str | None = None,
    target: str | None = None,
) -> list[Finding]:
    """Audit a train/test split. No ERROR/WARNING (only SPLIT_OK) means clean."""
    findings: list[Finding] = []
    findings += _check_size(train, test)
    if time_col is not None:
        findings += _check_time(train, test, time_col)
    if group_col is not None:
        findings += _check_group(train, test, group_col)
    if target is not None:
        findings += _check_target(train, test, target)

    if not any(f.severity >= Severity.WARNING for f in findings):
        findings.append(Finding(
            Severity.INFO, "SPLIT_OK",
            "No time, group, or distribution leakage detected for the columns checked.",
            skill=SKILL,
        ))
    return findings
