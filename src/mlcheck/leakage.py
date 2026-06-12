"""Leakage: two scans for the two leak surfaces.

  scan_source(path_or_text) -> static AST scan for preprocessing that runs BEFORE
                               the split (fit/fit_transform/fit_resample, manual
                               scaling, global-stat imputation, get_dummies).
  scan_data(train, test, target) -> runtime scan for train/test overlap, duplicate
                               rows, and a feature that almost perfectly predicts
                               the target.

Backs the `detecting-data-leakage` skill.
"""
from __future__ import annotations

import ast
import hashlib
import os

import numpy as np
import pandas as pd

from .core import Finding, Severity

SKILL = "detecting-data-leakage"

TARGET_LEAK_CORR = 0.999
_FIT_METHODS = {"fit", "fit_transform", "fit_resample", "fit_sample"}
_STAT_METHODS = {"mean", "std", "median", "var", "min", "max", "quantile", "mode"}
_SPLIT_NAMES = {"train_test_split", "TimeSeriesSplit", "GroupShuffleSplit",
                "GroupKFold", "StratifiedKFold", "KFold", "ShuffleSplit"}


def _read(path_or_text: str) -> str:
    if "\n" not in path_or_text and os.path.exists(path_or_text):
        with open(path_or_text, "r", encoding="utf-8") as fh:
            return fh.read()
    return path_or_text


def _attr_name(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return "?"


class _SourceVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.split_lines: list[int] = []
        # each entry: (lineno, code, message, fix)
        self.events: list[tuple[int, str, str, str]] = []

    def _is_split(self, func: ast.AST) -> bool:
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        return name in _SPLIT_NAMES

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if self._is_split(func):
            self.split_lines.append(node.lineno)

        if isinstance(func, ast.Attribute):
            attr = func.attr
            if attr in _FIT_METHODS:
                self.events.append((
                    node.lineno, "FIT_BEFORE_SPLIT",
                    f"`{_attr_name(func.value)}.{attr}(...)` learns from the whole dataset "
                    f"before the split.",
                    "Move the transform into a Pipeline fit AFTER the split (or inside CV).",
                ))
            elif attr == "get_dummies":
                self.events.append((
                    node.lineno, "GET_DUMMIES_BEFORE_SPLIT",
                    "`get_dummies(...)` on the full data fixes the category set using test rows.",
                    "One-hot encode inside a Pipeline (OneHotEncoder) fit on train only.",
                ))
            elif attr == "fillna" and self._args_use_stat(node):
                self.events.append((
                    node.lineno, "IMPUTE_LEAK",
                    "`fillna(...)` uses a global statistic computed over the whole dataset.",
                    "Impute with SimpleImputer inside a Pipeline fit on train only.",
                ))
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        # Manual scaling like (X - X.mean()) / X.std() before the split.
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in _STAT_METHODS):
                self.events.append((
                    node.lineno, "MANUAL_SCALE_BEFORE_SPLIT",
                    f"Arithmetic using a global statistic "
                    f"(`.{sub.func.attr}()`) computed over the whole dataset.",
                    "Scale with StandardScaler/MinMaxScaler inside a Pipeline fit on train only.",
                ))
                break
        self.generic_visit(node)

    @staticmethod
    def _args_use_stat(node: ast.Call) -> bool:
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in _STAT_METHODS):
                return True
        return False


def scan_source(path_or_text: str) -> list[Finding]:
    """Flag preprocessing that runs before the split. No ERROR/WARNING means clean."""
    source = _read(path_or_text)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding(Severity.WARNING, "PARSE_ERROR", f"could not parse source: {exc}",
                        skill=SKILL)]

    visitor = _SourceVisitor()
    visitor.visit(tree)

    if not visitor.split_lines:
        return [Finding(
            Severity.WARNING, "NO_SPLIT_FOUND",
            "No train/test split call found; cannot assess fit-before-split.",
            fix="Split your data before fitting anything, then audit the split.",
            skill="designing-validation-splits",
        )]

    first_split = min(visitor.split_lines)
    findings: list[Finding] = []
    seen: set[tuple[int, str]] = set()
    severity = {"FIT_BEFORE_SPLIT": Severity.ERROR}
    for lineno, code, message, fix in visitor.events:
        if lineno < first_split and (lineno, code) not in seen:
            seen.add((lineno, code))
            findings.append(Finding(
                severity.get(code, Severity.WARNING), code, message,
                fix=fix, skill=SKILL, location=f"line {lineno}",
            ))

    if not findings:
        findings.append(Finding(
            Severity.INFO, "SOURCE_CLEAN",
            "No preprocessing runs before the split.", skill=SKILL,
        ))
    return findings


def _row_hashes(df: pd.DataFrame) -> list[str]:
    digests = []
    for _, row in df.iterrows():
        digests.append(hashlib.sha1(
            pd.util.hash_pandas_object(row, index=False).values.tobytes()
        ).hexdigest())
    return digests


def scan_data(train: pd.DataFrame, test: pd.DataFrame, target: str) -> list[Finding]:
    """Flag train/test overlap, duplicate rows, and target-correlated features."""
    findings: list[Finding] = []
    features = [c for c in train.columns if c != target]

    train_hashes = _row_hashes(train[features])
    test_hashes = _row_hashes(test[features])
    overlap = set(train_hashes) & set(test_hashes)
    if overlap:
        findings.append(Finding(
            Severity.ERROR, "TRAIN_TEST_OVERLAP",
            f"{len(overlap)} identical feature row(s) appear in BOTH train and test.",
            fix="Deduplicate the dataset BEFORE splitting; check for repeated records.",
            skill=SKILL,
        ))

    dup_count = len(train_hashes) - len(set(train_hashes))
    if dup_count > 0:
        findings.append(Finding(
            Severity.WARNING, "DUPLICATE_ROWS",
            f"{dup_count} duplicate feature row(s) within train inflate apparent data size.",
            fix="Drop or aggregate duplicates; confirm they aren't a join/export bug.",
            skill=SKILL,
        ))

    y = pd.to_numeric(train[target], errors="coerce")
    if y.notna().any() and y.nunique(dropna=True) >= 2:
        for col in features:
            x = pd.to_numeric(train[col], errors="coerce")
            if x.nunique(dropna=True) < 2 or x.isna().all():
                continue
            corr = np.corrcoef(x.fillna(x.mean()), y.fillna(y.mean()))[0, 1]
            if abs(corr) >= TARGET_LEAK_CORR:
                findings.append(Finding(
                    Severity.ERROR, "TARGET_LEAK",
                    f"Feature '{col}' has |correlation|={abs(corr):.4f} with target '{target}'.",
                    fix="Prove the feature exists with this value at prediction time, or drop it.",
                    skill=SKILL, location=col,
                ))

    if not any(f.severity >= Severity.WARNING for f in findings):
        findings.append(Finding(
            Severity.INFO, "DATA_CLEAN",
            "No train/test overlap or target-correlated feature detected.", skill=SKILL,
        ))
    return findings
