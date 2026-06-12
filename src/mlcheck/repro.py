"""Reproducibility: a source scan for unseeded randomness, plus a data fingerprint.

Backs the `ensuring-reproducibility` skill. An unseeded run can't be reproduced,
compared, or debugged — its number is a one-off anecdote.
"""
from __future__ import annotations

import ast
import hashlib
import os

from .core import Finding, Severity

SKILL = "ensuring-reproducibility"

# Stochastic sklearn callables that should be pinned with random_state=.
_NEEDS_RANDOM_STATE = {
    "train_test_split", "KFold", "ShuffleSplit", "StratifiedShuffleSplit",
    "GroupShuffleSplit", "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor", "GradientBoostingClassifier",
    "GradientBoostingRegressor", "KMeans", "MiniBatchKMeans", "TSNE",
    "DecisionTreeClassifier", "DecisionTreeRegressor", "MLPClassifier", "MLPRegressor",
    "permutation_test_score", "SGDClassifier", "SGDRegressor",
}


def _read(path_or_text: str) -> str:
    if "\n" not in path_or_text and os.path.exists(path_or_text):
        with open(path_or_text, "r", encoding="utf-8") as fh:
            return fh.read()
    return path_or_text


def _call_name(func: ast.AST) -> str | None:
    return getattr(func, "id", None) or getattr(func, "attr", None)


def _has_kwarg(node: ast.Call, name: str) -> bool:
    return any(kw.arg == name for kw in node.keywords)


def scan_reproducibility(path_or_text: str) -> list[Finding]:
    """Flag unseeded randomness. No ERROR/WARNING (only REPRO_OK) means reproducible."""
    source = _read(path_or_text)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding(Severity.WARNING, "PARSE_ERROR", f"could not parse source: {exc}",
                        skill=SKILL)]

    findings: list[Finding] = []
    seeds_global = False
    uses_np_random = False
    np_random_seeded = False
    imports_torch = False
    seeds_torch = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] == "torch":
                    imports_torch = True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "torch":
            imports_torch = True

        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            full = ""
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                full = f"{node.func.value.id}.{node.func.attr}"

            if name in ("seed", "manual_seed") or full in ("np.random.seed",):
                seeds_global = True
            if full in ("np.random.seed",) or name == "default_rng":
                np_random_seeded = True
            if name == "manual_seed":
                seeds_torch = True

            if full.startswith("np.random.") and node.func.attr not in ("seed", "default_rng"):
                uses_np_random = True

            if name in _NEEDS_RANDOM_STATE:
                if not (_has_kwarg(node, "random_state") or _has_kwarg(node, "seed")):
                    findings.append(Finding(
                        Severity.WARNING, "MISSING_SEED",
                        f"`{name}(...)` is stochastic but has no random_state - the result "
                        f"changes every run.",
                        fix=f"Pass random_state= to {name}(...) (and seed numpy/torch globally).",
                        skill=SKILL, location=f"line {node.lineno}",
                    ))

    if uses_np_random and not (np_random_seeded or seeds_global):
        findings.append(Finding(
            Severity.WARNING, "MISSING_SEED",
            "np.random is used without a seed (np.random.seed(...) or default_rng(<int>)).",
            fix="Seed numpy once: rng = np.random.default_rng(0), or np.random.seed(0).",
            skill=SKILL,
        ))
    if imports_torch and not seeds_torch:
        findings.append(Finding(
            Severity.WARNING, "MISSING_TORCH_SEED",
            "torch is imported but torch.manual_seed(...) is never called.",
            fix="Call torch.manual_seed(0) (and seed CUDA) at startup.",
            skill=SKILL,
        ))

    if not findings:
        findings.append(Finding(Severity.INFO, "REPRO_OK",
                                "Randomness appears seeded.", skill=SKILL))
    return findings


def data_fingerprint(df) -> str:
    """A stable short hash of a DataFrame's contents — log it to version a dataset."""
    import pandas as pd

    digest = hashlib.sha1(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
    return digest[:16]
