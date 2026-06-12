"""Runtime detectors — the harder-to-fool checks that the static scanner cannot do.

These reuse scikit-learn so they are principled, not heuristic:

  adversarial_validation(train_X, test_X)   -> is train distinguishable from test?
  signal_is_real(estimator, X, y)           -> is the score better than chance? (permutation test)
  target_leak_scan(train, target)           -> does one feature predict the target nearly perfectly?
  can_overfit_small_sample(estimator, X, y) -> can the model even memorize a handful of rows?

All take a full sklearn estimator/Pipeline where relevant (so they are pipeline-aware),
return `list[Finding]`, and are deterministic via fixed random_state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Finding, Severity

LEAK_SKILL = "detecting-data-leakage"
SPLIT_SKILL = "designing-validation-splits"
BASELINE_SKILL = "establishing-baselines"
TRAIN_SKILL = "debugging-model-training"

# Adversarial AUC thresholds: above WARN the splits are distinguishable.
ADV_WARN = 0.75
ADV_ERROR = 0.90
# A single feature scoring at/above this against the target is a leak.
SINGLE_FEATURE_LEAK = 0.999
# permutation-test p-value above this == indistinguishable from chance.
NO_SIGNAL_PVALUE = 0.05


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number").copy()
    return num.fillna(num.mean(numeric_only=True))


def _is_classification(y) -> bool:
    y = np.asarray(y)
    if y.dtype.kind in "OUS" or y.dtype.kind == "b":
        return True
    return y.dtype.kind in "iu" and len(np.unique(y)) <= max(20, int(0.05 * len(y)))


def adversarial_validation(train_X: pd.DataFrame, test_X: pd.DataFrame, *, cv: int = 5) -> list[Finding]:
    """Train a classifier to tell train rows from test rows. High AUC means the
    two are distinguishable — distribution shift, or an id/time feature leaking.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cols = [c for c in train_X.columns if c in test_X.columns]
    a = _numeric_frame(train_X[cols])
    b = _numeric_frame(test_X[cols])
    shared = [c for c in a.columns if c in b.columns]
    if not shared:
        return [Finding(Severity.INFO, "ADVERSARIAL_SKIPPED",
                        "No shared numeric columns to compare.", skill=SPLIT_SKILL)]

    X = pd.concat([a[shared], b[shared]], ignore_index=True).to_numpy()
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    n_splits = min(cv, int(min(np.bincount(y.astype(int)))))
    if n_splits < 2:
        return [Finding(Severity.INFO, "ADVERSARIAL_SKIPPED",
                        "Too few rows on one side to cross-validate.", skill=SPLIT_SKILL)]

    clf = RandomForestClassifier(n_estimators=60, random_state=0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    auc = float(cross_val_score(clf, X, y, scoring="roc_auc", cv=skf).mean())

    if auc >= ADV_ERROR:
        sev, code = Severity.ERROR, "DISTRIBUTION_MISMATCH"
    elif auc >= ADV_WARN:
        sev, code = Severity.WARNING, "DISTRIBUTION_MISMATCH"
    else:
        return [Finding(Severity.INFO, "ADVERSARIAL_OK",
                        f"Train and test look like the same distribution (adversarial AUC={auc:.2f}).",
                        skill=SPLIT_SKILL)]
    return [Finding(
        sev, code,
        f"A classifier separates train from test with AUC={auc:.2f}. They are not "
        f"interchangeable - distribution shift, or an id/time column leaking.",
        fix="Find the distinguishing feature (drop ids/timestamps); make the split mirror production.",
        skill=SPLIT_SKILL,
    )]


def signal_is_real(estimator, X, y, *, scoring=None, cv: int = 5, n_permutations: int = 100,
                   random_state: int = 0) -> list[Finding]:
    """Permutation test: is the model's score better than it would be on shuffled
    labels? A high p-value means the 'result' is indistinguishable from chance.
    """
    from sklearn.model_selection import permutation_test_score

    score, _, pvalue = permutation_test_score(
        estimator, X, y, scoring=scoring, cv=cv,
        n_permutations=n_permutations, random_state=random_state,
    )
    if pvalue > NO_SIGNAL_PVALUE:
        return [Finding(
            Severity.WARNING, "NO_REAL_SIGNAL",
            f"Score={score:.3f} is not distinguishable from chance "
            f"(permutation p={pvalue:.3f}). The model has found no real signal.",
            fix="Don't ship this. Get more signal/data, or accept the problem isn't learnable here.",
            skill=BASELINE_SKILL,
        )]
    return [Finding(
        Severity.INFO, "SIGNAL_REAL",
        f"Score={score:.3f} beats shuffled-label chance (permutation p={pvalue:.3f}).",
        skill=BASELINE_SKILL,
    )]


def target_leak_scan(train: pd.DataFrame, target: str, *, max_features: int = 50) -> list[Finding]:
    """Deep target-leak scan: fit a small tree on EACH single feature. A lone
    feature that predicts the target nearly perfectly is a leak — and this catches
    NONLINEAR leaks that a Pearson-correlation check misses.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    y_raw = train[target]
    classification = _is_classification(y_raw)
    y = y_raw.to_numpy()
    features = [c for c in train.columns if c != target][:max_features]

    findings: list[Finding] = []
    binary = classification and len(np.unique(y)) == 2
    for col in features:
        x = pd.to_numeric(train[col], errors="coerce")
        if x.nunique(dropna=True) < 2 or x.isna().all():
            continue
        xx = x.fillna(x.mean()).to_numpy().reshape(-1, 1)
        try:
            if classification:
                est = DecisionTreeClassifier(max_depth=3, random_state=0)
                scoring = "roc_auc" if binary else "accuracy"
                score = float(cross_val_score(est, xx, y, scoring=scoring, cv=3).mean())
            else:
                est = DecisionTreeRegressor(max_depth=3, random_state=0)
                score = float(cross_val_score(est, xx, y, scoring="r2", cv=3).mean())
        except ValueError:
            continue
        if score >= SINGLE_FEATURE_LEAK:
            findings.append(Finding(
                Severity.ERROR, "TARGET_LEAK",
                f"Feature '{col}' alone predicts '{target}' nearly perfectly "
                f"(single-feature score={score:.4f}).",
                fix="Prove it exists with this value at prediction time, or drop it.",
                skill=LEAK_SKILL, location=col,
            ))
    if not findings:
        findings.append(Finding(
            Severity.INFO, "DEEP_LEAK_CLEAN",
            "No single feature predicts the target nearly perfectly.", skill=LEAK_SKILL,
        ))
    return findings


def can_overfit_small_sample(estimator, X, y, *, n: int = 20, threshold: float = 0.99,
                             random_state: int = 0) -> list[Finding]:
    """Fit a clone on a tiny sample and score it on those SAME rows. A model that
    cannot even memorize a handful of examples has a capacity or wiring bug —
    debug that before trusting any training curve.
    """
    from sklearn.base import clone

    X = np.asarray(X)
    y = np.asarray(y)
    n = min(n, len(y))
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(y))[:n]
    Xs, ys = X[idx], y[idx]

    model = clone(estimator)
    model.fit(Xs, ys)
    train_score = float(model.score(Xs, ys))

    if train_score < threshold:
        return [Finding(
            Severity.ERROR, "CANNOT_OVERFIT",
            f"The model scores only {train_score:.3f} on {n} samples it just trained on. "
            f"It cannot memorize a handful of rows - capacity or wiring bug.",
            fix="Increase capacity / lower regularization, and check the target is wired correctly.",
            skill=TRAIN_SKILL,
        )]
    return [Finding(
        Severity.INFO, "CAN_OVERFIT",
        f"The model memorizes {n} samples (score={train_score:.3f}) - capacity/wiring is sane.",
        skill=TRAIN_SKILL,
    )]
