"""Impact quantifier — turn "you have a leak" into "your real number is 0.84".

`split_impact` measures how much a naive random split inflates your score versus
an honest (grouped / time-ordered) split. The honest score IS your real number;
the gap is what the leak was buying you.

Backs `designing-validation-splits` and the "does this actually improve accuracy?"
question — by showing the true accuracy, not the inflated one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Finding, Severity

SKILL = "designing-validation-splits"

# Inflation (naive minus honest) above these thresholds is worth flagging.
INFLATION_WARN = 0.05
INFLATION_ERROR = 0.15


def split_impact(estimator, X: pd.DataFrame, y, *, group_col: str | None = None,
                 time_col: str | None = None, feature_cols: list | None = None,
                 cv: int = 5, scoring=None, random_state: int = 0) -> list[Finding]:
    """Compare a naive random split against an honest one and report the inflation.

    Provide `group_col` (rows that share an entity) or `time_col` (predict the
    future). The honest score is your real number.
    """
    from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit, cross_val_score

    exclude = {c for c in (group_col, time_col) if c}
    if feature_cols is None:
        feature_cols = [c for c in X.columns if c not in exclude]
    Xf = X[feature_cols].to_numpy()
    yy = np.asarray(y)

    naive_cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    naive = float(cross_val_score(estimator, Xf, yy, cv=naive_cv, scoring=scoring).mean())

    if group_col is not None:
        groups = np.asarray(X[group_col])
        n_splits = min(cv, len(np.unique(groups)))
        honest = float(cross_val_score(estimator, Xf, yy, cv=GroupKFold(n_splits=n_splits),
                                       groups=groups, scoring=scoring).mean())
        kind = f"grouped-by-'{group_col}'"
    elif time_col is not None:
        order = np.argsort(np.asarray(X[time_col]), kind="stable")
        honest = float(cross_val_score(estimator, Xf[order], yy[order],
                                       cv=TimeSeriesSplit(n_splits=cv), scoring=scoring).mean())
        kind = f"time-ordered-by-'{time_col}'"
    else:
        raise ValueError("split_impact needs group_col or time_col")

    delta = naive - honest
    if delta >= INFLATION_WARN:
        sev = Severity.ERROR if delta >= INFLATION_ERROR else Severity.WARNING
        return [Finding(
            sev, "SCORE_INFLATED_BY_SPLIT",
            f"Naive random split scores {naive:.3f}, but an honest {kind} split scores "
            f"{honest:.3f}. The naive split inflates your score by {delta:.3f} - your real "
            f"number is about {honest:.3f}.",
            fix=f"Trust and report the honest ({kind}) score; validate with that split going forward.",
            skill=SKILL,
        )]
    return [Finding(
        Severity.INFO, "SPLIT_IMPACT_OK",
        f"Naive and honest ({kind}) splits agree (gap={delta:.3f}); the split is not "
        f"inflating your score. Real number is about {honest:.3f}.",
        skill=SKILL,
    )]
