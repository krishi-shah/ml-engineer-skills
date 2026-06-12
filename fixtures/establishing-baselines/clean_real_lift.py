"""CLEAN fixture: a model with a real, defensible lift over the baseline.

Same exact 95/5 imbalance (baseline = 0.95), but the reported model scores
0.972 — a meaningful, plausible gain (~44% of the available headroom). The
diagnostic should report LIFT_OK and raise no warning.
"""
import numpy as np


def build():
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:50] = 1  # exactly 5% positives -> majority-class baseline = 0.95
    X = np.arange(n, dtype=float).reshape(-1, 1)
    reported_model_score = 0.972  # ~2.2 pts over baseline: real but not absurd
    return {
        "X": X,
        "y": y,
        "model_score": reported_model_score,
        "task": "classification",
        "metric": "accuracy",
    }
