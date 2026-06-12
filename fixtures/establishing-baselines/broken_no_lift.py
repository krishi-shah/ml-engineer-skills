"""BROKEN fixture: a "great" 95% accuracy that is actually the baseline.

Notebook-style cell. The target is exactly 95/5 imbalanced, so always predicting
the majority class already scores 0.95. The reported model also scores 0.95 —
zero real skill — but the headline number looks impressive.

`build()` returns the inputs the baseline_runner diagnostic consumes. The class
counts are exact (not random) so the baseline, and therefore the test, are
deterministic.
"""
import numpy as np


def build():
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:50] = 1  # exactly 5% positives -> majority-class baseline = 0.95
    X = np.arange(n, dtype=float).reshape(-1, 1)  # one uninformative feature
    reported_model_score = 0.95  # identical to the baseline: no real skill
    return {
        "X": X,
        "y": y,
        "model_score": reported_model_score,
        "task": "classification",
        "metric": "accuracy",
    }
