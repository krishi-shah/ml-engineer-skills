"""CLEAN fixture: the same 99/1 imbalance, but scored with PR-AUC (average
precision) — a metric that punishes missing the rare class. The advisor should
report METRIC_OK with no warning.
"""
import numpy as np


def build():
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:10] = 1  # exactly 1% positives
    return {"y_true": y, "metric_name": "average_precision"}
