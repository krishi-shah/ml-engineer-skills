"""BROKEN fixture: reporting accuracy on a 99/1 imbalanced target.

Notebook-style cell. "All negative" already scores 99% accuracy, so the metric
rewards a model that catches none of the rare positive class. The advisor should
flag ACCURACY_ON_IMBALANCE.
"""
import numpy as np


def build():
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:10] = 1  # exactly 1% positives
    return {"y_true": y, "metric_name": "accuracy"}
