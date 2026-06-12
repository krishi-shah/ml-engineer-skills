"""BROKEN fixture: random labels. A permutation test should conclude the model's
score is indistinguishable from chance (NO_REAL_SIGNAL).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def build():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = rng.integers(0, 2, size=120)  # labels unrelated to X
    return {"estimator": LogisticRegression(max_iter=1000), "X": X, "y": y}
