"""CLEAN fixture: a genuinely learnable target. The permutation test should find
real signal (SIGNAL_REAL).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def build():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # clearly determined by X
    return {"estimator": LogisticRegression(max_iter=1000), "X": X, "y": y}
