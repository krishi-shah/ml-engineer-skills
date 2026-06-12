"""CLEAN fixture: a high-capacity model that can memorize a tiny sample.
can_overfit_small_sample should report CAN_OVERFIT.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def build():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))            # distinct continuous rows
    y = rng.integers(0, 2, size=100)
    return {"estimator": DecisionTreeClassifier(random_state=0), "X": X, "y": y}
