"""BROKEN fixture: a no-capacity model (DummyClassifier) that cannot memorize even
a tiny sample. can_overfit_small_sample should raise CANNOT_OVERFIT.
"""
import numpy as np
from sklearn.dummy import DummyClassifier


def build():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.array([0, 1] * 50)  # balanced -> majority predictor scores ~0.5
    return {"estimator": DummyClassifier(strategy="most_frequent"), "X": X, "y": y}
