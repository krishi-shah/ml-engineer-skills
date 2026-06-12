"""CLEAN fixture: train and test come from the same distribution, so no classifier
can separate them (adversarial AUC ~0.5).
"""
import numpy as np
import pandas as pd


def build():
    rng = np.random.default_rng(0)
    both = rng.normal(0, 1, size=(400, 2))
    train_X = pd.DataFrame(both[:200], columns=["a", "b"])
    test_X = pd.DataFrame(both[200:], columns=["a", "b"])
    return {"train_X": train_X, "test_X": test_X}
