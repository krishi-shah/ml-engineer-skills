"""BROKEN fixture: train and test are drawn from different distributions, so a
classifier can tell them apart (high adversarial AUC). In real life this is a
covariate shift or an id/time column leaking into the split.
"""
import numpy as np
import pandas as pd


def build():
    rng = np.random.default_rng(0)
    train_X = pd.DataFrame({
        "a": rng.normal(0, 1, size=200),
        "b": rng.normal(0, 1, size=200),
    })
    test_X = pd.DataFrame({
        "a": rng.normal(5, 1, size=200),   # shifted +5
        "b": rng.normal(5, 1, size=200),
    })
    return {"train_X": train_X, "test_X": test_X}
