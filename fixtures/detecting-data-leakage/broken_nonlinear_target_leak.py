"""BROKEN fixture: a feature that predicts the target nearly perfectly through a
NONLINEAR relationship the Pearson-correlation check misses.

For y==1 the feature is +5 or -5 (alternating); for y==0 it is ~0. The mean
feature value is the same for both classes, so linear correlation ~0 -- but a
depth-2 tree splitting on |feature| separates them perfectly.
"""
import numpy as np
import pandas as pd


def build():
    rng = np.random.default_rng(0)
    n = 120
    y = np.array([0, 1] * (n // 2))
    feature = np.where(y == 1, np.where(np.arange(n) % 4 < 2, 5.0, -5.0), 0.0)
    feature = feature + rng.normal(0, 0.01, size=n)  # tiny jitter, keeps it deterministic-ish
    df = pd.DataFrame({
        "sneaky": feature,
        "noise": rng.normal(size=n),
        "label": y,
    })
    return {"train": df, "target": "label"}
