"""CLEAN fixture: only noise features, none predictive of the target. The deep
single-feature scan should report DEEP_LEAK_CLEAN.
"""
import numpy as np
import pandas as pd


def build():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "label": np.array([0, 1] * (n // 2)),
    })
    return {"train": df, "target": "label"}
