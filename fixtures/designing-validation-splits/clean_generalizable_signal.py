"""CLEAN fixture: a genuinely generalizable feature. The naive and the honest
(grouped) splits agree, so the split is not inflating the score.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def build():
    rng = np.random.default_rng(0)
    n = 180
    x = rng.normal(size=n)
    y = (x > 0).astype(int)                 # real signal in the feature itself
    group = rng.integers(0, 30, size=n)     # group unrelated to the label
    X = pd.DataFrame({"x": x, "group": group})
    return {
        "estimator": DecisionTreeClassifier(max_depth=3, random_state=0),
        "X": X,
        "y": y,
        "group_col": "group",
    }
