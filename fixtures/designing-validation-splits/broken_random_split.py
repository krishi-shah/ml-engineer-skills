"""BROKEN fixture: a random split on time-ordered, per-user data.

Notebook-style cell. The data is a sequence of user events over time. A plain
shuffled train_test_split scatters each user's events across both sides AND lets
the model train on events that occur after the test events — both group leakage
and time leakage at once.

`build()` returns train/test DataFrames plus the audit kwargs.
"""
import numpy as np
import pandas as pd


def _data():
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame({
        "ts": np.arange(n),                       # strictly increasing time
        "user_id": rng.integers(0, 50, size=n),   # 50 recurring users
        "feature": rng.normal(size=n),
        "label": (rng.random(n) < 0.3).astype(int),
    })


def build():
    df = _data()
    # The classic mistake: shuffle everything, then cut.
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cut = int(0.8 * len(shuffled))
    train = shuffled.iloc[:cut]
    test = shuffled.iloc[cut:]
    return {
        "train": train,
        "test": test,
        "time_col": "ts",
        "group_col": "user_id",
        "target": "label",
    }
