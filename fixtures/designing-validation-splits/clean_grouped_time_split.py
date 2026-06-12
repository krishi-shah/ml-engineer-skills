"""CLEAN fixture: a split that is both grouped (by user) and time-ordered.

Users are partitioned so no user appears on both sides, and every test event
occurs strictly after every train event. The auditor should report SPLIT_OK.
"""
import numpy as np
import pandas as pd


def _data():
    rng = np.random.default_rng(0)
    rows = []
    ts = 0
    # First 40 users live in the "past" (train), last 10 in the "future" (test).
    for user in range(50):
        for _ in range(12):
            rows.append({
                "ts": ts,
                "user_id": user,
                "feature": float(rng.normal()),
                "label": int(rng.random() < 0.3),
            })
            ts += 1
    return pd.DataFrame(rows)


def build():
    df = _data().sort_values("ts").reset_index(drop=True)
    train_users = set(range(40))
    train = df[df["user_id"].isin(train_users)]
    test = df[~df["user_id"].isin(train_users)]
    # Re-base time so all test timestamps are strictly after all train timestamps,
    # mirroring a real chronological + grouped split.
    train = train.copy()
    test = test.copy()
    train["ts"] = range(len(train))
    test["ts"] = range(len(train), len(train) + len(test))
    return {
        "train": train,
        "test": test,
        "time_col": "ts",
        "group_col": "user_id",
        "target": "label",
    }
