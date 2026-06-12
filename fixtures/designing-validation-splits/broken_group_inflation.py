"""BROKEN fixture: a feature that is really an entity fingerprint. A naive random
split lets the model memorize each user's label and score ~perfectly; an honest
GroupKFold on the user shows the real (chance-level) score.

`build()` returns the estimator + X (with the fingerprint feature and the user_id
group column) + y, for split_impact.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def build():
    rng = np.random.default_rng(0)
    n_users = 30
    rows_per_user = 6
    user_label = rng.integers(0, 2, size=n_users)  # each user's label is fixed but random
    user_id, fingerprint, y = [], [], []
    for u in range(n_users):
        for _ in range(rows_per_user):
            user_id.append(u)
            fingerprint.append(float(u))      # feature == user identity (no real signal)
            y.append(int(user_label[u]))
    X = pd.DataFrame({"user_fingerprint": fingerprint, "user_id": user_id})
    return {
        "estimator": DecisionTreeClassifier(random_state=0),
        "X": X,
        "y": np.array(y),
        "group_col": "user_id",
    }
