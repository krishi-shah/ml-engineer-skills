"""BROKEN fixture: a preprocessing step fit on the whole dataset before the split.

This file is the SUBJECT of the static scan — `scan_source` reads its source and
should flag the `fit_transform` on line below that runs before train_test_split.
It is also runnable, and demonstrates the real mistake.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.random.normal(size=(200, 5))
y = (X[:, 0] > 0).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # LEAK: fit sees train + test together

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)
