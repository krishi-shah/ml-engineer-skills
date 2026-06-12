"""CLEAN fixture: every source of randomness is seeded. scan_reproducibility
should report REPRO_OK.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)         # seeded numpy
X = rng.normal(size=(200, 5))
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
