"""BROKEN fixture: unseeded randomness everywhere. This file is the SUBJECT of the
reproducibility scan -- scan_reproducibility reads its source and should flag the
unseeded split and the unseeded np.random usage.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = np.random.normal(size=(200, 5))   # np.random without a seed
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y)   # no random_state

model = RandomForestClassifier()       # no random_state
model.fit(X_train, y_train)
