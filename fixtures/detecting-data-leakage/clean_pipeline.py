"""CLEAN fixture: the scaler is fit AFTER the split, inside a Pipeline.

`scan_source` reads this source and should report SOURCE_CLEAN — every fit runs
after train_test_split, so no test statistics leak into training.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X = np.random.normal(size=(200, 5))
y = (X[:, 0] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)                 # fit sees train only
