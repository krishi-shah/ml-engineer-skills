"""A realistic-looking training script with THREE planted leaks.

Run the scanner on it to see them caught:

    mlcheck scan-source --source example/leaky_pipeline.py
    # or:  python -m mlcheck scan-source --source example/leaky_pipeline.py
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("customers.csv")

# LEAK 1: one-hot encode the whole dataset (category set fixed using test rows)
df = pd.get_dummies(df)

# LEAK 2: impute with a statistic computed over train AND test
df = df.fillna(df.mean())

X = df.drop(columns=["churned"])
y = df["churned"]

# LEAK 3: scale before the split, so test statistics leak into training
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LogisticRegression().fit(X_train, y_train)
print("accuracy:", model.score(X_test, y_test))  # looks great, is a lie
