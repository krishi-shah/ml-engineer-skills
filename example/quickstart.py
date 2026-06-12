"""30-second quickstart: build a deliberately-broken result and watch mlcheck
catch every problem at once.

    python example/quickstart.py
"""
import numpy as np
import pandas as pd

from mlcheck import (
    advise_metric,
    adversarial_validation,
    audit_split,
    evaluate_baseline,
    format_report,
    scan_data,
    scan_reproducibility,
    scan_source,
)

# A 99/1 imbalanced, time-ordered dataset of per-user events.
rng = np.random.default_rng(0)
n = 1000
df = pd.DataFrame({
    "ts": np.arange(n),
    "user_id": rng.integers(0, 50, size=n),
    "feature": rng.normal(size=n),
    "label": (np.arange(n) < 10).astype(int),  # 1% positives
})

# The classic mistake: a shuffled split on time-ordered, grouped data.
shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
cut = int(0.8 * n)
train, test = shuffled.iloc[:cut], shuffled.iloc[cut:]

feature_cols = ["ts", "user_id", "feature"]
sections = {
    "Source scan (leakage)": scan_source("example/leaky_pipeline.py"),
    "Reproducibility": scan_reproducibility("example/leaky_pipeline.py"),
    "Split audit": audit_split(train, test, time_col="ts", group_col="user_id", target="label"),
    "Data scan (leakage)": scan_data(train, test, target="label"),
    "Adversarial validation": adversarial_validation(train[feature_cols], test[feature_cols]),
    "Baseline": evaluate_baseline(
        df.drop(columns=["label"]), df["label"],
        model_score=0.99, task="classification", metric="accuracy",
    ),
    "Metric": advise_metric(df["label"], "accuracy"),
}

print(format_report(sections))
