---
name: detecting-data-leakage
description: Use whenever a preprocessing step is fit, a feature is engineered, or a metric looks suspiciously good — to confirm no information crossed from test into train or from the target into the features. Required before trusting any score, especially after a refactor that "improved" results. Run on the code AND the data.
---

# Detecting Data Leakage

Leakage is the model learning something it will not know at prediction time. It is the single most common reason a great validation score collapses in production.

It hides in two places: the **code** (a transform fit on data it shouldn't see) and the **features** (a column that secretly encodes the answer). This skill checks both.

## The Iron Law

**FIT ON TRAIN ONLY — EVERYTHING ELSE IS LEAKAGE.**

Every statistic the pipeline learns — a scaler's mean, an encoder's categories, an imputer's median, a feature-selection ranking, a target encoding — must be computed on the training fold alone and then *applied* to validation/test. Computing it on the full dataset leaks the test distribution into training.

## The Two Leak Surfaces

**1. Code leakage — fit before split.** The classic:

```python
scaler = StandardScaler().fit(X)          # LEAK: fit sees the whole dataset
X = scaler.transform(X)
X_train, X_test = train_test_split(X)     # too late — test stats already baked in
```

The fix is always a `Pipeline` fit *after* the split (or inside cross-validation), so every fit sees train only.

**2. Feature leakage — the column knows the answer.** A feature that is a proxy for the target, recorded after the outcome, or computed using future information. Symptom: one feature correlates almost perfectly with the target, or removing it tanks the score to baseline.

## The Workflow

1. **Scan the code** for any `fit`/`fit_transform` that runs before the split. Hand off the fix to a `Pipeline`.
2. **Scan the features** for any column with near-perfect association to the target, and for train/test row overlap (duplicates that straddle the split).
3. **Interrogate every "too-good" feature:** would it exist, with this value, at prediction time? If it is recorded at or after the moment of the outcome, it leaks.
4. **Fix the leak, then re-run the score.** The drop is the measure of how much was leaking.

## Run the diagnostic

`mlcheck` does both scans.

```bash
mlcheck scan-source --source train.py
mlcheck scan-data --train train.csv --test test.csv --target label
```

```python
from mlcheck import scan_source, scan_data
scan_source("train.py")                       # static: preprocessing before the split
scan_data(train_df, test_df, target="label")  # runtime: overlap, duplicates, target proxy
```

`scan_source` flags `FIT_BEFORE_SPLIT` (scalers, encoders, imputers, `fit_resample`/SMOTE), plus `MANUAL_SCALE_BEFORE_SPLIT`, `IMPUTE_LEAK` (global-stat `fillna`), and `GET_DUMMIES_BEFORE_SPLIT`. `scan_data` flags `TRAIN_TEST_OVERLAP`, `DUPLICATE_ROWS`, and `TARGET_LEAK` (a feature almost perfectly predicting the target).

For leaks the fast checks miss, go deeper:

```python
from mlcheck import target_leak_scan, adversarial_validation
target_leak_scan(train_df, target="label")        # nonlinear single-feature leaks
adversarial_validation(train_X, test_X)            # train/test distinguishable?
```

`target_leak_scan` (also `mlcheck scan-data --deep`) fits a small tree on each single feature, catching **nonlinear** near-deterministic leaks that a Pearson-correlation check misses. `adversarial_validation` trains a classifier to tell train rows from test rows — a high AUC (`DISTRIBUTION_MISMATCH`) means a distribution shift or an id/time column leaking.

## Anti-Rationalization

| The thought | The reality |
|---|---|
| "I scaled everything up front so it's consistent." | 'Consistent' here means the test mean is baked into training. Fit after the split. |
| "It's just normalization, it can't leak much." | Normalization, imputation, encoding, feature selection — all learn from data. All leak if fit on the full set. |
| "This feature is just really predictive." | A feature that predicts almost perfectly is usually the answer in disguise. Prove it exists *before* the outcome. |
| "The score went up after my change, so the change is good." | A score that jumps after a refactor is a leak until proven otherwise. Re-scan. |
| "Cross-validation handles this for me." | Only if every transform is inside the CV pipeline. A pre-split `fit_transform` leaks through every fold. |

> **From the trenches (a medical risk-scoring model):** A near-perfect AUC traced to a feature populated by a downstream clinical workflow that only ran *after* the risk event it was meant to predict. At prediction time the column would always be empty — the model was reading the answer key. Removing it dropped the AUC to its honest range and pointed the project at the real, harder problem.

## Verification Checklist

- [ ] `scan_source` reports no `FIT_BEFORE_SPLIT`; every transform lives in a `Pipeline` fit after the split.
- [ ] `scan_data` reports no `TRAIN_TEST_OVERLAP`.
- [ ] Any feature flagged `TARGET_LEAK` is proven to exist, with that value, at prediction time — or removed.
- [ ] You re-ran the score after fixing leaks and recorded the (lower, honest) number.

## After This

Code and features clean → finish the `ml-result-skepticism` gate with `choosing-evaluation-metrics` to confirm the now-trustworthy score is also the *right* score.
