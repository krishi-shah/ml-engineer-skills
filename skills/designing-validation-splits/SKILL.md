---
name: designing-validation-splits
description: Use before trusting any validation score and whenever you create a train/test or cross-validation split — to confirm the split mirrors how the model is used in production. Required for time-series, grouped/clustered data (per-user, per-patient, per-session), and imbalanced targets. Run before train_test_split feels "good enough".
---

# Designing Validation Splits

The validation split is a simulation of the future. If the simulation cheats, the score is fiction.

A `train_test_split(X, y)` is the right tool only when rows are independent and identically distributed. The moment your data has time, groups, or imbalance, a random split lets information cross from test into train — and the score measures memorization, not generalization.

## The Iron Law

**THE SPLIT MUST MIRROR HOW THE MODEL IS USED IN PRODUCTION.**

Ask: at prediction time, what does the model actually know? The split must reproduce exactly that information boundary. If production predicts *tomorrow* from *today*, the test set must be strictly in the future. If production sees a *new patient*, no row from a test patient may appear in train.

## The Three Failure Modes

| Mode | What it looks like | The honest split |
|---|---|---|
| **Time leakage** | Random split on time-ordered data; the model trains on the future to predict the past. | Chronological split — all test timestamps strictly after all train timestamps (`TimeSeriesSplit`). |
| **Group leakage** | The same entity (user, patient, session, device) has rows in both train and test. | `GroupKFold` / `GroupShuffleSplit` on the entity id — no group spans the boundary. |
| **Distribution skew** | A random split leaves a rare class barely present (or absent) in one side. | Stratified split (`StratifiedKFold`) on the target. |

These compound: per-user *time-series* data needs a split that is both grouped *and* time-ordered.

## The Workflow

1. **State the production boundary in one sentence.** "Predict next-day churn for a user we've seen before" / "score a brand-new applicant." This sentence determines the split.
2. **Match the split to the boundary** using the table above.
3. **Audit the split you actually made** with the diagnostic — do not trust that the code "looks right."
4. **If the audit flags a leak, fix the split, not the score.** A higher score from a leaky split is the problem, not the goal.

## Run the diagnostic

`mlcheck` inspects an existing train/test split for all the failure modes.

```bash
mlcheck audit-split --train train.csv --test test.csv \
    --time-col ts --group-col user_id --target label
```

```python
from mlcheck import audit_split
for f in audit_split(train_df, test_df, time_col="ts", group_col="user_id", target="label"):
    print(f)
```

It flags `TIME_LEAK` when test is not strictly after train, `GROUP_LEAK` when an entity appears on both sides, `DISTRIBUTION_SKEW`/`MISSING_CLASS` when stratification is needed but absent, and `SMALL_TEST` when the test set is too small to trust.

## Anti-Rationalization

| The thought | The reality |
|---|---|
| "`train_test_split` with a fixed seed is reproducible, so it's fine." | Reproducible leakage is still leakage. A fixed seed makes the wrong split *consistently* wrong. |
| "Shuffling makes the split fairer." | Shuffling time-ordered or grouped data is exactly how you leak the future / the entity. |
| "Cross-validation already protects me." | Plain k-fold shuffles too. Without `Group`/`TimeSeries` folds, every fold leaks. |
| "The classes are roughly balanced overall." | 'Overall' is not 'in each fold'. A rare class can vanish from a fold and quietly break the metric. |
| "It's the same split everyone uses." | Convention is not correctness. Match the split to *your* production boundary. |

> **From the trenches (a medical risk-scoring model):** Patient records had multiple visits per patient. A random split scattered a single patient's visits across train and test, so the model "recognized" patients it had effectively already seen — validation looked excellent. A `GroupKFold` on patient id dropped the score to its honest level and revealed how much work was left. The leak was invisible until the split was audited.

## Verification Checklist

- [ ] You wrote the one-sentence production boundary before choosing the split.
- [ ] Time-ordered data uses a chronological split; the audit reports no `TIME_LEAK`.
- [ ] Grouped data is split on the entity id; the audit reports no `GROUP_LEAK`.
- [ ] Imbalanced targets are stratified; no `MISSING_CLASS` / `DISTRIBUTION_SKEW`.

## After This

Split audited clean → continue the `ml-result-skepticism` gate with `detecting-data-leakage` (a clean split can still leak through features), then `choosing-evaluation-metrics`.
