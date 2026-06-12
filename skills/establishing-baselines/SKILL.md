---
name: establishing-baselines
description: Use before training or trusting any model, and whenever you report a score — to establish the dumb baseline the model must beat. Required the moment you have a metric and want to know if it is good. Run before celebrating accuracy/AUC/RMSE or comparing models.
---

# Establishing Baselines

A model's score is meaningless until you know what a model with no skill scores.

92% accuracy is a triumph on balanced 10-class data and a failure on a 95/5 split — there the do-nothing baseline is 95%. You cannot tell which world you're in without computing the baseline. The baseline is not optional context; it is the denominator that turns a number into a result.

## The Iron Law

**NO MODEL WITHOUT A DUMB BASELINE IT MUST BEAT.**

Before you trust a score, you compute what a trivial predictor scores on the *same split and metric*. The result you report is the **gap**, not the raw number. No baseline → no result.

## What "dumb baseline" means

| Task | Dumb baseline |
|---|---|
| Classification | Predict the majority class (`DummyClassifier(strategy="most_frequent")`); also check `stratified` and `prior`. |
| Regression | Predict the mean/median (`DummyRegressor(strategy="mean"/"median")`). |
| Time-series forecast | Predict the last observed value (persistence / naive forecast). |
| Ranking / recommendation | Most-popular-item ordering. |
| Anything with a known heuristic | The existing rule-of-thumb already in use. |

The baseline must use the **same split and the same metric** as the model. A baseline computed on different data answers a different question.

## The Workflow

1. **Identify the task and metric** (classification/regression/forecast; the metric you'll report).
2. **Compute the dumb baseline on the validation split** using the table above.
3. **Compute the gap:** `model_score − baseline_score`.
4. **Judge the gap, not the score.** A tiny gap means the model learned almost nothing — no matter how high the absolute number. A large gap on a problem experts call hard is a leak signal (hand off to `detecting-data-leakage`).
5. **Report the lift.** "0.91 vs majority 0.88 (+0.03)" — never "0.91" alone.

## Run the diagnostic

The `mlcheck` toolkit computes the baseline for you and flags a missing or trivial lift.

```bash
mlcheck baseline --data data.csv --target label --task classification \
    --metric accuracy --model-score 0.91
# or, without installing:  python -m mlcheck baseline ...
```

Programmatically:

```python
from mlcheck import evaluate_baseline
for f in evaluate_baseline(X, y, model_score=0.91, task="classification", metric="accuracy"):
    print(f)        # severity, code, message, and a concrete fix
```

It returns a finding when the model fails to beat the baseline, when the lift is negligible, or when the lift is so large on imbalanced data that leakage is the likelier explanation. Supports `accuracy`, `f1`, `balanced_accuracy`, `roc_auc` (classification) and `r2`, `mae` (regression).

For the rigorous, non-heuristic version of "is the lift real?", run a permutation test — it compares your score against scores on shuffled labels:

```python
from mlcheck import signal_is_real
for f in signal_is_real(model, X, y, n_permutations=100):
    print(f)   # NO_REAL_SIGNAL if the score is indistinguishable from chance
```

## Anti-Rationalization

| The thought | The reality |
|---|---|
| "The accuracy is high, I don't need a baseline." | High accuracy on imbalanced data *is* the baseline. Without it you can't tell skill from class frequency. |
| "Baselines are for benchmarks, not real projects." | Real projects are exactly where an unbeaten baseline wastes months. The baseline is the cheapest check you own. |
| "Obviously my model beats predicting the mean." | Then it costs ten seconds to prove. 'Obviously' is how unbeaten baselines ship. |
| "The lift is huge, we're great." | A huge lift on a hard problem is a leak signal, not a victory. Verify with `detecting-data-leakage`. |
| "I'll add the baseline later for the writeup." | By the writeup the number is already a 'fact'. Compute it before you believe the score. |

> **From the trenches (a retrieval-augmented (RAG) system):** A retrieval-augmented answer system reported a headline accuracy that looked strong — until the majority-class baseline (always answer "no relevant document") was computed on the same split and landed only two points lower. The "strong" model was barely doing anything. The baseline turned a celebration into the actual research question.

## Verification Checklist

- [ ] Baseline computed on the **same split and metric** as the model.
- [ ] You report the **gap**, not the raw score.
- [ ] A negligible gap is treated as "the model learned nothing," regardless of absolute score.
- [ ] A suspiciously large gap is escalated to `detecting-data-leakage`.

## After This

Baseline beaten by a defensible margin → continue the `ml-result-skepticism` gate: audit the split (`designing-validation-splits`), then rule out leakage (`detecting-data-leakage`).
