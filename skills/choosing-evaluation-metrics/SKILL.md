---
name: choosing-evaluation-metrics
description: Use before reporting or optimizing any metric — to confirm the metric reflects the real cost of being wrong. Required for imbalanced data, ranking/probability outputs, and any decision with asymmetric error costs. Run before accuracy/AUC are taken at face value.
---

# Choosing Evaluation Metrics

The wrong metric makes a useless model look excellent. Optimizing it makes the model worse at the thing you actually care about.

A metric is a definition of success. If it doesn't encode the cost of the mistakes you fear, a high score is not good news — it's a measurement of the wrong thing.

## The Iron Law

**THE METRIC MUST MATCH THE COST OF BEING WRONG.**

Before you trust or optimize a number, state which error is expensive — a false positive or a false negative — and confirm the metric punishes that error. If it doesn't, change the metric, not the model.

## The Common Traps

| Situation | The trap | The honest metric |
|---|---|---|
| Imbalanced classes (fraud, disease, churn) | Accuracy — predicting "all negative" scores 99% and catches nothing. | Precision/recall, F1, **PR-AUC** (average precision), recall at fixed precision. |
| Asymmetric costs (a missed cancer ≫ a false alarm) | A symmetric metric that weighs both errors equally. | Cost-weighted metric, recall at a precision floor, F-beta with β set by the cost ratio. |
| Probability outputs used for decisions | Reporting ROC-AUC (ranking only) while thresholding probabilities. | Calibration (reliability curve, Brier score) **in addition to** ranking. |
| Ranking / recommendation | Accuracy@top-1 when users see a list. | NDCG, MAP, recall@k. |
| Regression with outliers | RMSE dominated by a few extreme points. | MAE, or quantile loss if the tails matter asymmetrically. |

## The Workflow

1. **Name the expensive error in one sentence.** "A missed fraudulent transaction costs 100× a false alarm."
2. **Pick the metric that punishes that error** using the table.
3. **Reject accuracy on imbalanced data outright** — it is the most common lie in applied ML.
4. **If you output probabilities and act on them, check calibration**, not just ranking.
5. **Optimize and report the chosen metric** — and its baseline (`establishing-baselines`).

## Run the diagnostic

`mlcheck` checks whether your metric fits your labels.

```bash
mlcheck metric --data data.csv --target label --metric accuracy
```

```python
from mlcheck import advise_metric
for f in advise_metric(y_true, metric_name="accuracy", y_score=probs):
    print(f)
```

It flags `ACCURACY_ON_IMBALANCE` when accuracy is used on skewed labels, `CALIBRATION_UNCHECKED` when probabilities are scored only by ranking, and `OUTLIER_SENSITIVE_METRIC` when RMSE/R² is reported on a heavy-tailed regression target.

## Anti-Rationalization

| The thought | The reality |
|---|---|
| "Accuracy is the standard metric." | Accuracy is the default, not the right choice. On imbalanced data it rewards ignoring the rare class — the one you built the model for. |
| "AUC is high, the model is great." | ROC-AUC measures ranking, not the threshold you deploy. A high-AUC model can be badly calibrated and make poor decisions. |
| "F1 is a single number, easier to report." | F1 weights precision and recall equally. If your costs are asymmetric, F1 hides the error you care about. Set β. |
| "Both errors are about equally bad." | Say that out loud about your actual problem. Usually one error is far worse — and the metric must reflect it. |
| "The metric is fine, the model just needs tuning." | Tuning toward the wrong metric makes the real outcome worse. Fix the metric first. |

> **From the trenches (a retrieval-augmented (RAG) system):** Answer quality was first tracked with exact-match accuracy. On a corpus where most queries had no answerable document, "always abstain" scored high while the system helped no one. Switching to recall-at-precision on the answerable subset exposed the true gap and redirected the work toward retrieval, where the real problem was.

## Verification Checklist

- [ ] You named the expensive error before choosing the metric.
- [ ] The metric punishes that error (no accuracy on imbalanced data).
- [ ] Probability outputs used for decisions are checked for calibration, not just ranking.
- [ ] You report the metric alongside its baseline (`establishing-baselines`).

## After This

Metric matches the cost → the `ml-result-skepticism` gate is complete. The reported number is now trustworthy, honestly split, leak-free, and the *right* number. Report it with its baseline and its split design.
