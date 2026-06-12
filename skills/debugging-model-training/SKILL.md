---
name: debugging-model-training
description: Use when a model trains but something seems off — loss plateaus, the metric is implausibly low OR high, or you're unsure the training loop is even wired correctly. Required before trusting a training curve or concluding "the model can't learn this". Run before tuning hyperparameters on a model you haven't sanity-checked.
---

# Debugging Model Training

A training curve is not evidence that training works. A model can produce a smooth, plausible loss curve while learning the wrong thing — or nothing.

Before you tune learning rates for a week, prove the loop can learn *at all*. Most "the model won't converge" problems are wiring bugs: shuffled labels, a detached target, a frozen layer, a loss computed on the wrong tensor.

## The Iron Law

**PROVE THE MODEL CAN OVERFIT A TINY SAMPLE BEFORE YOU TRUST A TRAINING CURVE.**

If your model cannot reach near-perfect score on 10–20 examples it is allowed to memorize, it has a capacity or wiring bug. No amount of tuning on the full dataset will fix that — and every curve you read until you fix it is noise.

## The Two Sanity Checks

**1. Overfit a tiny sample.** Take a handful of rows. Train until the model *memorizes* them. A correctly-wired model with enough capacity will hit ~100% train accuracy / ~0 loss on a dozen examples. If it can't, stop — the bug is in the model or the plumbing, not the data size.

**2. Shuffle the labels.** Train on randomly-permuted labels. A correct setup should now do no better than chance. If it still "learns" (low loss, high score) on noise, you have **leakage in the evaluation itself** — information is reaching the model that shouldn't.

These two bracket the failure space: check 1 catches "can't learn even when it should," check 2 catches "learns even when it shouldn't."

## The Workflow

1. **Run the overfit-a-tiny-sample check.** Until it passes, do nothing else.
2. **Run the label-shuffle / permutation check.** Confirm the score collapses to chance.
3. **Only then** read the real training curve, tune, or conclude anything about learnability.
4. **If check 1 fails:** increase capacity, lower regularization, verify the target is connected to the loss, check for frozen/zeroed gradients.
5. **If check 2 fails:** you have leakage — hand off to `detecting-data-leakage` and `designing-validation-splits`.

## Run the diagnostic

```python
from mlcheck.detectors import can_overfit_small_sample, signal_is_real

# Check 1: can it memorize a handful of rows?
for f in can_overfit_small_sample(model, X, y, n=20):
    print(f)

# Check 2: is the score better than shuffled-label chance?
for f in signal_is_real(model, X, y, n_permutations=100):
    print(f)
```

`can_overfit_small_sample` raises `CANNOT_OVERFIT` when the model can't memorize a small sample. `signal_is_real` (a permutation test) raises `NO_REAL_SIGNAL` when the score is indistinguishable from chance — and, run on a "too-good" result, a *low* p-value with an implausible score points you back to leakage.

## Anti-Rationalization

| The thought | The reality |
|---|---|
| "The loss is going down, training works." | A decreasing loss can come from learning the wrong target or memorizing an artifact. Overfit a tiny sample to prove the loop is wired. |
| "It won't converge — I need a better architecture." | First prove the current one can overfit 12 rows. If it can't, the bug is plumbing, not architecture. |
| "The score is low, this problem is just hard." | Maybe — or a label is detached / a layer is frozen. The tiny-sample check separates 'hard' from 'broken'. |
| "Shuffling labels is a waste of time." | It's the only check that catches a loop that 'learns' from noise — i.e. leakage hiding inside your evaluation. |
| "I'll sanity-check after I get a baseline number." | The number is meaningless until the loop is proven sane. Sanity first, number second. |

> **From the trenches (an image-classification project):** An image classifier's loss fell beautifully but val accuracy was stuck near chance. The overfit-a-tiny-sample check failed — the model couldn't memorize even 16 images. The cause wasn't the architecture; a label tensor was being detached before the loss. Once the wiring was fixed it overfit 16 images instantly, and only then was the real curve worth reading.

## Verification Checklist

- [ ] The model reaches ~perfect score on a tiny memorizable sample (`can_overfit_small_sample` passes).
- [ ] On shuffled labels, the score collapses to chance (no `NO_REAL_SIGNAL`-style "learning from noise").
- [ ] Only after both pass did you tune or judge learnability.
- [ ] A failure of check 2 was escalated to `detecting-data-leakage`.

## After This

Loop proven sane → trust the curve, then run the `ml-result-skepticism` gate on the resulting metric.
