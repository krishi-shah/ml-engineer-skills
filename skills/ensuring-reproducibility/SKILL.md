---
name: ensuring-reproducibility
description: Use before reporting, comparing, or committing any result, and whenever randomness enters a pipeline (splitting, shuffling, weight init, augmentation, sampling). Required before you say "this run scored X" or compare two runs. Run when a number can't be reproduced on a second run.
---

# Ensuring Reproducibility

A result you can't reproduce isn't a result — it's an anecdote. If the number changes on the next run, you can't compare models, you can't bisect a regression, and you can't trust your own conclusions.

Two runs that disagree because of an unseeded shuffle look exactly like two runs that disagree because your change worked. You cannot tell signal from noise without pinning the randomness.

## The Iron Law

**UNSEEDED, UNVERSIONED RESULTS DON'T EXIST.**

Before a number counts: every source of randomness is seeded, and the exact data that produced it is identified by a fingerprint. If you can't regenerate the number, you may not report or compare it.

## The Three Things to Pin

| Source | What goes wrong | Fix |
|---|---|---|
| **Randomness** | Unseeded splits, shuffles, weight init, augmentation, sampling → the score wobbles run to run. | Seed everything: `np.random.default_rng(0)`, `random_state=` on every estimator/split, `torch.manual_seed(0)` (+ CUDA determinism). |
| **Data version** | "It worked yesterday" because the dataset silently changed. | Fingerprint the exact data (`data_fingerprint(df)`) and log it next to the metric. |
| **Environment** | Different library versions produce different numbers. | Pin versions (lockfile / `pip freeze`) and record them with the result. |

## The Workflow

1. **Scan for unseeded randomness** before you trust a run.
2. **Seed every source** the scan reports (and set a single global seed at the top of the script).
3. **Fingerprint the data** and log the hash alongside the metric.
4. **Record the environment** (versions) with the result.
5. **Reproduce once:** run twice; identical inputs must give an identical number. If they don't, something is still unseeded.

## Run the diagnostic

```bash
mlcheck scan-repro train.py
# or:  python -m mlcheck scan-repro train.py
```

```python
from mlcheck import scan_reproducibility, data_fingerprint
for f in scan_reproducibility("train.py"):
    print(f)
print("data version:", data_fingerprint(df))
```

`scan_reproducibility` flags `MISSING_SEED` for stochastic calls without `random_state`, for `np.random` used without a seed, and `MISSING_TORCH_SEED` when torch is imported but never seeded. `data_fingerprint` gives a stable short hash to log with every result.

## Anti-Rationalization

| The thought | The reality |
|---|---|
| "The score only moves a little between runs." | 'A little' is exactly the size of the improvement you're trying to measure. Pin the seed or you can't tell them apart. |
| "I'll set seeds later when it matters." | It matters the moment you compare two numbers — which is now. Seed first. |
| "The data is the same, I just loaded it." | 'The same' is a claim until a fingerprint proves it. Datasets change silently. |
| "Reproducibility is for papers, not experiments." | Experiments are *where* you compare runs. Unseeded experiments send you chasing noise. |
| "It's just the train/test split, one split is fine." | An unseeded split reshuffles every run; your metric and your baseline move together and independently. Pin it. |

> **From the trenches (a retrieval-augmented (RAG) system):** Two retrieval configs were "compared" across runs with an unseeded shuffle and no record of the eval set version. The 1.5-point difference that drove a decision evaporated when both were rerun with a fixed seed on a fingerprinted eval set — it had been split noise the whole time. Seeding and a logged data hash turned a phantom result into a real, repeatable comparison.

## Verification Checklist

- [ ] `scan_reproducibility` reports no `MISSING_SEED` / `MISSING_TORCH_SEED`.
- [ ] A single global seed is set, and every split/estimator passes `random_state=`.
- [ ] The dataset `data_fingerprint` is logged next to the metric.
- [ ] Library versions are recorded with the result.
- [ ] Running twice produces an identical number.

## After This

Result reproducible and data versioned → it can now be trusted and compared. Feed it into the `ml-result-skepticism` gate.
