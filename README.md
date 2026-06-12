# ML Engineer Skills

**A spell-checker for machine-learning mistakes.** Point it at your code and data, and it tells you — in plain English — what's wrong and how to fix it. The tool you install and run is called **`mlcheck`**.

## Why you'd want this

When you train a model, you get a score like *"98% accurate!"*. The problem: a score that looks amazing is very often **fake** — not because you cheated, but because of small, invisible mistakes almost everyone makes (your model accidentally saw the answers, or you tested it on data it already studied). The model looks brilliant in your notebook, then falls apart on real data.

`mlcheck` catches those mistakes early, explains them, and even shows you your model's **real** score.

## Install

```bash
git clone https://github.com/krishi-shah/ml-engineer-skills
cd ml-engineer-skills
pip install -e .
```

That gives you the `mlcheck` command.

## Try it in 30 seconds

```bash
python example/quickstart.py
```

You'll see it catch 5 real bugs at once, each with a fix:

```
[x] FIT_BEFORE_SPLIT (line 26): scaling happened before the train/test split.
    -> fix: Move the transform into a Pipeline fit AFTER the split.
[x] NO_LIFT: accuracy 0.99 does NOT beat the dumb baseline (0.99) - the model learned nothing.
[x] ACCURACY_ON_IMBALANCE: 99% of rows are one class, so "accuracy" is meaningless here.
...
```

## Use it on your own project

Check a training script for leaks:

```bash
mlcheck scan-source train.py
```

Run every check at once and get one report (great for CI — it exits with an error if something's wrong):

```bash
mlcheck audit --source train.py --train train.csv --test test.csv \
    --target label --fail-on error
```

Want the corrected code, not just the warning? Add `--suggest`:

```bash
mlcheck scan-source train.py --suggest
```

## What it checks for

| Mistake | What it means (plain English) |
|---|---|
| **Data leakage** | Your model secretly saw information it won't have in real life. |
| **Bad train/test split** | You tested the model on data it already studied, so the score is inflated. |
| **No baseline** | "98%" sounds great — until you learn that *always guessing the same answer* also scores 98%. |
| **Wrong metric** | You're measuring the wrong thing (e.g. "accuracy" on rare-event data). |
| **Not reproducible** | The score changes every run because randomness wasn't pinned down. |
| **Broken training** | The model can't even memorize a few examples — a wiring bug, not bad luck. |

## Know your *real* score

This is the part that actually helps your accuracy. Instead of just saying "you have a leak," it measures what the leak cost you:

```python
from mlcheck import split_impact
for finding in split_impact(model, X, y, group_col="user_id"):
    print(finding)
# Naive split scores 1.000, but an honest split scores 0.733.
# Your real number is about 0.733.
```

That honest number is the one that holds up in the real world.

## Run it automatically (so you never forget)

**In your tests** — fail the build if there's a leak:
```python
from mlcheck.integrations.pytest_plugin import assert_no_leakage
def test_no_leakage():
    assert_no_leakage("train.py")
```

**Before every commit** — add to `.pre-commit-config.yaml`:
```yaml
- repo: https://github.com/krishi-shah/ml-engineer-skills
  rev: v0.4.0
  hooks:
    - id: mlcheck-scan-source
```

**In a Jupyter notebook** — flag leaks as you write them:
```python
%load_ext mlcheck.integrations.jupyter
%%mlcheck
scaler = StandardScaler().fit(X)        # flagged right after the cell runs
```

**With an AI assistant (Claude Code)** — install the skills so your assistant checks its own ML work:
```
/plugin marketplace add krishi-shah/ml-engineer-skills
/plugin install ml-engineer-skills
```

## Is it for me?

- **New to ML / a student?** Yes — it catches the exact mistakes you don't yet know to look for, and teaches you as it goes.
- **On a team?** Yes — wire it into CI so a leaked model can't get merged.
- **Already an expert?** You'll get less out of it; you've built these habits. It's still a useful safety net.

It's a smoke detector, not a guarantee: it catches the *common* mistakes, which is most of them.

## What's inside

```
src/mlcheck/        the tool (plain Python: numpy, pandas, scikit-learn)
skills/             the playbooks that teach each check (and guide an AI assistant)
example/            quickstart.py + a deliberately-buggy script to scan
tests/              64 tests, each proving a check catches a real bug
```

Verify it yourself:

```bash
pip install -e ".[dev]"
pytest -q          # 64 passed
```

## License

MIT
