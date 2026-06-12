"""The unified CLI: end-to-end audit, exit codes, and subcommands."""
import numpy as np
import pandas as pd
from mlcheck.cli import main


def _write_source(tmp_path, body):
    p = tmp_path / "train.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_audit_source_leak_fails(tmp_path, capsys):
    src = _write_source(
        tmp_path,
        "X = StandardScaler().fit_transform(X)\n"
        "X_tr, X_te = train_test_split(X, y)\n",
    )
    rc = main(["audit", "--source", src, "--fail-on", "error"])
    out = capsys.readouterr().out
    assert "FIT_BEFORE_SPLIT" in out
    assert "FAIL" in out
    assert rc == 1


def test_audit_clean_source_passes(tmp_path, capsys):
    src = _write_source(
        tmp_path,
        "X_tr, X_te = train_test_split(X, y)\n"
        "model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)\n",
    )
    rc = main(["audit", "--source", src])
    out = capsys.readouterr().out
    assert "SOURCE_CLEAN" in out
    assert "PASS" in out
    assert rc == 0


def test_baseline_subcommand_flags_no_lift(tmp_path, capsys):
    df = pd.DataFrame({"f": np.arange(200), "label": [0] * 190 + [1] * 10})
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    rc = main([
        "baseline", "--data", str(csv), "--target", "label",
        "--task", "classification", "--metric", "accuracy", "--model-score", "0.95",
    ])
    out = capsys.readouterr().out
    assert "NO_LIFT" in out
    assert rc == 1


def test_audit_with_nothing_returns_2(capsys):
    rc = main(["audit"])
    assert rc == 2
