"""Proof for the v0.3 CLI surface: scan-repro, adversarial, scan-data --deep,
and audit --repro."""
import numpy as np
import pandas as pd
from mlcheck.cli import main

from helpers import REPO_ROOT


def test_scan_repro_positional_file(capsys):
    path = str(REPO_ROOT / "fixtures/ensuring-reproducibility/broken_unseeded.py")
    rc = main(["scan-repro", path])
    out = capsys.readouterr().out
    assert "MISSING_SEED" in out
    assert rc == 0  # repro findings are warnings, not errors


def test_scan_source_positional_files(capsys):
    path = str(REPO_ROOT / "fixtures/detecting-data-leakage/broken_fit_before_split.py")
    rc = main(["scan-source", path])
    out = capsys.readouterr().out
    assert "FIT_BEFORE_SPLIT" in out
    assert rc == 1  # leakage is an error


def test_adversarial_subcommand(tmp_path, capsys):
    rng = np.random.default_rng(0)
    pd.DataFrame({"a": rng.normal(0, 1, 150), "b": rng.normal(0, 1, 150)}).to_csv(
        tmp_path / "tr.csv", index=False)
    pd.DataFrame({"a": rng.normal(5, 1, 150), "b": rng.normal(5, 1, 150)}).to_csv(
        tmp_path / "te.csv", index=False)
    rc = main(["adversarial", "--train", str(tmp_path / "tr.csv"), "--test", str(tmp_path / "te.csv")])
    out = capsys.readouterr().out
    assert "DISTRIBUTION_MISMATCH" in out
    assert rc == 1


def test_scan_data_deep(tmp_path, capsys):
    f = __import__("helpers").load("fixtures/detecting-data-leakage/broken_nonlinear_target_leak.py")
    df = f.build()["train"]
    tr = df.iloc[:90]
    te = df.iloc[90:]
    tr.to_csv(tmp_path / "tr.csv", index=False)
    te.to_csv(tmp_path / "te.csv", index=False)
    rc = main(["scan-data", "--train", str(tmp_path / "tr.csv"),
               "--test", str(tmp_path / "te.csv"), "--target", "label", "--deep"])
    out = capsys.readouterr().out
    assert "TARGET_LEAK" in out
