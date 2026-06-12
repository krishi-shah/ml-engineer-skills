"""Proof for the integrations: pytest assertions raise on broken / pass on clean,
and the Jupyter cell-source audit catches a planted leak.
"""
import numpy as np
import pytest
from mlcheck.integrations import audit_cell_source
from mlcheck.integrations.pytest_plugin import (
    assert_beats_baseline,
    assert_no_leakage,
    assert_reproducible,
)

from helpers import REPO_ROOT


def test_assert_no_leakage_raises_on_leak():
    path = str(REPO_ROOT / "fixtures/detecting-data-leakage/broken_fit_before_split.py")
    with pytest.raises(AssertionError):
        assert_no_leakage(path)


def test_assert_no_leakage_passes_clean():
    path = str(REPO_ROOT / "fixtures/detecting-data-leakage/clean_pipeline.py")
    assert_no_leakage(path)  # must not raise


def test_assert_beats_baseline_raises_when_no_lift():
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:50] = 1
    X = np.arange(n, dtype=float).reshape(-1, 1)
    with pytest.raises(AssertionError):
        assert_beats_baseline(X, y, model_score=0.95, task="classification", metric="accuracy")


def test_assert_reproducible_raises_on_unseeded():
    path = str(REPO_ROOT / "fixtures/ensuring-reproducibility/broken_unseeded.py")
    with pytest.raises(AssertionError):
        assert_reproducible(path)


def test_audit_cell_source_reports_leak_and_seed():
    src = (
        "import numpy as np\n"
        "X = np.random.normal(size=(10, 2))\n"
        "X = StandardScaler().fit_transform(X)\n"
        "X_tr, X_te = train_test_split(X)\n"
    )
    report = audit_cell_source(src)
    assert "FIT_BEFORE_SPLIT" in report
    assert "MISSING_SEED" in report
