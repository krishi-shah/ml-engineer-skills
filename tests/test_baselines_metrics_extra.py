"""Extended coverage: more baseline metrics and the regression-metric outlier check."""
import numpy as np
from mlcheck import advise_metric, baseline_score, evaluate_baseline

from helpers import codes


def _imbalanced():
    y = np.zeros(200, dtype=int)
    y[:10] = 1
    X = np.arange(200, dtype=float).reshape(-1, 1)
    return X, y


def test_roc_auc_baseline_is_half():
    X, y = _imbalanced()
    assert baseline_score(X, y, "classification", "roc_auc") == 0.5


def test_regression_mae_baseline_and_lift():
    X = np.arange(4, dtype=float).reshape(-1, 1)
    y = np.array([1.0, 2.0, 3.0, 4.0])  # mean predictor MAE = 1.0
    assert abs(baseline_score(X, y, "regression", "mae") - 1.0) < 1e-9
    assert "NO_LIFT" in codes(evaluate_baseline(X, y, 1.2, "regression", "mae"))
    assert "LIFT_OK" in codes(evaluate_baseline(X, y, 0.4, "regression", "mae"))


def test_rmse_on_heavy_tailed_target_is_flagged():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(size=300), np.array([60.0, -55.0, 70.0])])
    assert "OUTLIER_SENSITIVE_METRIC" in codes(advise_metric(y, "rmse"))


def test_mae_on_heavy_tailed_target_is_ok():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(size=300), np.array([60.0, -55.0, 70.0])])
    assert "OUTLIER_SENSITIVE_METRIC" not in codes(advise_metric(y, "mae"))
