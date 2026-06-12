"""Proof for choosing-evaluation-metrics: the advisor flags accuracy on
imbalanced labels and approves a PR-AUC metric on the same labels.
"""
import numpy as np
from mlcheck import advise_metric

from helpers import codes, load

broken = load("fixtures/choosing-evaluation-metrics/broken_accuracy_on_imbalance.py")
clean = load("fixtures/choosing-evaluation-metrics/clean_pr_auc.py")


def test_flags_accuracy_on_imbalance():
    assert "ACCURACY_ON_IMBALANCE" in codes(advise_metric(**broken.build()))


def test_passes_pr_auc_on_imbalance():
    found = codes(advise_metric(**clean.build()))
    assert "ACCURACY_ON_IMBALANCE" not in found
    assert "METRIC_OK" in found, found


def test_calibration_note_for_ranking_metric():
    y = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    assert "CALIBRATION_UNCHECKED" in codes(advise_metric(y, "roc_auc", y_score=scores))
