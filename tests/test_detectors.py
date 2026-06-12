"""Proof for the runtime detectors: adversarial validation, permutation test,
nonlinear target leak, and the overfit-a-tiny-sample check.
"""
from mlcheck import (
    adversarial_validation,
    can_overfit_small_sample,
    signal_is_real,
    target_leak_scan,
)

from helpers import codes, load


def test_adversarial_flags_distribution_shift():
    f = load("fixtures/detecting-data-leakage/broken_distribution_shift.py")
    assert "DISTRIBUTION_MISMATCH" in codes(adversarial_validation(**f.build()))


def test_adversarial_passes_same_distribution():
    f = load("fixtures/detecting-data-leakage/clean_same_distribution.py")
    found = codes(adversarial_validation(**f.build()))
    assert "DISTRIBUTION_MISMATCH" not in found
    assert "ADVERSARIAL_OK" in found, found


def test_target_leak_scan_catches_nonlinear_leak():
    f = load("fixtures/detecting-data-leakage/broken_nonlinear_target_leak.py")
    assert "TARGET_LEAK" in codes(target_leak_scan(**f.build()))


def test_target_leak_scan_clean_on_noise():
    f = load("fixtures/detecting-data-leakage/clean_noise_features.py")
    assert codes(target_leak_scan(**f.build())) == {"DEEP_LEAK_CLEAN"}


def test_signal_is_real_flags_random_labels():
    f = load("fixtures/debugging-model-training/broken_no_signal.py")
    assert "NO_REAL_SIGNAL" in codes(signal_is_real(**f.build(), n_permutations=50))


def test_signal_is_real_passes_learnable():
    f = load("fixtures/debugging-model-training/clean_learnable.py")
    assert "SIGNAL_REAL" in codes(signal_is_real(**f.build(), n_permutations=50))


def test_cannot_overfit_flagged():
    f = load("fixtures/debugging-model-training/broken_cannot_overfit.py")
    assert "CANNOT_OVERFIT" in codes(can_overfit_small_sample(**f.build()))


def test_can_overfit_passes():
    f = load("fixtures/debugging-model-training/clean_can_overfit.py")
    assert "CAN_OVERFIT" in codes(can_overfit_small_sample(**f.build()))
