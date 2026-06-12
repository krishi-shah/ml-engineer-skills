"""Proof for establishing-baselines: the diagnostic flags the unbeaten baseline
in the broken fixture and stays quiet on the clean one.
"""
from mlcheck import baseline_score, evaluate_baseline

from helpers import codes, load

broken = load("fixtures/establishing-baselines/broken_no_lift.py")
clean = load("fixtures/establishing-baselines/clean_real_lift.py")


def test_flags_unbeaten_baseline():
    findings = evaluate_baseline(**broken.build())
    assert "NO_LIFT" in codes(findings), [str(f) for f in findings]


def test_passes_real_lift():
    found = codes(evaluate_baseline(**clean.build()))
    assert "NO_LIFT" not in found
    assert "NEGLIGIBLE_LIFT" not in found
    assert "LIFT_OK" in found, found


def test_baseline_score_matches_majority_share():
    data = broken.build()
    base = baseline_score(data["X"], data["y"], "classification", "accuracy")
    assert abs(base - 0.95) < 1e-9


def test_findings_carry_a_fix():
    f = evaluate_baseline(**broken.build())[0]
    assert f.fix and f.skill == "establishing-baselines"
