"""Proof for the impact quantifier: it reveals how much a bad split inflates the
score, and stays quiet when the split is honest.
"""
from mlcheck import split_impact

from helpers import codes, load


def test_flags_grouped_inflation():
    f = load("fixtures/designing-validation-splits/broken_group_inflation.py")
    findings = split_impact(**f.build())
    assert "SCORE_INFLATED_BY_SPLIT" in codes(findings), [str(x) for x in findings]


def test_reports_real_number_in_message():
    f = load("fixtures/designing-validation-splits/broken_group_inflation.py")
    msg = split_impact(**f.build())[0].message
    assert "real number is about" in msg


def test_clean_signal_is_not_inflated():
    f = load("fixtures/designing-validation-splits/clean_generalizable_signal.py")
    assert "SCORE_INFLATED_BY_SPLIT" not in codes(split_impact(**f.build()))
