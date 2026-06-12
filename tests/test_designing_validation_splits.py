"""Proof for designing-validation-splits: the auditor flags the time + group
leakage in the random split and stays quiet on the grouped chronological split.
"""
import pandas as pd
from mlcheck import audit_split

from helpers import codes, load

broken = load("fixtures/designing-validation-splits/broken_random_split.py")
clean = load("fixtures/designing-validation-splits/clean_grouped_time_split.py")


def test_flags_time_and_group_leak():
    found = codes(audit_split(**broken.build()))
    assert "TIME_LEAK" in found, found
    assert "GROUP_LEAK" in found, found


def test_passes_grouped_time_split():
    found = codes(audit_split(**clean.build()))
    assert found == {"SPLIT_OK"}, found


def test_missing_class_detected():
    train = pd.DataFrame({"label": [0, 1, 0, 1]})
    test = pd.DataFrame({"label": [0, 0, 0]})  # class 1 absent from test
    assert "MISSING_CLASS" in codes(audit_split(train, test, target="label"))


def test_small_test_set_flagged():
    train = pd.DataFrame({"label": list(range(100))})
    test = pd.DataFrame({"label": [0, 1]})  # 2% of the data
    assert "SMALL_TEST" in codes(audit_split(train, test, target="label"))
