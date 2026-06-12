"""Proof for detecting-data-leakage: the source scan catches preprocessing
before the split, the clean pipeline passes, and the data scan catches overlap,
duplicates, and target leak.
"""
import pandas as pd
from mlcheck import scan_data, scan_source

from helpers import REPO_ROOT, codes


def test_source_scan_flags_fit_before_split():
    path = str(REPO_ROOT / "fixtures/detecting-data-leakage/broken_fit_before_split.py")
    assert "FIT_BEFORE_SPLIT" in codes(scan_source(path))


def test_source_scan_passes_pipeline():
    path = str(REPO_ROOT / "fixtures/detecting-data-leakage/clean_pipeline.py")
    found = codes(scan_source(path))
    assert "FIT_BEFORE_SPLIT" not in found
    assert "SOURCE_CLEAN" in found, found


def test_data_scan_flags_target_leak_and_overlap():
    train = pd.DataFrame({
        "leaky": [0, 1, 0, 1, 0, 1],
        "noise": [5, 9, 2, 7, 3, 8],
        "label": [0, 1, 0, 1, 0, 1],
    })
    test = pd.DataFrame({"leaky": [0, 1], "noise": [5, 9], "label": [0, 1]})
    found = codes(scan_data(train, test, target="label"))
    assert "TARGET_LEAK" in found, found
    assert "TRAIN_TEST_OVERLAP" in found, found


def test_data_scan_clean():
    train = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6],
        "f2": [9, 1, 4, 2, 7, 3],
        "label": [0, 1, 0, 1, 1, 0],
    })
    test = pd.DataFrame({"f1": [10, 11, 12], "f2": [8, 6, 5], "label": [1, 0, 1]})
    assert codes(scan_data(train, test, target="label")) == {"DATA_CLEAN"}
