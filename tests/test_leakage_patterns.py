"""The broadened source scan: catches the leaky preprocessing patterns people
actually write, beyond the textbook fit_transform-before-split.
"""
from mlcheck import scan_data, scan_source

from helpers import codes

import pandas as pd

SPLIT = "X_tr, X_te = train_test_split(X, y)\n"


def test_manual_scaling_before_split():
    src = "X = (X - X.mean()) / X.std()\n" + SPLIT
    assert "MANUAL_SCALE_BEFORE_SPLIT" in codes(scan_source(src))


def test_global_stat_imputation_before_split():
    src = "df = df.fillna(df.mean())\n" + SPLIT
    assert "IMPUTE_LEAK" in codes(scan_source(src))


def test_get_dummies_before_split():
    src = "X = pd.get_dummies(X)\n" + SPLIT
    assert "GET_DUMMIES_BEFORE_SPLIT" in codes(scan_source(src))


def test_resample_before_split_is_fit_leak():
    src = "X, y = SMOTE().fit_resample(X, y)\n" + SPLIT
    assert "FIT_BEFORE_SPLIT" in codes(scan_source(src))


def test_no_split_found():
    src = "model = LogisticRegression().fit(X, y)\n"
    assert "NO_SPLIT_FOUND" in codes(scan_source(src))


def test_preprocessing_after_split_is_clean():
    src = SPLIT + "scaler = StandardScaler().fit(X_tr)\n"
    assert "SOURCE_CLEAN" in codes(scan_source(src))


def test_duplicate_rows_within_train():
    train = pd.DataFrame({"f": [1, 1, 2, 3], "label": [0, 0, 1, 1]})
    test = pd.DataFrame({"f": [9, 8], "label": [0, 1]})
    assert "DUPLICATE_ROWS" in codes(scan_data(train, test, target="label"))
