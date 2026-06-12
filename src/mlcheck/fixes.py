"""Fix suggestions — for each finding code, a concrete corrected code template you
can paste, not just prose. Turns "you have a leak" into "here's the right code".
"""
from __future__ import annotations

from .core import Finding

# Code -> ready-to-adapt corrected snippet.
_SNIPPETS: dict[str, str] = {
    "FIT_BEFORE_SPLIT": (
        "# Fit inside a Pipeline AFTER the split, so every transform sees train only:\n"
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)\n"
        "model = make_pipeline(StandardScaler(), YourEstimator())\n"
        "model.fit(X_train, y_train)\n"
    ),
    "MANUAL_SCALE_BEFORE_SPLIT": (
        "# Don't scale by hand on the full data. Let a Pipeline learn the stats on train:\n"
        "model = make_pipeline(StandardScaler(), YourEstimator())\n"
        "model.fit(X_train, y_train)   # scaler.mean_/scale_ come from train only\n"
    ),
    "IMPUTE_LEAK": (
        "# Impute inside the Pipeline so the fill value is computed on train only:\n"
        "from sklearn.impute import SimpleImputer\n"
        "model = make_pipeline(SimpleImputer(strategy='mean'), YourEstimator())\n"
        "model.fit(X_train, y_train)\n"
    ),
    "GET_DUMMIES_BEFORE_SPLIT": (
        "# One-hot encode inside the Pipeline so categories are fixed on train only:\n"
        "from sklearn.preprocessing import OneHotEncoder\n"
        "from sklearn.compose import make_column_transformer\n"
        "pre = make_column_transformer((OneHotEncoder(handle_unknown='ignore'), cat_cols),\n"
        "                              remainder='passthrough')\n"
        "model = make_pipeline(pre, YourEstimator()).fit(X_train, y_train)\n"
    ),
    "MISSING_SEED": (
        "# Pin every source of randomness:\n"
        "import numpy as np\n"
        "rng = np.random.default_rng(0)\n"
        "train_test_split(X, y, random_state=0)\n"
        "YourEstimator(random_state=0)\n"
    ),
}


def suggest_for(findings: list[Finding]) -> str:
    """Return paste-ready fix snippets for the codes present in `findings`."""
    seen: list[str] = []
    blocks: list[str] = []
    for f in findings:
        if f.code in _SNIPPETS and f.code not in seen:
            seen.append(f.code)
            blocks.append(f"--- suggested fix for {f.code} ---\n{_SNIPPETS[f.code]}")
    if not blocks:
        return ""
    return "\n".join(blocks)
