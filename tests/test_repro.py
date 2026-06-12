"""Proof for ensuring-reproducibility: the scan flags unseeded randomness and
passes a fully-seeded script; the fingerprint is stable.
"""
import pandas as pd
from mlcheck import data_fingerprint, scan_reproducibility

from helpers import REPO_ROOT, codes


def test_flags_unseeded_source():
    path = str(REPO_ROOT / "fixtures/ensuring-reproducibility/broken_unseeded.py")
    assert "MISSING_SEED" in codes(scan_reproducibility(path))


def test_passes_seeded_source():
    path = str(REPO_ROOT / "fixtures/ensuring-reproducibility/clean_seeded.py")
    assert codes(scan_reproducibility(path)) == {"REPRO_OK"}


def test_torch_without_seed_flagged():
    src = "import torch\nx = torch.randn(3)\n"
    assert "MISSING_TORCH_SEED" in codes(scan_reproducibility(src))


def test_data_fingerprint_is_stable_and_content_sensitive():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert data_fingerprint(df) == data_fingerprint(df.copy())
    other = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]})
    assert data_fingerprint(df) != data_fingerprint(other)
