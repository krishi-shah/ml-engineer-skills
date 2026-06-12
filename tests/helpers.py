"""Test helpers: load diagnostic and fixture modules by repo-relative path.

Skill directories use hyphens (not importable as packages), so we load modules
directly from their file path via importlib.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load(relpath: str):
    """Load a module from a path relative to the repo root."""
    path = REPO_ROOT / relpath
    name = "_mlrig_" + path.stem + "_" + str(abs(hash(str(path))))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"could not load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def codes(findings) -> set[str]:
    """Collect the `.code` of every finding for easy assertions."""
    return {f.code for f in findings}
