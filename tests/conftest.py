"""Anchors the tests/ directory on sys.path so `import helpers` resolves under
pytest's default (prepend) import mode.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
