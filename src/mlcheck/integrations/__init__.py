"""Integrations that make mlcheck run where ML actually happens — pytest, Jupyter,
and pre-commit. Import assertion helpers from `mlcheck.integrations.pytest_plugin`,
or `%load_ext mlcheck.integrations.jupyter` in a notebook.
"""
from .jupyter import audit_cell_source

__all__ = ["audit_cell_source"]
