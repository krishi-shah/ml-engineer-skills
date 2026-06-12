"""Jupyter integration: catch leaks in the notebook, where they're born.

    %load_ext mlcheck.integrations.jupyter

    %%mlcheck
    scaler = StandardScaler().fit(X)        # flagged before you even run the next cell
    X_train, X_test = train_test_split(X)

The `%%mlcheck` cell magic runs the cell, then scans its source for leakage and
unseeded randomness and prints a report.
"""
from __future__ import annotations

from ..core import format_report


def audit_cell_source(src: str) -> str:
    """Scan a chunk of notebook/source code and return a formatted report.

    Pure function (no IPython needed) so it is unit-testable.
    """
    from ..leakage import scan_source
    from ..repro import scan_reproducibility

    sections = {
        "Source scan (leakage)": scan_source(src),
        "Reproducibility": scan_reproducibility(src),
    }
    return format_report(sections)


def load_ipython_extension(ipython):  # pragma: no cover - requires a live IPython
    """Register the %%mlcheck cell magic. Called by `%load_ext`."""
    from IPython.core.magic import register_cell_magic

    @register_cell_magic("mlcheck")
    def _mlcheck(line, cell):
        ipython.run_cell(cell)
        print(audit_cell_source(cell))

    ipython.register_magic_function(_mlcheck, magic_kind="cell", magic_name="mlcheck")
