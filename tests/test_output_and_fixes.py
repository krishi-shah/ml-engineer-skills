"""Proof for v0.4 output: JSON, markdown report, fix suggestions, and the CLI
report/--format/--suggest surface.
"""
import json

from mlcheck import Finding, Severity, format_json, format_markdown
from mlcheck.cli import main
from mlcheck.fixes import suggest_for

from helpers import REPO_ROOT

LEAK = str(REPO_ROOT / "fixtures/detecting-data-leakage/broken_fit_before_split.py")


def _findings():
    return {"Source scan (leakage)": [
        Finding(Severity.ERROR, "FIT_BEFORE_SPLIT", "leak", fix="use a Pipeline", skill="x"),
    ]}


def test_format_json_is_valid_and_structured():
    data = json.loads(format_json(_findings()))
    assert data["worst_severity"] == "error"
    assert data["summary"]["error"] == 1
    assert data["sections"]["Source scan (leakage)"][0]["code"] == "FIT_BEFORE_SPLIT"


def test_format_markdown_has_table():
    md = format_markdown(_findings())
    assert "| `FIT_BEFORE_SPLIT`" in md
    assert "error(s)" in md


def test_suggest_for_returns_snippet():
    snippet = suggest_for(_findings()["Source scan (leakage)"])
    assert "make_pipeline" in snippet
    assert "FIT_BEFORE_SPLIT" in snippet


def test_cli_audit_json(capsys):
    rc = main(["audit", "--source", LEAK, "--format", "json"])
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["worst_severity"] == "error"
    assert rc == 1


def test_cli_scan_source_suggest(capsys):
    main(["scan-source", LEAK, "--suggest"])
    out = capsys.readouterr().out
    assert "suggested fix for FIT_BEFORE_SPLIT" in out
    assert "make_pipeline" in out


def test_cli_report_to_file(tmp_path):
    out = tmp_path / "report.md"
    rc = main(["report", "--source", LEAK, "--out", str(out)])
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "# mlcheck report" in text
    assert "FIT_BEFORE_SPLIT" in text
