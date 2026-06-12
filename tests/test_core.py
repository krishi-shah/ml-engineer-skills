"""The shared core: severity ordering, finding normalization, report rendering."""
from mlcheck import Finding, Severity, format_report, worst_severity


def test_severity_parse_and_order():
    assert Severity.parse("error") is Severity.ERROR
    assert Severity.parse(Severity.INFO) is Severity.INFO
    assert Severity.ERROR > Severity.WARNING > Severity.INFO


def test_finding_normalizes_string_severity():
    f = Finding("warning", "C", "m")
    assert f.severity is Severity.WARNING


def test_worst_severity():
    findings = [Finding(Severity.INFO, "A", "x"), Finding(Severity.ERROR, "B", "y")]
    assert worst_severity(findings) is Severity.ERROR
    assert worst_severity([]) is Severity.INFO


def test_format_report_shows_fix_and_skill():
    report = format_report({
        "Section": [Finding(Severity.ERROR, "LEAK", "bad thing", fix="do this", skill="x")],
    })
    assert "LEAK" in report
    assert "do this" in report
    assert "1 error(s)" in report
