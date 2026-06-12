"""pytest integration: drop these assertions into your test suite so a leak, an
unbeaten baseline, or a non-reproducible script fails CI like any other bug.

    from mlcheck.integrations.pytest_plugin import assert_no_leakage, assert_beats_baseline

    def test_training_script_has_no_leakage():
        assert_no_leakage("train.py")

    def test_model_beats_baseline():
        assert_beats_baseline(X, y, model_score=0.91, task="classification")

Registered as a pytest plugin via the `pytest11` entry point, so the helpers are
importable wherever pytest runs.
"""
from __future__ import annotations

from ..core import Finding, Severity, format_report, worst_severity


def _assert(findings: list[Finding], fail_on: "str | Severity", section: str) -> None:
    threshold = Severity.parse(fail_on)
    if worst_severity(findings) >= threshold:
        raise AssertionError("\n" + format_report({section: findings}))


def assert_no_leakage(source, fail_on: "str | Severity" = Severity.ERROR) -> None:
    from ..leakage import scan_source
    _assert(scan_source(source), fail_on, "Source scan (leakage)")


def assert_data_clean(train, test, target, fail_on: "str | Severity" = Severity.ERROR) -> None:
    from ..leakage import scan_data
    _assert(scan_data(train, test, target), fail_on, "Data scan (leakage)")


def assert_split_clean(train, test, *, time_col=None, group_col=None, target=None,
                       fail_on: "str | Severity" = Severity.ERROR) -> None:
    from ..splits import audit_split
    findings = audit_split(train, test, time_col=time_col, group_col=group_col, target=target)
    _assert(findings, fail_on, "Split audit")


def assert_beats_baseline(X, y, model_score, task, metric="accuracy",
                          fail_on: "str | Severity" = Severity.ERROR) -> None:
    from ..baselines import evaluate_baseline
    _assert(evaluate_baseline(X, y, model_score, task, metric), fail_on, "Baseline")


def assert_metric_appropriate(y_true, metric, y_score=None,
                              fail_on: "str | Severity" = Severity.ERROR) -> None:
    from ..metrics import advise_metric
    _assert(advise_metric(y_true, metric, y_score=y_score), fail_on, "Metric")


def assert_signal_is_real(estimator, X, y, *, fail_on: "str | Severity" = Severity.WARNING,
                          **kwargs) -> None:
    from ..detectors import signal_is_real
    _assert(signal_is_real(estimator, X, y, **kwargs), fail_on, "Permutation test")


def assert_can_overfit(estimator, X, y, *, fail_on: "str | Severity" = Severity.ERROR,
                       **kwargs) -> None:
    from ..detectors import can_overfit_small_sample
    _assert(can_overfit_small_sample(estimator, X, y, **kwargs), fail_on, "Overfit check")


def assert_reproducible(source, fail_on: "str | Severity" = Severity.WARNING) -> None:
    from ..repro import scan_reproducibility
    _assert(scan_reproducibility(source), fail_on, "Reproducibility")


def assert_split_not_inflating(estimator, X, y, *, fail_on: "str | Severity" = Severity.WARNING,
                               **kwargs) -> None:
    from ..impact import split_impact
    _assert(split_impact(estimator, X, y, **kwargs), fail_on, "Split impact")


def pytest_configure(config):  # pragma: no cover - pytest hook
    config.addinivalue_line("markers", "mlcheck: marks tests using mlcheck assertions")
