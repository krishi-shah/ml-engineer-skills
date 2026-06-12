"""Command-line interface for mlcheck.

The headline command is `mlcheck audit`: point it at a training script and/or a
dataset, and it runs every applicable check, prints one consolidated report with
fixes, and exits non-zero in CI when something is wrong.

    mlcheck audit --source train.py --train train.csv --test test.csv \
        --target label --time-col ts --group-col user_id \
        --task classification --metric accuracy --model-score 0.97 --fail-on error
"""
from __future__ import annotations

import argparse
from typing import Optional

from .core import Finding, Severity, format_report, worst_severity


def _load_csv(path: str):
    import pandas as pd
    return pd.read_csv(path)


def _collect_sections(args) -> dict[str, list[Finding]]:
    from .baselines import evaluate_baseline
    from .leakage import scan_data, scan_source
    from .metrics import advise_metric
    from .splits import audit_split

    sections: dict[str, list[Finding]] = {}

    if args.source:
        sections["Source scan (leakage)"] = scan_source(args.source)
        if args.repro:
            from .repro import scan_reproducibility
            sections["Reproducibility"] = scan_reproducibility(args.source)

    train_df = test_df = None
    if args.train and args.test:
        train_df, test_df = _load_csv(args.train), _load_csv(args.test)
        from .detectors import adversarial_validation, target_leak_scan
        adv_train = train_df.drop(columns=[args.target], errors="ignore") if args.target else train_df
        adv_test = test_df.drop(columns=[args.target], errors="ignore") if args.target else test_df
        sections["Adversarial validation"] = adversarial_validation(adv_train, adv_test)
        if args.target:
            sections["Split audit"] = audit_split(
                train_df, test_df, time_col=args.time_col,
                group_col=args.group_col, target=args.target,
            )
            sections["Data scan (leakage)"] = scan_data(train_df, test_df, args.target)
            sections["Deep target-leak scan"] = target_leak_scan(train_df, args.target)

    full_df = None
    if args.data:
        full_df = _load_csv(args.data)

    # Baseline check needs a full dataset, a task, and the reported score.
    if full_df is not None and args.target and args.task and args.model_score is not None:
        X = full_df.drop(columns=[args.target])
        y = full_df[args.target]
        sections["Baseline"] = evaluate_baseline(
            X, y, args.model_score, args.task, args.metric or "accuracy",
        )

    # Metric advice needs labels + a metric name.
    if args.metric:
        labels = None
        if full_df is not None and args.target:
            labels = full_df[args.target]
        elif train_df is not None and args.target:
            labels = train_df[args.target]
        if labels is not None:
            sections["Metric"] = advise_metric(labels, args.metric)

    return sections


def _run_audit(args) -> int:
    from .core import format_json

    sections = _collect_sections(args)
    if not sections:
        print("Nothing to audit. Provide --source and/or --train/--test/--data with --target.")
        return 2

    if getattr(args, "format", "text") == "json":
        print(format_json(sections))
    else:
        print(format_report(sections))
        if getattr(args, "suggest", False):
            from .fixes import suggest_for
            snippets = suggest_for([f for fs in sections.values() for f in fs])
            if snippets:
                print("\n" + snippets)

    worst = worst_severity([f for findings in sections.values() for f in findings])
    threshold = {"never": None, "warning": Severity.WARNING, "error": Severity.ERROR}[args.fail_on]
    if threshold is not None and worst >= threshold:
        if getattr(args, "format", "text") != "json":
            print(f"\nFAIL: worst severity '{worst}' >= --fail-on '{args.fail_on}'.")
        return 1
    if getattr(args, "format", "text") != "json":
        print(f"\nPASS (--fail-on {args.fail_on}).")
    return 0


def _run_report(args) -> int:
    import sys
    from .core import format_markdown

    sections = _collect_sections(args)
    if not sections:
        print("Nothing to report. Provide --source and/or --train/--test/--data with --target.")
        return 2
    md = format_markdown(sections, title=args.title)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"Wrote report to {args.out}")
    else:
        sys.stdout.buffer.write(md.encode("utf-8"))  # utf-8 safe on Windows consoles
        sys.stdout.buffer.write(b"\n")
    return 0


def _print(findings: list[Finding]) -> int:
    for f in findings:
        print(f)
    return 1 if worst_severity(findings) >= Severity.ERROR else 0


def _scan_paths(args, scan_fn) -> int:
    """Scan every path from positional `files` and/or `--source` (pre-commit friendly)."""
    paths = list(getattr(args, "files", None) or [])
    if getattr(args, "source", None):
        paths.append(args.source)
    if not paths:
        print("No files to scan.")
        return 0
    rc = 0
    for path in paths:
        findings = scan_fn(path)
        print(f"# {path}")
        for f in findings:
            print(f)
        if getattr(args, "suggest", False):
            from .fixes import suggest_for
            snippets = suggest_for(findings)
            if snippets:
                print("\n" + snippets)
        rc = max(rc, 1 if worst_severity(findings) >= Severity.ERROR else 0)
    return rc


def _run_scan_source(args) -> int:
    from .leakage import scan_source
    return _scan_paths(args, scan_source)


def _run_scan_repro(args) -> int:
    from .repro import scan_reproducibility
    return _scan_paths(args, scan_reproducibility)


def _run_scan_data(args) -> int:
    from .leakage import scan_data
    findings = scan_data(_load_csv(args.train), _load_csv(args.test), args.target)
    if getattr(args, "deep", False):
        from .detectors import target_leak_scan
        findings = findings + target_leak_scan(_load_csv(args.train), args.target)
    return _print(findings)


def _run_adversarial(args) -> int:
    from .detectors import adversarial_validation
    train, test = _load_csv(args.train), _load_csv(args.test)
    if args.target:
        train = train.drop(columns=[args.target], errors="ignore")
        test = test.drop(columns=[args.target], errors="ignore")
    return _print(adversarial_validation(train, test))


def _run_audit_split(args) -> int:
    from .splits import audit_split
    return _print(audit_split(
        _load_csv(args.train), _load_csv(args.test),
        time_col=args.time_col, group_col=args.group_col, target=args.target,
    ))


def _run_baseline(args) -> int:
    from .baselines import evaluate_baseline
    df = _load_csv(args.data)
    return _print(evaluate_baseline(
        df.drop(columns=[args.target]), df[args.target],
        args.model_score, args.task, args.metric or "accuracy",
    ))


def _run_metric(args) -> int:
    from .metrics import advise_metric
    df = _load_csv(args.data)
    return _print(advise_metric(df[args.target], args.metric))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlcheck", description="Catch the silent ML mistakes.")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="run every applicable check and emit one report")
    a.add_argument("--source", help="training script to static-scan for leakage")
    a.add_argument("--data", help="full dataset CSV (for baseline/metric checks)")
    a.add_argument("--train", help="train split CSV")
    a.add_argument("--test", help="test split CSV")
    a.add_argument("--target", help="target column name")
    a.add_argument("--time-col")
    a.add_argument("--group-col")
    a.add_argument("--task", choices=["classification", "regression"])
    a.add_argument("--metric")
    a.add_argument("--model-score", type=float)
    a.add_argument("--repro", action="store_true", help="also scan --source for unseeded randomness")
    a.add_argument("--format", choices=["text", "json"], default="text", help="output format")
    a.add_argument("--suggest", action="store_true", help="print paste-ready fix snippets")
    a.add_argument("--fail-on", choices=["error", "warning", "never"], default="error")
    a.set_defaults(func=_run_audit)

    rep = sub.add_parser("report", help="write a shareable markdown report (for a PR / writeup)")
    for flag in ("--source", "--data", "--train", "--test", "--target", "--time-col",
                 "--group-col", "--task", "--metric"):
        rep.add_argument(flag)
    rep.add_argument("--model-score", type=float)
    rep.add_argument("--repro", action="store_true")
    rep.add_argument("--title", default="mlcheck report")
    rep.add_argument("--out", help="output file (default: stdout)")
    rep.set_defaults(func=_run_report)

    s = sub.add_parser("scan-source", help="static scan a script for fit-before-split leakage")
    s.add_argument("files", nargs="*", help="files to scan (pre-commit passes these)")
    s.add_argument("--source", help="single file to scan")
    s.add_argument("--suggest", action="store_true", help="print paste-ready fix snippets")
    s.set_defaults(func=_run_scan_source)

    r = sub.add_parser("scan-repro", help="scan a script for unseeded randomness")
    r.add_argument("files", nargs="*", help="files to scan (pre-commit passes these)")
    r.add_argument("--source", help="single file to scan")
    r.set_defaults(func=_run_scan_repro)

    d = sub.add_parser("scan-data", help="scan train/test data for overlap and target leakage")
    d.add_argument("--train", required=True)
    d.add_argument("--test", required=True)
    d.add_argument("--target", required=True)
    d.add_argument("--deep", action="store_true", help="add single-feature (nonlinear) target-leak scan")
    d.set_defaults(func=_run_scan_data)

    adv = sub.add_parser("adversarial", help="test whether train and test are distinguishable")
    adv.add_argument("--train", required=True)
    adv.add_argument("--test", required=True)
    adv.add_argument("--target", help="target column to exclude from the comparison")
    adv.set_defaults(func=_run_adversarial)

    sp = sub.add_parser("audit-split", help="audit a train/test split")
    sp.add_argument("--train", required=True)
    sp.add_argument("--test", required=True)
    sp.add_argument("--target")
    sp.add_argument("--time-col")
    sp.add_argument("--group-col")
    sp.set_defaults(func=_run_audit_split)

    b = sub.add_parser("baseline", help="compute the dumb baseline a model must beat")
    b.add_argument("--data", required=True)
    b.add_argument("--target", required=True)
    b.add_argument("--task", choices=["classification", "regression"], required=True)
    b.add_argument("--metric")
    b.add_argument("--model-score", type=float, required=True)
    b.set_defaults(func=_run_baseline)

    m = sub.add_parser("metric", help="check whether a metric fits the labels")
    m.add_argument("--data", required=True)
    m.add_argument("--target", required=True)
    m.add_argument("--metric", required=True)
    m.set_defaults(func=_run_metric)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
