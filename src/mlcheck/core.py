"""Shared core: the Finding type, severity ordering, and report formatting.

Every diagnostic returns a list of `Finding`. A finding carries not just *what*
is wrong but *how to fix it* and *which skill* explains why — so the output is
actionable on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Severity(IntEnum):
    """Ordered so the worst finding is the maximum."""

    INFO = 0
    WARNING = 1
    ERROR = 2

    @classmethod
    def parse(cls, value: "str | Severity") -> "Severity":
        if isinstance(value, Severity):
            return value
        return cls[str(value).strip().upper()]

    def __str__(self) -> str:  # noqa: D105
        return self.name.lower()


# ASCII markers so output never crashes on a Windows cp1252 console.
_ICON = {Severity.INFO: "[-]", Severity.WARNING: "[!]", Severity.ERROR: "[x]"}


@dataclass
class Finding:
    """One result from a diagnostic.

    severity : INFO / WARNING / ERROR
    code     : stable machine-readable identifier (e.g. "FIT_BEFORE_SPLIT")
    message  : what is wrong, in plain language
    fix      : the concrete remediation (optional but strongly encouraged)
    skill    : the skill that explains the principle (optional)
    location : file:line or column name the finding refers to (optional)
    """

    severity: Severity
    code: str
    message: str
    fix: str = ""
    skill: str = ""
    location: str = ""

    def __post_init__(self) -> None:
        self.severity = Severity.parse(self.severity)

    def as_dict(self) -> dict:
        """JSON-serializable form (severity rendered as a lowercase string)."""
        return {
            "severity": str(self.severity),
            "code": self.code,
            "message": self.message,
            "fix": self.fix,
            "skill": self.skill,
            "location": self.location,
        }

    def __str__(self) -> str:
        head = f"[{str(self.severity).upper()}] {self.code}"
        if self.location:
            head += f" ({self.location})"
        out = f"{head}: {self.message}"
        if self.fix:
            out += f"\n    fix: {self.fix}"
        return out


def worst_severity(findings: list[Finding]) -> Severity:
    """The maximum severity across findings (INFO if empty)."""
    return max((f.severity for f in findings), default=Severity.INFO)


def format_report(sections: dict[str, list[Finding]], *, color: bool = False) -> str:
    """Render a multi-section report. `sections` maps a check name to its findings."""
    lines: list[str] = []
    counts = {Severity.ERROR: 0, Severity.WARNING: 0, Severity.INFO: 0}

    for name, findings in sections.items():
        if not findings:
            continue
        lines.append(f"-- {name} " + "-" * max(0, 56 - len(name)))
        for f in sorted(findings, key=lambda x: -int(x.severity)):
            counts[f.severity] += 1
            icon = _ICON[f.severity]
            loc = f" ({f.location})" if f.location else ""
            lines.append(f"  {icon} {f.code}{loc}: {f.message}")
            if f.fix:
                lines.append(f"      -> fix: {f.fix}")
            if f.skill:
                lines.append(f"      -> skill: ml-engineer-skills:{f.skill}")
        lines.append("")

    summary = (
        f"{counts[Severity.ERROR]} error(s), "
        f"{counts[Severity.WARNING]} warning(s), "
        f"{counts[Severity.INFO]} note(s)"
    )
    lines.append("=" * 60)
    lines.append(f"  {summary}")
    return "\n".join(lines)


def format_json(sections: dict[str, list[Finding]]) -> str:
    """Machine-readable report for CI/tooling: {section: [finding dicts...]}."""
    import json

    payload = {
        "summary": _counts(sections),
        "worst_severity": str(worst_severity([f for fs in sections.values() for f in fs])),
        "sections": {name: [f.as_dict() for f in fs] for name, fs in sections.items()},
    }
    return json.dumps(payload, indent=2)


def format_markdown(sections: dict[str, list[Finding]], *, title: str = "mlcheck report") -> str:
    """A shareable markdown report — attach to a PR or an assignment writeup."""
    # ASCII badges (not emoji) so the report is safe to print on any console.
    icons = {Severity.ERROR: "**ERROR**", Severity.WARNING: "**WARN**", Severity.INFO: "note"}
    out = [f"# {title}", ""]
    c = _counts(sections)
    out.append(f"**{c['error']} error(s), {c['warning']} warning(s), {c['info']} note(s)**")
    out.append("")
    for name, findings in sections.items():
        if not findings:
            continue
        out.append(f"## {name}")
        out.append("")
        out.append("| | Code | What | Fix |")
        out.append("|---|---|---|---|")
        for f in sorted(findings, key=lambda x: -int(x.severity)):
            loc = f" ({f.location})" if f.location else ""
            fix = f.fix.replace("|", "\\|") if f.fix else ""
            msg = f.message.replace("|", "\\|")
            out.append(f"| {icons[f.severity]} | `{f.code}`{loc} | {msg} | {fix} |")
        out.append("")
    return "\n".join(out)


def _counts(sections: dict[str, list[Finding]]) -> dict:
    c = {"error": 0, "warning": 0, "info": 0}
    for findings in sections.values():
        for f in findings:
            c[str(f.severity)] += 1
    return c
