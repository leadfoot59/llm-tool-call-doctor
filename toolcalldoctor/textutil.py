"""Tiny text formatting helpers for CLI reports."""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["table", "pct", "truncate", "bar"]


def truncate(text: Any, width: int = 60) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) <= width:
        return flat
    return flat[: width - 1] + "\u2026"


def pct(numerator: float, denominator: float) -> str:
    if not denominator:
        return "n/a"
    return f"{100.0 * numerator / denominator:.0f}%"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], indent: str = "  ") -> str:
    """Render a plain-text table sized to its content."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))
    lines = [indent + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    lines.append(indent + "  ".join("-" * w for w in widths))
    for row in cells:
        lines.append(
            indent
            + "  ".join(
                (row[i] if i < len(row) else "").ljust(widths[i]) for i in range(len(widths))
            ).rstrip()
        )
    return "\n".join(lines)


def bar(value: float, total: float, width: int = 20) -> str:
    if not total:
        return ""
    filled = int(round(width * value / total))
    return "#" * filled + "." * (width - filled)
