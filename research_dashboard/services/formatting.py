from __future__ import annotations


def fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.{digits}f}%"


def fmt_num(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def fmt_ticks(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f} ticks"


def fmt_currency(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"${x:,.{digits}f}"
