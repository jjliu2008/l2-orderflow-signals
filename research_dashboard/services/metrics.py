from __future__ import annotations

import numpy as np
import pandas as pd


def _as_numeric_series(obj) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(dtype=float)
        obj = obj.iloc[:, 0]
    return pd.to_numeric(obj, errors="coerce")


def ev_per_trade(pnl: pd.Series) -> float | None:
    s = _as_numeric_series(pnl).dropna()
    if s.empty:
        return None
    return float(s.mean())


def win_rate(pnl: pd.Series) -> float | None:
    s = _as_numeric_series(pnl).dropna()
    if s.empty:
        return None
    return float((s > 0).mean())


def profit_factor(pnl: pd.Series) -> float | None:
    s = _as_numeric_series(pnl).dropna()
    if s.empty:
        return None
    gross_win = float(s[s > 0].sum())
    gross_loss = float(-s[s <= 0].sum())
    if gross_loss <= 0:
        return None
    return gross_win / gross_loss


def max_drawdown_ticks(pnl: pd.Series) -> float | None:
    s = _as_numeric_series(pnl).dropna()
    if s.empty:
        return None
    equity = s.cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def rolling_ev(pnl: pd.Series, window: int = 10) -> pd.Series:
    s = _as_numeric_series(pnl)
    return s.rolling(window=window, min_periods=window).mean()


def rolling_hit_rate(pnl: pd.Series, window: int = 10) -> pd.Series:
    s = _as_numeric_series(pnl)
    return (s > 0).astype(float).rolling(window=window, min_periods=window).mean()


def rolling_drawdown(pnl: pd.Series, window: int = 10) -> pd.Series:
    s = _as_numeric_series(pnl).fillna(0.0)
    eq = s.cumsum()
    runmax = eq.rolling(window=window, min_periods=1).max()
    return eq - runmax


def contribution_shares(pnl: pd.Series) -> dict[str, float | None]:
    s = _as_numeric_series(pnl).dropna().sort_values(ascending=False)
    if s.empty:
        return {
            "top1_share": None,
            "top3_share": None,
            "top5_share": None,
            "top1_ticks": None,
            "top3_ticks": None,
            "top5_ticks": None,
            "total_ticks": 0.0,
        }
    total = float(s.sum())
    top1 = float(s.head(1).sum())
    top3 = float(s.head(3).sum())
    top5 = float(s.head(5).sum())
    if total > 0:
        top1_share = top1 / total
        top3_share = top3 / total
        top5_share = top5 / total
    else:
        top1_share = None
        top3_share = None
        top5_share = None
    return {
        "top1_share": top1_share,
        "top3_share": top3_share,
        "top5_share": top5_share,
        "top1_ticks": top1,
        "top3_ticks": top3,
        "top5_ticks": top5,
        "total_ticks": total,
    }


def outlier_removed_metrics(pnl: pd.Series, remove_top_n: int = 0, remove_bottom_n: int = 0) -> dict:
    s = _as_numeric_series(pnl).dropna().sort_values(ascending=False).reset_index(drop=True)
    if remove_top_n > 0:
        s = s.iloc[remove_top_n:]
    if remove_bottom_n > 0 and len(s) > 0:
        s = s.iloc[: len(s) - remove_bottom_n]
    return {
        "n": int(len(s)),
        "ev_per_trade": ev_per_trade(s),
        "win_rate": win_rate(s),
        "profit_factor": profit_factor(s),
        "max_drawdown_ticks": max_drawdown_ticks(s),
        "concentration": contribution_shares(s),
    }


def monthly_breakdown(df: pd.DataFrame, pnl_col: str = "ticks_pnl_net") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_pnl_work"] = _as_numeric_series(tmp[pnl_col])
    out = (
        tmp.groupby("month", as_index=False)["_pnl_work"]
        .agg(
            n="count",
            total_ticks="sum",
            ev_per_trade="mean",
            win_rate=lambda s: float((s > 0).mean()),
            profit_factor=lambda s: profit_factor(s),
        )
        .sort_values("month")
    )
    return out


def session_breakdown(df: pd.DataFrame, pnl_col: str = "ticks_pnl_net") -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["_pnl_work"] = _as_numeric_series(tmp[pnl_col])
    out = (
        tmp.groupby("session", as_index=False)["_pnl_work"]
        .agg(
            n="count",
            total_ticks="sum",
            ev_per_trade="mean",
            win_rate=lambda s: float((s > 0).mean()),
            profit_factor=lambda s: profit_factor(s),
        )
        .sort_values("session")
    )
    return out


def compute_core_metrics(df: pd.DataFrame, pnl_col: str = "ticks_pnl_net") -> dict:
    if df.empty:
        return {
            "n_trades": 0,
            "ev_per_trade": None,
            "win_rate": None,
            "profit_factor": None,
            "max_drawdown_ticks": None,
            "total_ticks": 0.0,
            "top1_share": None,
            "top3_share": None,
            "top5_share": None,
        }

    pnl = _as_numeric_series(df[pnl_col]).dropna()
    contrib = contribution_shares(pnl)
    return {
        "n_trades": int(len(pnl)),
        "ev_per_trade": ev_per_trade(pnl),
        "win_rate": win_rate(pnl),
        "profit_factor": profit_factor(pnl),
        "max_drawdown_ticks": max_drawdown_ticks(pnl),
        "total_ticks": float(pnl.sum()),
        "top1_share": contrib["top1_share"],
        "top3_share": contrib["top3_share"],
        "top5_share": contrib["top5_share"],
    }


def add_cost_column(
    df: pd.DataFrame,
    base_cost_ticks: float,
    override_enabled: bool = False,
    override_ticks: float = 0.0,
    gross_col: str = "ticks_pnl_gross",
    out_col: str = "ticks_pnl_net",
) -> pd.DataFrame:
    out = df.copy()
    cost = float(override_ticks if override_enabled else base_cost_ticks)
    out[out_col] = pd.to_numeric(out[gross_col], errors="coerce") - cost
    out["applied_cost_ticks"] = cost
    return out
