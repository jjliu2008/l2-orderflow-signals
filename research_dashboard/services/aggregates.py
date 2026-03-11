from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def bucket_series(s: pd.Series, bins: int = 5) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.dropna().nunique() < 2:
        return pd.Series(["single"] * len(x), index=x.index)
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = x.quantile(q).dropna().unique()
    if len(edges) < 2:
        return pd.Series(["single"] * len(x), index=x.index)
    labels = [f"Q{i}" for i in range(1, len(edges))]
    return pd.cut(x, bins=edges, labels=labels, include_lowest=True, duplicates="drop").astype(str)


def grouped_ev(df: pd.DataFrame, group_col: str, pnl_col: str = "ticks_pnl_net") -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "n", "ev_per_trade", "win_rate", "profit_factor"])
    rows = []
    for key, g in df.groupby(group_col):
        pnl = g[pnl_col]
        rows.append(
            {
                group_col: key,
                "n": int(len(g)),
                "ev_per_trade": metrics.ev_per_trade(pnl),
                "win_rate": metrics.win_rate(pnl),
                "profit_factor": metrics.profit_factor(pnl),
                "total_ticks": float(pd.to_numeric(pnl, errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("ev_per_trade", ascending=False)


def feature_bucket_summary(
    merged_df: pd.DataFrame,
    feature_col: str,
    bins: int = 5,
    pnl_col: str = "ticks_pnl_net",
) -> pd.DataFrame:
    if merged_df.empty or feature_col not in merged_df.columns:
        return pd.DataFrame(columns=["bucket", "n", "ev_per_trade", "win_rate", "profit_factor"])
    tmp = merged_df.copy()
    tmp["bucket"] = bucket_series(tmp[feature_col], bins=bins)
    return grouped_ev(tmp, "bucket", pnl_col=pnl_col)


def winner_loser_distributions(
    merged_df: pd.DataFrame,
    feature_cols: list[str],
    pnl_col: str = "ticks_pnl_net",
) -> pd.DataFrame:
    if merged_df.empty:
        return pd.DataFrame(columns=["feature", "winner_mean", "loser_mean", "winner_median", "loser_median"])
    out = []
    winners = merged_df[merged_df[pnl_col] > 0]
    losers = merged_df[merged_df[pnl_col] <= 0]
    for c in feature_cols:
        if c not in merged_df.columns:
            continue
        w = pd.to_numeric(winners[c], errors="coerce").dropna()
        l = pd.to_numeric(losers[c], errors="coerce").dropna()
        out.append(
            {
                "feature": c,
                "winner_mean": float(w.mean()) if len(w) else None,
                "loser_mean": float(l.mean()) if len(l) else None,
                "winner_median": float(w.median()) if len(w) else None,
                "loser_median": float(l.median()) if len(l) else None,
                "winner_n": int(len(w)),
                "loser_n": int(len(l)),
            }
        )
    return pd.DataFrame(out)


def variant_comparison(
    summary_variants: pd.DataFrame,
    strategy_name: str | None = None,
) -> pd.DataFrame:
    if summary_variants.empty:
        return summary_variants
    out = summary_variants.copy()
    if strategy_name:
        out = out[out["strategy_name"] == strategy_name]
    cols = [
        "strategy_name",
        "variant_name",
        "n_trades",
        "ev_net",
        "wr",
        "pf",
        "max_dd_ticks",
        "top1_share",
        "top3_share",
        "top5_share",
        "oos_ev",
        "jan_ev",
        "feb_ev",
        "notes",
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values("ev_net", ascending=False)
