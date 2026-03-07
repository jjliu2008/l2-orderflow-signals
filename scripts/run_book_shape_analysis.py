"""
Book-shape follow-through analysis.

For each gated trade entry, look up the new level-2-10 features from the
rebuilt parquet at the exact entry bar and test whether book shape / sweep
cost discriminates follow-through vs dead-on-arrival entries.

Primary test: split entries into terciles by entry-side sweep cost
(sweep_cost_buy5_ticks for long, sweep_cost_sell5_ticks for short).
Report median MAE and follow-through rate per tercile.

Secondary: Pearson/Spearman correlations of each new book feature vs MAE
and vs follow_plus1 (mfe_ticks >= 1).

Usage:
    python scripts/run_book_shape_analysis.py \\
        --trades-root artifacts/experiment_orchestrator \\
        --parquet-root data/processed \\
        --out artifacts/book_shape_analysis.json

    # or point at a specific run directory
    python scripts/run_book_shape_analysis.py \\
        --trades-root "artifacts/experiment_agent/pflft_v8_regime_v1/runs/A_cd0_prog0_none/artifacts/run_20260304_214204" \\
        --parquet-root data/processed \\
        --out artifacts/book_shape_analysis.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── feature groups ────────────────────────────────────────────────────────────

# These are pulled from the parquet and aligned to entry side.
# "bid_side" features are used for short entries; "ask_side" for long entries.
_BOOK_FEATURES_ALIGNED = {
    "sweep_cost_5ticks": ("sweep_cost_buy5_ticks", "sweep_cost_sell5_ticks"),
    "sweep_cost_1ticks": ("sweep_cost_buy1_ticks", "sweep_cost_sell1_ticks"),
    "depth_shape_ratio": ("depth_shape_ratio_ask", "depth_shape_ratio_bid"),
    "book_depth_slope": ("book_depth_slope_ask", "book_depth_slope_bid"),
}

_PARQUET_COLS_NEEDED = [
    "Time",
    "sweep_cost_buy5_ticks", "sweep_cost_sell5_ticks",
    "sweep_cost_buy1_ticks", "sweep_cost_sell1_ticks",
    "depth_shape_ratio_bid", "depth_shape_ratio_ask",
    "book_depth_slope_bid", "book_depth_slope_ask",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_trades(root: Path, pattern: str) -> pd.DataFrame:
    paths = [Path(p) for p in glob.glob(str(root / pattern), recursive=True)]
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "entry_time" not in df.columns:
            continue
        # Infer date from path if 'date' column is missing
        if "date" not in df.columns:
            # path looks like .../ES_2025-12-02/trades_gated.csv
            stem = p.parent.name  # e.g. ES_2025-12-02
            parts = stem.split("_", 1)
            df["date"] = parts[1] if len(parts) == 2 else ""
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    return out


def _load_parquets(root: Path, dates: List[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        p = root / f"instrument=ES" / f"date={date}" / "features_labels.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=[c for c in _PARQUET_COLS_NEEDED if c != "Time"] + ["Time"])
        except Exception:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["Time"] = pd.to_datetime(out["Time"], utc=True, errors="coerce")
    return out


def _align_feature(df: pd.DataFrame, feat_name: str) -> pd.Series:
    """Return the entry-side-aligned version of a (buy_col, sell_col) pair."""
    buy_col, sell_col = _BOOK_FEATURES_ALIGNED[feat_name]
    is_long = df["side"].astype(str).str.lower() == "long"
    buy_vals = pd.to_numeric(df.get(buy_col, pd.Series(np.nan, index=df.index)), errors="coerce")
    sell_vals = pd.to_numeric(df.get(sell_col, pd.Series(np.nan, index=df.index)), errors="coerce")
    return buy_vals.where(is_long, sell_vals)


def _corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 4:
        return float("nan"), float("nan"), n
    return (
        float(x[mask].corr(y[mask], method="pearson")),
        float(x[mask].corr(y[mask], method="spearman")),
        n,
    )


def _tercile_stats(df: pd.DataFrame, signal: pd.Series, label: str) -> Optional[Dict]:
    """Split df into terciles by signal; compute follow-through + MAE per tercile."""
    mask = signal.notna() & df["mfe_ticks"].notna() & df["mae_ticks"].notna()
    sub = df[mask].copy()
    sig_sub = signal[mask].copy()

    if len(sub) < 9:  # need at least 3 per bin
        return None

    try:
        bins = pd.qcut(sig_sub, q=3, labels=["low", "mid", "high"], duplicates="drop")
    except Exception:
        return None

    if bins.nunique() < 2:
        return None

    sub["_bin"] = bins.values
    sub["_mfe"] = pd.to_numeric(sub["mfe_ticks"], errors="coerce")
    sub["_mae"] = pd.to_numeric(sub["mae_ticks"], errors="coerce")
    sub["_follow"] = (sub["_mfe"] >= 1.0).astype(float)
    sub["_pnl"] = pd.to_numeric(sub["pnl_ticks"], errors="coerce")

    rows = []
    for bin_name in ["low", "mid", "high"]:
        grp = sub[sub["_bin"] == bin_name]
        if grp.empty:
            continue
        rows.append({
            "bin": bin_name,
            "n": int(len(grp)),
            "follow_plus1_rate": float(grp["_follow"].mean()),
            "median_mfe_ticks": float(grp["_mfe"].median()),
            "median_mae_ticks": float(grp["_mae"].median()),
            "median_pnl_ticks": float(grp["_pnl"].median()) if grp["_pnl"].notna().any() else float("nan"),
            "mean_pnl_ticks": float(grp["_pnl"].mean()) if grp["_pnl"].notna().any() else float("nan"),
            "signal_p25": float(sig_sub[sub["_bin"] == bin_name].quantile(0.25)),
            "signal_p75": float(sig_sub[sub["_bin"] == bin_name].quantile(0.75)),
        })

    if len(rows) < 2:
        return None

    # Primary test: monotonicity of follow rate low→high
    follow_rates = [r["follow_plus1_rate"] for r in rows]
    n_total = sum(r["n"] for r in rows)
    follow_low = rows[0]["follow_plus1_rate"]
    follow_high = rows[-1]["follow_plus1_rate"]

    return {
        "feature": label,
        "n_total": n_total,
        "tercile_bins": rows,
        "follow_rate_low_minus_high": round(follow_low - follow_high, 4),
        "monotone_ascending": bool(all(
            rows[i]["follow_plus1_rate"] <= rows[i + 1]["follow_plus1_rate"]
            for i in range(len(rows) - 1)
        )),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    trades_root = Path(args.trades_root)
    parquet_root = Path(args.parquet_root)

    trades = _load_trades(trades_root, args.pattern)
    if trades.empty:
        print("No trades found.")
        return 1

    # Filter to PFLFT_v8 gated entries only
    if "strategy" in trades.columns:
        trades = trades[trades["strategy"].astype(str).str.lower() == "gated"]
    if "entry_family" in trades.columns:
        trades = trades[trades["entry_family"].astype(str).str.startswith("PFLFT_v8")]

    if trades.empty:
        print("No gated PFLFT_v8 trades found after filtering.")
        return 1

    n_raw = len(trades)
    print(f"Loaded {n_raw} gated PFLFT_v8 trade entries.")

    # Identify dates to load from parquet
    dates = sorted(trades["date"].dropna().astype(str).unique().tolist())
    print(f"Loading parquets for {len(dates)} dates: {dates}")
    parquet = _load_parquets(parquet_root, dates)

    if parquet.empty:
        print("No rebuilt parquets found — run the full reprocess first.")
        return 1

    new_cols = [c for c in _PARQUET_COLS_NEEDED if c != "Time"]

    # Build a per-date index so each trade is joined only to its own date's bars.
    parquet["_date"] = parquet["Time"].dt.strftime("%Y-%m-%d")
    date_indices: Dict[str, pd.DataFrame] = {}
    for date_str, grp in parquet.groupby("_date"):
        date_indices[str(date_str)] = grp.set_index("Time").sort_index()

    rebuilt_dates = set(date_indices.keys())
    print(f"Rebuilt dates available: {sorted(rebuilt_dates)}")

    # Join trades → parquet on entry_time == Time (nearest bar, same date only)
    joined_rows = []
    skipped_no_parquet = 0
    for _, trade in trades.iterrows():
        ts = trade["entry_time"]
        date_str = str(trade.get("date", ""))
        if pd.isnull(ts) or not date_str:
            continue
        if date_str not in date_indices:
            skipped_no_parquet += 1
            continue
        idx_df = date_indices[date_str]
        try:
            idx = idx_df.index.get_indexer([ts], method="nearest")[0]
            if idx < 0:
                continue
            bar = idx_df.iloc[idx]
            row = trade.to_dict()
            for col in new_cols:
                row[col] = float(bar[col]) if col in bar.index else np.nan
            joined_rows.append(row)
        except Exception:
            continue

    if skipped_no_parquet:
        print(f"Skipped {skipped_no_parquet} trades: date not in rebuilt parquets yet.")

    if not joined_rows:
        print("Join produced no rows — check entry_time format vs parquet Time.")
        return 1

    df = pd.DataFrame(joined_rows)
    n_joined = len(df)
    print(f"Joined {n_joined}/{n_raw} trades to parquet bars.")

    # Align features to entry side
    for feat_name in _BOOK_FEATURES_ALIGNED:
        df[f"aligned_{feat_name}"] = _align_feature(df, feat_name)

    # ── Tercile analysis ────────────────────────────────────────────────────
    tercile_results = []
    for feat_name in _BOOK_FEATURES_ALIGNED:
        sig = df[f"aligned_{feat_name}"].astype(float)
        result = _tercile_stats(df, sig, label=feat_name)
        if result:
            tercile_results.append(result)
            low_r = result["tercile_bins"][0]["follow_plus1_rate"]
            high_r = result["tercile_bins"][-1]["follow_plus1_rate"]
            mid_r = result["tercile_bins"][1]["follow_plus1_rate"] if len(result["tercile_bins"]) > 1 else float("nan")
            print(f"  {feat_name:<25s}  follow low={low_r:.3f} mid={mid_r:.3f} high={high_r:.3f}  d={result['follow_rate_low_minus_high']:+.3f}")

    # ── Correlation analysis ─────────────────────────────────────────────────
    follow_plus1 = (pd.to_numeric(df["mfe_ticks"], errors="coerce") >= 1.0).astype(float)
    mae = pd.to_numeric(df["mae_ticks"], errors="coerce")

    corr_rows = []
    for feat_name in _BOOK_FEATURES_ALIGNED:
        sig = df[f"aligned_{feat_name}"].astype(float)
        p_fol, s_fol, n_fol = _corr(sig, follow_plus1)
        p_mae, s_mae, n_mae = _corr(sig, mae)
        corr_rows.append({
            "feature": feat_name,
            "pearson_vs_follow_plus1": round(p_fol, 4) if np.isfinite(p_fol) else None,
            "spearman_vs_follow_plus1": round(s_fol, 4) if np.isfinite(s_fol) else None,
            "pearson_vs_mae_ticks": round(p_mae, 4) if np.isfinite(p_mae) else None,
            "spearman_vs_mae_ticks": round(s_mae, 4) if np.isfinite(s_mae) else None,
            "n": n_fol,
        })

    base_follow_rate = float(follow_plus1.mean()) if follow_plus1.notna().any() else float("nan")
    base_mae = float(mae.mean()) if mae.notna().any() else float("nan")

    out = {
        "n_trades_raw": n_raw,
        "n_trades_joined": n_joined,
        "dates": dates,
        "base_follow_plus1_rate": round(base_follow_rate, 4),
        "base_mean_mae_ticks": round(base_mae, 4),
        "tercile_analysis": tercile_results,
        "correlation_analysis": corr_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Book-shape follow-through analysis using rebuilt MBP-10 parquets.")
    p.add_argument(
        "--trades-root",
        default="artifacts/experiment_agent/pflft_v8_regime_v1/runs/A_cd0_prog0_none/artifacts/run_20260304_214204",
        help="Root directory to search for trades_gated.csv files.",
    )
    p.add_argument(
        "--pattern",
        default="**/trades_gated.csv",
        help="Glob pattern for trade files under --trades-root.",
    )
    p.add_argument(
        "--parquet-root",
        default="data/processed",
        help="Root of rebuilt processed parquets (must contain level-2-10 columns).",
    )
    p.add_argument(
        "--out",
        default="artifacts/book_shape_analysis.json",
        help="Output report JSON path.",
    )
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(_parse_args()))
