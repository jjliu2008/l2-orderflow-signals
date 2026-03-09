"""
Raw MFE/MAE landscape analysis at 5, 15, 30 minute horizons on ESH6 Jan 2026.

No signal filter — unconditional distribution of forward favorable and adverse
excursions from every bar. This shows what TP/SL structures are structurally
viable at the 5-30 minute timescale before any signal is layered on.

Usage:
    python scripts/analyze_mfe_mae_horizons.py

Outputs:
    artifacts/mfe_mae_analysis/horizon_landscape.json
    (prints summary table to stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/mfe_mae_analysis"

TICK = 0.25

# Horizons in bars (100ms bars)
HORIZONS = {
    "5min":  3_000,
    "15min": 9_000,
    "30min": 18_000,
}

# RTH: 09:30-16:00 ET = 14:30-21:00 UTC
RTH_START_UTC = "14:30"
RTH_END_UTC   = "21:00"

# Candidate TP and SL values (in ticks) for viability table
TP_CANDIDATES = [6, 8, 10, 12, 15, 18, 20, 25, 30]
SL_CANDIDATES = [3, 4, 5, 6, 8, 10]


# ── Efficient forward rolling max/min via reverse-rolling trick ──────────────
def _fwd_max(series: pd.Series, H: int) -> np.ndarray:
    """Forward rolling max over next H bars (inclusive of current bar)."""
    rev = series.values[::-1]
    result = pd.Series(rev).rolling(H, min_periods=1).max().values
    return result[::-1]


def _fwd_min(series: pd.Series, H: int) -> np.ndarray:
    """Forward rolling min over next H bars (inclusive of current bar)."""
    rev = series.values[::-1]
    result = pd.Series(rev).rolling(H, min_periods=1).min().values
    return result[::-1]


def load_date(date_dir: Path) -> pd.DataFrame | None:
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq, columns=["Time", "mid"])
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    return df


def compute_horizons(df: pd.DataFrame) -> pd.DataFrame:
    """Add forward MFE/MAE columns at each horizon to full-day df."""
    mid = df["mid"]
    for name, H in HORIZONS.items():
        fmax = _fwd_max(mid, H)
        fmin = _fwd_min(mid, H)
        df[f"mfe_long_{name}"]  = (fmax - mid.values) / TICK
        df[f"mae_long_{name}"]  = (mid.values - fmin) / TICK
        df[f"mfe_short_{name}"] = (mid.values - fmin) / TICK
        df[f"mae_short_{name}"] = (fmax - mid.values) / TICK
    return df


def rth_filter(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    lo = pd.Timestamp(f"{date_str} {RTH_START_UTC}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END_UTC}:00",   tz="UTC")
    return df[(df["Time"] >= lo) & (df["Time"] < hi)]


def percentile_row(arr: np.ndarray) -> dict:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    return {
        "n":   len(arr),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def viability_table(
    mfe: np.ndarray,
    mae: np.ndarray,
    tp_vals: list[int],
    sl_vals: list[int],
) -> dict:
    """
    For each (TP, SL) pair:
      p_tp  = P(MFE >= TP)  — price reaches TP within horizon
      p_sl  = P(MAE >= SL)  — price reaches SL within horizon
      p_win = P(MFE >= TP and MAE < SL)  — path-feasible win proxy
    """
    n = len(mfe)
    result = {}
    for tp in tp_vals:
        for sl in sl_vals:
            p_tp  = float((mfe >= tp).mean())
            p_sl  = float((mae >= sl).mean())
            p_win = float(((mfe >= tp) & (mae < sl)).mean())
            # Breakeven WR for this TP/SL ratio
            be_wr = sl / (tp + sl)
            result[f"TP{tp}_SL{sl}"] = {
                "tp": tp, "sl": sl,
                "ratio": round(tp / sl, 2),
                "breakeven_wr": round(be_wr, 3),
                "p_tp_hit":    round(p_tp,  3),
                "p_sl_hit":    round(p_sl,  3),
                "p_path_win":  round(p_win, 3),
                # How many times more likely to hit TP than (TP AND no SL) — path purity
                "path_purity": round(p_win / p_tp, 3) if p_tp > 0 else 0,
            }
    return result


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-01-*"))
    if not date_dirs:
        print(f"No Jan parquets found under {DATA_ROOT}")
        return
    print(f"Loading {len(date_dirs)} Jan 2026 dates...")

    all_rth: list[pd.DataFrame] = []

    for dd in date_dirs:
        date_str = dd.name.replace("date=", "")
        df = load_date(dd)
        if df is None:
            print(f"  {date_str}: missing")
            continue
        df = compute_horizons(df)
        rth = rth_filter(df, date_str)
        all_rth.append(rth)
        print(f"  {date_str}: {len(rth):,} RTH bars")

    if not all_rth:
        print("No data loaded.")
        return

    combined = pd.concat(all_rth, ignore_index=True)
    print(f"\nTotal RTH bars across all dates: {len(combined):,}")

    results: dict = {
        "metadata": {
            "instrument":  "ESH6",
            "period":      "2026-01 RTH",
            "dates":       len(date_dirs),
            "total_bars":  len(combined),
            "bar_size_ms": 100,
            "session":     "RTH 09:30-16:00 ET",
            "tick_size":   TICK,
        },
        "horizons": {},
    }

    for name in HORIZONS:
        mfe_l = combined[f"mfe_long_{name}"].values
        mae_l = combined[f"mae_long_{name}"].values
        mfe_s = combined[f"mfe_short_{name}"].values
        mae_s = combined[f"mae_short_{name}"].values

        # For direction-agnostic analysis, take mean of long+short
        # (ES is symmetric so results should be similar)
        results["horizons"][name] = {
            "mfe_long":  percentile_row(mfe_l),
            "mae_long":  percentile_row(mae_l),
            "mfe_short": percentile_row(mfe_s),
            "mae_short": percentile_row(mae_s),
            "viability_long":  viability_table(mfe_l, mae_l, TP_CANDIDATES, SL_CANDIDATES),
            "viability_short": viability_table(mfe_s, mae_s, TP_CANDIDATES, SL_CANDIDATES),
        }

    # ── Print landscape summary ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("MFE/MAE LANDSCAPE - ESH6 Jan 2026 RTH (unconditional, all bars)")
    print("="*70)

    for name in HORIZONS:
        h = results["horizons"][name]
        print(f"\n-- {name} horizon ------------------------------------------")
        print(f"{'':12} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6}  (ticks)")
        for side in ("long", "short"):
            for metric in ("mfe", "mae"):
                key = f"{metric}_{side}"
                r = h[key]
                if not r:
                    continue
                print(f"  {key:12} {r['p25']:6.1f} {r['p50']:6.1f} {r['p75']:6.1f} {r['p90']:6.1f} {r['p95']:6.1f}")

        # Viability table for key combos (long only; short is symmetric)
        print(f"\n  Viability (long, no path assumption - P(MFE>=TP) and P(MFE>=TP & MAE<SL)):")
        print(f"  {'TP/SL':10} {'BE_WR':>7} {'P(TP)':>7} {'P(SL)':>7} {'P(path_win)':>12} {'path_pur':>9}")
        key_combos = [
            (15, 5), (18, 6), (20, 6), (20, 8), (25, 8), (30, 8), (30, 10),
        ]
        for tp, sl in key_combos:
            k = f"TP{tp}_SL{sl}"
            if k in h["viability_long"]:
                v = h["viability_long"][k]
                print(f"  {tp}t/{sl}t  ({v['ratio']:.1f}:1)  {v['breakeven_wr']:>7.1%}  {v['p_tp_hit']:>7.1%}  {v['p_sl_hit']:>7.1%}  {v['p_path_win']:>12.1%}  {v['path_purity']:>9.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "horizon_landscape.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
