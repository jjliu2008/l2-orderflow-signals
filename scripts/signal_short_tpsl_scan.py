"""
TP/SL structure scan on the short setup population (rp>=0.85, vd>=50).

For each (TP, SL) combo, compute:
  - Unconditional path_win (baseline)
  - path_win on signal population
  - Lift vs baseline
  - Breakeven WR = SL / (TP + SL)
  - Whether signal population clears breakeven
  - Jan/Feb stability

One data pass — all structures evaluated from the same MFE/MAE values.

Usage:
    python scripts/signal_short_tpsl_scan.py

Outputs:
    artifacts/signal_research/short_tpsl_scan.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/signal_research"

TICK    = 0.25
H_BARS  = 9_000    # 15 min at 100ms

RTH_START = "14:30"
RTH_END   = "21:00"

LOAD_COLS = ["Time", "mid", "trade_volume"]

# TP/SL structures to evaluate (in ticks)
STRUCTURES = [
    # (TP, SL, label)
    # --- 3:1 family (BE = 25%) ---
    ( 9,  3, "9t/3t   3:1  BE=25.0%"),
    (12,  4, "12t/4t  3:1  BE=25.0%"),
    (15,  5, "15t/5t  3:1  BE=25.0%"),
    (18,  6, "18t/6t  3:1  BE=25.0%  [current]"),
    (24,  8, "24t/8t  3:1  BE=25.0%"),
    (30, 10, "30t/10t 3:1  BE=25.0%"),
    # --- 2:1 family (BE = 33.3%) ---
    ( 8,  4, "8t/4t   2:1  BE=33.3%"),
    (12,  6, "12t/6t  2:1  BE=33.3%"),
    (16,  8, "16t/8t  2:1  BE=33.3%"),
    (18,  9, "18t/9t  2:1  BE=33.3%"),
    (24, 12, "24t/12t 2:1  BE=33.3%"),
    # --- 1.5:1 family (BE = 40.0%) ---
    ( 9,  6, "9t/6t   1.5:1  BE=40.0%"),
    (12,  8, "12t/8t  1.5:1  BE=40.0%"),
    (15, 10, "15t/10t 1.5:1  BE=40.0%"),
    # --- 4:1 family (BE = 20.0%) ---
    (12,  3, "12t/3t  4:1  BE=20.0%"),
    (20,  5, "20t/5t  4:1  BE=20.0%"),
    (24,  6, "24t/6t  4:1  BE=20.0%"),
]


def _fwd_max(s: pd.Series, H: int) -> np.ndarray:
    rev = s.values[::-1]
    return pd.Series(rev).rolling(H, min_periods=1).max().values[::-1]


def _fwd_min(s: pd.Series, H: int) -> np.ndarray:
    rev = s.values[::-1]
    return pd.Series(rev).rolling(H, min_periods=1).min().values[::-1]


def load_and_process(date_dir: Path) -> pd.DataFrame | None:
    date_str = date_dir.name.replace("date=", "")
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None

    df = pd.read_parquet(pq, columns=LOAD_COLS)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)

    # Single-pass MFE/MAE at 15 min
    fmax = _fwd_max(df["mid"], H_BARS)
    fmin = _fwd_min(df["mid"], H_BARS)
    df["mfe_short"] = (df["mid"].values - fmin) / TICK
    df["mae_short"] = (fmax - df["mid"].values) / TICK

    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    rth = df[(df["Time"] >= lo) & (df["Time"] < hi)].copy()
    rth["month"] = rth["Time"].dt.month

    # VWAP deviation
    vol  = rth["trade_volume"].clip(lower=0)
    vwap = (rth["mid"] * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    vwap = vwap.ffill()
    rth["vwap_dev"] = (rth["mid"] - vwap) / TICK

    # Session range position
    sess_range = (rth["mid"].cummax() - rth["mid"].cummin()).replace(0, np.nan)
    rth["range_pos"] = (rth["mid"] - rth["mid"].cummin()) / sess_range

    return rth


def path_win(mfe: np.ndarray, mae: np.ndarray, tp: int, sl: int) -> np.ndarray:
    return (mfe >= tp) & (mae < sl)


def pop_stats(df: pd.DataFrame, mask: pd.Series, tp: int, sl: int) -> dict:
    sub    = df[mask]
    n      = len(sub)
    n_jan  = int((sub["month"] == 1).sum())
    n_feb  = int((sub["month"] == 2).sum())
    pw_col = path_win(sub["mfe_short"].values, sub["mae_short"].values, tp, sl)
    pw_all = float(pw_col.mean()) if n > 0 else 0.0
    pw_jan_arr = pw_col[(sub["month"] == 1).values]
    pw_feb_arr = pw_col[(sub["month"] == 2).values]
    pw_jan = float(pw_jan_arr.mean()) if n_jan > 0 else 0.0
    pw_feb = float(pw_feb_arr.mean()) if n_feb > 0 else 0.0
    return {
        "n": n, "n_jan": n_jan, "n_feb": n_feb,
        "pw": round(pw_all, 5),
        "pw_jan": round(pw_jan, 5),
        "pw_feb": round(pw_feb, 5),
        "jan_feb_gap": round(abs(pw_jan - pw_feb), 5),
        "stable": abs(pw_jan - pw_feb) <= 0.05,
    }


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    print(f"Processing {len(date_dirs)} dates (Jan+Feb 2026)...")

    all_rth: list[pd.DataFrame] = []
    for dd in date_dirs:
        df = load_and_process(dd)
        if df is None or len(df) == 0:
            continue
        all_rth.append(df)

    combined = pd.concat(all_rth, ignore_index=True)
    n_dates  = combined["Time"].dt.date.nunique()
    print(f"Total RTH bars: {len(combined):,}  over {n_dates} dates")

    rp = combined["range_pos"]
    vd = combined["vwap_dev"]
    signal_mask = (rp >= 0.85) & (vd >= 50)
    all_mask    = pd.Series(True, index=combined.index)

    n_signal = signal_mask.sum()
    bpd      = round(n_signal / n_dates, 0)
    print(f"Signal population (rp>=0.85, vd>=50): {n_signal:,} bars  ({bpd:.0f}/day)")

    print(f"\n{'='*80}")
    print(f"TP/SL structure scan — short path_win at 15-min horizon")
    print(f"{'='*80}")
    print(f"{'Structure':30} {'BE':>6} {'base_pw':>8} {'sig_pw':>8} {'lift':>8} "
          f"{'clears?':>8} {'Jan':>8} {'Feb':>8} {'gap':>7} {'st':>4}")

    results = {
        "metadata": {
            "instrument": "ESH6", "period": "Jan+Feb 2026 RTH",
            "dates": n_dates, "total_bars": len(combined),
            "horizon_bars": H_BARS,
            "signal_n": int(n_signal),
            "signal_bpd": int(bpd),
            "signal_def": "range_pos>=0.85 AND vwap_dev>=50t",
        },
        "structures": [],
    }

    for tp, sl, label in STRUCTURES:
        be = sl / (tp + sl)

        base  = pop_stats(combined, all_mask,    tp, sl)
        sig   = pop_stats(combined, signal_mask, tp, sl)

        lift    = sig["pw"] - base["pw"]
        clears  = sig["pw"] >= be
        flag    = " ***" if clears else ""

        print(f"  {label:28} {be:>6.1%} {base['pw']:>8.3%} {sig['pw']:>8.3%} "
              f"{lift:>+8.3%} {'YES' + flag if clears else 'no':>8} "
              f"{sig['pw_jan']:>8.3%} {sig['pw_feb']:>8.3%} "
              f"{sig['jan_feb_gap']:>7.3%} {'Y' if sig['stable'] else 'N':>4}")

        results["structures"].append({
            "label": label, "tp_ticks": tp, "sl_ticks": sl,
            "ratio": round(tp / sl, 2),
            "breakeven": round(be, 5),
            "baseline": base,
            "signal": sig,
            "lift": round(lift, 5),
            "clears_breakeven": clears,
        })

    # Group summary by ratio family
    print(f"\n--- 3:1 structures: which TP clears 25%? ---")
    for r in results["structures"]:
        if abs(r["ratio"] - 3.0) < 0.1:
            flag = " <-- CLEARS" if r["clears_breakeven"] else ""
            print(f"  {r['label']:35}  sig_pw={r['signal']['pw']:.3%}  "
                  f"lift={r['lift']:+.3%}{flag}")

    print(f"\n--- 4:1 structures: which TP clears 20%? ---")
    for r in results["structures"]:
        if abs(r["ratio"] - 4.0) < 0.1:
            flag = " <-- CLEARS" if r["clears_breakeven"] else ""
            print(f"  {r['label']:35}  sig_pw={r['signal']['pw']:.3%}  "
                  f"lift={r['lift']:+.3%}{flag}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "short_tpsl_scan.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
