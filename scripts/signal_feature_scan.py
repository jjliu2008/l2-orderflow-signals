"""
Feature scan: longer-window cumulative delta and book imbalance features
vs 15-minute forward path_win at 18t TP / 6t SL.

Features tested (independently, no combinations):
  A. Cumulative signed volume over 3 min (1800 bars) and 5 min (3000 bars)
  B. order_book_imbalance   — bid/(bid+ask) at top of book, [0,1]
  C. depth_imbalance_top5   — (bid5-ask5)/(bid5+ask5), symmetric [-1,1]
  D. imbalance_delta        — 1-bar change in book imbalance

Each feature gets: correlation, decile table, directional lift.
Results kept separate — no combinations.

Usage:
    python scripts/signal_feature_scan.py

Outputs:
    artifacts/signal_research/feature_scan.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/signal_research"

TICK     = 0.25
TP_TICKS = 18
SL_TICKS = 6
H_BARS   = 9_000   # 15 min at 100ms

RTH_START = "14:30"
RTH_END   = "21:00"

LOAD_COLS = [
    "Time", "mid", "signed_volume",
    "order_book_imbalance", "depth_imbalance_top5",
    "imbalance_delta",
]

DELTA_WINDOWS = {
    "cum_delta_1800": 1_800,   # 3 min
    "cum_delta_3000": 3_000,   # 5 min
}

IMBALANCE_FEATURES = {
    "order_book_imbalance": {
        "direction": "positive",   # high OBI → expect long path_win elevated
        "description": "bid/(bid+ask) top-of-book, [0,1]",
    },
    "depth_imbalance_top5": {
        "direction": "positive",
        "description": "(bid5-ask5)/(bid5+ask5), symmetric [-1,1]",
    },
    "imbalance_delta": {
        "direction": "positive",
        "description": "1-bar change in book imbalance",
    },
}


def _fwd_max(series: pd.Series, H: int) -> np.ndarray:
    rev    = series.values[::-1]
    result = pd.Series(rev).rolling(H, min_periods=1).max().values
    return result[::-1]


def _fwd_min(series: pd.Series, H: int) -> np.ndarray:
    rev    = series.values[::-1]
    result = pd.Series(rev).rolling(H, min_periods=1).min().values
    return result[::-1]


def load_and_process(date_dir: Path) -> pd.DataFrame | None:
    date_str = date_dir.name.replace("date=", "")
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None

    df = pd.read_parquet(pq, columns=LOAD_COLS)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)

    # Forward outcomes (full day for edge accuracy at RTH close)
    fmax = _fwd_max(df["mid"], H_BARS)
    fmin = _fwd_min(df["mid"], H_BARS)
    df["mfe_long"]  = (fmax - df["mid"].values) / TICK
    df["mae_long"]  = (df["mid"].values - fmin) / TICK
    df["mfe_short"] = (df["mid"].values - fmin) / TICK
    df["mae_short"] = (fmax - df["mid"].values) / TICK
    df["path_win_long"]  = (df["mfe_long"]  >= TP_TICKS) & (df["mae_long"]  < SL_TICKS)
    df["path_win_short"] = (df["mfe_short"] >= TP_TICKS) & (df["mae_short"] < SL_TICKS)

    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    rth = df[(df["Time"] >= lo) & (df["Time"] < hi)].copy()

    # Longer-window delta (within RTH; no cross-day leakage)
    for name, w in DELTA_WINDOWS.items():
        rth[name] = rth["signed_volume"].rolling(w, min_periods=w).sum()

    return rth


def corr_biserial(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return {"r": 0.0, "p": 1.0, "n": 0}
    r, p = stats.pointbiserialr(y.astype(int), x)
    return {"r": round(float(r), 6), "p": round(float(p), 6), "n": int(len(x))}


def decile_table(df: pd.DataFrame, feat_col: str, outcome_col: str) -> list[dict]:
    valid = df[[feat_col, outcome_col]].dropna()
    if len(valid) < 100:
        return []
    valid = valid.copy()
    try:
        valid["dec"] = pd.qcut(valid[feat_col], 10, labels=False, duplicates="drop")
    except Exception:
        return []
    rows = []
    for dec, grp in valid.groupby("dec"):
        rows.append({
            "decile":       int(dec),
            "n":            int(len(grp)),
            "feat_p50":     round(float(valid.loc[grp.index, feat_col].median()), 4),
            "path_win":     round(float(grp[outcome_col].mean()), 5),
        })
    return rows


def directional_split(df: pd.DataFrame, feat_col: str, midpoint: float) -> dict:
    """Split on feat > midpoint (positive signal) vs feat < midpoint (negative)."""
    valid = df[[feat_col, "path_win_long", "path_win_short"]].dropna()
    pos  = valid[valid[feat_col] > midpoint]
    neg  = valid[valid[feat_col] < midpoint]
    mid  = valid[valid[feat_col] == midpoint]
    result = {}
    for label, sub in [("above_mid", pos), ("at_mid", mid), ("below_mid", neg)]:
        result[label] = {
            "n":              int(len(sub)),
            "path_win_long":  round(float(sub["path_win_long"].mean()),  5) if len(sub) else 0.0,
            "path_win_short": round(float(sub["path_win_short"].mean()), 5) if len(sub) else 0.0,
        }
    return result


def analyse_feature(
    combined: pd.DataFrame,
    feat_col: str,
    baseline_long: float,
    baseline_short: float,
) -> dict:
    valid = combined[[feat_col, "path_win_long", "path_win_short"]].dropna()
    midpoint = float(valid[feat_col].median())
    pctiles  = valid[feat_col].quantile([.1, .25, .5, .75, .9]).round(4).to_dict()

    corr_long  = corr_biserial(valid[feat_col].values, valid["path_win_long"].values)
    corr_short = corr_biserial(-valid[feat_col].values, valid["path_win_short"].values)

    dec_long  = decile_table(combined, feat_col, "path_win_long")
    dec_short = decile_table(combined, feat_col, "path_win_short")

    split = directional_split(combined, feat_col, midpoint)

    above_long_lift  = split["above_mid"]["path_win_long"]  - baseline_long
    below_short_lift = split["below_mid"]["path_win_short"] - baseline_short

    return {
        "n_valid":       int(len(valid)),
        "percentiles":   {str(k): float(v) for k, v in pctiles.items()},
        "median":        midpoint,
        "corr_long":     corr_long,
        "corr_short":    corr_short,
        "directional_split": split,
        "above_long_lift":   round(above_long_lift,  5),
        "below_short_lift":  round(below_short_lift, 5),
        "decile_long":   dec_long,
        "decile_short":  dec_short,
    }


def print_feature(name: str, r: dict, baseline_long: float, baseline_short: float) -> None:
    print(f"\n  {name}")
    print(f"    n={r['n_valid']:,}  median={r['median']:.4f}")
    print(f"    corr_long:  r={r['corr_long']['r']:.4f}  p={r['corr_long']['p']:.4f}")
    print(f"    corr_short: r={r['corr_short']['r']:.4f}  p={r['corr_short']['p']:.4f}")

    sp = r["directional_split"]
    print(f"    Directional split (baseline long={baseline_long:.3%}, short={baseline_short:.3%}):")
    for label, s in sp.items():
        long_flag  = " *" if abs(s["path_win_long"]  - baseline_long)  > 0.005 else ""
        short_flag = " *" if abs(s["path_win_short"] - baseline_short) > 0.005 else ""
        print(f"      {label:12} n={s['n']:>8,}  "
              f"pw_long={s['path_win_long']:.3%}{long_flag}  "
              f"pw_short={s['path_win_short']:.3%}{short_flag}")

    print(f"    Lift: above_mid->long  {r['above_long_lift']:+.3%}"
          f"   below_mid->short {r['below_short_lift']:+.3%}")

    dec_l = r["decile_long"]
    dec_s = r["decile_short"]
    if dec_l:
        print(f"    Decile table (0=low, 9=high {name}):")
        print(f"    {'dec':>5} {'n':>8} {'feat_p50':>10} {'pw_long':>9} {'pw_short':>9}")
        for dl, ds in zip(dec_l, dec_s):
            flag = " *" if (abs(dl['path_win'] - baseline_long) > 0.005 or
                            abs(ds['path_win'] - baseline_short) > 0.005) else ""
            print(f"    {dl['decile']:>5} {dl['n']:>8,} {dl['feat_p50']:>10.4f} "
                  f"{dl['path_win']:>9.3%} {ds['path_win']:>9.3%}{flag}")


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    print(f"Processing {len(date_dirs)} dates (Jan+Feb 2026)...")

    all_rth: list[pd.DataFrame] = []
    for dd in date_dirs:
        df = load_and_process(dd)
        if df is None or len(df) == 0:
            continue
        all_rth.append(df)

    if not all_rth:
        print("No data loaded.")
        return

    combined = pd.concat(all_rth, ignore_index=True)
    baseline_long  = float(combined["path_win_long"].mean())
    baseline_short = float(combined["path_win_short"].mean())
    print(f"Total RTH bars: {len(combined):,}")
    print(f"Baseline path_win_long:  {baseline_long:.3%}")
    print(f"Baseline path_win_short: {baseline_short:.3%}")

    results: dict = {
        "metadata": {
            "instrument": "ESH6", "period": "Jan+Feb 2026 RTH",
            "dates": len(date_dirs), "total_bars": len(combined),
            "tp_ticks": TP_TICKS, "sl_ticks": SL_TICKS, "horizon_bars": H_BARS,
            "baseline_path_win_long":  round(baseline_long,  4),
            "baseline_path_win_short": round(baseline_short, 4),
        },
        "features": {},
    }

    # ── A. Longer-window cumulative delta ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("A. Longer-window cumulative delta")
    for name in DELTA_WINDOWS:
        r = analyse_feature(combined, name, baseline_long, baseline_short)
        results["features"][name] = r
        print_feature(name, r, baseline_long, baseline_short)

    # ── B. Book imbalance features ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("B. Book imbalance features")
    for name, meta in IMBALANCE_FEATURES.items():
        print(f"\n  [{meta['description']}]")
        r = analyse_feature(combined, name, baseline_long, baseline_short)
        r["description"] = meta["description"]
        results["features"][name] = r
        print_feature(name, r, baseline_long, baseline_short)

    # ── Summary lift table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Summary: directional lift above baseline")
    print(f"  {'feature':30} {'r_long':>9} {'above->long':>12} {'below->short':>13}")
    for name, r in results["features"].items():
        print(f"  {name:30} {r['corr_long']['r']:>9.4f} "
              f"{r['above_long_lift']:>+12.3%} {r['below_short_lift']:>+13.3%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "feature_scan.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
