"""
Signal feasibility: cumulative signed volume (trailing N-bar delta) vs
15-minute forward path_win at 18t TP / 6t SL.

Hypothesis: bars with strongly positive cum_delta predict cleaner long-side
moves over the next 15 minutes; negative cum_delta predicts short-side moves.

Output:
  - Decile table: path_win_long and path_win_short by cum_delta decile
  - Directional lift: cum_delta > 0 bars vs cum_delta < 0 bars vs flat
  - Multiple window lengths: 3, 5, 10 bars (300ms, 500ms, 1s)
  - Correlation coefficients
  artifacts/signal_research/cumulative_delta_signal.json

Usage:
    python scripts/signal_cumulative_delta.py
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

TICK      = 0.25
TP_TICKS  = 18
SL_TICKS  = 6
H_BARS    = 9_000   # 15 min at 100ms
WINDOWS   = [3, 5, 10]   # bar windows for rolling delta

RTH_START = "14:30"
RTH_END   = "21:00"


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

    df = pd.read_parquet(pq, columns=["Time", "mid", "signed_volume"])
    df["Time"] = pd.to_datetime(df["Time"], utc=True)

    # Forward MFE/MAE computed on full day (avoid day-edge truncation within RTH)
    fmax = _fwd_max(df["mid"], H_BARS)
    fmin = _fwd_min(df["mid"], H_BARS)
    df["mfe_long"]  = (fmax - df["mid"].values) / TICK
    df["mae_long"]  = (df["mid"].values - fmin) / TICK
    df["mfe_short"] = (df["mid"].values - fmin) / TICK
    df["mae_short"] = (fmax - df["mid"].values) / TICK

    df["path_win_long"]  = (df["mfe_long"]  >= TP_TICKS) & (df["mae_long"]  < SL_TICKS)
    df["path_win_short"] = (df["mfe_short"] >= TP_TICKS) & (df["mae_short"] < SL_TICKS)

    # Filter to RTH for the entry-bar analysis
    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    rth = df[(df["Time"] >= lo) & (df["Time"] < hi)].copy()

    # Rolling delta windows — computed within RTH (no cross-day leakage)
    for w in WINDOWS:
        rth[f"cum_delta_{w}"] = (
            rth["signed_volume"]
            .rolling(w, min_periods=w)
            .sum()
        )

    return rth


def decile_analysis(df: pd.DataFrame, delta_col: str, outcome_col: str) -> list[dict]:
    """Bin delta_col into deciles, return path_win rate per decile."""
    valid = df[[delta_col, outcome_col]].dropna()
    if len(valid) == 0:
        return []
    try:
        valid = valid.copy()
        valid["decile"] = pd.qcut(valid[delta_col], 10, labels=False, duplicates="drop")
    except Exception:
        return []
    grouped = valid.groupby("decile")[outcome_col]
    rows = []
    for dec, grp in grouped:
        rows.append({
            "decile":    int(dec),
            "n":         int(len(grp)),
            "path_win":  float(grp.mean()),
            "delta_p50": float(valid.loc[grp.index, delta_col].median()),
        })
    return rows


def directional_split(df: pd.DataFrame, delta_col: str) -> dict:
    """Compare path_win in positive / zero / negative delta bars."""
    valid = df[[delta_col, "path_win_long", "path_win_short"]].dropna()
    pos  = valid[valid[delta_col] > 0]
    neg  = valid[valid[delta_col] < 0]
    flat = valid[valid[delta_col] == 0]
    result = {}
    for label, sub in [("pos_delta", pos), ("zero_delta", flat), ("neg_delta", neg)]:
        result[label] = {
            "n":              int(len(sub)),
            "path_win_long":  float(sub["path_win_long"].mean())  if len(sub) else 0,
            "path_win_short": float(sub["path_win_short"].mean()) if len(sub) else 0,
        }
    return result


def corr_stats(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return {"r": 0, "p": 1, "n": 0}
    r, p = stats.pointbiserialr(y.astype(int), x)
    return {"r": float(r), "p": float(p), "n": int(len(x))}


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    print(f"Processing {len(date_dirs)} dates (Jan+Feb 2026)...")

    all_rth: list[pd.DataFrame] = []
    for dd in date_dirs:
        df = load_and_process(dd)
        if df is None or len(df) == 0:
            continue
        all_rth.append(df)
        print(f"  {dd.name}: {len(df):,} RTH bars")

    if not all_rth:
        print("No data loaded.")
        return

    combined = pd.concat(all_rth, ignore_index=True)
    total_n  = len(combined)
    print(f"\nTotal RTH bars: {total_n:,}")
    print(f"Baseline path_win_long:  {combined['path_win_long'].mean():.3%}")
    print(f"Baseline path_win_short: {combined['path_win_short'].mean():.3%}")

    results: dict = {
        "metadata": {
            "instrument": "ESH6",
            "period": "Jan+Feb 2026 RTH",
            "dates": len(date_dirs),
            "total_bars": total_n,
            "tp_ticks": TP_TICKS,
            "sl_ticks": SL_TICKS,
            "horizon_bars": H_BARS,
            "baseline_path_win_long":  round(combined["path_win_long"].mean(), 4),
            "baseline_path_win_short": round(combined["path_win_short"].mean(), 4),
        },
        "windows": {},
    }

    for w in WINDOWS:
        col = f"cum_delta_{w}"
        print(f"\n{'='*60}")
        print(f"Window: {w} bars ({w*100}ms)")

        valid = combined[[col, "path_win_long", "path_win_short"]].dropna()
        print(f"  Valid bars: {len(valid):,}")
        print(f"  {col} p10/p25/p50/p75/p90: "
              f"{valid[col].quantile([.1,.25,.5,.75,.9]).values.round(1)}")

        # Correlations
        corr_long  = corr_stats(valid[col].values, valid["path_win_long"].values)
        corr_short = corr_stats(-valid[col].values, valid["path_win_short"].values)
        print(f"  Corr(delta, path_win_long):       r={corr_long['r']:.4f}  p={corr_long['p']:.4f}")
        print(f"  Corr(-delta, path_win_short):     r={corr_short['r']:.4f}  p={corr_short['p']:.4f}")

        # Directional split
        split = directional_split(combined, col)
        print(f"\n  Directional split (baseline long={results['metadata']['baseline_path_win_long']:.1%}, short={results['metadata']['baseline_path_win_short']:.1%}):")
        for label, s in split.items():
            print(f"  {label:15} n={s['n']:>8,}  "
                  f"pw_long={s['path_win_long']:.3%}  "
                  f"pw_short={s['path_win_short']:.3%}")

        # Decile tables
        dec_long  = decile_analysis(combined, col, "path_win_long")
        dec_short = decile_analysis(combined, col, "path_win_short")

        print(f"\n  Decile table (dec 0=most_neg, 9=most_pos delta):")
        print(f"  {'dec':>5} {'n':>8} {'delta_p50':>10} {'pw_long':>9} {'pw_short':>9}")
        for dl, ds in zip(dec_long, dec_short):
            print(f"  {dl['decile']:>5} {dl['n']:>8,} {dl['delta_p50']:>10.1f} "
                  f"{dl['path_win']:>9.3%} {ds['path_win']:>9.3%}")

        results["windows"][f"{w}bar"] = {
            "corr_long":  corr_long,
            "corr_short": corr_short,
            "directional_split": split,
            "decile_long":  dec_long,
            "decile_short": dec_short,
        }

    # Summary across windows
    print(f"\n{'='*60}")
    print("Summary: correlation and directional lift by window")
    print(f"  {'window':>8} {'r_long':>9} {'p_long':>9} {'r_short':>9} {'p_short':>9}"
          f"  pos->long  neg->short")
    for w in WINDOWS:
        k = f"{w}bar"
        r = results["windows"][k]
        sp = r["directional_split"]
        pos_long  = sp["pos_delta"]["path_win_long"]
        neg_short = sp["neg_delta"]["path_win_short"]
        bl = results["metadata"]["baseline_path_win_long"]
        bs = results["metadata"]["baseline_path_win_short"]
        print(f"  {w*100:>5}ms  "
              f"{r['corr_long']['r']:>9.4f} {r['corr_long']['p']:>9.4f}  "
              f"{r['corr_short']['r']:>9.4f} {r['corr_short']['p']:>9.4f}  "
              f"{pos_long:.3%}({pos_long-bl:+.3%})  "
              f"{neg_short:.3%}({neg_short-bs:+.3%})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "cumulative_delta_signal.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
