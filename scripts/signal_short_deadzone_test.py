"""
Dead-zone exclusion test: short setup (rp>=0.85 + vd>=50) with |cum_delta_5min| filter.

Base population: range_pos >= 0.85 AND vwap_dev >= 50 ticks (stable across regimes)
Layer: exclude bars where |cum_delta_5min| < threshold (near-zero, balanced flow)

Tests multiple absolute thresholds plus directional split (negative delta only).
Reports n, bars_per_day, path_win_short, Jan/Feb split for each variant.

Usage:
    python scripts/signal_short_deadzone_test.py

Outputs:
    artifacts/signal_research/short_deadzone_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/signal_research"

TICK      = 0.25
TP_TICKS  = 18
SL_TICKS  = 6
H_BARS    = 9_000    # 15 min at 100ms
DELTA_WIN = 3_000    # 5 min at 100ms

RTH_START = "14:30"
RTH_END   = "21:00"

LOAD_COLS = ["Time", "mid", "trade_volume", "signed_volume"]

# Dead-zone absolute thresholds to test
DZ_THRESHOLDS = [50, 100, 200, 300, 392, 500, 750]  # 392 ≈ p25 of |cum_delta_3000|


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

    # Forward outcomes on full-day data
    fmax = _fwd_max(df["mid"], H_BARS)
    fmin = _fwd_min(df["mid"], H_BARS)
    df["mfe_short"] = (df["mid"].values - fmin) / TICK
    df["mae_short"] = (fmax - df["mid"].values) / TICK
    df["path_win_short"] = (df["mfe_short"] >= TP_TICKS) & (df["mae_short"] < SL_TICKS)

    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    rth = df[(df["Time"] >= lo) & (df["Time"] < hi)].copy()
    rth["month"] = rth["Time"].dt.month

    # VWAP deviation (cumulative from RTH open)
    vol  = rth["trade_volume"].clip(lower=0)
    vwap = (rth["mid"] * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    vwap = vwap.ffill()
    rth["vwap_dev"] = (rth["mid"] - vwap) / TICK

    # Session range position
    sess_range = (rth["mid"].cummax() - rth["mid"].cummin()).replace(0, np.nan)
    rth["range_pos"] = (rth["mid"] - rth["mid"].cummin()) / sess_range

    # 5-min cumulative delta (within RTH, no cross-day leakage)
    rth["cum_delta_5min"] = rth["signed_volume"].rolling(DELTA_WIN, min_periods=DELTA_WIN).sum()

    return rth


def gate_stats(df: pd.DataFrame, mask: pd.Series, label: str, n_dates: int) -> dict:
    sub   = df[mask]
    n     = len(sub)
    n_jan = int((sub["month"] == 1).sum())
    n_feb = int((sub["month"] == 2).sum())
    pw_all = float(sub["path_win_short"].mean()) if n > 0 else 0.0
    pw_jan = float(sub.loc[sub["month"] == 1, "path_win_short"].mean()) if n_jan > 0 else 0.0
    pw_feb = float(sub.loc[sub["month"] == 2, "path_win_short"].mean()) if n_feb > 0 else 0.0
    bpd    = round(n / n_dates, 1) if n_dates > 0 else 0
    gap    = abs(pw_jan - pw_feb)
    stable = gap <= 0.05
    return {
        "label":    label,
        "n":        n,
        "n_jan":    n_jan,
        "n_feb":    n_feb,
        "bars_per_day": bpd,
        "pw":       round(pw_all, 5),
        "pw_jan":   round(pw_jan, 5),
        "pw_feb":   round(pw_feb, 5),
        "jan_feb_gap": round(gap, 5),
        "stable":   stable,
    }


def print_gate(g: dict, baseline: float, breakeven: float) -> None:
    lift   = g["pw"] - baseline
    be_str = " *** ABOVE BREAKEVEN" if g["pw"] >= breakeven else ""
    stab   = "stable" if g["stable"] else "UNSTABLE"
    print(f"  {g['label']}")
    print(f"    n={g['n']:>9,}  bpd={g['bars_per_day']:>8.1f}  "
          f"pw={g['pw']:.3%} ({lift:+.3%} vs baseline){be_str}")
    print(f"    Jan: n={g['n_jan']:,}  pw={g['pw_jan']:.3%}  |  "
          f"Feb: n={g['n_feb']:,}  pw={g['pw_feb']:.3%}  "
          f"[gap={g['jan_feb_gap']:.3%}  {stab}]")


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
    baseline = float(combined["path_win_short"].mean())
    breakeven = 0.25

    print(f"Total RTH bars: {len(combined):,}  over {n_dates} dates")
    print(f"Baseline path_win_short: {baseline:.3%}")

    # --- Base short population ---
    rp = combined["range_pos"]
    vd = combined["vwap_dev"]
    cd = combined["cum_delta_5min"]

    short_base = (rp >= 0.85) & (vd >= 50)

    # cum_delta stats within base population
    base_df = combined[short_base & cd.notna()]
    cd_abs = base_df["cum_delta_5min"].abs()
    print(f"\nWithin short base (rp>=0.85, vd>=50), cum_delta_5min (n={len(base_df):,}):")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  p{int(q*100):>2}: {base_df['cum_delta_5min'].quantile(q):>8.1f}  "
              f"|abs| p{int(q*100):>2}: {cd_abs.quantile(q):>8.1f}")

    print(f"\n{'='*68}")
    print(f"SHORT setup dead-zone exclusion test")
    print(f"Base: range_pos>=0.85, vwap_dev>=50t  (baseline={baseline:.3%}  breakeven={breakeven:.3%})")
    print(f"{'='*68}")

    gates = []

    # Base (no delta filter)
    g = gate_stats(combined, short_base, "Base: rp>=0.85 + vd>=50 (no delta filter)", n_dates)
    gates.append(g)
    print_gate(g, baseline, breakeven)

    # Base AND cum_delta is not NaN (requires 5min of RTH history)
    has_delta = short_base & cd.notna()
    g = gate_stats(combined, has_delta, "Base + has cum_delta (>=5min into session)", n_dates)
    gates.append(g)
    print_gate(g, baseline, breakeven)

    print()

    # Dead-zone exclusion: |cum_delta_5min| >= threshold
    for thr in DZ_THRESHOLDS:
        mask = has_delta & (cd.abs() >= thr)
        label = f"Base + |cd5| >= {thr:>4} (non-quiet flow)"
        g = gate_stats(combined, mask, label, n_dates)
        gates.append(g)
        print_gate(g, baseline, breakeven)

    print()

    # Directional tests: negative delta (net selling pressure)
    for thr in [0, 100, 200, 392]:
        mask = has_delta & (cd <= -thr)
        label = f"Base + cd5 <= {-thr:>5} (net selling)"
        g = gate_stats(combined, mask, label, n_dates)
        gates.append(g)
        print_gate(g, baseline, breakeven)

    print()

    # Directional: positive delta (buying into resistance — could confirm short)
    for thr in [0, 100, 200]:
        mask = has_delta & (cd >= thr)
        label = f"Base + cd5 >= +{thr:>4} (buying into resistance)"
        g = gate_stats(combined, mask, label, n_dates)
        gates.append(g)
        print_gate(g, baseline, breakeven)

    # Summary
    print(f"\n{'='*68}")
    print(f"{'Gate':52} {'pw':>8} {'lift':>8} {'bpd':>6} {'stable':>7}")
    for g in gates:
        lift  = g["pw"] - baseline
        stab  = "Y" if g["stable"] else "N"
        above = " **" if g["pw"] >= breakeven else ""
        print(f"  {g['label']:50} {g['pw']:>8.3%} {lift:>+8.3%} "
              f"{g['bars_per_day']:>6.0f} {stab:>7}{above}")

    results = {
        "metadata": {
            "instrument": "ESH6", "period": "Jan+Feb 2026 RTH",
            "dates": n_dates, "total_bars": len(combined),
            "tp_ticks": TP_TICKS, "sl_ticks": SL_TICKS, "horizon_bars": H_BARS,
            "delta_window_bars": DELTA_WIN,
            "baseline_path_win_short": round(baseline, 5),
            "breakeven": breakeven,
        },
        "gates": [
            {**g, "lift_vs_baseline": round(g["pw"] - baseline, 5)}
            for g in gates
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "short_deadzone_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
