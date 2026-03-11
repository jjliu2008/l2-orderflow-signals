"""
Combination gate test: range_pos x vwap_dev zones vs 15-min path_win.

Two independent setups tested separately:

  LONG:  range_pos in [0.30, 0.55] (mid-range pullback)
         AND vwap_dev in [-30, +30] ticks (near VWAP, not stretched)

  SHORT: range_pos >= 0.85 (near session high)
         AND vwap_dev >= 50 ticks (well above VWAP)

For each setup:
  - Individual gate A alone, gate B alone, A AND B together
  - Path_win rate and n qualifying bars
  - January vs February split
  - Sensitivity: a few threshold variants around primary hypothesis

Usage:
    python scripts/signal_combination_test.py

Outputs:
    artifacts/signal_research/combination_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/signal_research"

TICK     = 0.25
TP_TICKS = 18
SL_TICKS = 6
H_BARS   = 9_000    # 15 min at 100ms

RTH_START = "14:30"
RTH_END   = "21:00"

LOAD_COLS = ["Time", "mid", "trade_volume"]


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
    df["mfe_long"]  = (fmax - df["mid"].values) / TICK
    df["mae_long"]  = (df["mid"].values - fmin) / TICK
    df["mfe_short"] = (df["mid"].values - fmin) / TICK
    df["mae_short"] = (fmax - df["mid"].values) / TICK
    df["path_win_long"]  = (df["mfe_long"]  >= TP_TICKS) & (df["mae_long"]  < SL_TICKS)
    df["path_win_short"] = (df["mfe_short"] >= TP_TICKS) & (df["mae_short"] < SL_TICKS)

    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    rth = df[(df["Time"] >= lo) & (df["Time"] < hi)].copy()
    rth["month"] = rth["Time"].dt.month

    # VWAP deviation (cumulative from RTH open)
    vol     = rth["trade_volume"].clip(lower=0)
    vwap    = (rth["mid"] * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    vwap    = vwap.ffill()
    rth["vwap_dev"] = (rth["mid"] - vwap) / TICK

    # Session range position
    sess_range = (rth["mid"].cummax() - rth["mid"].cummin()).replace(0, np.nan)
    rth["range_pos"] = (rth["mid"] - rth["mid"].cummin()) / sess_range

    return rth


def gate_stats(df: pd.DataFrame, mask: pd.Series, outcome: str, label: str) -> dict:
    sub  = df[mask]
    n    = len(sub)
    n_jan = int((sub["month"] == 1).sum())
    n_feb = int((sub["month"] == 2).sum())
    pw_all = float(sub[outcome].mean()) if n > 0 else 0.0
    pw_jan = float(sub.loc[sub["month"] == 1, outcome].mean()) if n_jan > 0 else 0.0
    pw_feb = float(sub.loc[sub["month"] == 2, outcome].mean()) if n_feb > 0 else 0.0
    # Bars per day
    n_dates = df["Time"].dt.date.nunique()
    bpd     = round(n / n_dates, 1) if n_dates > 0 else 0
    return {
        "label":    label,
        "n":        n,
        "n_jan":    n_jan,
        "n_feb":    n_feb,
        "bars_per_day": bpd,
        "pw":       round(pw_all, 5),
        "pw_jan":   round(pw_jan, 5),
        "pw_feb":   round(pw_feb, 5),
        "lift_vs_baseline": None,   # filled in later
    }


def print_gate(g: dict, baseline: float, breakeven: float) -> None:
    lift    = g["pw"] - baseline
    above   = " *** ABOVE BREAKEVEN" if g["pw"] >= breakeven else ""
    stable  = abs(g["pw_jan"] - g["pw_feb"]) <= 0.05
    stab_s  = "stable" if stable else "UNSTABLE"
    print(f"    {g['label']}")
    print(f"      n={g['n']:>8,}  bars/day={g['bars_per_day']:>5.1f}  "
          f"pw={g['pw']:.3%} ({lift:+.3%} vs baseline){above}")
    print(f"      Jan: n={g['n_jan']:,}  pw={g['pw_jan']:.3%}  |  "
          f"Feb: n={g['n_feb']:,}  pw={g['pw_feb']:.3%}  [{stab_s}]")


def run_setup(
    combined: pd.DataFrame,
    setup_name: str,
    outcome_col: str,
    baseline: float,
    breakeven: float,
    gate_a_mask: pd.Series,
    gate_b_mask: pd.Series,
    gate_a_label: str,
    gate_b_label: str,
    variants: list[tuple[str, pd.Series]],
) -> dict:
    print(f"\n{'='*64}")
    print(f"{setup_name}  (baseline={baseline:.3%}  breakeven={breakeven:.3%})")
    print(f"{'='*64}")

    gates = [
        gate_stats(combined, gate_a_mask,              outcome_col, f"A: {gate_a_label}"),
        gate_stats(combined, gate_b_mask,              outcome_col, f"B: {gate_b_label}"),
        gate_stats(combined, gate_a_mask & gate_b_mask, outcome_col, "A AND B (primary)"),
    ]
    for v_label, v_mask in variants:
        gates.append(gate_stats(combined, v_mask, outcome_col, v_label))

    for g in gates:
        g["lift_vs_baseline"] = round(g["pw"] - baseline, 5)
        print_gate(g, baseline, breakeven)

    return {
        "setup":    setup_name,
        "baseline": round(baseline, 5),
        "breakeven": breakeven,
        "gates":    gates,
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

    combined      = pd.concat(all_rth, ignore_index=True)
    n_dates       = combined["Time"].dt.date.nunique()
    baseline_long  = float(combined["path_win_long"].mean())
    baseline_short = float(combined["path_win_short"].mean())
    print(f"Total RTH bars: {len(combined):,}  over {n_dates} dates")
    print(f"Baseline path_win_long:  {baseline_long:.3%}")
    print(f"Baseline path_win_short: {baseline_short:.3%}")

    rp  = combined["range_pos"]
    vd  = combined["vwap_dev"]

    # ── LONG setup: mid-range pullback near VWAP ──────────────────────────────
    long_a = (rp >= 0.30) & (rp <= 0.55)          # range position gate
    long_b = (vd >= -30)  & (vd <= 30)             # near-VWAP gate

    long_variants = [
        ("A AND B (tight rp 0.35-0.50)",
         ((rp >= 0.35) & (rp <= 0.50)) & long_b),
        ("A AND B (wider vd -40 to +40)",
         long_a & ((vd >= -40) & (vd <= 40))),
        ("A AND B (rp 0.35-0.50, vd -20 to +20)",
         ((rp >= 0.35) & (rp <= 0.50)) & ((vd >= -20) & (vd <= 20))),
    ]

    long_result = run_setup(
        combined, "LONG: mid-range pullback near VWAP",
        "path_win_long", baseline_long, breakeven=0.25,
        gate_a_mask=long_a, gate_b_mask=long_b,
        gate_a_label="range_pos [0.30, 0.55]",
        gate_b_label="vwap_dev [-30, +30] ticks",
        variants=long_variants,
    )

    # ── SHORT setup: near session high + far above VWAP ──────────────────────
    short_a = rp >= 0.85                            # near session high
    short_b = vd >= 50                              # well above VWAP

    short_variants = [
        ("A AND B (rp>=0.90, vd>=50)",
         (rp >= 0.90) & (vd >= 50)),
        ("A AND B (rp>=0.85, vd>=40)",
         short_a & (vd >= 40)),
        ("A AND B (rp>=0.85, vd>=70)",
         short_a & (vd >= 70)),
        ("A AND B (rp>=0.90, vd>=70)",
         (rp >= 0.90) & (vd >= 70)),
    ]

    short_result = run_setup(
        combined, "SHORT: near session high + well above VWAP",
        "path_win_short", baseline_short, breakeven=0.25,
        gate_a_mask=short_a, gate_b_mask=short_b,
        gate_a_label="range_pos >= 0.85",
        gate_b_label="vwap_dev >= 50 ticks",
        variants=short_variants,
    )

    # ── Print quick reference summary ────────────────────────────────────────
    print(f"\n{'='*64}")
    print("Summary: path_win for primary AND combination gates")
    print(f"  breakeven=25.0%   baseline_long={baseline_long:.3%}   baseline_short={baseline_short:.3%}")
    print(f"\n  {'Gate':40} {'pw':>8} {'lift':>8} {'bpd':>6} {'n':>8}")
    for setup in [long_result, short_result]:
        for g in setup["gates"]:
            above = " **" if g["pw"] >= 0.25 else ""
            print(f"  {g['label']:40} {g['pw']:>8.3%} {g['lift_vs_baseline']:>+8.3%} "
                  f"{g['bars_per_day']:>6.1f} {g['n']:>8,}{above}")

    results = {
        "metadata": {
            "instrument": "ESH6", "period": "Jan+Feb 2026 RTH",
            "dates": n_dates, "total_bars": len(combined),
            "tp_ticks": TP_TICKS, "sl_ticks": SL_TICKS, "horizon_bars": H_BARS,
            "baseline_long": round(baseline_long, 5),
            "baseline_short": round(baseline_short, 5),
        },
        "long_setup":  long_result,
        "short_setup": short_result,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "combination_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
