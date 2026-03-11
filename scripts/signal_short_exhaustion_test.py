"""
Momentum exhaustion timing filter: within the stable short setup (rp>=0.85, vd>=50),
split bars by whether price is still making new highs vs failing to advance.

At 100ms snapshots (no OHLC), "bar high" is approximated two ways:
  - mid:       midpoint of best bid/ask
  - ask_px:    best ask = where the market can be lifted (upper bound of bar price)

"New high" bar:    current > prior (price still advancing)
"Failed new high": current <= prior (price not making new highs)

Also tests session-cummax exhaustion: bars that are in the high zone (rp>=0.85)
but no longer AT the session cummax (price has already rolled off the exact peak).

Usage:
    python scripts/signal_short_exhaustion_test.py

Outputs:
    artifacts/signal_research/short_exhaustion_test.json
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

RTH_START = "14:30"
RTH_END   = "21:00"

LOAD_COLS = ["Time", "mid", "trade_volume", "ask_price_1"]


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

    # VWAP deviation
    vol  = rth["trade_volume"].clip(lower=0)
    vwap = (rth["mid"] * vol).cumsum() / vol.cumsum().replace(0, np.nan)
    vwap = vwap.ffill()
    rth["vwap_dev"] = (rth["mid"] - vwap) / TICK

    # Session range position
    sess_high  = rth["mid"].cummax()
    sess_low   = rth["mid"].cummin()
    sess_range = (sess_high - sess_low).replace(0, np.nan)
    rth["range_pos"] = (rth["mid"] - sess_low) / sess_range

    # Session cummax (for "at high" vs "off high" split)
    rth["sess_cummax"] = sess_high

    # Bar-over-bar comparisons — shift(1) within RTH (no cross-day leakage since per-date)
    rth["mid_prev"]    = rth["mid"].shift(1)
    rth["ask_prev"]    = rth["ask_price_1"].shift(1)

    # New high flags
    rth["mid_new_high"] = rth["mid"] > rth["mid_prev"]        # mid advanced
    rth["ask_new_high"] = rth["ask_price_1"] > rth["ask_prev"] # ask (upper bound) advanced

    # At-session-high flag: mid within 1 tick of the session cummax
    rth["at_sess_high"] = (rth["sess_cummax"] - rth["mid"]) <= TICK

    # Bars since last new session high (streak of non-new-highs)
    rth["is_new_sess_high"] = rth["mid"] >= rth["sess_cummax"]
    # Count consecutive bars NOT making a new session high
    streak = []
    count = 0
    for v in rth["is_new_sess_high"]:
        if v:
            count = 0
        else:
            count += 1
        streak.append(count)
    rth["bars_since_new_high"] = streak

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
    return {
        "label":       label,
        "n":           n,
        "n_jan":       n_jan,
        "n_feb":       n_feb,
        "bars_per_day": bpd,
        "pw":          round(pw_all, 5),
        "pw_jan":      round(pw_jan, 5),
        "pw_feb":      round(pw_feb, 5),
        "jan_feb_gap": round(gap, 5),
        "stable":      gap <= 0.05,
    }


def print_gate(g: dict, baseline: float, breakeven: float) -> None:
    lift   = g["pw"] - baseline
    be_str = " *** ABOVE BREAKEVEN" if g["pw"] >= breakeven else ""
    stab   = "stable" if g["stable"] else "UNSTABLE"
    print(f"  {g['label']}")
    print(f"    n={g['n']:>9,}  bpd={g['bars_per_day']:>7.0f}  "
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

    rp  = combined["range_pos"]
    vd  = combined["vwap_dev"]
    short_base = (rp >= 0.85) & (vd >= 50)

    mid_nh  = combined["mid_new_high"]
    ask_nh  = combined["ask_new_high"]
    at_high = combined["at_sess_high"]
    streak  = combined["bars_since_new_high"]

    print(f"\n{'='*70}")
    print(f"Momentum exhaustion timing test — within rp>=0.85 + vd>=50")
    print(f"(baseline={baseline:.3%}  breakeven={breakeven:.3%})")
    print(f"{'='*70}")

    gates = []

    # --- Reference ---
    g = gate_stats(combined, short_base, "BASE: rp>=0.85 + vd>=50", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    print("\n--- Mid price: bar-over-bar ---")

    g = gate_stats(combined, short_base & mid_nh,
                   "Base + mid new high (still advancing)", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    g = gate_stats(combined, short_base & ~mid_nh,
                   "Base + mid FAILED new high (exhausted)", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    print("\n--- Ask price: bar-over-bar ---")

    g = gate_stats(combined, short_base & ask_nh,
                   "Base + ask new high (still advancing)", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    g = gate_stats(combined, short_base & ~ask_nh,
                   "Base + ask FAILED new high (exhausted)", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    print("\n--- Session cummax exhaustion ---")

    g = gate_stats(combined, short_base & at_high,
                   "Base + at session high (within 1t of cummax)", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    g = gate_stats(combined, short_base & ~at_high,
                   "Base + off session high (>1t below cummax)", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    print("\n--- Streak: N bars since last new session high ---")
    for thr in [1, 3, 5, 10, 20, 50]:
        g = gate_stats(combined, short_base & (streak >= thr),
                       f"Base + no new high for >= {thr:>3} bars", n_dates)
        gates.append(g); print_gate(g, baseline, breakeven)

    # Combined: failed mid new high AND off session high
    print("\n--- Combined exhaustion signals ---")
    g = gate_stats(combined, short_base & ~mid_nh & ~at_high,
                   "Base + mid failed AND off sess high", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    g = gate_stats(combined, short_base & ~ask_nh & ~at_high,
                   "Base + ask failed AND off sess high", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    g = gate_stats(combined, short_base & ~mid_nh & (streak >= 3),
                   "Base + mid failed AND streak >= 3", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    g = gate_stats(combined, short_base & ~mid_nh & (streak >= 5),
                   "Base + mid failed AND streak >= 5", n_dates)
    gates.append(g); print_gate(g, baseline, breakeven)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Gate':52} {'pw':>8} {'lift':>8} {'bpd':>6} {'st':>4}")
    for g in gates:
        lift  = g["pw"] - baseline
        stab  = "Y" if g["stable"] else "N"
        above = " **" if g["pw"] >= breakeven else ""
        print(f"  {g['label']:50} {g['pw']:>8.3%} {lift:>+8.3%} "
              f"{g['bars_per_day']:>6.0f} {stab:>4}{above}")

    results = {
        "metadata": {
            "instrument": "ESH6", "period": "Jan+Feb 2026 RTH",
            "dates": n_dates, "total_bars": len(combined),
            "tp_ticks": TP_TICKS, "sl_ticks": SL_TICKS, "horizon_bars": H_BARS,
            "baseline_path_win_short": round(baseline, 5),
            "breakeven": breakeven,
        },
        "gates": [
            {**g, "lift_vs_baseline": round(g["pw"] - baseline, 5)}
            for g in gates
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "short_exhaustion_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
