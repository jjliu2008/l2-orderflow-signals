"""
Dynamic exit simulation on the ESH6 short setup (rp>=0.85, vd>=50).

For each sampled entry bar, simulate the trade path at 100ms resolution and
apply different exit strategies. Compare realized EV vs fixed TP/SL baseline.

Exit strategies tested:
  1. Fixed TP=18t / SL=6t                          (baseline)
  2. Fixed TP=15t / SL=5t                          (tighter baseline)
  3. Fixed TP=12t / SL=4t                          (even tighter)
  4. Trailing stop: initial=6t, trail=3t (tight)
  5. Trailing stop: initial=6t, trail=6t (equal to initial)
  6. Breakeven + trail: move to BE at +3t, trail 3t after +6t
  7. Breakeven + trail: move to BE at +6t, trail 3t after +9t
  8. Breakeven + trail: move to BE at +6t, trail 6t after +9t
  9. Time exit only (hold 15 min, exit at mid)     (distribution reference)
 10. Flow gate: TP=18t/SL=6t + exit if OBI > 0.6 post-entry (buying pressure)

Entry subsampling: every SAMPLE_EVERY bars within signal zone to reduce compute
and path overlap. Mean EV is unbiased; variance understates true variance due to
overlapping paths.

Usage:
    python scripts/signal_dynamic_exit_test.py

Outputs:
    artifacts/signal_research/dynamic_exit_test.json
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
H         = 9_000       # 15-min forward window (100ms bars)
RTH_START = "14:30"
RTH_END   = "21:00"

# Subsample every N bars within signal zone (reduces compute + correlation)
# 300 bars = every 30 sec; yields ~160 entries/day, ~6000 entries total
SAMPLE_EVERY = 300

LOAD_COLS = ["Time", "mid", "trade_volume", "order_book_imbalance"]


def _fwd_max(s: pd.Series, n: int) -> np.ndarray:
    rev = s.values[::-1]
    return pd.Series(rev).rolling(n, min_periods=1).max().values[::-1]


def _fwd_min(s: pd.Series, n: int) -> np.ndarray:
    rev = s.values[::-1]
    return pd.Series(rev).rolling(n, min_periods=1).min().values[::-1]


def simulate_fixed(fav: np.ndarray, adv: np.ndarray, tp: float, sl: float) -> float:
    """Fixed TP/SL. Returns realized PnL in ticks."""
    for t in range(len(fav)):
        if fav[t] >= tp:
            return tp
        if adv[t] >= sl:
            return -sl
    return float(fav[-1])  # time exit


def simulate_trail(fav: np.ndarray, adv: np.ndarray,
                   initial_sl: float, trail: float) -> float:
    """
    Trailing stop from entry.
    Initial hard stop at -initial_sl.
    Once price moves favorably, trail by 'trail' ticks from peak.
    No fixed TP — let the trail capture the exit.
    """
    peak = 0.0
    for t in range(len(fav)):
        if adv[t] >= initial_sl:       # initial hard stop
            return -initial_sl
        peak = max(peak, fav[t])
        trail_level = peak - trail
        if t > 0 and fav[t] < trail_level:  # trail triggered
            return max(trail_level, -initial_sl)
    return float(fav[-1])


def simulate_be_trail(fav: np.ndarray, adv: np.ndarray,
                      initial_sl: float, be_at: float,
                      trail_activate: float, trail: float) -> float:
    """
    Phase 1: initial hard stop at -initial_sl.
    Phase 2: at be_at ticks favorable, stop moves to 0 (breakeven).
    Phase 3: at trail_activate ticks favorable, trail by 'trail' ticks.
    """
    stop = -initial_sl
    trail_active = False
    peak = 0.0
    for t in range(len(fav)):
        f = fav[t]
        a = adv[t]
        # Update stop
        if f >= trail_activate:
            trail_active = True
        if f >= be_at and not trail_active:
            stop = max(stop, 0.0)
        if trail_active:
            peak = max(peak, f)
            stop = max(stop, peak - trail)
        # Check exit
        if f <= stop:
            return stop
        if a >= -stop and stop < 0:  # adverse stop (when stop is negative)
            return stop
        # Simpler: exit if current favorable falls below current stop
        if t > 0 and f < stop:
            return stop
    return float(fav[-1])


def simulate_flow_gate(fav: np.ndarray, adv: np.ndarray,
                       obi: np.ndarray, tp: float, sl: float,
                       obi_exit_threshold: float) -> float:
    """
    Fixed TP/SL with a flow gate: also exit if order book imbalance
    flips strongly bullish (OBI > threshold) — buying pressure into short.
    """
    for t in range(len(fav)):
        if fav[t] >= tp:
            return tp
        if adv[t] >= sl:
            return -sl
        # OBI > threshold: buyers dominating, exit short defensively
        if t > 0 and obi[t] > obi_exit_threshold:
            return float(fav[t])
    return float(fav[-1])


def load_and_process(date_dir: Path) -> pd.DataFrame | None:
    date_str = date_dir.name.replace("date=", "")
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq, columns=LOAD_COLS)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)

    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    # Keep full day for forward windows, tag RTH bars
    df["in_rth"] = (df["Time"] >= lo) & (df["Time"] < hi)

    # VWAP deviation
    rth_mask = df["in_rth"]
    vol  = df["trade_volume"].where(rth_mask, 0).clip(lower=0)
    vol  = vol.where(rth_mask, np.nan)
    vwap = (df["mid"] * vol.fillna(0)).cumsum() / vol.fillna(0).cumsum().replace(0, np.nan)
    vwap = vwap.ffill()
    df["vwap_dev"] = (df["mid"] - vwap) / TICK

    # Range position
    mid_rth = df["mid"].where(rth_mask, np.nan)
    sess_high  = mid_rth.cummax()
    sess_low   = mid_rth.cummin()
    sess_range = (sess_high - sess_low).replace(0, np.nan)
    df["range_pos"] = (df["mid"] - sess_low) / sess_range

    return df


def collect_entries(df: pd.DataFrame) -> list[int]:
    """Return subsampled indices of signal bars (rp>=0.85, vd>=50) within RTH."""
    rth_df = df[df["in_rth"]]
    signal = rth_df[(rth_df["range_pos"] >= 0.85) & (rth_df["vwap_dev"] >= 50)]
    indices = signal.index.tolist()
    # Subsample every SAMPLE_EVERY to reduce path overlap
    return indices[::SAMPLE_EVERY]


def run_simulations(df: pd.DataFrame, entry_indices: list[int]) -> dict[str, list[float]]:
    """Simulate all strategies for all entries. Returns dict of pnl lists."""
    mid_arr = df["mid"].values
    obi_arr = df["order_book_imbalance"].values
    n_total = len(mid_arr)

    results: dict[str, list[float]] = {
        "fixed_18_6":       [],
        "fixed_15_5":       [],
        "fixed_12_4":       [],
        "trail_6_3":        [],
        "trail_6_6":        [],
        "be3_trail3_act6":  [],
        "be6_trail3_act9":  [],
        "be6_trail6_act9":  [],
        "time_exit":        [],
        "flow_gate_18_6":   [],
    }

    for idx in entry_indices:
        end = min(idx + 1 + H, n_total)
        if end - idx - 1 < 100:    # too close to end of day
            continue

        entry_mid = mid_arr[idx]
        future    = mid_arr[idx + 1 : end]
        fav       = (entry_mid - future) / TICK   # short: favorable = price down
        adv       = (future - entry_mid) / TICK   # short: adverse = price up

        future_obi = obi_arr[idx + 1 : end]
        future_obi = np.where(np.isfinite(future_obi), future_obi, 0.5)

        results["fixed_18_6"].append(simulate_fixed(fav, adv, 18, 6))
        results["fixed_15_5"].append(simulate_fixed(fav, adv, 15, 5))
        results["fixed_12_4"].append(simulate_fixed(fav, adv, 12, 4))
        results["trail_6_3"].append(simulate_trail(fav, adv, initial_sl=6, trail=3))
        results["trail_6_6"].append(simulate_trail(fav, adv, initial_sl=6, trail=6))
        results["be3_trail3_act6"].append(
            simulate_be_trail(fav, adv, initial_sl=6, be_at=3, trail_activate=6, trail=3))
        results["be6_trail3_act9"].append(
            simulate_be_trail(fav, adv, initial_sl=6, be_at=6, trail_activate=9, trail=3))
        results["be6_trail6_act9"].append(
            simulate_be_trail(fav, adv, initial_sl=6, be_at=6, trail_activate=9, trail=6))
        results["time_exit"].append(float(fav[-1]) if len(fav) >= H else float(fav[-1]))
        results["flow_gate_18_6"].append(
            simulate_flow_gate(fav, adv, future_obi, tp=18, sl=6, obi_exit_threshold=0.60))

    return results


def summarise(pnls: list[float], label: str) -> dict:
    a = np.array(pnls)
    wins  = a[a > 0]
    losses = a[a <= 0]
    n = len(a)
    return {
        "label":    label,
        "n":        n,
        "ev":       round(float(a.mean()), 4) if n > 0 else 0.0,
        "win_rate": round(float((a > 0).mean()), 4) if n > 0 else 0.0,
        "avg_win":  round(float(wins.mean()),   4) if len(wins)   > 0 else 0.0,
        "avg_loss": round(float(losses.mean()), 4) if len(losses) > 0 else 0.0,
        "p10": round(float(np.percentile(a, 10)), 2) if n > 0 else 0.0,
        "p25": round(float(np.percentile(a, 25)), 2) if n > 0 else 0.0,
        "p50": round(float(np.percentile(a, 50)), 2) if n > 0 else 0.0,
        "p75": round(float(np.percentile(a, 75)), 2) if n > 0 else 0.0,
        "p90": round(float(np.percentile(a, 90)), 2) if n > 0 else 0.0,
    }


def print_summary(stats: list[dict], baseline_ev: float) -> None:
    print(f"\n{'='*78}")
    print(f"{'Strategy':28} {'EV':>7} {'vs base':>8} {'WR':>7} {'avgW':>7} "
          f"{'avgL':>7} {'p50':>6} {'p90':>6}")
    print(f"{'='*78}")
    for s in stats:
        lift = s["ev"] - baseline_ev
        flag = " ***" if s["ev"] > 0 else ""
        print(f"  {s['label']:26} {s['ev']:>7.3f} {lift:>+8.3f} "
              f"{s['win_rate']:>7.1%} {s['avg_win']:>7.2f} "
              f"{s['avg_loss']:>7.2f} {s['p50']:>6.2f} {s['p90']:>6.2f}{flag}")


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    print(f"Processing {len(date_dirs)} ESH6 dates (Jan+Feb 2026)...")
    print(f"Entry: rp>=0.85, vd>=50 (short setup), subsample every {SAMPLE_EVERY} bars")
    print(f"Forward window: {H} bars = 15 min")

    all_pnls: dict[str, list[float]] = {}
    total_entries = 0

    for dd in date_dirs:
        df = load_and_process(dd)
        if df is None or len(df) == 0:
            continue
        entries = collect_entries(df)
        if not entries:
            continue
        pnls = run_simulations(df, entries)
        total_entries += len(next(iter(pnls.values())))
        for k, v in pnls.items():
            all_pnls.setdefault(k, []).extend(v)
        date_str = dd.name.replace("date=", "")
        n = len(next(iter(pnls.values())))
        print(f"  {date_str}: {n} entries")

    print(f"\nTotal entries simulated: {total_entries:,}")
    print(f"(subsampled every {SAMPLE_EVERY} bars from signal zone; paths overlap)")

    STRATEGY_LABELS = {
        "fixed_18_6":      "Fixed TP=18t / SL=6t  [baseline]",
        "fixed_15_5":      "Fixed TP=15t / SL=5t",
        "fixed_12_4":      "Fixed TP=12t / SL=4t",
        "trail_6_3":       "Trail init=6t trail=3t",
        "trail_6_6":       "Trail init=6t trail=6t",
        "be3_trail3_act6": "BE+trail: BE@3t trail3t@6t",
        "be6_trail3_act9": "BE+trail: BE@6t trail3t@9t",
        "be6_trail6_act9": "BE+trail: BE@6t trail6t@9t",
        "time_exit":       "Time exit (15 min, no TP/SL)",
        "flow_gate_18_6":  "Flow gate: TP18/SL6+OBI>0.6 exit",
    }

    stats = []
    for k, label in STRATEGY_LABELS.items():
        if k in all_pnls:
            stats.append(summarise(all_pnls[k], label))

    baseline_ev = next(s["ev"] for s in stats if "baseline" in s["label"])
    print_summary(stats, baseline_ev)

    print(f"\nNote: EV in ticks (CL tick=$0.25). Positive EV = profitable direction.")
    print(f"BE thresholds do not include spread/commission costs.")
    print(f"\nKey question: does any dynamic strategy produce EV > 0?")
    best = max(stats, key=lambda s: s["ev"])
    print(f"Best EV: {best['label']} = {best['ev']:.3f} ticks")

    results = {
        "metadata": {
            "instrument": "ESH6", "period": "Jan+Feb 2026 RTH",
            "entry_signal": "rp>=0.85 AND vwap_dev>=50t (short)",
            "forward_window_bars": H,
            "subsample_every": SAMPLE_EVERY,
            "total_entries": total_entries,
            "tick_size": TICK,
            "note": "Simulation uses mid price; no spread/commission; paths overlap due to subsampling",
        },
        "strategies": stats,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "dynamic_exit_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
