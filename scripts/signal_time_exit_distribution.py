"""
Distribution analysis: pure time exit on the ESH6 short setup (rp>=0.85, vd>=50).

Tests hold times: 10, 15, 20, 30 minutes.
Reports full PnL distribution, after-cost viability, and Sharpe ratio.

The question: is +2.57t mean EV driven by a few large winners (low Sharpe,
fragile) or a tight positive distribution (robust, survives friction)?

Usage:
    python scripts/signal_time_exit_distribution.py

Outputs:
    artifacts/signal_research/time_exit_distribution.json
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
RTH_START = "14:30"
RTH_END   = "21:00"

SAMPLE_EVERY = 300   # subsample within signal zone

HOLD_TIMES = {
    "10min":  6_000,
    "15min":  9_000,
    "20min": 12_000,
    "30min": 18_000,
}

# Friction estimates in ticks (round-trip, ES)
# ES spread = 1 tick bid-ask, so 0.5t per side = 1t round-trip
# Commission ~$4.50 round-trip = 0.36 ticks at $12.50/tick
FRICTION_SPREAD    = 1.00   # ticks round-trip (0.5t each side)
FRICTION_COMMISSION = 0.36  # ticks round-trip (~$4.50 at $12.50/tick)
FRICTION_TOTAL     = FRICTION_SPREAD + FRICTION_COMMISSION   # ~1.36 ticks

LOAD_COLS = ["Time", "mid", "trade_volume"]


def _load_and_process(date_dir: Path) -> pd.DataFrame | None:
    date_str = date_dir.name.replace("date=", "")
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq, columns=LOAD_COLS)
    df["Time"] = pd.to_datetime(df["Time"], utc=True)

    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    df["in_rth"] = (df["Time"] >= lo) & (df["Time"] < hi)

    vol  = df["trade_volume"].where(df["in_rth"], np.nan).clip(lower=0)
    vwap = (df["mid"] * vol.fillna(0)).cumsum() / vol.fillna(0).cumsum().replace(0, np.nan)
    vwap = vwap.ffill()
    df["vwap_dev"] = (df["mid"] - vwap) / TICK

    mid_rth    = df["mid"].where(df["in_rth"], np.nan)
    sess_range = (mid_rth.cummax() - mid_rth.cummin()).replace(0, np.nan)
    df["range_pos"] = (df["mid"] - mid_rth.cummin()) / sess_range

    return df


def _collect_entries(df: pd.DataFrame) -> list[int]:
    rth = df[df["in_rth"]]
    sig = rth[(rth["range_pos"] >= 0.85) & (rth["vwap_dev"] >= 50)]
    return sig.index.tolist()[::SAMPLE_EVERY]


def _dist_stats(pnls: np.ndarray, label: str, friction: float) -> dict:
    n = len(pnls)
    if n == 0:
        return {}
    net = pnls - friction
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "label":          label,
        "n":              n,
        # Gross (no friction)
        "gross_ev":       round(float(pnls.mean()), 4),
        "gross_std":      round(float(pnls.std()),  4),
        "gross_sharpe":   round(float(pnls.mean() / pnls.std()), 4) if pnls.std() > 0 else 0,
        "gross_wr":       round(float((pnls > 0).mean()), 4),
        "gross_pcts":     {f"p{p}": round(float(np.percentile(pnls, p)), 2) for p in pcts},
        # Net (after friction)
        "friction_ticks": friction,
        "net_ev":         round(float(net.mean()), 4),
        "net_std":        round(float(net.std()),  4),
        "net_sharpe":     round(float(net.mean() / net.std()), 4) if net.std() > 0 else 0,
        "net_wr":         round(float((net > 0).mean()), 4),
        # Outcome buckets (gross)
        "pct_gt_plus3":   round(float((pnls >  3.0).mean()), 4),
        "pct_plus1_3":    round(float(((pnls >= 1.0) & (pnls <= 3.0)).mean()), 4),
        "pct_near_zero":  round(float((np.abs(pnls) < 1.0).mean()), 4),
        "pct_minus1_3":   round(float(((pnls <= -1.0) & (pnls >= -3.0)).mean()), 4),
        "pct_lt_minus3":  round(float((pnls < -3.0).mean()), 4),
    }


def _print_dist(s: dict) -> None:
    print(f"\n  {s['label']}  (n={s['n']:,})")
    print(f"    Gross:  EV={s['gross_ev']:+.3f}t  std={s['gross_std']:.2f}t  "
          f"Sharpe={s['gross_sharpe']:+.4f}  WR={s['gross_wr']:.1%}")
    print(f"    Net ({s['friction_ticks']:.2f}t friction):  "
          f"EV={s['net_ev']:+.3f}t  WR={s['net_wr']:.1%}")
    p = s["gross_pcts"]
    print(f"    Percentiles (ticks): "
          f"p1={p['p1']:.1f}  p10={p['p10']:.1f}  p25={p['p25']:.1f}  "
          f"p50={p['p50']:.1f}  p75={p['p75']:.1f}  p90={p['p90']:.1f}  p99={p['p99']:.1f}")
    print(f"    PnL buckets (gross):  "
          f">+3t={s['pct_gt_plus3']:.1%}  +1to3={s['pct_plus1_3']:.1%}  "
          f"|<1t|={s['pct_near_zero']:.1%}  -1to-3={s['pct_minus1_3']:.1%}  "
          f"<-3t={s['pct_lt_minus3']:.1%}")


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    print(f"Processing {len(date_dirs)} ESH6 dates (Jan+Feb 2026)...")
    print(f"Entry: short at rp>=0.85 + vd>=50, subsample every {SAMPLE_EVERY} bars")
    print(f"Exit: pure time exit at hold_time (mid price, no TP/SL)")
    print(f"Friction assumption: {FRICTION_TOTAL:.2f} ticks round-trip "
          f"(1.00t spread + {FRICTION_COMMISSION:.2f}t commission)")

    all_pnls: dict[str, list[float]] = {k: [] for k in HOLD_TIMES}
    total_entries = 0

    for dd in date_dirs:
        df = _load_and_process(dd)
        if df is None:
            continue
        entries = _collect_entries(df)
        if not entries:
            continue

        mid_arr = df["mid"].values
        n_total = len(mid_arr)
        n_valid = 0

        for idx in entries:
            entry_mid = mid_arr[idx]
            for hold_name, H in HOLD_TIMES.items():
                exit_idx = min(idx + H, n_total - 1)
                if exit_idx <= idx:
                    continue
                exit_mid = mid_arr[exit_idx]
                pnl = (entry_mid - exit_mid) / TICK   # short: positive when price falls
                all_pnls[hold_name].append(pnl)
            n_valid += 1

        total_entries += n_valid

    print(f"\nTotal entries: {total_entries:,}")

    print(f"\n{'='*70}")
    print("TIME EXIT DISTRIBUTION — short setup (rp>=0.85, vd>=50) on ESH6")
    print(f"{'='*70}")

    stats = []
    for hold_name, H in HOLD_TIMES.items():
        pnls = np.array(all_pnls[hold_name])
        s = _dist_stats(pnls, f"{hold_name} hold ({H} bars)", FRICTION_TOTAL)
        stats.append(s)
        _print_dist(s)

    # Comparison table
    print(f"\n{'='*70}")
    print("Comparison summary:")
    print(f"  {'Hold':8} {'GrossEV':>9} {'Std':>7} {'Sharpe':>8} {'WR':>7} "
          f"{'NetEV':>8} {'NetWR':>7}")
    for s in stats:
        be_flag = " ***" if s["net_ev"] > 0 else ""
        print(f"  {s['label'][:8]:8} {s['gross_ev']:>+9.3f} {s['gross_std']:>7.2f} "
              f"{s['gross_sharpe']:>+8.4f} {s['gross_wr']:>7.1%} "
              f"{s['net_ev']:>+8.3f} {s['net_wr']:>7.1%}{be_flag}")

    # Unconditional baseline for comparison
    print(f"\n  For reference: unconditional ESH6 RTH short EV at 15 min:")
    # Load all RTH bars and compute time-exit pnl unconditionally
    all_uncond: list[float] = []
    for dd in date_dirs:
        df = _load_and_process(dd)
        if df is None:
            continue
        rth = df[df["in_rth"]]
        mid_arr = df["mid"].values
        n_total = len(mid_arr)
        for idx in rth.index[::SAMPLE_EVERY]:
            exit_idx = min(idx + 9_000, n_total - 1)
            if exit_idx > idx:
                all_uncond.append((mid_arr[idx] - mid_arr[exit_idx]) / TICK)

    uncond = np.array(all_uncond)
    print(f"  Unconditional: EV={uncond.mean():+.3f}t  std={uncond.std():.2f}t  "
          f"WR={float((uncond > 0).mean()):.1%}  n={len(uncond):,}")
    print(f"  Signal lift:   {stats[1]['gross_ev'] - uncond.mean():+.3f}t EV  "
          f"{stats[1]['gross_wr'] - float((uncond > 0).mean()):+.1%} WR")

    results = {
        "metadata": {
            "instrument": "ESH6", "period": "Jan+Feb 2026 RTH",
            "entry": "rp>=0.85 AND vwap_dev>=50t (short)",
            "exit": "pure time exit at mid",
            "subsample_every": SAMPLE_EVERY,
            "total_entries": total_entries,
            "tick_dollar": 12.50,
            "friction_ticks": FRICTION_TOTAL,
            "friction_breakdown": {
                "spread_rt": FRICTION_SPREAD,
                "commission_rt": FRICTION_COMMISSION,
            },
        },
        "hold_times": stats,
        "unconditional_15min": {
            "ev":  round(float(uncond.mean()), 4),
            "std": round(float(uncond.std()),  4),
            "wr":  round(float((uncond > 0).mean()), 4),
            "n":   len(uncond),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "time_exit_distribution.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
