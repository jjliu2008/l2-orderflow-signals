"""
Early path analysis: does the first 1-5 minutes distinguish large winners from large losers?

For a short entry at rp>=0.85 + vd>=50, the mean reversion thesis is:
  price near or at the session high, above VWAP, will revert downward.

Thesis invalidation: if price makes a NEW session high post-entry, the reversal
is not happening — price is breaking out, not reverting.

This script:
  1. Collects all short setup entries and their 15-min time-exit PnL (baseline).
  2. For each entry, computes early-path features at 1/3/5 min:
       - new_high: did price exceed the session high that existed at entry?
       - max_adv: maximum adverse move (price up, bad for short)
       - max_fav: maximum favorable move (price down, good for short)
       - net_sv: net signed volume (positive = buying dominant) [if column exists]
  3. Compares these features across outcome groups (large win vs large loss).
  4. Backtests a conditional stop: exit at the first new-session-high bar within
     N minutes; otherwise hold to 15 min.

The conditional stop targets the left tail (large losers) without touching the
right tail (winners that develop quietly over 15 min).

Usage:
    python scripts/signal_early_path_analysis.py

Outputs:
    artifacts/signal_research/early_path_analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/signal_research"

TICK         = 0.25
H            = 9_000      # 15-min forward window (100ms bars)
RTH_START    = "14:30"
RTH_END      = "21:00"
SAMPLE_EVERY = 300
FRICTION     = 1.36       # round-trip ticks

# Early observation windows (bars at 100ms each)
EARLY_WINDOWS: dict[str, int] = {
    "1min": 600,
    "3min": 1_800,
    "5min": 3_000,
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_date(date_dir: Path) -> pd.DataFrame | None:
    date_str = date_dir.name.replace("date=", "")
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None

    base_cols = ["Time", "mid", "trade_volume"]
    try:
        df = pd.read_parquet(pq, columns=base_cols + ["signed_volume"])
    except Exception:
        df = pd.read_parquet(pq, columns=base_cols)

    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    df["in_rth"] = (df["Time"] >= lo) & (df["Time"] < hi)

    # VWAP deviation (ticks)
    vol  = df["trade_volume"].where(df["in_rth"], np.nan).clip(lower=0)
    vwap = (df["mid"] * vol.fillna(0)).cumsum() / vol.fillna(0).cumsum().replace(0, np.nan)
    df["vwap_dev"] = (df["mid"] - vwap.ffill()) / TICK

    # Session range position and running session high
    mid_rth        = df["mid"].where(df["in_rth"], np.nan)
    sess_high      = mid_rth.cummax()
    sess_low       = mid_rth.cummin()
    sess_range     = (sess_high - sess_low).replace(0, np.nan)
    df["range_pos"] = (df["mid"] - sess_low) / sess_range
    df["sess_high"] = sess_high   # session high INCLUDING current bar

    return df


# ── Per-entry feature extraction ──────────────────────────────────────────────

def _process_date(df: pd.DataFrame) -> list[dict]:
    mid_arr = df["mid"].values
    sh_arr  = df["sess_high"].values   # running session high at each bar
    sv_arr  = df["signed_volume"].values if "signed_volume" in df.columns else None
    n       = len(mid_arr)

    rth = df[df["in_rth"]]
    sig = rth[(rth["range_pos"] >= 0.85) & (rth["vwap_dev"] >= 50)]
    entries = sig.index.tolist()[::SAMPLE_EVERY]

    records: list[dict] = []
    for idx in entries:
        exit_idx = min(idx + H, n - 1)
        if exit_idx - idx < H // 2:   # too close to end of day
            continue

        entry_mid = float(mid_arr[idx])
        entry_sh  = float(sh_arr[idx])   # session high at entry (>= entry_mid)
        pnl_15    = (entry_mid - float(mid_arr[exit_idx])) / TICK

        rec: dict = {"pnl_15": round(pnl_15, 4)}

        for win, W in EARLY_WINDOWS.items():
            end   = min(idx + W + 1, n)
            early = mid_arr[idx + 1 : end]
            if len(early) == 0:
                continue

            # For short: adverse = price up, favorable = price down
            adv    = (early - entry_mid) / TICK
            fav    = (entry_mid - early) / TICK
            nh_mask = early > entry_sh     # new session high post-entry

            new_high = bool(nh_mask.any())
            rec[f"new_high_{win}"] = new_high
            rec[f"max_adv_{win}"]  = round(float(adv.max()), 2)
            rec[f"max_fav_{win}"]  = round(float(fav.max()), 2)

            # Conditional stop: exit at first new-high bar
            if new_high:
                first_nh_idx      = int(np.argmax(nh_mask))
                first_nh_price    = float(early[first_nh_idx])
                rec[f"nh_pnl_{win}"] = round((entry_mid - first_nh_price) / TICK, 4)
            else:
                rec[f"nh_pnl_{win}"] = None   # rule not triggered

            if sv_arr is not None:
                rec[f"net_sv_{win}"] = round(float(np.nansum(sv_arr[idx + 1 : end])), 0)

        records.append(rec)
    return records


# ── Summary helpers ───────────────────────────────────────────────────────────

def _group_stats(group: list[dict], win: str) -> dict:
    """Compute early-path stats for an outcome group."""
    if not group:
        return {"n": 0}
    nh_vals  = [r.get(f"new_high_{win}", False) for r in group]
    adv_vals = [r[f"max_adv_{win}"] for r in group if f"max_adv_{win}" in r]
    fav_vals = [r[f"max_fav_{win}"] for r in group if f"max_fav_{win}" in r]
    sv_vals  = [r[f"net_sv_{win}"]  for r in group if f"net_sv_{win}"  in r]
    out = {
        "n":           len(group),
        "nh_rate":     round(float(np.mean(nh_vals)), 4),
        "med_max_adv": round(float(np.median(adv_vals)), 2) if adv_vals else None,
        "med_max_fav": round(float(np.median(fav_vals)), 2) if fav_vals else None,
    }
    if sv_vals:
        out["med_net_sv"] = round(float(np.median(sv_vals)), 0)
    return out


def _dist_stats(pnls: np.ndarray, label: str) -> dict:
    n = len(pnls)
    if n == 0:
        return {"label": label, "n": 0}
    net = pnls - FRICTION
    return {
        "label":         label,
        "n":             n,
        "gross_ev":      round(float(pnls.mean()), 4),
        "gross_std":     round(float(pnls.std()),  4),
        "gross_sharpe":  round(float(pnls.mean() / pnls.std()), 4) if pnls.std() > 0 else 0,
        "gross_wr":      round(float((pnls > 0).mean()), 4),
        "net_ev":        round(float(net.mean()),  4),
        "pct_gt_plus3":  round(float((pnls >  3.0).mean()), 4),
        "pct_near_zero": round(float((np.abs(pnls) < 1.0).mean()), 4),
        "pct_lt_minus3": round(float((pnls < -3.0).mean()), 4),
        "percentiles": {
            f"p{p}": round(float(np.percentile(pnls, p)), 2)
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    print(f"Processing {len(date_dirs)} ESH6 dates (Jan+Feb 2026)...")
    print(f"Entry: short at rp>=0.85 + vd>=50, subsample every {SAMPLE_EVERY} bars")
    print(f"Baseline exit: pure time exit at 15 min")

    all_records: list[dict] = []
    for dd in date_dirs:
        df = _load_date(dd)
        if df is None:
            continue
        recs = _process_date(df)
        all_records.extend(recs)
        print(f"  {dd.name.replace('date=', '')}: {len(recs)} entries")

    n_total = len(all_records)
    print(f"\nTotal entries collected: {n_total:,}")

    pnl_15 = np.array([r["pnl_15"] for r in all_records])

    # ── Outcome groups ────────────────────────────────────────────────────────
    large_win  = [r for r in all_records if r["pnl_15"] >  3.0]
    neutral    = [r for r in all_records if -3.0 <= r["pnl_15"] <= 3.0]
    large_loss = [r for r in all_records if r["pnl_15"] < -3.0]

    print(f"\n{'='*72}")
    print("EARLY PATH DIAGNOSTIC — outcome groups defined by 15-min PnL")
    print(f"{'='*72}")
    print(f"  Large win  (pnl > +3t):  n={len(large_win):,} ({len(large_win)/n_total:.1%})")
    print(f"  Neutral  (|pnl| <= 3t):  n={len(neutral):,} ({len(neutral)/n_total:.1%})")
    print(f"  Large loss (pnl < -3t):  n={len(large_loss):,} ({len(large_loss)/n_total:.1%})")

    diag_results: dict = {}
    for win in EARLY_WINDOWS:
        lw = _group_stats(large_win,  win)
        nt = _group_stats(neutral,    win)
        ll = _group_stats(large_loss, win)

        print(f"\n  -- {win} early window --")
        print(f"  {'Metric':30} {'LargeWin':>10} {'Neutral':>10} {'LargeLoss':>10}")
        print(f"  {'New session high rate':30} "
              f"{lw['nh_rate']:>10.1%} {nt['nh_rate']:>10.1%} {ll['nh_rate']:>10.1%}")
        if lw.get("med_max_adv") is not None:
            print(f"  {'Median max adverse (ticks)':30} "
                  f"{lw['med_max_adv']:>10.2f} {nt['med_max_adv']:>10.2f} {ll['med_max_adv']:>10.2f}")
            print(f"  {'Median max favorable (ticks)':30} "
                  f"{lw['med_max_fav']:>10.2f} {nt['med_max_fav']:>10.2f} {ll['med_max_fav']:>10.2f}")
        if "med_net_sv" in lw:
            print(f"  {'Median net signed volume':30} "
                  f"{lw['med_net_sv']:>10.0f} {nt.get('med_net_sv',0):>10.0f} {ll.get('med_net_sv',0):>10.0f}")

        # Separation ratio: nh_rate(loser) / nh_rate(winner) — higher = better separator
        sep = ll["nh_rate"] / lw["nh_rate"] if lw["nh_rate"] > 0 else float("inf")
        print(f"  Separation ratio (loss_nh_rate / win_nh_rate): {sep:.2f}x")

        diag_results[win] = {
            "large_win":  lw,
            "neutral":    nt,
            "large_loss": ll,
            "separation_ratio": round(sep, 3),
        }

    # ── Conditional stop rule backtest ────────────────────────────────────────
    print(f"\n{'='*72}")
    print("CONDITIONAL STOP BACKTEST — exit if new session high within N min")
    print(f"Conditional stop exits at the PRICE of the first new-high bar.")
    print(f"If no new high: hold to 15 min (standard time exit).")
    print(f"{'='*72}")

    baseline = _dist_stats(pnl_15, "Baseline 15min  ")
    all_rule_stats: list[dict] = []

    for win in EARLY_WINDOWS:
        pnls_rule = np.array([
            r[f"nh_pnl_{win}"] if r.get(f"nh_pnl_{win}") is not None else r["pnl_15"]
            for r in all_records
            if f"nh_pnl_{win}" in r
        ])
        n_triggered = sum(
            1 for r in all_records
            if r.get(f"nh_pnl_{win}") is not None
        )
        s = _dist_stats(pnls_rule, f"Stop @ {win}     ")
        s["n_triggered"]     = n_triggered
        s["pct_triggered"]   = round(n_triggered / n_total, 4) if n_total > 0 else 0
        all_rule_stats.append(s)

    # Print comparison table
    hdr = f"  {'Strategy':20} {'GrossEV':>9} {'Std':>7} {'Sharpe':>8} {'WR':>7} {'NetEV':>8} {'>+3t':>6} {'<-3t':>6} {'%Trig':>7}"
    print(f"\n{hdr}")
    print(f"  {'-'*70}")

    def _row(s: dict, trig: str = "") -> str:
        return (f"  {s['label']:20} {s['gross_ev']:>+9.3f} {s['gross_std']:>7.2f} "
                f"{s['gross_sharpe']:>+8.4f} {s['gross_wr']:>7.1%} {s['net_ev']:>+8.3f} "
                f"{s['pct_gt_plus3']:>6.1%} {s['pct_lt_minus3']:>6.1%} {trig:>7}")

    print(_row(baseline, ""))
    for s in all_rule_stats:
        trig_str = f"{s['pct_triggered']:.1%}"
        print(_row(s, trig_str))

    # EV delta and Sharpe delta vs baseline
    print(f"\n  Deltas vs baseline (Sharpe improvement = key metric):")
    for s in all_rule_stats:
        d_ev     = s["gross_ev"]     - baseline["gross_ev"]
        d_sharpe = s["gross_sharpe"] - baseline["gross_sharpe"]
        d_std    = s["gross_std"]    - baseline["gross_std"]
        d_ltail  = s["pct_lt_minus3"] - baseline["pct_lt_minus3"]
        print(f"  {s['label'].strip():16}:  "
              f"dEV={d_ev:>+7.3f}t  dStd={d_std:>+7.2f}t  "
              f"dSharpe={d_sharpe:>+7.4f}  dLargeL={d_ltail:>+6.1%}")

    # ── Detailed percentile comparison ────────────────────────────────────────
    print(f"\n  Percentile comparison (ticks):")
    print(f"  {'Strategy':20} {'p1':>6} {'p5':>6} {'p10':>6} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6} {'p99':>6}")
    for s in [baseline] + all_rule_stats:
        p = s["percentiles"]
        print(f"  {s['label']:20} {p['p1']:>6.1f} {p['p5']:>6.1f} {p['p10']:>6.1f} "
              f"{p['p25']:>6.1f} {p['p50']:>6.1f} {p['p75']:>6.1f} {p['p90']:>6.1f} {p['p99']:>6.1f}")

    # ── Winners preserved / losers saved analysis ─────────────────────────────
    print(f"\n{'='*72}")
    print("TRADE CLASSIFICATION UNDER EACH RULE")
    print(f"{'='*72}")
    print(f"{'':22} {'Baseline':>12}  {'Stop@1min':>12}  {'Stop@3min':>12}  {'Stop@5min':>12}")

    # For each rule: how many of the baseline large-losers are "saved" (exit early)?
    # A loser is "saved" if: baseline pnl < -3 AND new_high triggered AND nh_pnl > pnl_15
    for win in EARLY_WINDOWS:
        losers_saved = sum(
            1 for r in all_records
            if r["pnl_15"] < -3.0
            and r.get(f"nh_pnl_{win}") is not None
            and r[f"nh_pnl_{win}"] > r["pnl_15"]
        )
        winners_cut = sum(
            1 for r in all_records
            if r["pnl_15"] > 3.0
            and r.get(f"nh_pnl_{win}") is not None
            and r[f"nh_pnl_{win}"] < r["pnl_15"]
        )
        n_large_loss = len(large_loss)
        n_large_win  = len(large_win)
        print(f"  {win}: losers saved {losers_saved}/{n_large_loss} ({losers_saved/n_large_loss:.1%}), "
              f"winners cut {winners_cut}/{n_large_win} ({winners_cut/n_large_win:.1%})")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "instrument":      "ESH6",
            "period":          "Jan+Feb 2026 RTH",
            "entry":           "rp>=0.85 AND vwap_dev>=50t (short)",
            "n_entries":       n_total,
            "subsample_every": SAMPLE_EVERY,
            "tick_size":       TICK,
            "friction_ticks":  FRICTION,
            "note":            "Conditional stop exits at mid price of first bar exceeding entry session high",
        },
        "baseline_15min":              baseline,
        "early_path_diagnostics":      diag_results,
        "conditional_stop_results":    all_rule_stats,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "early_path_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
