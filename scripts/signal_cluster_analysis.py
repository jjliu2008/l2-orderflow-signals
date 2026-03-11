"""
Cluster analysis: how many independent entry opportunities exist per day?

A "cluster" is a contiguous group of qualifying signal bars (rp>=0.85, vd>=50)
where no internal gap exceeds 15 minutes (9,000 bars at 100ms). Between clusters,
the 15-minute forward windows do not overlap — entries from different clusters
are approximately independent.

Within a cluster, all bars have heavily overlapping 15-min windows, so taking
multiple entries is redundant. One entry per cluster = maximum independent bets.

Analysis:
  1. Cluster structure: count per day, duration, size, gaps between clusters.
  2. Entry strategies per cluster:
       first_bar  — first qualifying bar (earliest entry)
       last_bar   — last qualifying bar (signal persisted longest)
       peak_rp    — bar with highest range_pos in cluster
       peak_vd    — bar with highest vwap_dev in cluster
  3. PnL distribution under one-entry-per-cluster vs subsampled baseline.
  4. Implied daily Sharpe: per_trade_sharpe * sqrt(clusters_per_day).

Usage:
    python scripts/signal_cluster_analysis.py

Outputs:
    artifacts/signal_research/cluster_analysis.json
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
FRICTION     = 1.36

# Gap threshold: clusters separated by more than this are treated as independent
CLUSTER_GAP  = 9_000     # 15 min = 9,000 bars at 100ms


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_date(date_dir: Path) -> pd.DataFrame | None:
    date_str = date_dir.name.replace("date=", "")
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None

    df = pd.read_parquet(pq, columns=["Time", "mid", "trade_volume"])
    df["Time"] = pd.to_datetime(df["Time"], utc=True)

    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    df["in_rth"] = (df["Time"] >= lo) & (df["Time"] < hi)

    vol  = df["trade_volume"].where(df["in_rth"], np.nan).clip(lower=0)
    vwap = (df["mid"] * vol.fillna(0)).cumsum() / vol.fillna(0).cumsum().replace(0, np.nan)
    df["vwap_dev"] = (df["mid"] - vwap.ffill()) / TICK

    mid_rth    = df["mid"].where(df["in_rth"], np.nan)
    sess_high  = mid_rth.cummax()
    sess_low   = mid_rth.cummin()
    sess_range = (sess_high - sess_low).replace(0, np.nan)
    df["range_pos"] = (df["mid"] - sess_low) / sess_range

    return df


# ── Cluster detection ─────────────────────────────────────────────────────────

def find_clusters(signal_indices: list[int], gap: int = CLUSTER_GAP) -> list[list[int]]:
    """
    Group signal bar indices into clusters.
    A new cluster starts when the gap from the previous signal bar > gap bars.
    Within a cluster, entries share overlapping 15-min forward windows.
    Between clusters, entries are approximately independent.
    """
    if not signal_indices:
        return []
    clusters: list[list[int]] = [[signal_indices[0]]]
    for idx in signal_indices[1:]:
        if idx - clusters[-1][-1] <= gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return clusters


# ── Entry selection within cluster ────────────────────────────────────────────

def _pick_entries(clusters: list[list[int]],
                  rp_arr: np.ndarray,
                  vd_arr: np.ndarray) -> dict[str, list[int]]:
    """Return one entry index per cluster for each strategy."""
    strategies: dict[str, list[int]] = {
        "first_bar": [],
        "last_bar":  [],
        "peak_rp":   [],
        "peak_vd":   [],
    }
    for cluster in clusters:
        strategies["first_bar"].append(cluster[0])
        strategies["last_bar"].append(cluster[-1])
        strategies["peak_rp"].append(cluster[int(np.argmax(rp_arr[cluster]))])
        strategies["peak_vd"].append(cluster[int(np.argmax(vd_arr[cluster]))])
    return strategies


# ── PnL computation ───────────────────────────────────────────────────────────

def _compute_pnl(indices: list[int], mid_arr: np.ndarray, n: int) -> list[float]:
    """Time exit PnL (15 min, short) for each entry index."""
    pnls: list[float] = []
    for idx in indices:
        exit_idx = min(idx + H, n - 1)
        if exit_idx - idx < H // 2:
            continue
        pnl = (float(mid_arr[idx]) - float(mid_arr[exit_idx])) / TICK
        pnls.append(pnl)
    return pnls


# ── Summary statistics ────────────────────────────────────────────────────────

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
    print(f"Entry: short at rp>=0.85 + vd>=50")
    print(f"Cluster gap: {CLUSTER_GAP} bars = 15 min")
    print(f"Exit: pure time exit at 15 min")

    # Per-date cluster stats
    daily_stats: list[dict] = []

    # Accumulated PnLs per strategy
    all_pnls: dict[str, list[float]] = {
        "first_bar": [], "last_bar": [], "peak_rp": [], "peak_vd": []
    }

    for dd in date_dirs:
        date_str = dd.name.replace("date=", "")
        df = _load_date(dd)
        if df is None:
            continue

        mid_arr = df["mid"].values
        rp_arr  = df["range_pos"].values
        vd_arr  = df["vwap_dev"].values
        n       = len(mid_arr)

        # All qualifying signal bars within RTH
        rth = df[df["in_rth"]]
        sig = rth[(rth["range_pos"] >= 0.85) & (rth["vwap_dev"] >= 50)]
        sig_indices = sig.index.tolist()

        if not sig_indices:
            daily_stats.append({
                "date": date_str,
                "n_signal_bars": 0,
                "n_clusters": 0,
                "cluster_durations_min": [],
                "cluster_sizes": [],
                "gaps_between_min": [],
            })
            continue

        clusters = find_clusters(sig_indices)

        # Cluster properties
        durations = []
        sizes     = []
        for c in clusters:
            dur_bars = c[-1] - c[0] + 1
            durations.append(round(dur_bars / 600, 2))   # convert bars to minutes (600 bars/min)
            sizes.append(len(c))

        # Gaps between clusters (time from end of cluster N to start of cluster N+1)
        gaps = []
        for i in range(1, len(clusters)):
            gap_bars = clusters[i][0] - clusters[i - 1][-1]
            gaps.append(round(gap_bars / 600, 2))

        # Time of first bar of each cluster relative to RTH open
        rth_start_idx = rth.index[0] if len(rth) > 0 else 0
        cluster_start_times = []
        for c in clusters:
            offset_min = (c[0] - rth_start_idx) / 600
            cluster_start_times.append(round(offset_min, 1))

        n_clusters = len(clusters)
        daily_stats.append({
            "date":                    date_str,
            "n_signal_bars":           len(sig_indices),
            "n_clusters":              n_clusters,
            "cluster_durations_min":   durations,
            "cluster_sizes":           sizes,
            "gaps_between_min":        gaps,
            "cluster_start_offset_min": cluster_start_times,
            "gaps_between_min":         gaps,
        })

        # Entry strategies: one per cluster
        strat_entries = _pick_entries(clusters, rp_arr, vd_arr)
        for strat, indices in strat_entries.items():
            pnls = _compute_pnl(indices, mid_arr, n)
            all_pnls[strat].extend(pnls)

        # Print per-date summary
        dur_str = ", ".join(f"{d:.0f}min" for d in durations)
        print(f"  {date_str}: {n_clusters} cluster(s) | "
              f"sizes={sizes} | durations=[{dur_str}]")

    # ── Cluster count distribution ─────────────────────────────────────────────
    all_cluster_counts = [d["n_clusters"] for d in daily_stats]
    days_with_signal   = [d for d in daily_stats if d["n_clusters"] > 0]
    all_durations_flat = [dur for d in daily_stats for dur in d["cluster_durations_min"]]
    all_sizes_flat     = [sz  for d in daily_stats for sz  in d["cluster_sizes"]]
    all_gaps_flat      = [g   for d in daily_stats for g   in d["gaps_between_min"]]

    print(f"\n{'='*72}")
    print("CLUSTER STRUCTURE SUMMARY")
    print(f"{'='*72}")
    print(f"  Total dates:          {len(date_dirs)}")
    print(f"  Dates with signal:    {len(days_with_signal)}")
    print(f"  Dates with 0 clusters: {sum(1 for c in all_cluster_counts if c == 0)}")

    if all_cluster_counts:
        counts_nonzero = [c for c in all_cluster_counts if c > 0]
        print(f"\n  Clusters per day (all {len(date_dirs)} dates):")
        print(f"    min={min(all_cluster_counts)}  "
              f"p25={float(np.percentile(all_cluster_counts, 25)):.1f}  "
              f"median={float(np.median(all_cluster_counts)):.1f}  "
              f"p75={float(np.percentile(all_cluster_counts, 75)):.1f}  "
              f"max={max(all_cluster_counts)}  "
              f"mean={float(np.mean(all_cluster_counts)):.2f}")
        if counts_nonzero:
            print(f"  Clusters per day (signal days only, n={len(counts_nonzero)}):")
            print(f"    min={min(counts_nonzero)}  "
                  f"p25={float(np.percentile(counts_nonzero, 25)):.1f}  "
                  f"median={float(np.median(counts_nonzero)):.1f}  "
                  f"p75={float(np.percentile(counts_nonzero, 75)):.1f}  "
                  f"max={max(counts_nonzero)}  "
                  f"mean={float(np.mean(counts_nonzero)):.2f}")

        dist = {}
        for c in all_cluster_counts:
            dist[c] = dist.get(c, 0) + 1
        print(f"\n  Distribution of cluster counts:")
        for k in sorted(dist):
            bar = "#" * dist[k]
            print(f"    {k} clusters: {dist[k]:3d} days  {bar}")

    if all_durations_flat:
        print(f"\n  Cluster duration (minutes):")
        print(f"    p25={float(np.percentile(all_durations_flat, 25)):.1f}  "
              f"median={float(np.median(all_durations_flat)):.1f}  "
              f"p75={float(np.percentile(all_durations_flat, 75)):.1f}  "
              f"p90={float(np.percentile(all_durations_flat, 90)):.1f}  "
              f"max={max(all_durations_flat):.1f}")

    if all_gaps_flat:
        print(f"\n  Gap between clusters (minutes):")
        print(f"    p25={float(np.percentile(all_gaps_flat, 25)):.1f}  "
              f"median={float(np.median(all_gaps_flat)):.1f}  "
              f"p75={float(np.percentile(all_gaps_flat, 75)):.1f}  "
              f"p90={float(np.percentile(all_gaps_flat, 90)):.1f}  "
              f"max={max(all_gaps_flat):.1f}")

    # ── PnL comparison: one-per-cluster strategies ────────────────────────────
    print(f"\n{'='*72}")
    print("PNL COMPARISON: one entry per cluster vs subsampled baseline")
    print(f"Baseline (300-bar subsample) from earlier: EV=+2.57t, Std=25.44t, Sharpe=0.101")
    print(f"{'='*72}")

    strat_stats: list[dict] = []
    for strat, pnls_list in all_pnls.items():
        if not pnls_list:
            continue
        pnls = np.array(pnls_list)
        s = _dist_stats(pnls, strat)
        strat_stats.append(s)

    header = (f"  {'Strategy':14} {'n':>6} {'GrossEV':>9} {'Std':>7} "
              f"{'Sharpe':>8} {'WR':>7} {'NetEV':>8} {'>+3t':>6} {'<-3t':>6}")
    print(f"\n{header}")
    print(f"  {'-'*72}")
    for s in strat_stats:
        print(f"  {s['label']:14} {s['n']:>6,} {s['gross_ev']:>+9.3f} {s['gross_std']:>7.2f} "
              f"{s['gross_sharpe']:>+8.4f} {s['gross_wr']:>7.1%} {s['net_ev']:>+8.3f} "
              f"{s['pct_gt_plus3']:>6.1%} {s['pct_lt_minus3']:>6.1%}")

    # ── Daily Sharpe projection ────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("IMPLIED DAILY SHARPE (assuming cluster entries are independent)")
    print(f"{'='*72}")
    print(f"  Formula: daily_sharpe = per_trade_sharpe * sqrt(n_clusters_per_day)")
    print(f"  Prop firm viability targets: daily Sharpe >= 0.32 (~10 trades), 0.50 (~25 trades)")
    print()

    # Use first_bar strategy as representative (least lookahead bias)
    s_ref = next((s for s in strat_stats if s["label"] == "first_bar"), None)
    if s_ref and s_ref.get("gross_sharpe"):
        per_trade_sharpe = s_ref["gross_sharpe"]
        print(f"  Per-trade Sharpe (first_bar strategy): {per_trade_sharpe:+.4f}")
        print()
        print(f"  {'Clusters/day':14} {'Daily Sharpe':>14} {'Assessment':}")
        for k in [1, 2, 3, 5, 7, 10, 15, 20]:
            ds = per_trade_sharpe * (k ** 0.5)
            if k <= 2:
                assessment = "not viable"
            elif k <= 5:
                assessment = "marginal"
            elif k <= 10:
                assessment = "possible (aggressive challenge)"
            else:
                assessment = "viable"
            # Check if k is feasible given actual cluster counts
            median_clusters = float(np.median(all_cluster_counts)) if all_cluster_counts else 0
            feasible = "  <-- ~actual" if abs(k - median_clusters) < 0.5 else ""
            print(f"  {k:14d} {ds:>14.3f} {assessment}{feasible}")

    # ── Cluster timing: where in the session do clusters occur? ───────────────
    print(f"\n{'='*72}")
    print("CLUSTER TIMING WITHIN RTH SESSION (offset from 14:30 UTC open)")
    print(f"{'='*72}")
    all_offsets = [o for d in daily_stats for o in d.get("cluster_start_offset_min", [])]
    if all_offsets:
        # Bin into 30-min buckets
        max_offset = int(max(all_offsets) / 30) * 30 + 30
        bins = list(range(0, max_offset + 1, 30))
        counts = [0] * (len(bins) - 1)
        for o in all_offsets:
            for i, b in enumerate(bins[:-1]):
                if b <= o < bins[i + 1]:
                    counts[i] += 1
                    break
        print(f"  {'Offset bucket':16} {'Count':>6}  Bar chart")
        for i, b in enumerate(bins[:-1]):
            bar = "#" * counts[i]
            print(f"  {b:3d}-{bins[i+1]:3d} min:      {counts[i]:>4}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "instrument":       "ESH6",
            "period":           "Jan+Feb 2026 RTH",
            "entry":            "rp>=0.85 AND vwap_dev>=50t (short)",
            "cluster_gap_bars": CLUSTER_GAP,
            "cluster_gap_min":  CLUSTER_GAP / 600,
            "hold_bars":        H,
            "hold_min":         H / 600,
            "tick_size":        TICK,
            "friction_ticks":   FRICTION,
        },
        "cluster_structure_summary": {
            "all_cluster_counts":    all_cluster_counts,
            "mean_clusters_per_day": round(float(np.mean(all_cluster_counts)), 3) if all_cluster_counts else 0,
            "median_clusters_per_day": round(float(np.median(all_cluster_counts)), 1) if all_cluster_counts else 0,
            "cluster_duration_p50":  round(float(np.median(all_durations_flat)), 1) if all_durations_flat else 0,
            "cluster_duration_p90":  round(float(np.percentile(all_durations_flat, 90)), 1) if all_durations_flat else 0,
            "gap_between_p50":       round(float(np.median(all_gaps_flat)), 1) if all_gaps_flat else 0,
        },
        "daily_stats":          daily_stats,
        "strategy_pnl_stats":   strat_stats,
        "baseline_reference": {
            "label":        "subsampled 300-bar (previous analysis)",
            "gross_ev":     2.568,
            "gross_std":    25.44,
            "gross_sharpe": 0.1009,
            "net_ev":       1.208,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "cluster_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
