"""
ES-CL spread analysis: does the normalized return spread mean-revert at z-score extremes?

Uses ESH6 + CLH6 100ms bar data on the joint session (14:30-19:30 UTC).
Normalizes both to log returns to handle tick-size differences ($12.50/t vs $10/t).

For each lookback window W (30, 60, 120 min):
  - spread_W[t] = log(ES_t / ES_{t-W}) - log(CL_t / CL_{t-W})
      i.e., the cumulative W-min log return differential
  - z[t] = (spread_W[t] - rolling_mean(spread_W, 4W)) / rolling_std(spread_W, 4W)
  - At |z| >= 2.0 events: record z0 and forward z at 15/30/60 min
  - Events are pooled cross-day for aggregate statistics

Questions answered:
  1. Rolling correlation between ES and CL (if low, pair is noise)
  2. Frequency of z-score extremes (if rare, too few trades)
  3. Forward reversion rate at |z| >= 2 (if not > 55%, pair doesn't trade)
  4. Spread std vs individual leg std (tighter spread = structural edge)

Usage:
    python scripts/signal_escl_spread.py

Outputs:
    artifacts/signal_research/escl_spread_analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
ES_ROOT       = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
CL_ROOT       = PROJECT_ROOT / "data/processed_clh6/instrument=CL"
OUTPUT_DIR    = PROJECT_ROOT / "artifacts/signal_research"

SESSION_START = "14:30"
SESSION_END   = "19:30"

SPREAD_WINDOWS = {
    "30min":  18_000,
    "60min":  36_000,
    "120min": 72_000,
}
Z_MULTIPLIER = 4
Z_THRESH     = 2.0

FWD_WINDOWS = {
    "15min":  9_000,
    "30min": 18_000,
    "60min": 36_000,
}

CORR_BIN = 600   # 1-min bins for correlation estimation


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_date(date_str: str) -> pd.DataFrame | None:
    es_pq = ES_ROOT / f"date={date_str}" / "features_labels.parquet"
    cl_pq = CL_ROOT / f"date={date_str}" / "features_labels.parquet"
    if not es_pq.exists() or not cl_pq.exists():
        return None

    es = pd.read_parquet(es_pq, columns=["Time", "mid"])
    cl = pd.read_parquet(cl_pq, columns=["Time", "mid"])
    es["Time"] = pd.to_datetime(es["Time"], utc=True)
    cl["Time"] = pd.to_datetime(cl["Time"], utc=True)

    df = es.merge(cl, on="Time", suffixes=("_es", "_cl"))
    if len(df) == 0:
        return None

    lo = pd.Timestamp(f"{date_str} {SESSION_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {SESSION_END}:00",   tz="UTC")
    df = df[(df["Time"] >= lo) & (df["Time"] < hi)].reset_index(drop=True)
    if len(df) < 1000:
        return None

    df["lret_es"]     = np.log(df["mid_es"] / df["mid_es"].shift(1))
    df["lret_cl"]     = np.log(df["mid_cl"] / df["mid_cl"].shift(1))
    df["spread_lret"] = df["lret_es"] - df["lret_cl"]
    return df


def _add_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    spread_ret = df["spread_lret"].fillna(0)
    for name, W in SPREAD_WINDOWS.items():
        cum   = spread_ret.rolling(W, min_periods=W // 2).sum()
        Z_W   = W * Z_MULTIPLIER
        mu    = cum.rolling(Z_W, min_periods=W).mean()
        sigma = cum.rolling(Z_W, min_periods=W).std()
        df[f"z_{name}"] = (cum - mu) / sigma.replace(0, np.nan)
    return df


# ── Per-day correlation and leg distribution ──────────────────────────────────

def _daily_stats(df: pd.DataFrame) -> dict:
    lret_es = df["lret_es"].fillna(0)
    lret_cl = df["lret_cl"].fillna(0)

    # 1-min bin correlation
    n_bins = len(df) // CORR_BIN
    if n_bins >= 5:
        es_b = np.array([lret_es.iloc[i*CORR_BIN:(i+1)*CORR_BIN].sum() for i in range(n_bins)])
        cl_b = np.array([lret_cl.iloc[i*CORR_BIN:(i+1)*CORR_BIN].sum() for i in range(n_bins)])
        corr_1min = float(np.corrcoef(es_b, cl_b)[0, 1])
    else:
        corr_1min = np.nan

    # 15-min non-overlapping log returns for spread vs leg std
    H = 9_000
    n_w = len(df) // H
    if n_w >= 2:
        es_15 = np.array([lret_es.iloc[i*H:(i+1)*H].sum() for i in range(n_w)])
        cl_15 = np.array([lret_cl.iloc[i*H:(i+1)*H].sum() for i in range(n_w)])
        sp_15 = es_15 - cl_15
        es_std, cl_std, sp_std = es_15.std(), cl_15.std(), sp_15.std()
        corr_15 = float(np.corrcoef(es_15, cl_15)[0, 1])
    else:
        es_std = cl_std = sp_std = corr_15 = np.nan

    return {
        "corr_1min":       round(corr_1min, 4) if np.isfinite(corr_1min) else None,
        "corr_15min":      round(corr_15, 4)   if np.isfinite(corr_15)   else None,
        "es_ret_std_15":   round(es_std, 6)    if np.isfinite(es_std)    else None,
        "cl_ret_std_15":   round(cl_std, 6)    if np.isfinite(cl_std)    else None,
        "spread_ret_std_15": round(sp_std, 6)  if np.isfinite(sp_std)    else None,
        "spread_vs_es_std": round(sp_std / es_std, 4) if es_std > 0 else None,
        "spread_vs_cl_std": round(sp_std / cl_std, 4) if cl_std > 0 else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    es_dates = {d.name.replace("date=", "") for d in ES_ROOT.glob("date=2026-0[12]-*")}
    cl_dates = {d.name.replace("date=", "") for d in CL_ROOT.glob("date=2026-0[12]-*")}
    dates = sorted(es_dates & cl_dates)

    print(f"ES-CL Spread Analysis — ESH6 vs CLH6")
    print(f"Joint session: {SESSION_START}-{SESSION_END} UTC")
    print(f"Overlapping dates: {len(dates)},  Z threshold: |z| >= {Z_THRESH}")

    # Accumulators
    daily_corrs_1m:  list[float] = []
    daily_corrs_15m: list[float] = []
    daily_sp_vs_es:  list[float] = []
    daily_sp_vs_cl:  list[float] = []

    # Cross-day event pools: list of {"z0": float, "fwd": {"15min": float, ...}}
    event_pool: dict[str, list[dict]] = {w: [] for w in SPREAD_WINDOWS}
    events_per_day: dict[str, list[int]] = {w: [] for w in SPREAD_WINDOWS}

    for date_str in dates:
        df = _load_date(date_str)
        if df is None:
            print(f"  {date_str}: skip")
            continue

        df = _add_z_scores(df)
        ds = _daily_stats(df)

        if ds["corr_1min"]  is not None: daily_corrs_1m.append(ds["corr_1min"])
        if ds["corr_15min"] is not None: daily_corrs_15m.append(ds["corr_15min"])
        if ds["spread_vs_es_std"] is not None: daily_sp_vs_es.append(ds["spread_vs_es_std"])
        if ds["spread_vs_cl_std"] is not None: daily_sp_vs_cl.append(ds["spread_vs_cl_std"])

        n_ev_30 = 0
        for wname, W in SPREAD_WINDOWS.items():
            z_arr = df[f"z_{wname}"].values
            n     = len(z_arr)
            SUBS  = W   # subsample spacing: one event per spread window

            extreme_idx = np.where(np.abs(z_arr) >= Z_THRESH)[0]
            kept: list[int] = []
            last = -SUBS
            for idx in extreme_idx:
                if int(idx) - last >= SUBS:
                    kept.append(int(idx))
                    last = int(idx)

            n_valid = 0
            for idx in kept:
                z0 = float(z_arr[idx])
                if not np.isfinite(z0):
                    continue
                # Require all forward windows to be within session and non-NaN
                fwd_z: dict[str, float] = {}
                ok = True
                for fw_name, FWD in FWD_WINDOWS.items():
                    fwd_idx = idx + FWD
                    if fwd_idx >= n:
                        ok = False
                        break
                    zf = float(z_arr[fwd_idx])
                    if not np.isfinite(zf):
                        ok = False
                        break
                    fwd_z[fw_name] = zf
                if ok:
                    event_pool[wname].append({"z0": z0, "fwd": fwd_z})
                    n_valid += 1

            events_per_day[wname].append(n_valid)
            if wname == "30min":
                n_ev_30 = n_valid

        c = ds.get("corr_1min")
        c_str = f"{c:.3f}" if c is not None else "n/a"
        print(f"  {date_str}: z_events(30min)={n_ev_30}  corr1min={c_str}")

    # ── Correlation summary ────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print("ROLLING CORRELATION: ES vs CL")
    print(f"{'='*72}")

    if daily_corrs_1m:
        arr = np.array(daily_corrs_1m)
        print(f"  1-min bins  (n={len(arr)} days): "
              f"min={arr.min():.3f}  p25={np.percentile(arr,25):.3f}  "
              f"median={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  "
              f"max={arr.max():.3f}")
    if daily_corrs_15m:
        arr = np.array(daily_corrs_15m)
        print(f"  15-min bins (n={len(arr)} days): "
              f"min={arr.min():.3f}  p25={np.percentile(arr,25):.3f}  "
              f"median={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  "
              f"max={arr.max():.3f}")
    print(f"\n  Interpretation:")
    if daily_corrs_1m:
        med = float(np.median(daily_corrs_1m))
        if med > 0.70:
            print(f"    Median corr={med:.3f}: HIGH — pairs reduce risk (spread is tighter)")
        elif med > 0.30:
            print(f"    Median corr={med:.3f}: MODERATE — some co-movement, partial hedge")
        else:
            print(f"    Median corr={med:.3f}: LOW — instruments move independently, spread = noise")

    # ── Spread vs leg std ──────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print("SPREAD DISTRIBUTION vs INDIVIDUAL LEGS (15-min log returns, non-overlapping)")
    print(f"{'='*72}")
    if daily_sp_vs_es and daily_sp_vs_cl:
        spe = np.array(daily_sp_vs_es)
        spc = np.array(daily_sp_vs_cl)
        print(f"  spread_std / ES_std:  median={np.median(spe):.3f}  "
              f"(< 1.0 = spread tighter than ES alone)")
        print(f"  spread_std / CL_std:  median={np.median(spc):.3f}  "
              f"(< 1.0 = spread tighter than CL alone)")
        print()
        if np.median(spe) > 1.0 and np.median(spc) > 1.0:
            print(f"  VERDICT: Spread is NOISIER than either leg.")
            print(f"  The two instruments move independently — pairing adds variance, not removes it.")
        elif np.median(spe) < 1.0 and np.median(spc) < 1.0:
            print(f"  VERDICT: Spread is TIGHTER than both legs.")
            print(f"  High co-movement — pair reduces variance per unit of directional exposure.")
        else:
            print(f"  VERDICT: Mixed — spread tighter than one leg but not the other.")

    # ── Z-score reversion ──────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print(f"Z-SCORE REVERSION (pooled cross-day, |z0| >= {Z_THRESH})")
    print(f"{'='*72}")
    print(f"Benchmark: random walk predicts ~50% reversion rate, ~0% mean forward z")

    for wname in SPREAD_WINDOWS:
        pool = event_pool[wname]
        epd  = events_per_day[wname]

        print(f"\n  -- Spread window: {wname} --")
        print(f"  Total pooled events: {len(pool)}  "
              f"(per-day: mean={np.mean(epd):.1f}, "
              f"min={min(epd)}, max={max(epd)})")

        if len(pool) < 10:
            print(f"  Insufficient events for reliable statistics.")
            continue

        z0_arr  = np.array([e["z0"] for e in pool])
        pos_m   = z0_arr > 0   # ES outperformed CL

        print(f"  z0 dist: mean={np.mean(np.abs(z0_arr)):.2f}  "
              f"positive (ES>CL): {pos_m.sum()} ({pos_m.mean():.0%})  "
              f"negative (CL>ES): {(~pos_m).sum()} ({(~pos_m).mean():.0%})")

        print(f"\n  {'Forward':8} {'P(|zf|<|z0|)':>14}  {'P(cross 0)':>12}  "
              f"{'P(still ext.)':>14}  {'mean zf|pos':>12}  {'mean zf|neg':>12}  {'mean |zf|':>10}")

        fwd_rows: list[dict] = []
        for fw_name in FWD_WINDOWS:
            zf_arr = np.array([e["fwd"][fw_name] for e in pool])

            reverted     = np.abs(zf_arr) < np.abs(z0_arr)
            crossed      = (z0_arr * zf_arr) < 0   # sign flip
            still_ext    = np.abs(zf_arr) >= Z_THRESH
            mean_zf_pos  = float(zf_arr[pos_m].mean())   if pos_m.any()  else np.nan
            mean_zf_neg  = float(zf_arr[~pos_m].mean())  if (~pos_m).any() else np.nan

            p_rev  = float(reverted.mean())
            p_cross = float(crossed.mean())
            p_still = float(still_ext.mean())
            m_zf    = float(np.abs(zf_arr).mean())

            print(f"  {fw_name:8} {p_rev:>14.1%}  {p_cross:>12.1%}  "
                  f"{p_still:>14.1%}  {mean_zf_pos:>12.3f}  {mean_zf_neg:>12.3f}  {m_zf:>10.3f}")

            fwd_rows.append({
                "forward":              fw_name,
                "n":                    len(zf_arr),
                "pct_reverted":         round(p_rev,   4),
                "pct_crossed_zero":     round(p_cross, 4),
                "pct_still_extreme":    round(p_still, 4),
                "mean_abs_fwd_z":       round(m_zf,    4),
                "mean_fwd_z_pos_start": round(mean_zf_pos, 4) if np.isfinite(mean_zf_pos) else None,
                "mean_fwd_z_neg_start": round(mean_zf_neg, 4) if np.isfinite(mean_zf_neg) else None,
            })

        # Verdict on this window
        fwd_30 = next((r for r in fwd_rows if r["forward"] == "30min"), None)
        if fwd_30:
            p_rev30 = fwd_30["pct_reverted"]
            p_crs30 = fwd_30["pct_crossed_zero"]
            if p_rev30 > 0.60 and p_crs30 > 0.30:
                verdict = "STRONG mean reversion — worth pursuing"
            elif p_rev30 > 0.52:
                verdict = "WEAK mean reversion — marginal above random walk"
            else:
                verdict = "NO mean reversion — random walk at this window"
            print(f"  Verdict ({wname}): {verdict}")

    # ── Directionality test ────────────────────────────────────────────────────
    # When ES outperforms CL (z>0), do shorts on ES / longs on CL have positive EV?
    print(f"\n{'='*72}")
    print("DIRECTIONALITY: Does z>0 (ES out) predict ES underperformance going forward?")
    print(f"{'='*72}")
    print(f"(Pair trade: short ES / long CL at z>+2, expecting spread to narrow)")

    for wname in SPREAD_WINDOWS:
        pool = event_pool[wname]
        if len(pool) < 10:
            continue
        z0_arr = np.array([e["z0"] for e in pool])
        pos_ev = z0_arr > 0
        if not pos_ev.any():
            continue

        print(f"\n  {wname}: n={pos_ev.sum()} positive events (z>+2)")
        for fw_name in FWD_WINDOWS:
            zf_arr = np.array([e["fwd"][fw_name] for e in pool])
            # For z>0 events, mean reversion = forward z moves NEGATIVE
            zf_pos = zf_arr[pos_ev]
            zf_neg = zf_arr[~pos_ev]
            # Return: if z0>0, short spread = negative return if z stays pos, positive if z goes neg
            # Simplified: spread_ret = -(zf - z0) in z-score space
            spread_returns = -(zf_arr - z0_arr)    # positive = spread narrowed
            sr_pos = spread_returns[pos_ev]        # trades entered at z>+2
            sr_neg = spread_returns[~pos_ev]       # trades entered at z<-2
            combined = np.concatenate([sr_pos, -sr_neg])  # long spread return for both signs
            print(f"    {fw_name}: mean_fwd_z|pos={zf_pos.mean():.3f}  "
                  f"P(zf<z0)|pos={float((zf_pos < z0_arr[pos_ev]).mean()):.1%}  "
                  f"mean_spread_return={combined.mean():.3f}z  "
                  f"WR={float((combined > 0).mean()):.1%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "instruments":     ["ESH6", "CLH6"],
            "period":          f"Jan-Feb 2026 ({len(dates)} overlapping dates)",
            "session_utc":     f"{SESSION_START}-{SESSION_END}",
            "z_threshold":     Z_THRESH,
            "spread_windows":  list(SPREAD_WINDOWS.keys()),
            "fwd_windows":     list(FWD_WINDOWS.keys()),
        },
        "correlation": {
            "corr_1min_median":  round(float(np.median(daily_corrs_1m)), 4)  if daily_corrs_1m  else None,
            "corr_15min_median": round(float(np.median(daily_corrs_15m)), 4) if daily_corrs_15m else None,
        },
        "spread_vs_legs": {
            "spread_std_vs_es_median": round(float(np.median(daily_sp_vs_es)), 4) if daily_sp_vs_es else None,
            "spread_std_vs_cl_median": round(float(np.median(daily_sp_vs_cl)), 4) if daily_sp_vs_cl else None,
        },
        "z_reversion": {
            wname: {
                "n_events": len(event_pool[wname]),
                "forward":  [
                    {
                        "window":           fw_name,
                        "n":                len(event_pool[wname]),
                        "pct_reverted":     round(float(np.mean(
                            np.abs(np.array([e["fwd"][fw_name] for e in event_pool[wname]])) <
                            np.abs(np.array([e["z0"] for e in event_pool[wname]])))), 4)
                            if event_pool[wname] else None,
                        "pct_crossed_zero": round(float(np.mean(
                            np.array([e["z0"] for e in event_pool[wname]]) *
                            np.array([e["fwd"][fw_name] for e in event_pool[wname]]) < 0)), 4)
                            if event_pool[wname] else None,
                    }
                    for fw_name in FWD_WINDOWS
                ],
            }
            for wname in SPREAD_WINDOWS
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "escl_spread_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
