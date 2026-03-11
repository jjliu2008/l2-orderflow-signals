"""
ES-NQ spread analysis: correlation, spread variance, and z-score mean reversion.

Loads ESH6 mid prices from MBP-10 parquets and NQH6 last-price bars from
the TRADES-derived parquets built by build_nqh6_price_bars.py.

Joint session: 14:30-21:00 UTC (full ES RTH — NQ trades same hours).

Three checks:
  1. Correlation: expect 0.90+ at 1-min bins (vs ES-CL median 0.10)
  2. Spread std / ES std: expect < 0.50 (vs ES-CL ratio 2.40)
  3. Z-score at ±2: cross-zero rate within 30 min > 50% (vs ES-CL ~50%)

If all three hold, the spread is structurally tighter and z-score extremes
are genuinely mean-reverting — a materially better starting point than
either individual leg for path_win analysis.

Usage:
    python scripts/signal_esnq_spread.py

Outputs:
    artifacts/signal_research/esnq_spread_analysis.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
ES_ROOT       = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
NQ_ROOT       = PROJECT_ROOT / "data/processed_nqh6/instrument=NQ"
OUTPUT_DIR    = PROJECT_ROOT / "artifacts/signal_research"

SESSION_START = "14:30"
SESSION_END   = "21:00"    # full ES RTH; NQ trades same window

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

CORR_BIN = 600   # 1-min bins for correlation


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_date(date_str: str) -> pd.DataFrame | None:
    es_pq = ES_ROOT / f"date={date_str}" / "features_labels.parquet"
    nq_pq = NQ_ROOT / f"date={date_str}" / "prices.parquet"

    if not es_pq.exists() or not nq_pq.exists():
        return None

    es = pd.read_parquet(es_pq, columns=["Time", "mid"])
    nq = pd.read_parquet(nq_pq, columns=["Time", "mid"])

    es["Time"] = pd.to_datetime(es["Time"], utc=True)
    nq["Time"] = pd.to_datetime(nq["Time"], utc=True)

    df = es.merge(nq, on="Time", suffixes=("_es", "_nq"))
    if len(df) == 0:
        return None

    lo = pd.Timestamp(f"{date_str} {SESSION_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {SESSION_END}:00",   tz="UTC")
    df = df[(df["Time"] >= lo) & (df["Time"] < hi)].reset_index(drop=True)
    if len(df) < 1000:
        return None

    df["lret_es"] = np.log(df["mid_es"] / df["mid_es"].shift(1))
    df["lret_nq"] = np.log(df["mid_nq"] / df["mid_nq"].shift(1))
    df["spread_lret"] = df["lret_es"] - df["lret_nq"]
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


# ── Per-day statistics ────────────────────────────────────────────────────────

def _daily_stats(df: pd.DataFrame) -> dict:
    lret_es = df["lret_es"].fillna(0)
    lret_nq = df["lret_nq"].fillna(0)

    # 1-min bin correlation
    n_bins = len(df) // CORR_BIN
    if n_bins >= 5:
        es_b = np.array([lret_es.iloc[i*CORR_BIN:(i+1)*CORR_BIN].sum() for i in range(n_bins)])
        nq_b = np.array([lret_nq.iloc[i*CORR_BIN:(i+1)*CORR_BIN].sum() for i in range(n_bins)])
        corr_1min = float(np.corrcoef(es_b, nq_b)[0, 1])
    else:
        corr_1min = np.nan

    # 15-min non-overlapping returns for spread vs leg std
    H = 9_000
    n_w = len(df) // H
    if n_w >= 2:
        es_15 = np.array([lret_es.iloc[i*H:(i+1)*H].sum() for i in range(n_w)])
        nq_15 = np.array([lret_nq.iloc[i*H:(i+1)*H].sum() for i in range(n_w)])
        sp_15 = es_15 - nq_15
        es_std, nq_std, sp_std = es_15.std(), nq_15.std(), sp_15.std()
        corr_15 = float(np.corrcoef(es_15, nq_15)[0, 1])
    else:
        es_std = nq_std = sp_std = corr_15 = np.nan

    return {
        "corr_1min":          round(corr_1min, 4) if np.isfinite(corr_1min) else None,
        "corr_15min":         round(corr_15,   4) if np.isfinite(corr_15)   else None,
        "es_ret_std_15":      round(es_std,  6)   if np.isfinite(es_std)    else None,
        "nq_ret_std_15":      round(nq_std,  6)   if np.isfinite(nq_std)    else None,
        "spread_ret_std_15":  round(sp_std,  6)   if np.isfinite(sp_std)    else None,
        "spread_vs_es_std":   round(sp_std / es_std, 4) if es_std > 0 else None,
        "spread_vs_nq_std":   round(sp_std / nq_std, 4) if nq_std > 0 else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    es_dates = {d.name.replace("date=", "") for d in ES_ROOT.glob("date=2026-0[12]-*")}
    nq_dates = {d.name.replace("date=", "") for d in NQ_ROOT.glob("date=2026-0[12]-*")}
    dates = sorted(es_dates & nq_dates)

    print(f"ES-NQ Spread Analysis — ESH6 vs NQH6")
    print(f"Joint session: {SESSION_START}-{SESSION_END} UTC (ES RTH)")
    print(f"Overlapping dates: {len(dates)},  Z threshold: |z| >= {Z_THRESH}")
    print(f"NQ price source: TRADES last-price bars (100ms, forward-filled)")

    daily_corrs_1m:  list[float] = []
    daily_corrs_15m: list[float] = []
    daily_sp_vs_es:  list[float] = []
    daily_sp_vs_nq:  list[float] = []

    event_pool:     dict[str, list[dict]] = {w: [] for w in SPREAD_WINDOWS}
    events_per_day: dict[str, list[int]]  = {w: [] for w in SPREAD_WINDOWS}
    all_daily:      list[dict]            = []

    for date_str in dates:
        df = _load_date(date_str)
        if df is None:
            print(f"  {date_str}: skip (missing data)")
            continue

        df = _add_z_scores(df)
        ds = _daily_stats(df)

        if ds["corr_1min"]        is not None: daily_corrs_1m.append(ds["corr_1min"])
        if ds["corr_15min"]       is not None: daily_corrs_15m.append(ds["corr_15min"])
        if ds["spread_vs_es_std"] is not None: daily_sp_vs_es.append(ds["spread_vs_es_std"])
        if ds["spread_vs_nq_std"] is not None: daily_sp_vs_nq.append(ds["spread_vs_nq_std"])

        n_ev_30 = 0
        for wname, W in SPREAD_WINDOWS.items():
            z_arr = df[f"z_{wname}"].values
            n     = len(z_arr)
            SUBS  = W

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
                fwd_z: dict[str, float] = {}
                ok = True
                for fw_name, FWD in FWD_WINDOWS.items():
                    fwd_idx = idx + FWD
                    if fwd_idx >= n:
                        ok = False; break
                    zf = float(z_arr[fwd_idx])
                    if not np.isfinite(zf):
                        ok = False; break
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
        all_daily.append({"date": date_str, **ds})

    # ── Correlation ────────────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print("ROLLING CORRELATION: ES vs NQ")
    print(f"{'='*72}")
    print(f"  Benchmark: ES-CL median was 0.104  (ES-NQ target: > 0.90)")

    if daily_corrs_1m:
        arr = np.array(daily_corrs_1m)
        print(f"\n  1-min bins  (n={len(arr)} days):")
        print(f"    min={arr.min():.3f}  p25={np.percentile(arr,25):.3f}  "
              f"median={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  "
              f"max={arr.max():.3f}  mean={arr.mean():.3f}")
    if daily_corrs_15m:
        arr = np.array(daily_corrs_15m)
        print(f"  15-min bins (n={len(arr)} days):")
        print(f"    min={arr.min():.3f}  p25={np.percentile(arr,25):.3f}  "
              f"median={np.median(arr):.3f}  p75={np.percentile(arr,75):.3f}  "
              f"max={arr.max():.3f}  mean={arr.mean():.3f}")

    if daily_corrs_1m:
        med = float(np.median(daily_corrs_1m))
        if med > 0.85:
            verdict = f"CONFIRMED ({med:.3f}) — co-integrated pair, spread math works"
        elif med > 0.60:
            verdict = f"PARTIAL ({med:.3f}) — moderate co-movement, spread is somewhat tighter"
        else:
            verdict = f"FAILED ({med:.3f}) — instruments too independent, same problem as ES-CL"
        print(f"\n  Correlation check: {verdict}")

    # ── Spread vs legs ─────────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print("SPREAD DISTRIBUTION vs INDIVIDUAL LEGS (15-min log returns)")
    print(f"{'='*72}")
    print(f"  Benchmark: ES-CL spread_std/ES_std was 2.397  (target: < 0.50)")

    if daily_sp_vs_es and daily_sp_vs_nq:
        spe = np.array(daily_sp_vs_es)
        spn = np.array(daily_sp_vs_nq)
        print(f"\n  spread_std / ES_std:  median={np.median(spe):.3f}  "
              f"p25={np.percentile(spe,25):.3f}  p75={np.percentile(spe,75):.3f}")
        print(f"  spread_std / NQ_std:  median={np.median(spn):.3f}  "
              f"p25={np.percentile(spn,25):.3f}  p75={np.percentile(spn,75):.3f}")

        med_spe = float(np.median(spe))
        if med_spe < 0.50:
            verdict = f"CONFIRMED ({med_spe:.3f}) — spread is tighter than either leg"
        elif med_spe < 1.0:
            verdict = f"PARTIAL ({med_spe:.3f}) — spread tighter than ES but not by enough for clean pair"
        else:
            verdict = f"FAILED ({med_spe:.3f}) — spread is noisier than ES alone"
        print(f"\n  Spread tightness check: {verdict}")

    # ── Z-score reversion ──────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print(f"Z-SCORE REVERSION (pooled cross-day, |z0| >= {Z_THRESH})")
    print(f"{'='*72}")
    print(f"  Benchmark: ES-CL P(cross zero in 30min) = 51%  (target: > 55%)")

    all_verdicts: list[str] = []
    for wname in SPREAD_WINDOWS:
        pool = event_pool[wname]
        epd  = events_per_day[wname]

        print(f"\n  -- Spread window: {wname} --")
        print(f"  Total pooled events: {len(pool)}  "
              f"(per-day: mean={np.mean(epd):.1f}, min={min(epd)}, max={max(epd)})")

        if len(pool) < 10:
            print(f"  Insufficient events.")
            continue

        z0_arr = np.array([e["z0"] for e in pool])
        pos_m  = z0_arr > 0

        print(f"  z0: mean={np.mean(np.abs(z0_arr)):.2f}  "
              f"positive={pos_m.sum()} ({pos_m.mean():.0%})  "
              f"negative={(~pos_m).sum()} ({(~pos_m).mean():.0%})")
        print(f"\n  {'Forward':8} {'P(|zf|<|z0|)':>14}  {'P(cross 0)':>12}  "
              f"{'P(still ext)':>13}  {'mean|zf|':>10}")

        fwd_30_cross = None
        for fw_name, FWD in FWD_WINDOWS.items():
            zf_arr    = np.array([e["fwd"][fw_name] for e in pool])
            reverted  = np.abs(zf_arr) < np.abs(z0_arr)
            crossed   = (z0_arr * zf_arr) < 0
            still_ext = np.abs(zf_arr) >= Z_THRESH
            p_rev   = float(reverted.mean())
            p_cross = float(crossed.mean())
            p_still = float(still_ext.mean())
            m_zf    = float(np.abs(zf_arr).mean())
            print(f"  {fw_name:8} {p_rev:>14.1%}  {p_cross:>12.1%}  "
                  f"{p_still:>13.1%}  {m_zf:>10.3f}")
            if fw_name == "30min":
                fwd_30_cross = p_cross

        if fwd_30_cross is not None:
            if fwd_30_cross > 0.60:
                v = f"STRONG reversion ({fwd_30_cross:.1%} cross zero in 30min)"
            elif fwd_30_cross > 0.52:
                v = f"WEAK reversion ({fwd_30_cross:.1%}) — above random walk but marginal"
            else:
                v = f"NO reversion ({fwd_30_cross:.1%}) — consistent with random walk"
            print(f"  Verdict ({wname}): {v}")
            all_verdicts.append(v)

    # ── Overall assessment ─────────────────────────────────────────────────────

    print(f"\n{'='*72}")
    print("OVERALL ASSESSMENT vs ES-CL BASELINE")
    print(f"{'='*72}")
    checks = {
        "corr_1min_median":    round(float(np.median(daily_corrs_1m)), 4)  if daily_corrs_1m  else None,
        "spread_vs_es_median": round(float(np.median(daily_sp_vs_es)), 4) if daily_sp_vs_es  else None,
    }
    escl_baseline = {"corr": 0.104, "spread_vs_es": 2.397, "cross_zero_30": 0.51}
    print(f"  {'Metric':30} {'ES-CL':>10} {'ES-NQ':>10} {'Target':>10}")
    print(f"  {'Correlation (1-min)':30} {escl_baseline['corr']:>10.3f} "
          f"{checks.get('corr_1min_median', 0) or 0:>10.3f} {'> 0.90':>10}")
    print(f"  {'Spread std / ES std':30} {escl_baseline['spread_vs_es']:>10.3f} "
          f"{checks.get('spread_vs_es_median', 0) or 0:>10.3f} {'< 0.50':>10}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "instruments":       ["ESH6", "NQH6"],
            "nq_price_source":   "TRADES last-price 100ms bars (forward-filled)",
            "period":            f"Jan-Feb 2026 ({len(dates)} overlapping dates)",
            "session_utc":       f"{SESSION_START}-{SESSION_END}",
            "z_threshold":       Z_THRESH,
            "escl_baseline": {
                "corr_1min_median":    0.104,
                "spread_vs_es_median": 2.397,
                "cross_zero_30min":    0.51,
            },
        },
        "correlation": {
            "corr_1min_all":     daily_corrs_1m,
            "corr_1min_median":  round(float(np.median(daily_corrs_1m)), 4)  if daily_corrs_1m  else None,
            "corr_15min_median": round(float(np.median(daily_corrs_15m)), 4) if daily_corrs_15m else None,
        },
        "spread_vs_legs": {
            "spread_vs_es_all":    daily_sp_vs_es,
            "spread_vs_es_median": round(float(np.median(daily_sp_vs_es)), 4) if daily_sp_vs_es else None,
            "spread_vs_nq_median": round(float(np.median(daily_sp_vs_nq)), 4) if daily_sp_vs_nq else None,
        },
        "z_reversion": {
            wname: {
                "n_events":   len(event_pool[wname]),
                "cross_zero_30min": round(float(np.mean(
                    (np.array([e["z0"] for e in event_pool[wname]]) *
                     np.array([e["fwd"]["30min"] for e in event_pool[wname]])) < 0
                )), 4) if len(event_pool[wname]) >= 5 else None,
            }
            for wname in SPREAD_WINDOWS
        },
        "daily": all_daily,
    }

    def _to_python(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "esnq_spread_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=_to_python)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
