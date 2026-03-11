"""
MFE/MAE landscape analysis for CLH6 Jan-Feb 2026.

Session: 09:00-14:30 ET = 14:00-19:30 UTC (NYMEX pit equivalent)
Horizons: 5 / 15 / 30 minutes
EIA Wednesdays flagged and split separately.

Outputs:
    artifacts/mfe_mae_analysis/clh6_landscape.json
    (prints summary + direct ES comparison to stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_clh6/instrument=CL"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/mfe_mae_analysis"

# CL tick size: $0.01 / barrel
TICK = 0.01

# Session: 09:00-14:30 ET = 14:00-19:30 UTC
RTH_START_UTC = "14:00"
RTH_END_UTC   = "19:30"

# Horizons in bars (100ms bars)
HORIZONS = {
    "5min":  3_000,
    "15min": 9_000,
    "30min": 18_000,
}

# TP/SL candidates in CL ticks.
# CL 15-min range is typically 20-80 ticks; daily range 70-200 ticks.
# Include small values for ES comparison, larger values for CL-native structures.
TP_CANDIDATES = [6, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100]
SL_CANDIDATES = [3, 4, 5, 6, 8, 10, 15, 20, 25, 33]

# EIA report: every Wednesday at 10:30 ET = 15:30 UTC
EIA_WEEKDAY = 2  # Wednesday (0=Monday)


def _fwd_max(series: pd.Series, H: int) -> np.ndarray:
    rev = series.values[::-1]
    return pd.Series(rev).rolling(H, min_periods=1).max().values[::-1]


def _fwd_min(series: pd.Series, H: int) -> np.ndarray:
    rev = series.values[::-1]
    return pd.Series(rev).rolling(H, min_periods=1).min().values[::-1]


def load_date(date_dir: Path) -> pd.DataFrame | None:
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None
    df = pd.read_parquet(pq, columns=["Time", "mid"])
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    return df


def compute_horizons(df: pd.DataFrame) -> pd.DataFrame:
    mid = df["mid"]
    for name, H in HORIZONS.items():
        fmax = _fwd_max(mid, H)
        fmin = _fwd_min(mid, H)
        df[f"mfe_long_{name}"]  = (fmax - mid.values) / TICK
        df[f"mae_long_{name}"]  = (mid.values - fmin) / TICK
        df[f"mfe_short_{name}"] = (mid.values - fmin) / TICK
        df[f"mae_short_{name}"] = (fmax - mid.values) / TICK
    return df


def rth_filter(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    lo = pd.Timestamp(f"{date_str} {RTH_START_UTC}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END_UTC}:00",   tz="UTC")
    return df[(df["Time"] >= lo) & (df["Time"] < hi)].copy()


def percentile_row(arr: np.ndarray) -> dict:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {}
    return {
        "n":    len(arr),
        "p10":  float(np.percentile(arr, 10)),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "p90":  float(np.percentile(arr, 90)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
    }


def viability_table(mfe: np.ndarray, mae: np.ndarray,
                    tp_vals: list[int], sl_vals: list[int]) -> dict:
    result = {}
    for tp in tp_vals:
        for sl in sl_vals:
            if sl >= tp:
                continue
            p_tp  = float((mfe >= tp).mean())
            p_sl  = float((mae >= sl).mean())
            p_win = float(((mfe >= tp) & (mae < sl)).mean())
            be_wr = sl / (tp + sl)
            result[f"TP{tp}_SL{sl}"] = {
                "tp": tp, "sl": sl,
                "ratio": round(tp / sl, 2),
                "breakeven_wr": round(be_wr, 3),
                "p_tp_hit":   round(p_tp,  3),
                "p_sl_hit":   round(p_sl,  3),
                "p_path_win": round(p_win, 3),
                "path_purity": round(p_win / p_tp, 3) if p_tp > 0 else 0,
            }
    return result


def print_landscape(name: str, h: dict, key_combos: list[tuple[int, int]],
                    instrument: str) -> None:
    print(f"\n-- {name} horizon ({instrument}) --")
    print(f"{'':14} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6}  (ticks)")
    for side in ("long", "short"):
        for metric in ("mfe", "mae"):
            key = f"{metric}_{side}"
            r = h[key]
            if not r:
                continue
            print(f"  {key:14} {r['p25']:6.1f} {r['p50']:6.1f} {r['p75']:6.1f} "
                  f"{r['p90']:6.1f} {r['p95']:6.1f}")

    print(f"\n  Viability (long): BE=breakeven WR, pw=path_win, pur=path_purity")
    print(f"  {'TP/SL':12} {'BE':>7} {'P(TP)':>7} {'P(SL)':>7} {'pw':>8} {'purity':>7}")
    for tp, sl in key_combos:
        k = f"TP{tp}_SL{sl}"
        if k in h["viability_long"]:
            v = h["viability_long"][k]
            be_flag = " <--" if v["p_path_win"] >= v["breakeven_wr"] else ""
            print(f"  {tp}t/{sl}t ({v['ratio']:.1f}:1)  "
                  f"{v['breakeven_wr']:>7.1%}  {v['p_tp_hit']:>7.1%}  "
                  f"{v['p_sl_hit']:>7.1%}  {v['p_path_win']:>8.1%}  "
                  f"{v['path_purity']:>7.3f}{be_flag}")


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    if not date_dirs:
        print(f"No CLH6 parquets found under {DATA_ROOT}")
        return
    print(f"Loading {len(date_dirs)} CLH6 dates (Jan+Feb 2026)...")

    all_rth:     list[pd.DataFrame] = []
    eia_dates:   list[str] = []
    non_eia:     list[pd.DataFrame] = []
    eia_frames:  list[pd.DataFrame] = []

    for dd in date_dirs:
        date_str = dd.name.replace("date=", "")
        df = load_date(dd)
        if df is None:
            print(f"  {date_str}: missing")
            continue
        df = compute_horizons(df)
        rth = rth_filter(df, date_str)
        if len(rth) == 0:
            print(f"  {date_str}: 0 RTH bars, skipping")
            continue

        dt = pd.to_datetime(date_str)
        is_eia = (dt.weekday() == EIA_WEEKDAY)
        rth["is_eia"] = is_eia
        rth["date"]   = date_str

        all_rth.append(rth)
        if is_eia:
            eia_dates.append(date_str)
            eia_frames.append(rth)
        else:
            non_eia.append(rth)

        eia_tag = " [EIA Wed]" if is_eia else ""
        print(f"  {date_str}: {len(rth):,} RTH bars{eia_tag}")

    if not all_rth:
        print("No data loaded.")
        return

    combined    = pd.concat(all_rth,    ignore_index=True)
    eia_df      = pd.concat(eia_frames, ignore_index=True) if eia_frames  else pd.DataFrame()
    non_eia_df  = pd.concat(non_eia,    ignore_index=True) if non_eia     else pd.DataFrame()

    n_dates     = len(all_rth)
    n_eia       = len(eia_dates)
    n_bars      = len(combined)
    bars_per_day = round(n_bars / n_dates)

    print(f"\nTotal RTH bars: {n_bars:,}  over {n_dates} dates  ({bars_per_day:,}/day)")
    print(f"EIA Wednesdays: {n_eia}  ({', '.join(eia_dates)})")
    print(f"Non-EIA days:   {n_dates - n_eia}")

    # 3:1 family at CL-appropriate scales + ES comparison anchors
    key_combos_3to1 = [
        (12, 4), (15, 5), (18, 6), (20, 7), (30, 10), (40, 13), (50, 17), (60, 20),
    ]
    key_combos_2to1 = [
        (12, 6), (20, 10), (30, 15), (40, 20), (50, 25),
    ]
    all_key_combos = key_combos_3to1 + key_combos_2to1

    def _build_results(df: pd.DataFrame, label: str) -> dict:
        r: dict = {"label": label, "n": len(df), "horizons": {}}
        for name in HORIZONS:
            mfe_l = df[f"mfe_long_{name}"].values
            mae_l = df[f"mae_long_{name}"].values
            mfe_s = df[f"mfe_short_{name}"].values
            mae_s = df[f"mae_short_{name}"].values
            r["horizons"][name] = {
                "mfe_long":       percentile_row(mfe_l),
                "mae_long":       percentile_row(mae_l),
                "mfe_short":      percentile_row(mfe_s),
                "mae_short":      percentile_row(mae_s),
                "viability_long":  viability_table(mfe_l, mae_l, TP_CANDIDATES, SL_CANDIDATES),
                "viability_short": viability_table(mfe_s, mae_s, TP_CANDIDATES, SL_CANDIDATES),
            }
        return r

    full_results    = _build_results(combined,   f"CLH6 all {n_dates} days")
    eia_results     = _build_results(eia_df,      f"CLH6 EIA Wednesdays ({n_eia} days)") if not eia_df.empty else {}
    non_eia_results = _build_results(non_eia_df,  f"CLH6 non-EIA ({n_dates - n_eia} days)") if not non_eia_df.empty else {}

    # ── Print landscape ───────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("CLH6 MFE/MAE LANDSCAPE -- 09:00-14:30 ET RTH, Jan-Feb 2026")
    print("="*72)

    for name in HORIZONS:
        h = full_results["horizons"][name]
        print_landscape(name, h, all_key_combos, "CLH6 all days")

    # EIA vs non-EIA comparison at 15 min
    if eia_results and non_eia_results:
        print(f"\n{'='*72}")
        print("EIA Wednesday vs non-EIA comparison (15min horizon, long path_win):")
        print(f"  {'TP/SL':12} {'BE':>7}  {'all_pw':>8}  {'EIA_pw':>8}  {'non-EIA_pw':>11}  {'EIA_lift':>9}")
        for tp, sl in [(18, 6), (20, 7), (30, 10), (40, 13), (50, 17), (60, 20)]:
            k = f"TP{tp}_SL{sl}"
            h_all = full_results["horizons"]["15min"]["viability_long"].get(k, {})
            h_eia = eia_results["horizons"]["15min"]["viability_long"].get(k, {})
            h_non = non_eia_results["horizons"]["15min"]["viability_long"].get(k, {})
            if not h_all:
                continue
            eia_pw  = h_eia.get("p_path_win", 0)
            non_pw  = h_non.get("p_path_win", 0)
            all_pw  = h_all["p_path_win"]
            eia_lift = eia_pw - non_pw
            print(f"  {tp}t/{sl}t ({h_all['ratio']:.1f}:1)  "
                  f"{h_all['breakeven_wr']:>7.1%}  {all_pw:>8.1%}  "
                  f"{eia_pw:>8.1%}  {non_pw:>11.1%}  {eia_lift:>+9.1%}")

    # ── ES vs CL direct comparison ────────────────────────────────────────────
    es_landscape_path = PROJECT_ROOT / "artifacts/mfe_mae_analysis/horizon_landscape.json"
    es_landscape_feb  = PROJECT_ROOT / "artifacts/mfe_mae_analysis/horizon_landscape_feb.json"
    if es_landscape_path.exists():
        with open(es_landscape_path) as f:
            es_jan = json.load(f)
        es_feb = None
        if es_landscape_feb.exists():
            with open(es_landscape_feb) as f:
                es_feb = json.load(f)

        print(f"\n{'='*72}")
        print("DIRECT COMPARISON: CL vs ES at 15-min horizon (path_win, 3:1 structures)")
        print(f"  Session: CL=09:00-14:30ET  ES=09:30-16:00ET")
        print(f"  {'Structure':16} {'BE':>7}  {'CL_pw':>8}  {'ES_Jan_pw':>10}  {'ES_Feb_pw':>10}  {'CL_vs_ES':>9}")

        cl_15 = full_results["horizons"]["15min"]["viability_long"]
        es_15 = es_jan.get("horizons", {}).get("15min", {}).get("viability_long", {})
        es_15f = (es_feb or {}).get("horizons", {}).get("15min", {}).get("viability_long", {}) if es_feb else {}

        for tp, sl in [(12, 4), (15, 5), (18, 6), (20, 7), (30, 10), (40, 13)]:
            k = f"TP{tp}_SL{sl}"
            cl_v  = cl_15.get(k, {})
            es_v  = es_15.get(k, {})
            es_fv = es_15f.get(k, {})
            if not cl_v:
                continue
            cl_pw  = cl_v["p_path_win"]
            es_pw  = es_v.get("p_path_win", float("nan"))
            es_fpw = es_fv.get("p_path_win", float("nan"))
            diff   = cl_pw - es_pw if np.isfinite(es_pw) else float("nan")
            es_str  = f"{es_pw:>10.1%}" if np.isfinite(es_pw)  else f"{'n/a':>10}"
            esf_str = f"{es_fpw:>10.1%}" if np.isfinite(es_fpw) else f"{'n/a':>10}"
            diff_str = f"{diff:>+9.1%}" if np.isfinite(diff) else f"{'n/a':>9}"
            print(f"  {tp}t/{sl}t  ({cl_v['ratio']:.1f}:1)  "
                  f"{cl_v['breakeven_wr']:>7.1%}  {cl_pw:>8.1%}  "
                  f"{es_str}  {esf_str}  {diff_str}")

        # MFE p50 comparison
        print(f"\n  MFE p50 at 15min (ticks): CL vs ES")
        cl_mfe50 = full_results["horizons"]["15min"]["mfe_long"].get("p50", "n/a")
        es_mfe50_jan = es_jan.get("horizons", {}).get("15min", {}).get("mfe_long", {}).get("p50", "n/a")
        es_mfe50_feb = (es_feb or {}).get("horizons", {}).get("15min", {}).get("mfe_long", {}).get("p50", "n/a")
        print(f"    CL: {cl_mfe50:.1f}t   ES Jan: {es_mfe50_jan:.1f}t   ES Feb: {es_mfe50_feb:.1f}t")
        print(f"    (Note: CL and ES ticks are different dollar values: CL=$0.01, ES=$0.25)")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "instrument":   "CLH6",
            "period":       "Jan-Feb 2026",
            "dates":        n_dates,
            "total_bars":   n_bars,
            "bars_per_day": bars_per_day,
            "bar_size_ms":  100,
            "session_utc":  f"{RTH_START_UTC}-{RTH_END_UTC}",
            "session_et":   "09:00-14:30",
            "tick_size":    TICK,
            "eia_dates":    eia_dates,
        },
        "all_days":    full_results,
        "eia_days":    eia_results,
        "non_eia_days": non_eia_results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "clh6_landscape.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
