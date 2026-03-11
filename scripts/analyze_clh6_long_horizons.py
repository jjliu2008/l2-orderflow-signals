"""
MFE/MAE landscape for CLH6 at 1, 2, and 4 hour horizons.

Same session and data as the 15-min analysis (14:00-19:30 UTC).
Forward MFE/MAE computed on full 24h data so bars near session end
capture post-RTH price action — appropriate for longer holds.

Outputs:
    artifacts/mfe_mae_analysis/clh6_long_horizons.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_clh6/instrument=CL"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/mfe_mae_analysis"

TICK = 0.01

RTH_START_UTC = "14:00"
RTH_END_UTC   = "19:30"

HORIZONS = {
    "60min":  36_000,
    "120min": 72_000,
    "240min": 144_000,
}

# TP/SL grid scaled for 1-4 hour CL moves (typical range 30-200 ticks)
# Include 3:1 and 2:1 families across a wide range
TP_CANDIDATES = [15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]
SL_CANDIDATES = [5, 7, 10, 13, 17, 20, 25, 33, 40, 50, 67]

EIA_WEEKDAY = 2  # Wednesday


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
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "p90":  float(np.percentile(arr, 90)),
        "p95":  float(np.percentile(arr, 95)),
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
                "p_tp_hit":    round(p_tp,  4),
                "p_sl_hit":    round(p_sl,  4),
                "p_path_win":  round(p_win, 4),
                "path_purity": round(p_win / p_tp, 3) if p_tp > 0 else 0,
            }
    return result


def print_horizon(name: str, h: dict, key_combos: list[tuple[int, int]]) -> None:
    print(f"\n-- {name} horizon --")
    print(f"{'':14} {'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6}  (ticks)")
    for metric in ("mfe_long", "mae_long"):
        r = h[metric]
        if r:
            print(f"  {metric:14} {r['p25']:6.1f} {r['p50']:6.1f} {r['p75']:6.1f} "
                  f"{r['p90']:6.1f} {r['p95']:6.1f}")

    print(f"\n  {'TP/SL':14} {'BE':>7} {'P(TP)':>7} {'P(SL)':>7} {'pw':>8} {'purity':>7}")
    for tp, sl in key_combos:
        k = f"TP{tp}_SL{sl}"
        if k not in h["viability_long"]:
            continue
        v = h["viability_long"][k]
        be_flag = " <-- CLEARS BE" if v["p_path_win"] >= v["breakeven_wr"] else ""
        print(f"  {tp}t/{sl}t  ({v['ratio']:.1f}:1)  "
              f"{v['breakeven_wr']:>7.1%}  {v['p_tp_hit']:>7.1%}  "
              f"{v['p_sl_hit']:>7.1%}  {v['p_path_win']:>8.1%}  "
              f"{v['path_purity']:>7.3f}{be_flag}")


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    print(f"Loading {len(date_dirs)} CLH6 dates...")

    all_rth:    list[pd.DataFrame] = []
    eia_frames: list[pd.DataFrame] = []
    non_eia:    list[pd.DataFrame] = []
    eia_dates:  list[str] = []

    for dd in date_dirs:
        date_str = dd.name.replace("date=", "")
        df = load_date(dd)
        if df is None:
            continue
        df = compute_horizons(df)
        rth = rth_filter(df, date_str)
        if len(rth) == 0:
            continue
        is_eia = pd.to_datetime(date_str).weekday() == EIA_WEEKDAY
        rth["is_eia"] = is_eia
        all_rth.append(rth)
        if is_eia:
            eia_frames.append(rth)
            eia_dates.append(date_str)
        else:
            non_eia.append(rth)
        print(f"  {date_str}: {len(rth):,} RTH bars{'  [EIA]' if is_eia else ''}")

    combined   = pd.concat(all_rth,    ignore_index=True)
    eia_df     = pd.concat(eia_frames, ignore_index=True) if eia_frames else pd.DataFrame()
    non_eia_df = pd.concat(non_eia,    ignore_index=True) if non_eia    else pd.DataFrame()

    n_dates = len(all_rth)
    print(f"\nTotal RTH bars: {len(combined):,}  over {n_dates} dates")

    # Key combos: 3:1 and 2:1 at scales appropriate for 1-4h CL moves
    key_combos = [
        # 3:1 (BE=25%)
        (20, 7), (30, 10), (40, 13), (50, 17), (60, 20), (80, 27), (100, 33), (120, 40), (150, 50),
        # 2:1 (BE=33%)
        (30, 15), (40, 20), (50, 25), (60, 30), (80, 40), (100, 50),
    ]

    def _build(df: pd.DataFrame) -> dict:
        r: dict = {"horizons": {}}
        for name in HORIZONS:
            mfe_l = df[f"mfe_long_{name}"].values
            mae_l = df[f"mae_long_{name}"].values
            mfe_s = df[f"mfe_short_{name}"].values
            mae_s = df[f"mae_short_{name}"].values
            r["horizons"][name] = {
                "mfe_long":        percentile_row(mfe_l),
                "mae_long":        percentile_row(mae_l),
                "mfe_short":       percentile_row(mfe_s),
                "mae_short":       percentile_row(mae_s),
                "viability_long":  viability_table(mfe_l, mae_l, TP_CANDIDATES, SL_CANDIDATES),
                "viability_short": viability_table(mfe_s, mae_s, TP_CANDIDATES, SL_CANDIDATES),
            }
        return r

    full_res    = _build(combined)
    eia_res     = _build(eia_df)     if not eia_df.empty    else {}
    non_eia_res = _build(non_eia_df) if not non_eia_df.empty else {}

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("CLH6 LONG-HORIZON LANDSCAPE  09:00-14:30 ET  Jan-Feb 2026")
    print("Forward window uses full 24h data (includes overnight after RTH close)")
    print("="*72)

    for name in HORIZONS:
        print_horizon(name, full_res["horizons"][name], key_combos)

    # EIA vs non-EIA at 60min
    if eia_res and non_eia_res:
        print(f"\n{'='*72}")
        print("EIA Wednesday vs non-EIA (60min horizon, long):")
        print(f"  {'TP/SL':14} {'BE':>7}  {'all':>8}  {'EIA':>8}  {'non-EIA':>9}  {'lift':>7}")
        for tp, sl in [(30, 10), (40, 13), (50, 17), (60, 20), (80, 27), (100, 33)]:
            k = f"TP{tp}_SL{sl}"
            va = full_res["horizons"]["60min"]["viability_long"].get(k, {})
            ve = eia_res["horizons"]["60min"]["viability_long"].get(k, {})
            vn = non_eia_res["horizons"]["60min"]["viability_long"].get(k, {})
            if not va:
                continue
            lift = ve.get("p_path_win", 0) - vn.get("p_path_win", 0)
            print(f"  {tp}t/{sl}t ({va['ratio']:.1f}:1)  "
                  f"{va['breakeven_wr']:>7.1%}  {va['p_path_win']:>8.1%}  "
                  f"{ve.get('p_path_win', 0):>8.1%}  "
                  f"{vn.get('p_path_win', 0):>9.1%}  {lift:>+7.1%}")

    # ── Cross-horizon summary ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("Cross-horizon path_win summary (long, 3:1 structures):")
    print(f"  {'TP/SL':14} {'BE':>7}  {'15min':>8}  {'60min':>8}  {'120min':>8}  {'240min':>8}")

    # Load 15min results for comparison
    landscape_15 = {}
    lpath = OUTPUT_DIR / "clh6_landscape.json"
    if lpath.exists():
        with open(lpath) as f:
            d = json.load(f)
        landscape_15 = d.get("all_days", {}).get("horizons", {}).get("15min", {}).get("viability_long", {})

    for tp, sl in [(20, 7), (30, 10), (40, 13), (50, 17), (60, 20), (80, 27), (100, 33)]:
        k = f"TP{tp}_SL{sl}"
        v15  = landscape_15.get(k, {})
        v60  = full_res["horizons"]["60min"]["viability_long"].get(k, {})
        v120 = full_res["horizons"]["120min"]["viability_long"].get(k, {})
        v240 = full_res["horizons"]["240min"]["viability_long"].get(k, {})
        if not v60:
            continue
        be = v60["breakeven_wr"]
        pw15  = v15.get("p_path_win",  float("nan"))
        pw60  = v60.get("p_path_win",  float("nan"))
        pw120 = v120.get("p_path_win", float("nan"))
        pw240 = v240.get("p_path_win", float("nan"))

        def fmt(x, be):
            if not np.isfinite(x):
                return f"{'n/a':>8}"
            flag = "*" if x >= be else " "
            return f"{x:>7.1%}{flag}"

        print(f"  {tp}t/{sl}t ({v60['ratio']:.1f}:1)  "
              f"{be:>7.1%}  {fmt(pw15, be)}  {fmt(pw60, be)}  "
              f"{fmt(pw120, be)}  {fmt(pw240, be)}")
    print("  (* = clears breakeven)")

    # ── Save ─────────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "instrument": "CLH6", "period": "Jan-Feb 2026",
            "dates": n_dates, "total_bars": len(combined),
            "bar_size_ms": 100,
            "session_utc": f"{RTH_START_UTC}-{RTH_END_UTC}",
            "tick_size": TICK,
            "eia_dates": eia_dates,
            "note": "Forward window computed on full 24h data",
        },
        "all_days":     full_res,
        "eia_days":     eia_res,
        "non_eia_days": non_eia_res,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "clh6_long_horizons.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
