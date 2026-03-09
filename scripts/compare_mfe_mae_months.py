"""
Compare MFE/MAE landscape across months.

Runs the same horizon analysis on February ESH6 parquets, then prints a
side-by-side comparison with January to check distribution stability.

Usage:
    python scripts/compare_mfe_mae_months.py

Outputs:
    artifacts/mfe_mae_analysis/horizon_landscape_feb.json
    artifacts/mfe_mae_analysis/stability_comparison.json
    (prints comparison table to stdout)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse shared logic from analyze_mfe_mae_horizons
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mfe_mae_horizons import (
    _fwd_max, _fwd_min, load_date, compute_horizons, rth_filter,
    percentile_row, viability_table,
    HORIZONS, TP_CANDIDATES, SL_CANDIDATES, TICK,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/mfe_mae_analysis"

KEY_COMBOS = [(15, 5), (18, 6), (20, 6), (20, 8), (25, 8), (30, 10)]


def run_month(date_glob: str, label: str) -> dict:
    date_dirs = sorted(DATA_ROOT.glob(date_glob))
    print(f"\nLoading {len(date_dirs)} dates ({label})...")
    all_rth = []
    for dd in date_dirs:
        date_str = dd.name.replace("date=", "")
        df = load_date(dd)
        if df is None:
            print(f"  {date_str}: missing")
            continue
        df = compute_horizons(df)
        rth = rth_filter(df, date_str)
        all_rth.append(rth)
        print(f"  {date_str}: {len(rth):,} RTH bars")
    if not all_rth:
        return {}
    combined = pd.concat(all_rth, ignore_index=True)
    print(f"  Total: {len(combined):,} RTH bars")

    results = {
        "label": label,
        "dates": len(date_dirs),
        "total_bars": len(combined),
        "horizons": {},
    }
    for name in HORIZONS:
        mfe_l = combined[f"mfe_long_{name}"].values
        mae_l = combined[f"mae_long_{name}"].values
        mfe_s = combined[f"mfe_short_{name}"].values
        mae_s = combined[f"mae_short_{name}"].values
        results["horizons"][name] = {
            "mfe_long":       percentile_row(mfe_l),
            "mae_long":       percentile_row(mae_l),
            "mfe_short":      percentile_row(mfe_s),
            "mae_short":      percentile_row(mae_s),
            "viability_long": viability_table(mfe_l, mae_l, TP_CANDIDATES, SL_CANDIDATES),
        }
    return results


def print_comparison(jan: dict, feb: dict) -> None:
    print("\n" + "="*72)
    print("STABILITY CHECK: Jan vs Feb 2026 ESH6 RTH (unconditional)")
    print("="*72)

    for name in HORIZONS:
        jh = jan["horizons"].get(name, {})
        fh = feb["horizons"].get(name, {})
        if not jh or not fh:
            continue
        print(f"\n-- {name} horizon (percentiles in ticks) ----------------------")
        print(f"  {'metric':16} {'p25 J':>7} {'p25 F':>7} {'p50 J':>7} {'p50 F':>7} {'p75 J':>7} {'p75 F':>7} {'p90 J':>7} {'p90 F':>7}")
        for side in ("long", "short"):
            for metric in ("mfe", "mae"):
                key = f"{metric}_{side}"
                jr = jh.get(key, {})
                fr = fh.get(key, {})
                if not jr or not fr:
                    continue
                drift_p50 = fr["p50"] - jr["p50"]
                flag = " *" if abs(drift_p50) > 3 else ""
                print(f"  {key:16} {jr['p25']:7.1f} {fr['p25']:7.1f} "
                      f"{jr['p50']:7.1f} {fr['p50']:7.1f} "
                      f"{jr['p75']:7.1f} {fr['p75']:7.1f} "
                      f"{jr['p90']:7.1f} {fr['p90']:7.1f}{flag}")

        print(f"\n  Viability (path_win) - P(MFE>=TP & MAE<SL):")
        print(f"  {'combo':14} {'J_pwin':>8} {'F_pwin':>8} {'drift':>8} {'J_pur':>8} {'F_pur':>8} {'be_wr':>8}")
        for tp, sl in KEY_COMBOS:
            k = f"TP{tp}_SL{sl}"
            jv = jh["viability_long"].get(k, {})
            fv = fh["viability_long"].get(k, {})
            if not jv or not fv:
                continue
            drift = fv["p_path_win"] - jv["p_path_win"]
            flag = " *" if abs(drift) > 0.04 else ""
            print(f"  {tp}t/{sl}t ({tp/sl:.1f}:1)  "
                  f"{jv['p_path_win']:8.1%} {fv['p_path_win']:8.1%} "
                  f"{drift:+8.1%} "
                  f"{jv['path_purity']:8.3f} {fv['path_purity']:8.3f} "
                  f"{jv['breakeven_wr']:8.1%}{flag}")


def main() -> None:
    jan_json = OUTPUT_DIR / "horizon_landscape.json"
    if jan_json.exists():
        print("Loading Jan landscape from cached JSON...")
        with open(jan_json) as f:
            raw = json.load(f)
        # Reconstruct viability_long from raw (already has all keys)
        jan = {
            "label": "Jan 2026",
            "dates": raw["metadata"]["dates"],
            "total_bars": raw["metadata"]["total_bars"],
            "horizons": raw["horizons"],
        }
    else:
        jan = run_month("date=2026-01-*", "Jan 2026")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feb = run_month("date=2026-02-*", "Feb 2026")

    if not feb:
        print("No Feb data loaded.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feb_path = OUTPUT_DIR / "horizon_landscape_feb.json"
    with open(feb_path, "w") as f:
        json.dump(feb, f, indent=2)
    print(f"\nFeb landscape saved to: {feb_path}")

    print_comparison(jan, feb)

    # Stability summary
    stability: dict = {}
    for name in HORIZONS:
        jh = jan["horizons"].get(name, {})
        fh = feb["horizons"].get(name, {})
        if not jh or not fh:
            continue
        for tp, sl in KEY_COMBOS:
            k = f"TP{tp}_SL{sl}"
            jv = jh["viability_long"].get(k, {})
            fv = fh["viability_long"].get(k, {})
            if not jv or not fv:
                continue
            stability[f"{name}_TP{tp}_SL{sl}"] = {
                "horizon": name, "tp": tp, "sl": sl,
                "jan_path_win": jv["p_path_win"],
                "feb_path_win": fv["p_path_win"],
                "drift": round(fv["p_path_win"] - jv["p_path_win"], 4),
                "stable": abs(fv["p_path_win"] - jv["p_path_win"]) <= 0.04,
            }

    stab_path = OUTPUT_DIR / "stability_comparison.json"
    with open(stab_path, "w") as f:
        json.dump(stability, f, indent=2)
    print(f"Stability summary saved to: {stab_path}")

    # Verdict
    print("\n-- Stability Verdict -----------------------------------------------")
    for name in ("5min", "15min", "30min"):
        unstable = [
            v for k, v in stability.items()
            if v["horizon"] == name and not v["stable"]
        ]
        if unstable:
            print(f"  {name}: {len(unstable)}/{len(KEY_COMBOS)} combos UNSTABLE (drift > 4pp)")
            for v in unstable:
                print(f"    TP{v['tp']}/SL{v['sl']}: Jan={v['jan_path_win']:.1%} "
                      f"Feb={v['feb_path_win']:.1%} drift={v['drift']:+.1%}")
        else:
            print(f"  {name}: all combos STABLE (drift <= 4pp) -- landscape is consistent")


if __name__ == "__main__":
    main()
