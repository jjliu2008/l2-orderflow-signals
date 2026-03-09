"""
Signal feasibility: VWAP deviation and session range position vs
15-minute forward path_win at 18t TP / 6t SL.

Both features are structural anchors — price location relative to the
session's developing auction. Measured independently.

  vwap_dev_ticks:  (mid - session_VWAP) / tick, reset at RTH open each day
  range_pos:       (mid - session_low) / (session_high - session_low)
                   0 = at day low, 1 = at day high [cumulative since RTH open]

Usage:
    python scripts/signal_structure_features.py

Outputs:
    artifacts/signal_research/structure_features.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/signal_research"

TICK     = 0.25
TP_TICKS = 18
SL_TICKS = 6
H_BARS   = 9_000   # 15 min at 100ms

RTH_START = "14:30"
RTH_END   = "21:00"

LOAD_COLS = ["Time", "mid", "trade_volume"]


def _fwd_max(series: pd.Series, H: int) -> np.ndarray:
    rev    = series.values[::-1]
    result = pd.Series(rev).rolling(H, min_periods=1).max().values
    return result[::-1]


def _fwd_min(series: pd.Series, H: int) -> np.ndarray:
    rev    = series.values[::-1]
    result = pd.Series(rev).rolling(H, min_periods=1).min().values
    return result[::-1]


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
    df["mfe_long"]  = (fmax - df["mid"].values) / TICK
    df["mae_long"]  = (df["mid"].values - fmin) / TICK
    df["mfe_short"] = (df["mid"].values - fmin) / TICK
    df["mae_short"] = (fmax - df["mid"].values) / TICK
    df["path_win_long"]  = (df["mfe_long"]  >= TP_TICKS) & (df["mae_long"]  < SL_TICKS)
    df["path_win_short"] = (df["mfe_short"] >= TP_TICKS) & (df["mae_short"] < SL_TICKS)

    lo = pd.Timestamp(f"{date_str} {RTH_START}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {RTH_END}:00",   tz="UTC")
    rth = df[(df["Time"] >= lo) & (df["Time"] < hi)].copy()

    # ── VWAP deviation (cumulative from RTH open each day) ──────────────────
    vol  = rth["trade_volume"].clip(lower=0)
    cum_vol = vol.cumsum()
    cum_pv  = (rth["mid"] * vol).cumsum()
    # Avoid division by zero at first bar (no trades yet)
    vwap = (cum_pv / cum_vol.replace(0, np.nan)).ffill()
    rth["vwap_dev_ticks"] = (rth["mid"] - vwap) / TICK

    # ── Session range position (cumulative H/L since RTH open) ──────────────
    sess_high  = rth["mid"].cummax()
    sess_low   = rth["mid"].cummin()
    sess_range = (sess_high - sess_low).replace(0, np.nan)
    rth["range_pos"] = (rth["mid"] - sess_low) / sess_range

    return rth


def corr_biserial(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return {"r": 0.0, "p": 1.0, "n": 0}
    r, p = stats.pointbiserialr(y.astype(int), x)
    return {"r": round(float(r), 6), "p": round(float(p), 6), "n": int(len(x))}


def decile_table(df: pd.DataFrame, feat: str, outcome: str) -> list[dict]:
    valid = df[[feat, outcome]].dropna()
    if len(valid) < 100:
        return []
    valid = valid.copy()
    try:
        valid["dec"] = pd.qcut(valid[feat], 10, labels=False, duplicates="drop")
    except Exception:
        return []
    rows = []
    for dec, grp in valid.groupby("dec"):
        rows.append({
            "decile":   int(dec),
            "n":        int(len(grp)),
            "feat_p50": round(float(valid.loc[grp.index, feat].median()), 4),
            "path_win": round(float(grp[outcome].mean()), 5),
        })
    return rows


def quartile_split(df: pd.DataFrame, feat: str) -> dict:
    """Split into quartiles: Q1 (bottom 25%), Q2-Q3 (middle 50%), Q4 (top 25%)."""
    valid = df[[feat, "path_win_long", "path_win_short"]].dropna()
    q25   = valid[feat].quantile(0.25)
    q75   = valid[feat].quantile(0.75)
    q1    = valid[valid[feat] <= q25]
    mid   = valid[(valid[feat] > q25) & (valid[feat] < q75)]
    q4    = valid[valid[feat] >= q75]
    result = {}
    for label, sub in [("Q1_low25", q1), ("Q2Q3_mid50", mid), ("Q4_high25", q4)]:
        result[label] = {
            "n":              int(len(sub)),
            "feat_p50":       round(float(sub[feat].median()), 4),
            "path_win_long":  round(float(sub["path_win_long"].mean()),  5) if len(sub) else 0,
            "path_win_short": round(float(sub["path_win_short"].mean()), 5) if len(sub) else 0,
        }
    return result


def print_feature(name: str, r: dict, baseline_long: float, baseline_short: float) -> None:
    print(f"\n  {name}  (n_valid={r['n_valid']:,})")
    pcts = r["percentiles"]
    print(f"    range: [{pcts.get('0.1', 0):.2f}, {pcts.get('0.9', 0):.2f}]  "
          f"p25={pcts.get('0.25', 0):.2f}  p50={pcts.get('0.5', 0):.2f}  p75={pcts.get('0.75', 0):.2f}")
    print(f"    corr_long:  r={r['corr_long']['r']:+.5f}  p={r['corr_long']['p']:.4f}")
    print(f"    corr_short: r={r['corr_short']['r']:+.5f}  p={r['corr_short']['p']:.4f}")

    sp = r["quartile_split"]
    print(f"    Quartile split (baseline long={baseline_long:.3%}, short={baseline_short:.3%}):")
    for label, s in sp.items():
        ll = s["path_win_long"]  - baseline_long
        ls = s["path_win_short"] - baseline_short
        flag_l = " *" if abs(ll) > 0.005 else ""
        flag_s = " *" if abs(ls) > 0.005 else ""
        print(f"      {label:14} feat_p50={s['feat_p50']:8.2f}  "
              f"pw_long={s['path_win_long']:.3%}({ll:+.3%}){flag_l}  "
              f"pw_short={s['path_win_short']:.3%}({ls:+.3%}){flag_s}")

    dec_l = r["decile_long"]
    dec_s = r["decile_short"]
    if dec_l:
        print(f"    Decile table (0=lowest {name}):")
        print(f"    {'dec':>5} {'n':>8} {'feat_p50':>10} {'pw_long':>9}  {'pw_short':>9}")
        for dl, ds in zip(dec_l, dec_s):
            flag = ""
            if (abs(dl["path_win"] - baseline_long)  > 0.007 or
                abs(ds["path_win"] - baseline_short) > 0.007):
                flag = " *"
            print(f"    {dl['decile']:>5} {dl['n']:>8,} {dl['feat_p50']:>10.2f} "
                  f"{dl['path_win']:>9.3%}  {ds['path_win']:>9.3%}{flag}")


def analyse_feature(combined: pd.DataFrame, feat: str,
                    baseline_long: float, baseline_short: float) -> dict:
    valid   = combined[[feat, "path_win_long", "path_win_short"]].dropna()
    pctiles = valid[feat].quantile([.1, .25, .5, .75, .9]).round(4).to_dict()
    return {
        "n_valid":        int(len(valid)),
        "percentiles":    {str(k): float(v) for k, v in pctiles.items()},
        "corr_long":      corr_biserial(valid[feat].values, valid["path_win_long"].values),
        "corr_short":     corr_biserial(-valid[feat].values, valid["path_win_short"].values),
        "quartile_split": quartile_split(combined, feat),
        "decile_long":    decile_table(combined, feat, "path_win_long"),
        "decile_short":   decile_table(combined, feat, "path_win_short"),
    }


def main() -> None:
    date_dirs = sorted(DATA_ROOT.glob("date=2026-0[12]-*"))
    print(f"Processing {len(date_dirs)} dates (Jan+Feb 2026)...")

    all_rth: list[pd.DataFrame] = []
    for dd in date_dirs:
        df = load_and_process(dd)
        if df is None or len(df) == 0:
            continue
        all_rth.append(df)
    print(f"  Loaded {len(all_rth)} dates")

    combined      = pd.concat(all_rth, ignore_index=True)
    baseline_long  = float(combined["path_win_long"].mean())
    baseline_short = float(combined["path_win_short"].mean())
    print(f"Total RTH bars: {len(combined):,}")
    print(f"Baseline path_win_long:  {baseline_long:.3%}")
    print(f"Baseline path_win_short: {baseline_short:.3%}")

    features = {
        "vwap_dev_ticks": "VWAP deviation in ticks (negative=below VWAP)",
        "range_pos":      "Session range position [0=day_low, 1=day_high]",
    }

    results: dict = {
        "metadata": {
            "instrument": "ESH6", "period": "Jan+Feb 2026 RTH",
            "dates": len(date_dirs), "total_bars": len(combined),
            "tp_ticks": TP_TICKS, "sl_ticks": SL_TICKS, "horizon_bars": H_BARS,
            "baseline_path_win_long":  round(baseline_long,  4),
            "baseline_path_win_short": round(baseline_short, 4),
        },
        "features": {},
    }

    print(f"\n{'='*64}")
    print("Structure features vs 15-min path_win at 18t/6t")
    print(f"{'='*64}")

    for feat, desc in features.items():
        print(f"\n[{desc}]")
        r = analyse_feature(combined, feat, baseline_long, baseline_short)
        r["description"] = desc
        results["features"][feat] = r
        print_feature(feat, r, baseline_long, baseline_short)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("Summary — max lift by feature")
    print(f"  {'feature':22} {'r_long':>9} {'r_short':>9} "
          f"{'max_long_lift':>15} {'max_short_lift':>15}")
    for feat, r in results["features"].items():
        dec_l = r["decile_long"]
        dec_s = r["decile_short"]
        max_l = max((d["path_win"] - baseline_long  for d in dec_l), default=0)
        max_s = max((d["path_win"] - baseline_short for d in dec_s), default=0)
        print(f"  {feat:22} {r['corr_long']['r']:+9.5f} {r['corr_short']['r']:+9.5f} "
              f"{max_l:>+15.3%} {max_s:>+15.3%}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "structure_features.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
