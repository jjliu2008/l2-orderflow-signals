
"""
Calendar spread behavior diagnostics aligned to Phase 1-7 research questions.

Requires two processed legs with date=YYYY-MM-DD/features_labels.parquet.
Example (front vs second CL once both exist):
  python scripts/analyze_calendar_spread_behavior.py \
    --leg-a-root data/processed_clh6/instrument=CL \
    --leg-b-root data/processed_clj6/instrument=CL \
    --date-glob "date=2026-0[12]-*" \
    --session-start 14:00 --session-end 19:30 --step-ms 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fnum(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def series_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(len(s)),
        "mean": fnum(s.mean()),
        "std": fnum(s.std()),
        "skew": fnum(s.skew()),
        "kurtosis_fisher": fnum(s.kurt()),
        "p01": fnum(q.loc[0.01]),
        "p05": fnum(q.loc[0.05]),
        "p25": fnum(q.loc[0.25]),
        "p50": fnum(q.loc[0.5]),
        "p75": fnum(q.loc[0.75]),
        "p95": fnum(q.loc[0.95]),
        "p99": fnum(q.loc[0.99]),
    }


def corr(a: pd.Series, b: pd.Series):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    m = a.notna() & b.notna()
    if int(m.sum()) < 10:
        return None
    return fnum(a[m].corr(b[m]))


def slope_per_bar(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 3:
        return None
    x = np.arange(len(s), dtype=float)
    return fnum(np.polyfit(x, s.values.astype(float), 1)[0])


def bars(minutes: int, step_ms: int) -> int:
    return max(1, int((minutes * 60_000) // step_ms))


def date_dirs(root: Path, patt: str):
    out = {}
    for d in root.glob(patt):
        if d.is_dir() and d.name.startswith("date="):
            out[d.name.replace("date=", "")] = d
    return out

def load_day(date_dir: Path, suffix: str):
    pq = date_dir / "features_labels.parquet"
    if not pq.exists():
        return None
    raw = pd.read_parquet(pq)
    wanted = ["Time", "mid", "spread", "signed_volume", "trade_volume", "stress_ratio"]
    cols = [c for c in wanted if c in raw.columns]
    if "Time" not in cols or "mid" not in cols:
        return None
    df = raw[cols].copy()
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df = df.rename(columns={c: f"{c}_{suffix}" for c in cols if c != "Time"})
    return df


def apply_session(df: pd.DataFrame, date_str: str, start: str | None, end: str | None):
    if not start or not end:
        return df
    lo = pd.Timestamp(f"{date_str} {start}:00", tz="UTC")
    hi = pd.Timestamp(f"{date_str} {end}:00", tz="UTC")
    return df[(df["Time"] >= lo) & (df["Time"] < hi)].copy()


def resample_df(df: pd.DataFrame, step_ms: int):
    if step_ms <= 0:
        return df.sort_values("Time").reset_index(drop=True)
    rule = f"{step_ms}ms"
    last_cols = [c for c in ["mid_a", "mid_b", "spread_a", "spread_b", "stress_ratio_a", "stress_ratio_b"] if c in df.columns]
    sum_cols = [c for c in ["signed_volume_a", "signed_volume_b", "trade_volume_a", "trade_volume_b"] if c in df.columns]
    agg = {c: "last" for c in last_cols}
    agg.update({c: "sum" for c in sum_cols})
    out = (
        df.set_index("Time")
        .sort_index()
        .resample(rule)
        .agg(agg)
        .dropna(subset=["mid_a", "mid_b"])
        .reset_index()
    )
    return out


def build_dataset(leg_a_root: Path, leg_b_root: Path, patt: str, step_ms: int, session_start: str | None, session_end: str | None):
    a = date_dirs(leg_a_root, patt)
    b = date_dirs(leg_b_root, patt)
    common = sorted(set(a) & set(b))
    if not common:
        raise RuntimeError(f"No overlapping dates for roots: {leg_a_root} vs {leg_b_root}")

    frames = []
    skipped = 0
    for ds in common:
        da = load_day(a[ds], "a")
        db = load_day(b[ds], "b")
        if da is None or db is None:
            skipped += 1
            continue
        m = da.merge(db, on="Time", how="inner")
        if m.empty:
            skipped += 1
            continue
        m = apply_session(m, ds, session_start, session_end)
        if m.empty:
            skipped += 1
            continue
        m = resample_df(m, step_ms)
        if m.empty:
            skipped += 1
            continue
        m["date"] = ds
        frames.append(m)

    if not frames:
        raise RuntimeError("No data after merge/session/resample")

    df = pd.concat(frames, ignore_index=True).sort_values(["date", "Time"]).reset_index(drop=True)
    df.attrs["n_common_dates"] = len(common)
    df.attrs["n_loaded_dates"] = len(frames)
    df.attrs["n_skipped_dates"] = skipped
    return df


def roll_by_day(df: pd.DataFrame, col: str, w: int, fn: str):
    mp = min(w, max(1, w // 5))
    return df.groupby("date", group_keys=False)[col].transform(lambda s: getattr(s.rolling(w, min_periods=mp), fn)())


def fwd_by_day(df: pd.DataFrame, col: str, h: int):
    return df.groupby("date", group_keys=False)[col].shift(-h) - df[col]

def analyze(df: pd.DataFrame, step_ms: int):
    df = df.copy()
    h5, h30, h60, h120 = bars(5, step_ms), bars(30, step_ms), bars(60, step_ms), bars(120, step_ms)
    hday = int(df.groupby("date").size().median())

    df["spread"] = df["mid_a"] - df["mid_b"]
    df["ds1"] = df.groupby("date", group_keys=False)["spread"].diff()
    df["abs_ds1"] = df["ds1"].abs()
    rm60 = roll_by_day(df, "spread", h60, "mean")
    rs60 = roll_by_day(df, "spread", h60, "std")
    df["z60"] = (df["spread"] - rm60) / rs60.replace(0, np.nan)
    df["abs_z60"] = df["z60"].abs()
    df["past_5m"] = df["spread"] - df.groupby("date", group_keys=False)["spread"].shift(h5)
    df["vol30"] = roll_by_day(df, "ds1", h30, "std")
    abs_move30 = (df["spread"] - df.groupby("date", group_keys=False)["spread"].shift(h30)).abs()
    sum_abs30 = roll_by_day(df, "abs_ds1", h30, "sum")
    df["trend30"] = abs_move30 / sum_abs30.replace(0, np.nan)

    df["session_mean"] = df.groupby("date", group_keys=False)["spread"].expanding().mean().reset_index(level=0, drop=True)
    df["dist_session_mean"] = df["spread"] - df["session_mean"]

    if "trade_volume_a" in df.columns and "trade_volume_b" in df.columns:
        v = pd.to_numeric(df["trade_volume_a"], errors="coerce").fillna(0) + pd.to_numeric(df["trade_volume_b"], errors="coerce").fillna(0)
        num = (df["spread"] * v).groupby(df["date"]).cumsum()
        den = v.groupby(df["date"]).cumsum()
        df["dist_session_vwap"] = df["spread"] - (num / den.replace(0, np.nan))
    else:
        df["dist_session_vwap"] = np.nan

    horizons = {"5m": h5, "30m": h30, "60m": h60, "1day": max(1, hday)}
    for name, hb in horizons.items():
        df[f"fwd_{name}"] = fwd_by_day(df, "spread", hb)

    hh = df["Time"].dt.hour
    mm = df["Time"].dt.minute
    tod_min = hh * 60 + mm
    df["tod_30m"] = df["Time"].dt.floor("30min").dt.strftime("%H:%M")
    df["dow"] = df["Time"].dt.day_name()
    df["is_eia_window"] = (df["Time"].dt.weekday == 2) & ((tod_min - (15 * 60 + 30)).abs() <= 30)

    report = {"meta": {"rows": int(len(df)), "dates": sorted(df["date"].unique().tolist()), "step_ms": int(step_ms), "horizon_bars": horizons}}

    # Phase 1
    day_summary = []
    for d, g in df.groupby("date"):
        s = g["spread"]
        n = len(g)
        h = n // 2
        shift = None
        if h >= 10 and s.std() not in (0, np.nan):
            shift = fnum((s.iloc[h:].mean() - s.iloc[:h].mean()) / s.std())
        day_summary.append({"date": d, "n": int(n), "slope_per_bar": slope_per_bar(s), "half_shift_std_units": shift, "stats": series_stats(s)})
    report["phase1_distribution"] = {
        "overall": series_stats(df["spread"]),
        "overall_slope_per_bar": slope_per_bar(df["spread"]),
        "daily": day_summary,
        "days_abs_shift_gt_1std": int(sum(1 for r in day_summary if r["half_shift_std_units"] is not None and abs(r["half_shift_std_units"]) > 1.0)),
    }

    # Phase 2
    p2 = {}
    for k in [1, 2, 4, 8]:
        m = df["spread"] - df.groupby("date", group_keys=False)["spread"].shift(k)
        f = df.groupby("date", group_keys=False)["spread"].shift(-k) - df["spread"]
        ok = m.notna() & f.notna() & (m != 0) & (f != 0)
        if int(ok.sum()) < 20:
            p2[f"{k}bar"] = {"n": int(ok.sum())}
            continue
        sm, sf = np.sign(m[ok].values), np.sign(f[ok].values)
        p2[f"{k}bar"] = {
            "n": int(ok.sum()),
            "continuation_rate": fnum(np.mean(sm == sf)),
            "snapback_rate": fnum(np.mean(sm == -sf)),
            "corr_move_vs_forward": fnum(np.corrcoef(m[ok].values, f[ok].values)[0, 1]),
        }
    report["phase2_persistence"] = p2

    # Phase 3 + 4 + 7
    p3 = {}
    p4 = {}
    for h in ["5m", "30m", "60m", "1day"]:
        fwd = df[f"fwd_{h}"]
        rev = -np.sign(df["z60"]) * fwd
        ev = df["abs_z60"] >= 2
        p3[h] = {
            "corr_z60_vs_fwd": corr(df["z60"], fwd),
            "corr_ds1_vs_fwd": corr(df["ds1"], fwd),
            "extreme_reversion_rate_absz2": fnum((rev[ev] > 0).mean()) if int(ev.sum()) > 0 else None,
        }
    for thr in [1.0, 2.0, 3.0]:
        m = df["abs_z60"] >= thr
        p4[f"|z|>={thr:g}"] = {
            "count": int(m.sum()),
            "frequency": fnum(m.mean()),
            "forward_30m_reversion_rate": fnum(((-np.sign(df["z60"]) * df["fwd_30m"])[m] > 0).mean()) if int(m.sum()) > 0 else None,
            "forward_30m_stats": series_stats(((-np.sign(df["z60"]) * df["fwd_30m"])[m])),
        }
    report["phase3_predictability_by_horizon"] = p3
    report["phase4_extremes"] = p4

    # Phase 5/6 large widening/narrowing and Phase 8-12 regime/time/event
    q95, q05 = df["past_5m"].quantile(0.95), df["past_5m"].quantile(0.05)
    w = df["past_5m"] >= q95
    n = df["past_5m"] <= q05
    p56 = {"thresholds": {"q95": fnum(q95), "q05": fnum(q05)}, "rows": {}}
    for h in ["5m", "30m", "60m"]:
        fwd = df[f"fwd_{h}"]
        p56["rows"][h] = {
            "after_widen_reversion_rate": fnum((fwd[w] < 0).mean()) if int(w.sum()) > 0 else None,
            "after_widen_stats": series_stats(fwd[w]),
            "after_narrow_reversion_rate": fnum((fwd[n] > 0).mean()) if int(n.sum()) > 0 else None,
            "after_narrow_stats": series_stats(fwd[n]),
        }

    ev = df["abs_z60"] >= 2
    rev30 = -np.sign(df["z60"]) * df["fwd_30m"]
    vol_cut = float(df["vol30"].median(skipna=True)) if df["vol30"].notna().any() else np.nan
    trend_cut = float(df["trend30"].median(skipna=True)) if df["trend30"].notna().any() else np.nan
    df["regime2d"] = np.where(df["vol30"] >= vol_cut, "high_vol", "low_vol") + "|" + np.where(df["trend30"] >= trend_cut, "trend", "chop")

    reg_rows = []
    for rg, g in df[ev].groupby("regime2d"):
        rr = rev30.loc[g.index]
        reg_rows.append({"regime": rg, "n": int(rr.notna().sum()), "reversion_rate_30m": fnum((rr > 0).mean()), "mean_reversion_ret_30m": fnum(rr.mean())})

    tod_instability = (
        df.groupby("tod_30m", as_index=False)["ds1"]
        .agg(std_move="std", mean_abs_move=lambda x: np.nanmean(np.abs(x)))
        .sort_values("std_move", ascending=False)
        .to_dict(orient="records")
    )
    tod_rev = []
    for tod, g in df[ev].groupby("tod_30m"):
        rr = rev30.loc[g.index]
        tod_rev.append({"tod_30m": tod, "n": int(rr.notna().sum()), "reversion_rate_30m": fnum((rr > 0).mean()), "mean_reversion_ret_30m": fnum(rr.mean())})
    dow_rows = []
    for dw, g in df[ev].groupby("dow"):
        rr = rev30.loc[g.index]
        dow_rows.append({"dow": dw, "n": int(rr.notna().sum()), "reversion_rate_30m": fnum((rr > 0).mean()), "mean_reversion_ret_30m": fnum(rr.mean())})

    def block(mask):
        rr = rev30[ev & mask]
        return {"n": int(rr.notna().sum()), "reversion_rate_30m": fnum((rr > 0).mean()), "mean_reversion_ret_30m": fnum(rr.mean())}

    report["phase5_6_large_moves"] = p56
    report["phase8_12_regime_time_event"] = {
        "regime_cutoffs": {"vol30_median": fnum(vol_cut), "trend30_median": fnum(trend_cut)},
        "regime_rows": reg_rows,
        "tod_instability": tod_instability,
        "tod_reversion": sorted(tod_rev, key=lambda x: (x["reversion_rate_30m"] is None, -(x["reversion_rate_30m"] or -1))),
        "dow_effects": dow_rows,
        "event_vs_none": {"eia_window": block(df["is_eia_window"]), "non_event": block(~df["is_eia_window"])}
    }

    # Phase 13-22 compact diagnostics
    p13 = []
    for mins in [30, 60, 120]:
        wlb = bars(mins, step_ms)
        rm = roll_by_day(df, "spread", wlb, "mean")
        dist = df["spread"] - rm
        p13.append({"lookback_min": mins, "corr_dist_vs_fwd30": corr(dist, df["fwd_30m"]), "corr_absdist_vs_absfwd30": corr(dist.abs(), df["fwd_30m"].abs())})

    event_flag = (df["abs_z60"] >= 2).astype(float)
    acf = {f"lag_{i}": fnum(event_flag.autocorr(lag=i)) for i in range(1, 11)}
    idx = np.where(event_flag.values > 0.5)[0]
    inter = np.diff(idx) * step_ms / 60_000.0 if len(idx) >= 2 else np.array([])

    fwd30_d1 = df.groupby("date", group_keys=False)["spread"].shift(-(h30 + 1)) - df.groupby("date", group_keys=False)["spread"].shift(-1)
    rev30_d1 = -np.sign(df["z60"]) * fwd30_d1

    flow_a = pd.to_numeric(df.get("signed_volume_a", pd.Series(index=df.index, dtype=float)), errors="coerce")
    flow_b = pd.to_numeric(df.get("signed_volume_b", pd.Series(index=df.index, dtype=float)), errors="coerce")
    fd3 = roll_by_day(df.assign(flow_diff=flow_a - flow_b), "flow_diff", 3, "sum")
    aligned_flow = -np.sign(df["z60"]) * fd3

    report["phase13_22_compact"] = {
        "stretch_predictive": p13,
        "session_distance_predictive": {
            "corr_dist_session_mean_vs_fwd30": corr(df["dist_session_mean"], df["fwd_30m"]),
            "corr_dist_session_vwap_vs_fwd30": corr(df["dist_session_vwap"], df["fwd_30m"]),
        },
        "event_clustering": {
            "n_events": int(event_flag.sum()),
            "acf_lag1_10": acf,
            "interarrival_minutes": series_stats(pd.Series(inter)),
        },
        "execution_immediate_vs_wait1": {
            "immediate": {"n": int(rev30[ev].notna().sum()), "reversion_rate": fnum((rev30[ev] > 0).mean()), "mean_ret": fnum(rev30[ev].mean())},
            "wait1": {"n": int(rev30_d1[ev].notna().sum()), "reversion_rate": fnum((rev30_d1[ev] > 0).mean()), "mean_ret": fnum(rev30_d1[ev].mean())},
            "delta_mean_wait1_minus_now": fnum(rev30_d1[ev].mean() - rev30[ev].mean()),
        },
        "adverse_flow_split": {
            "aligned_half": {"n": int((ev & (aligned_flow >= aligned_flow.median())).sum()), "reversion_rate": fnum((rev30[ev & (aligned_flow >= aligned_flow.median())] > 0).mean())},
            "adverse_half": {"n": int((ev & (aligned_flow < aligned_flow.median())).sum()), "reversion_rate": fnum((rev30[ev & (aligned_flow < aligned_flow.median())] > 0).mean())},
        },
    }

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-a-root", default=str(PROJECT_ROOT / "data/processed_clh6/instrument=CL"))
    ap.add_argument("--leg-b-root", default=str(PROJECT_ROOT / "data/processed_clj6/instrument=CL"))
    ap.add_argument("--date-glob", default="date=2026-0[12]-*")
    ap.add_argument("--step-ms", type=int, default=1000)
    ap.add_argument("--session-start", default=None)
    ap.add_argument("--session-end", default=None)
    ap.add_argument("--output", default=str(PROJECT_ROOT / "artifacts/signal_research/calendar_spread_phase_report.json"))
    args = ap.parse_args()

    aroot = Path(args.leg_a_root)
    broot = Path(args.leg_b_root)
    if not aroot.exists():
        raise SystemExit(f"leg-a root not found: {aroot}")
    if not broot.exists():
        raise SystemExit(f"leg-b root not found: {broot}. Add second leg data first.")

    df = build_dataset(aroot, broot, args.date_glob, args.step_ms, args.session_start, args.session_end)
    rep = analyze(df, args.step_ms)
    rep["inputs"] = {
        "leg_a_root": str(aroot),
        "leg_b_root": str(broot),
        "date_glob": args.date_glob,
        "session_start": args.session_start,
        "session_end": args.session_end,
        "step_ms": int(args.step_ms),
        "n_common_dates": int(df.attrs.get("n_common_dates", 0)),
        "n_loaded_dates": int(df.attrs.get("n_loaded_dates", 0)),
        "n_skipped_dates": int(df.attrs.get("n_skipped_dates", 0)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"Wrote spread diagnostics to {out}")
    print(f"Loaded dates: {rep['inputs']['n_loaded_dates']}/{rep['inputs']['n_common_dates']} (skipped={rep['inputs']['n_skipped_dates']})")
    print(f"Rows analyzed: {rep['meta']['rows']}")


if __name__ == "__main__":
    main()
