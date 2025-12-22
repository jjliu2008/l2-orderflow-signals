"""
Horizon sweep to test EV_net tails under different gates (baseline, cheap, PFLFT).

Usage:
  python scripts/run_horizon_sweep.py

Env vars:
  TRAIN_DATA_DIRS               comma-separated data dirs (default: data/processed)
  TRAIN_MAX_READ_ROWS_PER_FILE  cap rows read per file (default: 0 = no cap)
  TRAIN_MAX_ROWS                optional cap on total rows after load (default: 0 = no cap)
  HORIZON_SWEEP_HORIZONS        comma-separated horizons, ticks (default: 10,20,50,100)
  HORIZON_SWEEP_QUANTILE        optional sweep_cost quantile cap (default: 0.0 = disabled)
  HORIZON_SWEEP_MIN_FLOOR       optional floor on sweep cutoff (default: 0.0)
  HORIZON_SWEEP_MAX_CAP         optional cap on sweep cutoff (default: 0.0)
  PFLFT_FLOW_WINDOW             rolling window (rows) for flow/lag features (default: 20)
  PFLFT_STEP_MS                 emit cadence (ms) for flow_intensity scaling (default: 100)
  PFLFT_REPRICE_BPS             repricing threshold in bps (default: 4.0)
  PFLFT_SPREAD_Q                low spread quantile for cheap gate (default: 0.3)
  PFLFT_SWEEP_Q                 low sweep quantile for cheap gate (default: 0.3)
  PFLFT_LAG_Q                   high lag_score quantile (default: 0.8)
  PFLFT_FLOW_Q                  high flow_intensity_roll quantile (default: 0.8)
  PFLFT_DMID_Q                  low |dmid_bps_roll| quantile (default: 0.3)
  PFLFT_DEPTH_Q                 low depth_total_top5 quantile floor (default: 0.2)
  PFLFT_MIN_RATE                min candidate rate for PFLFT gate (default: 0.01)
  PFLFT_MAX_RATE                max candidate rate for PFLFT gate (default: 0.30)
  PFLFT_MID_MIN                 optional min mid filter (default: 0 = disabled)
  PFLFT_MID_MAX                 optional max mid filter (default: 0 = disabled)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, add_orderflow_features


def _infer_step_ms(df: pd.DataFrame, fallback: int) -> int:
    if "Time" not in df.columns:
        return fallback
    dt = df["Time"].diff().dropna()
    if dt.empty:
        return fallback
    ms = int(dt.dt.total_seconds().median() * 1000)
    return ms if ms > 0 else fallback


def _add_pflft_features(df: pd.DataFrame, step_ms: int, window: int) -> pd.DataFrame:
    df = df.copy()
    if "signed_volume" in df.columns:
        df["abs_signed_volume"] = df["signed_volume"].abs()
    if {"signed_volume", "trade_volume"}.issubset(df.columns):
        denom = df["trade_volume"].astype(float).replace(0.0, np.nan)
        df["trade_imbalance_ratio"] = df["signed_volume"] / denom
    if "trade_volume" in df.columns:
        per_sec = 1000.0 / max(step_ms, 1)
        df["flow_intensity"] = df["trade_volume"] * per_sec
        df["flow_intensity_roll"] = df["flow_intensity"].rolling(window, min_periods=1).mean()
        df["flow_accel"] = df["flow_intensity_roll"].diff()
    if "mid" in df.columns:
        prev = df["mid"].shift(1)
        df["dmid_bps"] = ((df["mid"] - prev) / prev) * 1e4
        prev_w = df["mid"].shift(window)
        df["dmid_bps_roll"] = ((df["mid"] - prev_w) / prev_w) * 1e4
    if "abs_signed_volume" in df.columns and "dmid_bps" in df.columns:
        eps = 1e-9
        df["impact_per_flow"] = df["dmid_bps"] / (df["abs_signed_volume"] + eps)
    if "flow_intensity" in df.columns and "dmid_bps" in df.columns:
        eps = 1e-9
        df["lag_score"] = df["flow_intensity"] / (df["dmid_bps"].abs() + eps)
    if {"depth_bid_top5", "depth_ask_top5"}.issubset(df.columns):
        df["depth_total_top5"] = df["depth_bid_top5"] + df["depth_ask_top5"]
    if {"spread", "mid"}.issubset(df.columns):
        spread_bps = (df["spread"] / df["mid"]) * 1e4
        df["spread_stability"] = spread_bps.rolling(window, min_periods=1).std()
    if "signed_volume" in df.columns:
        df["signed_volume_roll"] = df["signed_volume"].rolling(window, min_periods=1).mean()
    return df


def compute_fav_excursion(mid: pd.Series, ref_dir: pd.Series, horizon: int = 2) -> pd.Series:
    """
    Directional favorable excursion over next `horizon` steps in bps:
      max_{k<=H} (ref_dir * (mid_{t+k} - mid_t) / mid_t) * 1e4
    """
    if horizon < 2:
        horizon = 2

    grouped = mid.groupby(level=0)
    fwd = grouped.apply(lambda x: x.shift(-1)).droplevel(0)
    rel = (fwd - mid) / mid
    rel_dir = rel * ref_dir
    fav = rel_dir.groupby(level=0).transform(lambda x: x.rolling(horizon, min_periods=horizon).max())
    return (fav * 1e4).rename(f"fav_excursion_{horizon}")


def compute_repriced_within(
    mid: pd.Series, ref_dir: pd.Series, horizon: int, reprice_bps: float
) -> pd.Series:
    if horizon < 2:
        horizon = 2
    grouped = mid.groupby(level=0)
    fwd = grouped.apply(lambda x: x.shift(-1)).droplevel(0)
    rel = (fwd - mid) / mid
    rel_dir = rel * ref_dir
    max_rel = rel_dir.groupby(level=0).transform(lambda x: x.rolling(horizon, min_periods=horizon).max())
    return (max_rel >= (reprice_bps / 1e4)).rename(f"repriced_within_{horizon}")


def _compute_cost_bps(df: pd.DataFrame, sweep_cost_mag: Optional[pd.Series]) -> pd.Series:
    spread_bps = (df["spread"] / df["mid"]) * 1e4
    if sweep_cost_mag is None:
        sweep_bps = pd.Series(0, index=df.index)
    else:
        sweep_bps = (sweep_cost_mag.reindex(df.index) * 1e4).fillna(0)
    return spread_bps / 2 + sweep_bps


def _quantile(series: pd.Series, q: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


def main():
    default_dirs = os.environ.get("TRAIN_DATA_DIRS", "data/processed")
    dir_list = [Path(d.strip()) for d in default_dirs.split(",") if d.strip()]

    horizons_env = os.environ.get("HORIZON_SWEEP_HORIZONS", "10,20,50,100")
    horizons: List[int] = [int(h.strip()) for h in horizons_env.split(",") if h.strip()]

    sweep_q = float(os.environ.get("HORIZON_SWEEP_QUANTILE", "0.0"))
    sweep_min_floor = float(os.environ.get("HORIZON_SWEEP_MIN_FLOOR", "0.0"))
    sweep_max_cap = float(os.environ.get("HORIZON_SWEEP_MAX_CAP", "0.0"))
    flow_window = int(os.environ.get("PFLFT_FLOW_WINDOW", "20"))
    step_ms = int(os.environ.get("PFLFT_STEP_MS", "100"))
    reprice_bps = float(os.environ.get("PFLFT_REPRICE_BPS", "4.0"))
    q_spread = float(os.environ.get("PFLFT_SPREAD_Q", "0.3"))
    q_sweep = float(os.environ.get("PFLFT_SWEEP_Q", "0.3"))
    q_lag = float(os.environ.get("PFLFT_LAG_Q", "0.8"))
    q_flow = float(os.environ.get("PFLFT_FLOW_Q", "0.8"))
    q_dmid = float(os.environ.get("PFLFT_DMID_Q", "0.3"))
    q_depth = float(os.environ.get("PFLFT_DEPTH_Q", "0.2"))
    min_rate = float(os.environ.get("PFLFT_MIN_RATE", "0.01"))
    max_rate = float(os.environ.get("PFLFT_MAX_RATE", "0.30"))
    mid_min = float(os.environ.get("PFLFT_MID_MIN", "0"))
    mid_max = float(os.environ.get("PFLFT_MID_MAX", "0"))

    max_rows_per_file_env = int(os.environ.get("TRAIN_MAX_READ_ROWS_PER_FILE", "0"))
    max_rows_per_file = max_rows_per_file_env if max_rows_per_file_env > 0 else None
    max_rows_total = int(os.environ.get("TRAIN_MAX_ROWS", "0"))

    frames = []
    for d in dir_list:
        if d.exists():
            print(f"Loading data from {d} ...")
            frames.append(load_all_raw_data(d, max_rows_per_file=max_rows_per_file))
        else:
            print(f"Skipping missing data dir: {d}")
    if not frames:
        raise FileNotFoundError("No training data directories contained usable files.")
    df = pd.concat(frames)
    if max_rows_total > 0 and len(df) > max_rows_total:
        df = df.iloc[:max_rows_total]
        print(f"Capped total rows to {len(df)} via TRAIN_MAX_ROWS={max_rows_total}")
    if "mid" not in df.columns or "spread" not in df.columns:
        df = add_basic_features(df)
    df = add_orderflow_features(df)
    step_ms = _infer_step_ms(df, step_ms)
    df = _add_pflft_features(df, step_ms=step_ms, window=flow_window)

    if "mid" not in df or "spread" not in df:
        raise ValueError("Required columns mid/spread missing after feature construction.")

    if mid_min > 0:
        df = df[df["mid"] >= mid_min]
    if mid_max > 0:
        df = df[df["mid"] <= mid_max]
    if df.empty:
        raise ValueError("No rows left after mid filtering; check PFLFT_MID_MIN/MAX.")

    sweep_cost_mag = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df.columns):
        sweep_cost_mag = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)

    cost_bps = _compute_cost_bps(df, sweep_cost_mag)
    spread_bps = (df["spread"] / df["mid"]) * 1e4
    sweep_bps = (sweep_cost_mag * 1e4) if sweep_cost_mag is not None else pd.Series(0, index=df.index)

    spread_cut = _quantile(spread_bps, q_spread)
    sweep_cut = _quantile(sweep_bps, q_sweep)
    lag_cut = _quantile(df["lag_score"], q_lag)
    flow_cut = _quantile(df["flow_intensity_roll"], q_flow)
    dmid_cut = _quantile(df["dmid_bps_roll"].abs(), q_dmid)
    depth_cut = _quantile(df["depth_total_top5"], q_depth)

    baseline_mask = pd.Series(True, index=df.index)
    cheap_mask = (spread_bps <= spread_cut) & (sweep_bps <= sweep_cut)
    pflft_mask = (
        cheap_mask
        & (df["lag_score"] >= lag_cut)
        & (df["flow_intensity_roll"] >= flow_cut)
        & (df["dmid_bps_roll"].abs() <= dmid_cut)
        & (df["depth_total_top5"] >= depth_cut)
    )

    def _run_gate(name: str, mask: pd.Series, enforce_rate: bool = False) -> None:
        total = len(df)
        count = int(mask.sum())
        rate = count / total if total else 0
        print(f"{name} gate: {count} / {total} ({rate:.2%})")
        if enforce_rate and (rate < min_rate or rate > max_rate):
            raise ValueError(
                f"{name} candidate rate {rate:.2%} outside [{min_rate:.0%},{max_rate:.0%}]; adjust PFLFT_* quantiles."
            )
        if count == 0:
            print(f"{name}: no rows; skipping")
            return
        sub = df.loc[mask].copy()
        ref_dir = np.sign(sub["signed_volume_roll"]).replace(0, np.nan).fillna(0)
        max_h = max(horizons) if horizons else 0
        if max_h > 0:
            fav_max = compute_fav_excursion(sub["mid"], ref_dir, horizon=max_h)
            base_idx = fav_max.dropna().index
            sub = sub.loc[base_idx]
            ref_dir = ref_dir.loc[base_idx]
        if sub.empty:
            print(f"{name}: no rows after constant-sample filter; skipping")
            return
        sub_cost = cost_bps.loc[sub.index]
        for h in horizons:
            fav = compute_fav_excursion(sub["mid"], ref_dir.loc[sub.index], horizon=h)
            rep = compute_repriced_within(sub["mid"], ref_dir.loc[sub.index], horizon=h, reprice_bps=reprice_bps)
            out = pd.DataFrame({
                "fav_exc_bps": fav,
                "cost_bps": sub_cost,
                "repriced": rep,
            }).dropna()
            if out.empty:
                print(f"{name} H={h}: no rows; skipping")
                continue
            ev_net = out["fav_exc_bps"] - out["cost_bps"]
            cost_desc = out["cost_bps"].quantile([0.5, 0.75, 0.9]).to_dict()
            print(
                f"{name} H={h}: rows={len(out)}, "
                f"repriced_rate={out['repriced'].mean():.3f}, "
                f"fav_exc_bps p50={out['fav_exc_bps'].median():.6g}, p75={out['fav_exc_bps'].quantile(0.75):.6g}, "
                f"p90={out['fav_exc_bps'].quantile(0.9):.6g}, "
                f"ev_net p50={ev_net.median():.6g}, p75={ev_net.quantile(0.75):.6g}, p90={ev_net.quantile(0.9):.6g}, "
                f"p(ev_net>0)={(ev_net > 0).mean():.3f}, "
                f"cost_bps p50={cost_desc.get(0.5, float('nan')):.6g}, p75={cost_desc.get(0.75, float('nan')):.6g}, p90={cost_desc.get(0.9, float('nan')):.6g}"
            )

    # Optional mild sweep filter (global) to drop ultra-thick regimes
    if sweep_cost_mag is not None and sweep_q > 0:
        sweep_cutoff = float(sweep_cost_mag.quantile(sweep_q))
        if sweep_min_floor > 0:
            sweep_cutoff = max(sweep_cutoff, sweep_min_floor)
        if sweep_max_cap > 0:
            sweep_cutoff = min(sweep_cutoff, sweep_max_cap)
        global_sweep_mask = (sweep_cost_mag <= sweep_cutoff).fillna(False)
        baseline_mask &= global_sweep_mask
        cheap_mask &= global_sweep_mask
        pflft_mask &= global_sweep_mask
        print(f"Global sweep cap: cutoff={sweep_cutoff:.6g}")
    else:
        print("Global sweep cap: disabled")

    print(
        f"Thresholds: spread_q={q_spread}({spread_cut:.6g}), sweep_q={q_sweep}({sweep_cut:.6g}), "
        f"lag_q={q_lag}({lag_cut:.6g}), flow_q={q_flow}({flow_cut:.6g}), "
        f"dmid_q={q_dmid}({dmid_cut:.6g}), depth_q={q_depth}({depth_cut:.6g})"
    )

    _run_gate("baseline", baseline_mask)
    _run_gate("cheap", cheap_mask)
    _run_gate("pflft", pflft_mask, enforce_rate=True)


if __name__ == "__main__":
    main()
