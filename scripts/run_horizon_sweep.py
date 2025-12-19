"""
Horizon sweep to measure realized range (and cost-adjusted EV proxy) on sweep_high candidates.

Usage:
  python scripts/run_horizon_sweep.py

Env vars:
  TRAIN_DATA_DIRS           comma-separated data dirs (default: data/processed)
  TRAIN_MAX_READ_ROWS_PER_FILE  cap rows read per file (default: 0 = no cap)
  TRAIN_MAX_ROWS            optional cap on total rows after load (default: 0 = no cap)
  HORIZON_SWEEP_HORIZONS    comma-separated horizons, ticks (default: 10,20,50,100)
  HORIZON_SWEEP_QUANTILE    sweep_cost quantile for sweep_high (default: 0.8)
  HORIZON_SWEEP_MIN_FLOOR   optional floor on sweep cutoff (default: 0.0)
  HORIZON_SWEEP_MAX_CAP     optional cap on sweep cutoff (default: 0.0)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, add_orderflow_features
from scripts.run_train_model import build_candidate_mask


def compute_expected_range(mid: pd.Series, horizon: int = 2) -> pd.Series:
    """
    Expected price range over the next `horizon` steps: (future_max - future_min) / current_mid.
    Uses a forward-looking rolling window per symbol. Horizon is clamped to >=2.
    """
    if horizon < 2:
        horizon = 2

    def _fwd_range(s: pd.Series) -> pd.Series:
        fwd = s.shift(-1)
        fwd_max = fwd.rolling(horizon, min_periods=horizon).max()
        fwd_min = fwd.rolling(horizon, min_periods=horizon).min()
        return (fwd_max - fwd_min) / s

    return mid.groupby(level=0).transform(_fwd_range).rename(f"expected_range_{horizon}")


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


def main():
    default_dirs = os.environ.get("TRAIN_DATA_DIRS", "data/processed")
    dir_list = [Path(d.strip()) for d in default_dirs.split(",") if d.strip()]

    horizons_env = os.environ.get("HORIZON_SWEEP_HORIZONS", "10,20,50,100")
    horizons: List[int] = [int(h.strip()) for h in horizons_env.split(",") if h.strip()]

    sweep_q = float(os.environ.get("HORIZON_SWEEP_QUANTILE", "0.0"))
    sweep_min_floor = float(os.environ.get("HORIZON_SWEEP_MIN_FLOOR", "0.0"))
    sweep_max_cap = float(os.environ.get("HORIZON_SWEEP_MAX_CAP", "0.0"))

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
    df = add_basic_features(df)
    df = add_orderflow_features(df)

    if "mid" not in df or "spread" not in df:
        raise ValueError("Required columns mid/spread missing after feature construction.")

    sweep_cost_mag = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df.columns):
        sweep_cost_mag = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)

    # Candidate gate (imbalance-driven)
    cand_mask = build_candidate_mask(df)
    if not cand_mask.any():
        raise ValueError("No candidates after applying imbalance/spread/activity gate. Loosen CAND_* envs.")
    df_cand = df[cand_mask]
    total = len(df)
    n_cand = len(df_cand)

    # Optional mild sweep filter to drop ultra-thick/dead regimes
    if sweep_cost_mag is not None and sweep_q > 0:
        sweep_cost_mag = sweep_cost_mag[cand_mask]
        sweep_cutoff = float(sweep_cost_mag.quantile(sweep_q))
        if sweep_min_floor > 0:
            sweep_cutoff = max(sweep_cutoff, sweep_min_floor)
        if sweep_max_cap > 0:
            sweep_cutoff = min(sweep_cutoff, sweep_max_cap)
        sweep_mask = (sweep_cost_mag <= sweep_cutoff).reindex(df_cand.index).fillna(False)
    else:
        sweep_mask = pd.Series(True, index=df_cand.index)
        sweep_cutoff = None

    cand_keep = df_cand[sweep_mask]
    if cand_keep.empty:
        raise ValueError("No rows after candidate gate (and optional sweep cap); aborting sweep.")

    # Cost proxy (relative bps) on kept candidates
    spread_rel = cand_keep["spread"] / cand_keep["mid"]
    spread_bps = spread_rel * 1e4
    if sweep_cost_mag is not None:
        sweep_bps = (sweep_cost_mag.reindex(cand_keep.index) * 1e4).fillna(0)
    else:
        sweep_bps = pd.Series(0, index=cand_keep.index)
    cost_proxy_bps = spread_bps / 2 + sweep_bps

    print(f"candidates: {n_cand} / {total} ({(n_cand/total if total else 0):.2%}), kept after sweep cap: {len(cand_keep)} ({(len(cand_keep)/n_cand if n_cand else 0):.2%}), sweep_cutoff={sweep_cutoff if sweep_cutoff is not None else 'none'}")

    ref_dir = np.sign((cand_keep["order_book_imbalance"].groupby(level=0).transform(lambda x: x.rolling(50, min_periods=25).mean())))
    # Constant sample across horizons: require forward coverage for max horizon
    max_h = max(horizons) if horizons else 0
    base_idx = cand_keep.index
    if max_h > 0:
        fav_max = compute_fav_excursion(cand_keep["mid"], ref_dir, horizon=max_h)
        base_idx = fav_max.dropna().index
        cand_keep = cand_keep.loc[base_idx]
        ref_dir = ref_dir.loc[base_idx]
        cost_proxy_bps = cost_proxy_bps.loc[base_idx]

    for h in horizons:
        fav = compute_fav_excursion(cand_keep["mid"], ref_dir.loc[cand_keep.index], horizon=h)
        sub = pd.DataFrame({
            "fav_exc_bps": fav,
            "cost_bps": cost_proxy_bps,
        }).dropna()
        if sub.empty:
            print(f"H={h}: no candidate rows; skipping")
            continue
        ev_net = sub["fav_exc_bps"] - sub["cost_bps"]
        cost_desc = sub["cost_bps"].quantile([0.5, 0.75, 0.9]).to_dict()
        print(
            f"H={h}: candidates={len(sub)}, "
            f"fav_exc_bps p50={sub['fav_exc_bps'].median():.6g}, p75={sub['fav_exc_bps'].quantile(0.75):.6g}, "
            f"p90={sub['fav_exc_bps'].quantile(0.9):.6g}, "
            f"ev_net p50={ev_net.median():.6g}, p75={ev_net.quantile(0.75):.6g}, p90={ev_net.quantile(0.9):.6g}, "
            f"cost_bps p50={cost_desc.get(0.5, float('nan')):.6g}, p75={cost_desc.get(0.75, float('nan')):.6g}, p90={cost_desc.get(0.9, float('nan')):.6g}"
        )


if __name__ == "__main__":
    main()
