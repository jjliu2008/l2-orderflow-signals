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


def main():
    default_dirs = os.environ.get("TRAIN_DATA_DIRS", "data/processed")
    dir_list = [Path(d.strip()) for d in default_dirs.split(",") if d.strip()]

    horizons_env = os.environ.get("HORIZON_SWEEP_HORIZONS", "10,20,50,100")
    horizons: List[int] = [int(h.strip()) for h in horizons_env.split(",") if h.strip()]

    sweep_q = float(os.environ.get("HORIZON_SWEEP_QUANTILE", "0.8"))
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
    if sweep_cost_mag is None:
        raise ValueError("Sweep cost columns missing; cannot define sweep_high.")

    sweep_cutoff = None
    if sweep_q > 0:
        sweep_cutoff = float(sweep_cost_mag.quantile(sweep_q))
    if sweep_cutoff is None or sweep_cutoff == 0:
        raise ValueError("Sweep cutoff collapsed to zero; set a nonzero HORIZON_SWEEP_QUANTILE.")
    if sweep_min_floor > 0:
        sweep_cutoff = max(sweep_cutoff, sweep_min_floor)
    if sweep_max_cap > 0:
        sweep_cutoff = min(sweep_cutoff, sweep_max_cap)

    sweep_high = sweep_cost_mag >= sweep_cutoff
    n_sweep = sweep_high.sum()
    total = len(df)
    print(f"sweep_high cutoff={sweep_cutoff:.6g}, count={n_sweep} / {total} ({n_sweep/total:.2%})")
    if n_sweep == 0:
        raise ValueError("No sweep_high rows after cutoff; aborting sweep.")

    # Cost proxy
    # cost proxy in relative terms
    spread_rel = df["spread"] / df["mid"]
    cost_proxy = spread_rel / 2 + sweep_cost_mag

    for h in horizons:
        rng = compute_expected_range(df["mid"], horizon=h)
        sub = pd.DataFrame({
            "range": rng,
            "cost_proxy": cost_proxy,
            "sweep_high": sweep_high,
        }).dropna()
        sub_high = sub[sub["sweep_high"]]
        if sub_high.empty:
            print(f"H={h}: no sweep_high rows with range; skipping")
            continue
        ev_net = sub_high["range"] - sub_high["cost_proxy"]
        print(
            f"H={h}: sweep_high rows={len(sub_high)}, "
            f"range p50={sub_high['range'].median():.6g}, p75={sub_high['range'].quantile(0.75):.6g}, "
            f"p90={sub_high['range'].quantile(0.9):.6g}, "
            f"ev_net p50={ev_net.median():.6g}, p75={ev_net.quantile(0.75):.6g}, p90={ev_net.quantile(0.9):.6g}"
        )


if __name__ == "__main__":
    main()
