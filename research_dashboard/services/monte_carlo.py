from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import numpy as np
import pandas as pd


def _to_array(trade_pnl_ticks: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(pd.to_numeric(trade_pnl_ticks, errors="coerce"), dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def run_trade_shuffle_mc(
    trade_pnl_ticks: pd.Series | np.ndarray,
    num_paths: int = 2000,
    horizon: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    base = _to_array(trade_pnl_ticks)
    if base.size == 0:
        return np.zeros((0, 0), dtype=float)

    n = base.size if horizon is None else int(horizon)
    rng = np.random.default_rng(seed)
    paths = np.empty((num_paths, n), dtype=float)
    for i in range(num_paths):
        perm = rng.permutation(base)
        if n <= base.size:
            draw = perm[:n]
        else:
            extra = rng.choice(base, size=n - base.size, replace=True)
            draw = np.concatenate([perm, extra])
        paths[i] = np.cumsum(draw)
    return paths


def run_bootstrap_mc(
    trade_pnl_ticks: pd.Series | np.ndarray,
    num_paths: int = 2000,
    horizon: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    base = _to_array(trade_pnl_ticks)
    if base.size == 0:
        return np.zeros((0, 0), dtype=float)

    n = base.size if horizon is None else int(horizon)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, base.size, size=(num_paths, n))
    draws = base[idx]
    return np.cumsum(draws, axis=1)


def compute_drawdowns(equity_paths_ticks: np.ndarray) -> np.ndarray:
    if equity_paths_ticks.size == 0:
        return np.zeros((0,), dtype=float)
    runmax = np.maximum.accumulate(equity_paths_ticks, axis=1)
    dd = equity_paths_ticks - runmax
    return dd.min(axis=1)


def compute_mc_summary(
    equity_paths_ticks: np.ndarray,
    tick_value: float,
    starting_capital: float,
    dd_threshold_currency: float = -800.0,
) -> dict:
    if equity_paths_ticks.size == 0:
        return {
            "n_paths": 0,
            "horizon": 0,
            "prob_end_positive": None,
            "prob_dd_breach": None,
            "final_pnl_p5": None,
            "final_pnl_p50": None,
            "final_pnl_p95": None,
            "max_dd_p5": None,
            "max_dd_p50": None,
            "max_dd_p95": None,
        }

    final_ticks = equity_paths_ticks[:, -1]
    final_pnl = final_ticks * tick_value
    max_dd_ticks = compute_drawdowns(equity_paths_ticks)
    max_dd_currency = max_dd_ticks * tick_value
    ending_capital = starting_capital + final_pnl
    dd_breach = max_dd_currency <= dd_threshold_currency

    return {
        "n_paths": int(equity_paths_ticks.shape[0]),
        "horizon": int(equity_paths_ticks.shape[1]),
        "prob_end_positive": float((final_pnl > 0).mean()),
        "prob_ruin_proxy": float((ending_capital <= 0).mean()),
        "prob_dd_breach": float(dd_breach.mean()),
        "final_pnl_p5": float(np.quantile(final_pnl, 0.05)),
        "final_pnl_p50": float(np.quantile(final_pnl, 0.50)),
        "final_pnl_p95": float(np.quantile(final_pnl, 0.95)),
        "max_dd_p5": float(np.quantile(max_dd_currency, 0.05)),
        "max_dd_p50": float(np.quantile(max_dd_currency, 0.50)),
        "max_dd_p95": float(np.quantile(max_dd_currency, 0.95)),
    }


def percentile_equity_paths(
    equity_paths_ticks: np.ndarray,
    percentiles: tuple[float, float, float] = (0.05, 0.50, 0.95),
) -> pd.DataFrame:
    if equity_paths_ticks.size == 0:
        return pd.DataFrame(columns=["step", "p05", "p50", "p95"])
    qs = np.quantile(equity_paths_ticks, q=list(percentiles), axis=0)
    return pd.DataFrame(
        {
            "step": np.arange(1, equity_paths_ticks.shape[1] + 1),
            "p05": qs[0],
            "p50": qs[1],
            "p95": qs[2],
        }
    )


def cache_key_for_mc(
    trade_pnl_ticks: pd.Series | np.ndarray,
    mc_type: str,
    num_paths: int,
    horizon: int | None,
    seed: int,
) -> str:
    arr = _to_array(trade_pnl_ticks)
    payload = {
        "mc_type": mc_type,
        "num_paths": int(num_paths),
        "horizon": None if horizon is None else int(horizon),
        "seed": int(seed),
        "arr_head": arr[:200].tolist(),
        "arr_len": int(arr.size),
        "arr_sum": float(arr.sum()) if arr.size else 0.0,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@lru_cache(maxsize=64)
def _run_mc_cached(
    cache_key: str,
    arr_tuple: tuple[float, ...],
    mc_type: str,
    num_paths: int,
    horizon: int | None,
    seed: int,
) -> np.ndarray:
    arr = np.array(arr_tuple, dtype=float)
    if mc_type == "shuffle":
        return run_trade_shuffle_mc(arr, num_paths=num_paths, horizon=horizon, seed=seed)
    return run_bootstrap_mc(arr, num_paths=num_paths, horizon=horizon, seed=seed)


def run_mc_cached(
    trade_pnl_ticks: pd.Series | np.ndarray,
    mc_type: str,
    num_paths: int,
    horizon: int | None,
    seed: int = 42,
) -> np.ndarray:
    arr = _to_array(trade_pnl_ticks)
    key = cache_key_for_mc(arr, mc_type, num_paths, horizon, seed)
    return _run_mc_cached(key, tuple(arr.tolist()), mc_type, int(num_paths), horizon, int(seed))
