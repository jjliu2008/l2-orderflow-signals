"""
Trade-level Monte Carlo robustness analysis.

This script resamples realized trade outcomes (pnl_ticks) for a single strategy
configuration and estimates the distribution of expectancy, total PnL, and max
drawdown under repeated paths.

Example:
  python scripts/run_trade_monte_carlo.py \
    --trades artifacts/.../trades_entry_alpha_v1_W10_gate_side_matched_baseline_vs_gated.csv \
    --strategy gated \
    --paths 5000 \
    --block-size 12 \
    --dd-limit 160

Cost resolution order:
  1) --cost-ticks (manual override)
  2) --config cost_mode + fixed_env (default config: configs/experiment_agent.json)
  3) environment fallback (PNL_* vars, round_trip mode)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _default_cost_ticks() -> float:
    tick_value = float(os.environ.get("PNL_TICK_VALUE", "12.5"))
    commission_rt = float(os.environ.get("PNL_COMMISSION_ROUND_TURN", "1.2"))
    slippage_ticks = float(os.environ.get("PNL_SLIPPAGE_TICKS", "1"))
    return _cost_ticks_from_parts(
        tick_value=tick_value,
        commission_rt=commission_rt,
        slippage_ticks=slippage_ticks,
        cost_mode="round_trip",
    )


def _cost_ticks_from_parts(
    tick_value: float,
    commission_rt: float,
    slippage_ticks: float,
    cost_mode: str,
) -> float:
    commission_ticks = commission_rt / tick_value if tick_value > 0 else 0.0
    mode = str(cost_mode).strip().lower()
    if mode == "none":
        return 0.0
    if mode == "commission_only":
        return float(commission_ticks)
    # default/fallback: round_trip
    return float(slippage_ticks) + float(commission_ticks)


def _cost_ticks_from_config(config_path: Path, cost_mode_override: str | None = None) -> Tuple[float, str]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    fixed_env = cfg.get("fixed_env", {})
    if not isinstance(fixed_env, dict):
        fixed_env = {}
    tick_value = float(fixed_env.get("PNL_TICK_VALUE", os.environ.get("PNL_TICK_VALUE", "12.5")))
    commission_rt = float(
        fixed_env.get("PNL_COMMISSION_ROUND_TURN", os.environ.get("PNL_COMMISSION_ROUND_TURN", "1.2"))
    )
    slippage_ticks = float(fixed_env.get("PNL_SLIPPAGE_TICKS", os.environ.get("PNL_SLIPPAGE_TICKS", "1")))
    mode = (
        str(cost_mode_override).strip().lower()
        if cost_mode_override and str(cost_mode_override).strip().lower() != "auto"
        else str(cfg.get("cost_mode", "round_trip")).strip().lower()
    )
    return (
        _cost_ticks_from_parts(
            tick_value=tick_value,
            commission_rt=commission_rt,
            slippage_ticks=slippage_ticks,
            cost_mode=mode,
        ),
        mode,
    )


def _resolve_cost_ticks(
    cost_ticks_arg: float | None,
    config_path_arg: str,
    cost_mode_arg: str,
) -> Tuple[float, str, str]:
    if cost_ticks_arg is not None:
        return float(cost_ticks_arg), "manual", "manual"
    config_path = Path(config_path_arg).expanduser().resolve() if str(config_path_arg).strip() else None
    if config_path and config_path.exists():
        ticks, mode = _cost_ticks_from_config(config_path, cost_mode_override=cost_mode_arg)
        return float(ticks), "config", mode
    mode = str(cost_mode_arg).strip().lower()
    if mode in {"auto", ""}:
        mode = "round_trip"
    tick_value = float(os.environ.get("PNL_TICK_VALUE", "12.5"))
    commission_rt = float(os.environ.get("PNL_COMMISSION_ROUND_TURN", "1.2"))
    slippage_ticks = float(os.environ.get("PNL_SLIPPAGE_TICKS", "1"))
    ticks = _cost_ticks_from_parts(
        tick_value=tick_value,
        commission_rt=commission_rt,
        slippage_ticks=slippage_ticks,
        cost_mode=mode,
    )
    return float(ticks), "env", mode


def _max_drawdown_ticks(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(equity)
    drawdown = running_peak - equity
    return float(drawdown.max()) if drawdown.size else 0.0


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def _sample_iid_paths(net_ticks: np.ndarray, n_paths: int, horizon_trades: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, net_ticks.size, size=(n_paths, horizon_trades))
    return net_ticks[idx]


def _sample_block_paths(
    net_ticks: np.ndarray,
    n_paths: int,
    horizon_trades: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_size <= 1:
        return _sample_iid_paths(net_ticks, n_paths, horizon_trades, rng)
    n = net_ticks.size
    steps = int(math.ceil(horizon_trades / block_size))
    starts = rng.integers(0, n, size=(n_paths, steps))
    offsets = np.arange(block_size, dtype=int)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    sampled = net_ticks[idx.reshape(n_paths, -1)]
    return sampled[:, :horizon_trades]


def _path_metrics(sampled_paths: np.ndarray) -> Dict[str, np.ndarray]:
    if sampled_paths.size == 0:
        empty = np.array([], dtype=float)
        return {"totals": empty, "ev": empty, "max_dd": empty}
    totals = sampled_paths.sum(axis=1)
    ev = totals / sampled_paths.shape[1]
    max_dd = np.zeros(sampled_paths.shape[0], dtype=float)
    for i in range(sampled_paths.shape[0]):
        max_dd[i] = _max_drawdown_ticks(sampled_paths[i])
    return {"totals": totals, "ev": ev, "max_dd": max_dd}


def _summarize_paths(metrics: Dict[str, np.ndarray], dd_limit: float | None) -> Dict[str, object]:
    totals = metrics["totals"]
    ev = metrics["ev"]
    max_dd = metrics["max_dd"]
    out: Dict[str, object] = {
        "paths": int(totals.size),
        "mean_total_ticks": float(totals.mean()) if totals.size else 0.0,
        "mean_ev_ticks_per_trade": float(ev.mean()) if ev.size else 0.0,
        "mean_max_dd_ticks": float(max_dd.mean()) if max_dd.size else 0.0,
        "total_ticks": _quantiles(totals),
        "ev_ticks_per_trade": _quantiles(ev),
        "max_dd_ticks": _quantiles(max_dd),
        "prob_total_positive": float((totals > 0).mean()) if totals.size else 0.0,
        "prob_ev_positive": float((ev > 0).mean()) if ev.size else 0.0,
    }
    if dd_limit is not None:
        out["prob_max_dd_breach"] = float((max_dd > dd_limit).mean()) if max_dd.size else 0.0
    return out


def _load_net_ticks(
    trades_path: Path,
    strategy: str,
    pnl_col: str,
    cost_ticks: float,
) -> np.ndarray:
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades file not found: {trades_path}")
    df = pd.read_csv(trades_path)
    if strategy and "strategy" in df.columns:
        df = df[df["strategy"].astype(str).str.lower() == strategy.lower()].copy()
    if pnl_col not in df.columns:
        raise ValueError(f"{pnl_col} column not found in {trades_path}")
    if "exit_time" in df.columns:
        sort_col = "exit_time"
    elif "entry_time" in df.columns:
        sort_col = "entry_time"
    else:
        sort_col = None
    if sort_col:
        df["_t"] = pd.to_datetime(df[sort_col], errors="coerce")
        df = df.sort_values("_t").drop(columns=["_t"])
    pnl_ticks = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    net_ticks = pnl_ticks - float(cost_ticks)
    net_ticks = net_ticks[np.isfinite(net_ticks)]
    return net_ticks


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trade-level Monte Carlo for one entry/config.")
    parser.add_argument("--trades", required=True, help="Path to trades CSV (baseline_vs_gated).")
    parser.add_argument(
        "--config",
        default="configs/experiment_agent.json",
        help="Optional experiment config path used to resolve cost model (default: configs/experiment_agent.json).",
    )
    parser.add_argument("--strategy", default="gated", help="Strategy filter if strategy column exists.")
    parser.add_argument("--pnl-col", default="pnl_ticks", help="PnL column name.")
    parser.add_argument(
        "--cost-ticks",
        type=float,
        default=None,
        help="Per-trade cost in ticks. If omitted, auto-resolve from --config cost_mode + fixed_env.",
    )
    parser.add_argument(
        "--cost-mode",
        default="auto",
        choices=["auto", "none", "commission_only", "round_trip"],
        help="Optional cost mode override when --cost-ticks is omitted.",
    )
    parser.add_argument("--paths", type=int, default=5000, help="Number of simulated paths.")
    parser.add_argument("--horizon-trades", type=int, default=0, help="Trades per simulated path (0=use sample size).")
    parser.add_argument("--block-size", type=int, default=12, help="Block size for block bootstrap.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dd-limit", type=float, default=None, help="Optional DD limit for breach probability.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    trades_path = Path(args.trades).expanduser().resolve()
    rng = np.random.default_rng(int(args.seed))
    resolved_cost_ticks, cost_source, cost_mode = _resolve_cost_ticks(
        cost_ticks_arg=args.cost_ticks,
        config_path_arg=str(args.config or ""),
        cost_mode_arg=str(args.cost_mode or "auto"),
    )

    net_ticks = _load_net_ticks(
        trades_path=trades_path,
        strategy=str(args.strategy),
        pnl_col=str(args.pnl_col),
        cost_ticks=float(resolved_cost_ticks),
    )
    if net_ticks.size == 0:
        raise ValueError("No trades available after filtering.")

    horizon = int(args.horizon_trades) if int(args.horizon_trades) > 0 else int(net_ticks.size)
    n_paths = max(1, int(args.paths))
    block_size = max(1, int(args.block_size))
    dd_limit = None if args.dd_limit is None else float(args.dd_limit)

    observed = {
        "n_trades": int(net_ticks.size),
        "total_net_ticks": float(net_ticks.sum()),
        "ev_ticks_per_trade": float(net_ticks.mean()),
        "win_rate": float((net_ticks > 0).mean()),
        "max_dd_ticks": float(_max_drawdown_ticks(net_ticks)),
    }

    iid_paths = _sample_iid_paths(net_ticks, n_paths=n_paths, horizon_trades=horizon, rng=rng)
    block_paths = _sample_block_paths(
        net_ticks,
        n_paths=n_paths,
        horizon_trades=horizon,
        block_size=block_size,
        rng=rng,
    )
    iid_summary = _summarize_paths(_path_metrics(iid_paths), dd_limit=dd_limit)
    block_summary = _summarize_paths(_path_metrics(block_paths), dd_limit=dd_limit)

    result = {
        "input": {
            "trades_path": str(trades_path),
            "config_path": str(Path(args.config).expanduser().resolve()) if str(args.config).strip() else "",
            "strategy": str(args.strategy),
            "pnl_col": str(args.pnl_col),
            "cost_ticks": float(resolved_cost_ticks),
            "cost_source": cost_source,
            "cost_mode": cost_mode,
            "paths": n_paths,
            "horizon_trades": horizon,
            "block_size": block_size,
            "seed": int(args.seed),
            "dd_limit": dd_limit,
        },
        "observed": observed,
        "monte_carlo": {
            "iid_bootstrap": iid_summary,
            "block_bootstrap": block_summary,
        },
    }

    print(json.dumps(result, indent=2))
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
