"""
Experiment agent for EntryAlpha families with strict guardrails.

Runs a coarse grid of configs (max N runs), collects artifacts, computes
train/lock metrics, and enforces disqualifiers.

Guardrails (defaults can be overridden by env):
  - Train/lock split fixed inside 2025-12-01..2025-12-18
  - Min trades thresholds
  - Daily loss + trailing DD disqualifiers
  - Fixed cost model (no changes during runs)
  - Fixed exits during entry development
"""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "run_lrams_gate_backtest.py"


@dataclass(frozen=True)
class Guardrails:
    train_start: str = "2025-12-01"
    train_end: str = "2025-12-12"
    lock_start: str = "2025-12-13"
    lock_end: str = "2025-12-18"
    min_trades_train: int = 120
    min_trades_lock: int = 50
    daily_loss_limit_ticks: float = 80.0
    trailing_dd_limit_ticks: float = 160.0


@dataclass(frozen=True)
class RunConfig:
    label: str
    flow_win_bars: int
    spread_max: int
    dmid_abs_max_ticks: int
    lag_score_min: float
    proof_max_bars: int
    proof_ticks: int
    confirm_min_abs_ofid: int
    confirm_min_flow_sum: int


def _parse_lag_score_steps(env_val: str | None) -> Dict[str, float]:
    if not env_val:
        return {"low": 5.0, "mid": 10.0, "high": 20.0}
    out: Dict[str, float] = {}
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    if not out:
        return {"low": 5.0, "mid": 10.0, "high": 20.0}
    return out


def _discover_days(data_dir: Path, instrument: str) -> List[str]:
    root = data_dir / f"instrument={instrument}"
    if not root.exists():
        return []
    days = []
    for day_dir in root.iterdir():
        if not day_dir.is_dir():
            continue
        if day_dir.name.startswith("date="):
            day = day_dir.name.replace("date=", "")
        else:
            day = day_dir.name
        if (day_dir / "features_labels.parquet").exists():
            days.append(day)
    return sorted(set(days))


def _select_window(days: List[str], start: str, end: str) -> List[str]:
    return [d for d in days if start <= d <= end]


def _ensure_days(days: List[str], start: str, end: str) -> List[str]:
    if not days:
        raise ValueError(f"No days available in range {start}..{end}")
    return days


def _compute_cost_ticks() -> float:
    tick_value = float(os.environ.get("PNL_TICK_VALUE", "12.5"))
    commission_rt = float(os.environ.get("PNL_COMMISSION_ROUND_TURN", "1.2"))
    slippage_ticks = float(os.environ.get("PNL_SLIPPAGE_TICKS", "1"))
    commission_ticks = commission_rt / tick_value if tick_value > 0 else 0.0
    return float(slippage_ticks) + float(commission_ticks)


def _find_latest_run_dir(output_dir: Path) -> Path:
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")
    run_dirs = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run_*/ dirs in {output_dir}")
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime)[-1]


def _find_first(path: Path, pattern: str) -> Path | None:
    matches = list(path.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _load_trades(run_dir: Path) -> pd.DataFrame:
    trades_path = _find_first(run_dir, "trades_entry_alpha_v1_W*_gate_*_baseline_vs_gated.csv")
    if trades_path is None:
        raise FileNotFoundError(f"Missing trades csv in {run_dir}")
    df = pd.read_csv(trades_path)
    if "strategy" in df.columns:
        df = df[df["strategy"] == "gated"].copy()
    return df


def _parse_date(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    return ts.dt.date.astype(str)


def _compute_mfe_rates(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty or "mfe_ticks" not in trades.columns:
        return {"pct_mfe_ge_1": 0.0, "pct_mfe_ge_4": 0.0}
    mfe = pd.to_numeric(trades["mfe_ticks"], errors="coerce")
    return {
        "pct_mfe_ge_1": float((mfe >= 1.0).mean()),
        "pct_mfe_ge_4": float((mfe >= 4.0).mean()),
    }


def _compute_expectancy(trades: pd.DataFrame, cost_ticks: float) -> Tuple[float, float, float, float, float]:
    if trades.empty or "pnl_ticks" not in trades.columns:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    pnl = pd.to_numeric(trades["pnl_ticks"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    net = pnl - float(cost_ticks)
    wins = net[net > 0]
    losses = net[net < 0]
    expectancy = float(net.mean()) if net.size else 0.0
    win_rate = float((net > 0).mean()) if net.size else 0.0
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0
    net_ticks = float(net.sum()) if net.size else 0.0
    return expectancy, win_rate, avg_win, avg_loss, net_ticks


def _equity_curve_from_trades(trades: pd.DataFrame, cost_ticks: float) -> Tuple[pd.Series, pd.Series]:
    if trades.empty:
        return pd.Series(dtype=float), pd.Series(dtype=str)
    df = trades.copy()
    if "exit_time" in df.columns and df["exit_time"].notna().any():
        df["event_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    else:
        df["event_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.sort_values("event_time")
    pnl = pd.to_numeric(df["pnl_ticks"], errors="coerce").fillna(0.0) - float(cost_ticks)
    equity = pnl.cumsum()
    dates = _parse_date(df["event_time"])
    return equity.reset_index(drop=True), dates.reset_index(drop=True)


def _daily_risk_stats(trades: pd.DataFrame, cost_ticks: float, guard: Guardrails) -> Dict[str, float]:
    equity, dates = _equity_curve_from_trades(trades, cost_ticks)
    if equity.empty:
        return {
            "daily_loss_breaches": 0,
            "trailing_dd_breaches": 0,
            "worst_day_ticks": 0.0,
            "max_dd_ticks": 0.0,
            "min_dd_buffer": 0.0,
        }
    df = pd.DataFrame({"equity": equity, "date": dates})
    day_groups = df.groupby("date", sort=False)

    daily_loss_breaches = 0
    worst_day_ticks = 0.0
    max_dd_ticks = 0.0

    eod_equity = 0.0
    eod_high = 0.0
    min_dd_buffer = float("inf")
    trailing_dd_breaches = 0

    for day, g in day_groups:
        day_start = eod_equity
        day_equity = day_start + g["equity"] - g["equity"].iloc[0]
        day_min = float(day_equity.min())
        day_end = float(day_equity.iloc[-1])
        day_net = day_end - day_start
        worst_day_ticks = min(worst_day_ticks, day_net)
        daily_loss = day_min - day_start
        if daily_loss <= -guard.daily_loss_limit_ticks:
            daily_loss_breaches += 1

        eod_high = max(eod_high, day_start)
        dd_floor = eod_high - guard.trailing_dd_limit_ticks
        day_min_dd = float(day_equity.min())
        max_dd_ticks = max(max_dd_ticks, float(eod_high - day_min_dd))
        min_dd_buffer = min(min_dd_buffer, float(day_min_dd - dd_floor))
        if day_min_dd <= dd_floor:
            trailing_dd_breaches += 1

        eod_equity = day_end
        eod_high = max(eod_high, eod_equity)

    if min_dd_buffer == float("inf"):
        min_dd_buffer = 0.0
    return {
        "daily_loss_breaches": daily_loss_breaches,
        "trailing_dd_breaches": trailing_dd_breaches,
        "worst_day_ticks": float(worst_day_ticks),
        "max_dd_ticks": float(max_dd_ticks),
        "min_dd_buffer": float(min_dd_buffer),
    }


def _parse_log_counts(log_path: Path) -> Dict[str, int]:
    counts = {
        "pflft_candidates": 0,
        "blocked_spread": 0,
        "blocked_depth": 0,
        "blocked_dmid": 0,
        "blocked_lag": 0,
        "confirm_failed": 0,
        "proof_failed": 0,
        "not_armed": 0,
        "candidates": 0,
    }
    if not log_path.exists():
        return counts
    pflft_re = re.compile(
        r"PFLFT_STATS .* candidates=(\d+) blocked_spread=(\d+) blocked_depth=(\d+) blocked_dmid=(\d+) blocked_lag=(\d+)"
    )
    confirm_re = re.compile(r"PFLFT_CONFIRM .* checked=(\d+) passed=(\d+) failed=(\d+) proof_failed=(\d+)")
    excl_re = re.compile(r"EA_BLOCK_REASON_EXCL .* not_armed=(\d+) .* candidates=(\d+) ")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pflft_re.search(line)
        if m:
            counts["pflft_candidates"] += int(m.group(1))
            counts["blocked_spread"] += int(m.group(2))
            counts["blocked_depth"] += int(m.group(3))
            counts["blocked_dmid"] += int(m.group(4))
            counts["blocked_lag"] += int(m.group(5))
        m = confirm_re.search(line)
        if m:
            counts["confirm_failed"] += int(m.group(3))
            counts["proof_failed"] += int(m.group(4))
        m = excl_re.search(line)
        if m:
            counts["not_armed"] += int(m.group(1))
            counts["candidates"] += int(m.group(2))
    return counts


def _run_backtest(env: Dict[str, str], output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{label}.log"
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, str(BACKTEST_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Backtest failed for {label}, see {log_path}")
    return log_path


def _build_grid(steps: Dict[str, float], max_runs: int) -> List[RunConfig]:
    flow_win = [10, 20, 40]
    spread_max = [1, 2]
    dmid_abs = [1, 2]
    lag_steps = [steps["low"], steps["mid"], steps["high"]]
    proof_bars = [0, 3, 5]
    proof_ticks = [1, 2]
    confirm_abs = [5, 8, 12]
    confirm_flow = [0, 5, 8]
    configs = []
    for idx, combo in enumerate(
        itertools.product(
            flow_win, spread_max, dmid_abs, lag_steps, proof_bars, proof_ticks, confirm_abs, confirm_flow
        )
    ):
        cfg = RunConfig(
            label=f"run_{idx+1:03d}",
            flow_win_bars=combo[0],
            spread_max=combo[1],
            dmid_abs_max_ticks=combo[2],
            lag_score_min=combo[3],
            proof_max_bars=combo[4],
            proof_ticks=combo[5],
            confirm_min_abs_ofid=combo[6],
            confirm_min_flow_sum=combo[7],
        )
        configs.append(cfg)
        if len(configs) >= max_runs:
            break
    return configs


def _split_trades(trades: pd.DataFrame, train_days: List[str], lock_days: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades, trades
    ts_col = "entry_time" if "entry_time" in trades.columns else "exit_time"
    dates = _parse_date(trades[ts_col])
    trades = trades.copy()
    trades["date"] = dates
    train = trades[trades["date"].isin(train_days)].copy()
    lock = trades[trades["date"].isin(lock_days)].copy()
    return train, lock


def main() -> int:
    guard = Guardrails(
        train_start=os.environ.get("TRAIN_START", "2025-12-01"),
        train_end=os.environ.get("TRAIN_END", "2025-12-12"),
        lock_start=os.environ.get("LOCK_START", "2025-12-13"),
        lock_end=os.environ.get("LOCK_END", "2025-12-18"),
        min_trades_train=int(os.environ.get("MIN_TRADES_TRAIN", "120")),
        min_trades_lock=int(os.environ.get("MIN_TRADES_LOCK", "50")),
        daily_loss_limit_ticks=float(os.environ.get("DAILY_LOSS_LIMIT_TICKS", "80")),
        trailing_dd_limit_ticks=float(os.environ.get("TRAILING_DD_LIMIT_TICKS", "160")),
    )
    data_dir = Path(os.environ.get("DATA_DIR", "data/processed")).expanduser().resolve()
    instrument = os.environ.get("INSTRUMENT", "ES").strip()
    days = _discover_days(data_dir, instrument)
    if not days:
        raise ValueError(f"No processed days found for {instrument} in {data_dir}")
    full_days = _select_window(days, guard.train_start, guard.lock_end)
    full_days = _ensure_days(full_days, guard.train_start, guard.lock_end)
    train_days = _select_window(full_days, guard.train_start, guard.train_end)
    lock_days = _select_window(full_days, guard.lock_start, guard.lock_end)
    if not train_days or not lock_days:
        raise ValueError("Train/lock split has no days; check available data.")

    lag_steps = _parse_lag_score_steps(os.environ.get("PFLFT_LAG_SCORE_STEPS"))
    max_runs = int(os.environ.get("MAX_RUNS", "8"))
    configs = _build_grid(lag_steps, max_runs)

    output_root = Path(os.environ.get("AGENT_OUTPUT_DIR", "artifacts/entry_alpha_agent")).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cost_ticks = _compute_cost_ticks()
    run_rows: List[Dict[str, object]] = []

    for cfg in configs:
        run_env = os.environ.copy()
        run_env["STRATEGY_MODE"] = "entry_alpha_v1"
        run_env["ENTRY_ALPHA_FAMILY_ALLOWLIST"] = run_env.get("ENTRY_ALPHA_FAMILY_ALLOWLIST", "PFLFT_v1")
        run_env["TEST_DAYS"] = ",".join(full_days)
        run_env["VALIDATE_DEBUG"] = "1"
        run_env["PFLFT_FLOW_WIN_BARS"] = str(cfg.flow_win_bars)
        run_env["PFLFT_SPREAD_MAX"] = str(cfg.spread_max)
        run_env["PFLFT_DMID_ABS_MAX_TICKS"] = str(cfg.dmid_abs_max_ticks)
        run_env["PFLFT_LAG_SCORE_MIN"] = str(cfg.lag_score_min)
        run_env["PFLFT_PROOF_MAX_BARS"] = str(cfg.proof_max_bars)
        run_env["PFLFT_PROOF_TICKS"] = str(cfg.proof_ticks)
        run_env["PFLFT_CONFIRM_MIN_ABS_OFID"] = str(cfg.confirm_min_abs_ofid)
        run_env["PFLFT_CONFIRM_MIN_FLOW_SUM"] = str(cfg.confirm_min_flow_sum)

        run_dir_root = output_root / cfg.label
        log_path = _run_backtest(run_env, run_dir_root, cfg.label)
        run_dir = _find_latest_run_dir(run_dir_root)

        trades = _load_trades(run_dir)
        train_trades, lock_trades = _split_trades(trades, train_days, lock_days)

        train_exp, train_win_rate, train_avg_win, train_avg_loss, train_net = _compute_expectancy(train_trades, cost_ticks)
        lock_exp, lock_win_rate, lock_avg_win, lock_avg_loss, lock_net = _compute_expectancy(lock_trades, cost_ticks)

        train_mfe = _compute_mfe_rates(train_trades)
        lock_mfe = _compute_mfe_rates(lock_trades)

        train_risk = _daily_risk_stats(train_trades, cost_ticks, guard)
        lock_risk = _daily_risk_stats(lock_trades, cost_ticks, guard)

        log_counts = _parse_log_counts(log_path)
        candidates = max(log_counts.get("candidates", 0), log_counts.get("pflft_candidates", 0))
        denom = float(candidates) if candidates else 1.0
        blocked_pct = {
            "pct_block_not_armed": float(log_counts.get("not_armed", 0) / denom),
            "pct_block_spread": float(log_counts.get("blocked_spread", 0) / denom),
            "pct_block_depth": float(log_counts.get("blocked_depth", 0) / denom),
            "pct_block_dmid": float(log_counts.get("blocked_dmid", 0) / denom),
            "pct_block_lag": float(log_counts.get("blocked_lag", 0) / denom),
            "pct_block_confirm_failed": float(log_counts.get("confirm_failed", 0) / denom),
            "pct_block_proof_failed": float(log_counts.get("proof_failed", 0) / denom),
        }

        train_starve = len(train_trades) < guard.min_trades_train
        lock_starve = len(lock_trades) < guard.min_trades_lock
        starvation = train_starve or lock_starve
        disqualified = (
            lock_risk["daily_loss_breaches"] > 0 or lock_risk["trailing_dd_breaches"] > 0
        )

        good = (
            lock_exp > 0
            and not starvation
            and lock_risk["daily_loss_breaches"] == 0
            and lock_risk["trailing_dd_breaches"] == 0
        )
        status = "GOOD" if good else "OK"
        if starvation:
            status = "TRADE_STARVATION"
        if disqualified:
            status = "DISQUALIFIED"

        run_rows.append(
            {
                "label": cfg.label,
                "run_dir": str(run_dir),
                "trades_train": int(len(train_trades)),
                "trades_lock": int(len(lock_trades)),
                "train_expectancy_ticks": train_exp,
                "lock_expectancy_ticks": lock_exp,
                "train_net_ticks": train_net,
                "lock_net_ticks": lock_net,
                "lock_win_rate": lock_win_rate,
                "lock_avg_win_ticks": lock_avg_win,
                "lock_avg_loss_ticks": lock_avg_loss,
                "lock_pct_mfe_ge_1": lock_mfe["pct_mfe_ge_1"],
                "lock_pct_mfe_ge_4": lock_mfe["pct_mfe_ge_4"],
                "lock_worst_day_ticks": lock_risk["worst_day_ticks"],
                "lock_max_dd_ticks": lock_risk["max_dd_ticks"],
                "lock_min_dd_buffer": lock_risk["min_dd_buffer"],
                "daily_loss_breaches": lock_risk["daily_loss_breaches"],
                "trailing_dd_breaches": lock_risk["trailing_dd_breaches"],
                "status": status,
                **blocked_pct,
            }
        )

    df = pd.DataFrame(run_rows)
    if not df.empty:
        df = df.sort_values(
            ["status", "lock_expectancy_ticks", "lock_net_ticks", "lock_min_dd_buffer", "lock_worst_day_ticks"],
            ascending=[True, False, False, False, False],
        )
    out_csv = output_root / "comparison_table.csv"
    df.to_csv(out_csv, index=False)

    print("Experiment complete.")
    print(f"Wrote comparison table: {out_csv}")
    if not df.empty:
        print(df.to_string(index=False))

    manifest = {
        "guardrails": guard.__dict__,
        "days": {
            "all": full_days,
            "train": train_days,
            "lock": lock_days,
        },
        "cost_ticks": cost_ticks,
        "runs": [r.__dict__ for r in configs],
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
