"""
Experiment agent CLI with hard guardrails for EntryAlpha strategy development.

Usage:
  python scripts/experiment_agent.py --config configs/experiment_agent.json --outdir artifacts/experiment_agent/<run_id>

Exports:
  - run_experiments(config_path, outdir, dry_run=False, max_runs=None)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "run_lrams_gate_backtest.py"


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        proc = subprocess.run(
            ["powershell", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return "ProcessName" in proc.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock(outdir: Path) -> Path:
    lock_path = outdir / ".experiment_agent.lock"
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", 0))
        except Exception:
            pid = 0
        if _pid_running(pid):
            raise RuntimeError(
                f"experiment_agent already running (pid={pid}). Remove {lock_path} if stale."
            )
    outdir.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "started_at": time.time()}, indent=2),
        encoding="utf-8",
    )
    return lock_path

DATA_RANGE_KEYS = {
    "TEST_DAYS",
    "START_DATE",
    "END_DATE",
    "DAYS_BACK",
    "USE_ALL_AVAILABLE_DAYS",
    "EXCLUDE_DATES",
    "QUICK_EXCLUDE_MAJOR",
}

COST_KEYS = {
    "PNL_TICK_VALUE",
    "PNL_COMMISSION_ROUND_TURN",
    "PNL_SLIPPAGE_TICKS",
}

EXIT_KEYS = {
    "ENTRY_ALPHA_MAX_SL_TICKS",
    "ENTRY_ALPHA_TIME_STOP_BARS",
    "ENTRY_ALPHA_TP_TICKS",
    "ENTRY_ALPHA_TP_TICKS_PBRA",
    "PFLFT_STOP_TICKS",
    "PFLFT_TP_TICKS",
    "PFLFT_TIME_STOP_BARS",
    "PBRA_SCRATCH_BARS",
    "PBRA_SCRATCH_MIN_MFE_TICKS",
    "PBRA_SCRATCH_MIN_ABS_OFID_MAX",
    "PBRA_RUNNER_ARM_BARS",
    "PBRA_RUNNER_ARM_MFE_TICKS",
    "PBRA_RUNNER_TP_TICKS",
    "PBRA_RUNNER_BE_STOP_TICKS",
}

RESULT_COLUMNS = [
    "name",
    "run_dir",
    "stdout_log",
    "exit_code",
    "trades_train",
    "trades_lock",
    "train_expectancy_ticks",
    "lock_expectancy_ticks",
    "lock_net_ticks",
    "lock_win_rate",
    "lock_avg_win_ticks",
    "lock_avg_loss_ticks",
    "lock_pct_mfe_ge_1",
    "lock_pct_mfe_ge_4",
    "lock_worst_day_ticks",
    "lock_max_dd_ticks",
    "daily_loss_breaches",
    "trailing_dd_breaches",
    "min_dd_buffer",
    "status",
    "missing_artifacts",
    "failure_reason",
]

@dataclass(frozen=True)
class Guardrails:
    train_start: str
    train_end: str
    lock_start: str
    lock_end: str
    min_trades_train: int
    min_trades_lock: int
    daily_loss_limit_ticks: float
    trailing_dd_limit_ticks: float


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


def _parse_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_guardrails(config: Dict[str, object]) -> Guardrails:
    data_days = config.get("data_days", {})
    train_start = data_days.get("train_start")
    train_end = data_days.get("train_end")
    lock_start = data_days.get("lock_start")
    lock_end = data_days.get("lock_end")
    if (train_start, train_end, lock_start, lock_end) != (
        "2025-12-01",
        "2025-12-12",
        "2025-12-13",
        "2025-12-18",
    ):
        raise ValueError("Train/lock split must remain 2025-12-01..12-12 and 2025-12-13..12-18.")

    min_trades = config.get("min_trades", {})
    risk_limits = config.get("risk_limits", {})
    return Guardrails(
        train_start=str(train_start),
        train_end=str(train_end),
        lock_start=str(lock_start),
        lock_end=str(lock_end),
        min_trades_train=int(min_trades.get("train", 120)),
        min_trades_lock=int(min_trades.get("lock", 50)),
        daily_loss_limit_ticks=float(risk_limits.get("daily_loss_limit_ticks", 80)),
        trailing_dd_limit_ticks=float(risk_limits.get("trailing_dd_limit_ticks", 160)),
    )


def _ensure_fixed_env(config: Dict[str, object]) -> Dict[str, str]:
    fixed_env = config.get("fixed_env", {})
    if not isinstance(fixed_env, dict):
        raise ValueError("fixed_env must be a dict")
    fixed_env = {str(k): str(v) for k, v in fixed_env.items()}
    for key in DATA_RANGE_KEYS:
        if key in fixed_env:
            raise ValueError(f"fixed_env may not set data range key: {key}")
    return fixed_env


def _check_grid_env(grid_env: Dict[str, str], fixed_env: Dict[str, str]) -> None:
    for key in DATA_RANGE_KEYS:
        if key in grid_env:
            raise ValueError(f"Grid env may not set data range key: {key}")
    for key in COST_KEYS | EXIT_KEYS:
        if key in grid_env:
            raise ValueError(f"Grid env may not override locked key: {key}")
    for key in ("STRATEGY_MODE", "ENTRY_ALPHA_FAMILY_ALLOWLIST"):
        if key in grid_env:
            raise ValueError(f"Grid env may not override fixed strategy key: {key}")


def _compute_cost_ticks(env: Dict[str, str]) -> float:
    for key in COST_KEYS:
        if key not in env and key not in os.environ:
            raise ValueError(f"Cost key {key} must be set in fixed_env or environment.")
    tick_value = float(env.get("PNL_TICK_VALUE", os.environ.get("PNL_TICK_VALUE", "12.5")))
    commission_rt = float(env.get("PNL_COMMISSION_ROUND_TURN", os.environ.get("PNL_COMMISSION_ROUND_TURN", "1.2")))
    slippage_ticks = float(env.get("PNL_SLIPPAGE_TICKS", os.environ.get("PNL_SLIPPAGE_TICKS", "1")))
    commission_ticks = commission_rt / tick_value if tick_value > 0 else 0.0
    return float(slippage_ticks) + float(commission_ticks)


def _order_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    df = trades.copy()
    if "exit_time" in df.columns and df["exit_time"].notna().any():
        df["event_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    else:
        df["event_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    return df.sort_values("event_time")


def _equity_curve(trades: pd.DataFrame, cost_ticks: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["event_time", "date", "equity"])
    df = _order_trades(trades)
    pnl = pd.to_numeric(df["pnl_ticks"], errors="coerce").fillna(0.0) - float(cost_ticks)
    equity = pnl.cumsum()
    df = df.assign(equity=equity)
    df["date"] = pd.to_datetime(df["event_time"], errors="coerce").dt.date.astype(str)
    return df[["event_time", "date", "equity"]]


def _compute_metrics(trades: pd.DataFrame, cost_ticks: float) -> Dict[str, float]:
    if trades.empty:
        return {
            "expectancy": 0.0,
            "net_ticks": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "pct_mfe_ge_1": 0.0,
            "pct_mfe_ge_4": 0.0,
        }
    pnl = pd.to_numeric(trades["pnl_ticks"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    net = pnl - float(cost_ticks)
    wins = net[net > 0]
    losses = net[net < 0]
    mfe = pd.to_numeric(trades.get("mfe_ticks", pd.Series(dtype=float)), errors="coerce")
    return {
        "expectancy": float(net.mean()) if net.size else 0.0,
        "net_ticks": float(net.sum()) if net.size else 0.0,
        "win_rate": float((net > 0).mean()) if net.size else 0.0,
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "pct_mfe_ge_1": float((mfe >= 1.0).mean()) if not mfe.empty else 0.0,
        "pct_mfe_ge_4": float((mfe >= 4.0).mean()) if not mfe.empty else 0.0,
    }


def _empty_result_row(
    name: str,
    status: str,
    *,
    run_dir: str = "",
    stdout_log: str = "",
    exit_code: int | None = None,
    missing_artifacts: str = "",
    failure_reason: str = "",
) -> Dict[str, object]:
    return {
        "name": name,
        "run_dir": run_dir,
        "stdout_log": stdout_log,
        "exit_code": exit_code,
        "trades_train": 0,
        "trades_lock": 0,
        "train_expectancy_ticks": 0.0,
        "lock_expectancy_ticks": 0.0,
        "lock_net_ticks": 0.0,
        "lock_win_rate": 0.0,
        "lock_avg_win_ticks": 0.0,
        "lock_avg_loss_ticks": 0.0,
        "lock_pct_mfe_ge_1": 0.0,
        "lock_pct_mfe_ge_4": 0.0,
        "lock_worst_day_ticks": 0.0,
        "lock_max_dd_ticks": 0.0,
        "daily_loss_breaches": 0,
        "trailing_dd_breaches": 0,
        "min_dd_buffer": 0.0,
        "status": status,
        "missing_artifacts": missing_artifacts,
        "failure_reason": failure_reason,
    }


def _risk_metrics(
    trades: pd.DataFrame,
    cost_ticks: float,
    guard: Guardrails,
) -> Dict[str, float]:
    eq = _equity_curve(trades, cost_ticks)
    if eq.empty:
        return {
            "worst_day_ticks": 0.0,
            "max_dd_ticks": 0.0,
            "daily_loss_breaches": 0,
            "trailing_dd_breaches": 0,
            "min_dd_buffer": 0.0,
        }
    daily_loss_breaches = 0
    trailing_dd_breaches = 0
    worst_day_ticks = 0.0
    max_dd_ticks = 0.0
    min_dd_buffer = float("inf")

    eod_equity = 0.0
    eod_high = 0.0
    for day, g in eq.groupby("date", sort=False):
        day_start = eod_equity
        day_equity = day_start + (g["equity"] - g["equity"].iloc[0])
        day_min = float(day_equity.min())
        day_end = float(day_equity.iloc[-1])
        day_net = day_end - day_start
        worst_day_ticks = min(worst_day_ticks, day_net)
        daily_loss = day_min - day_start
        if daily_loss <= -guard.daily_loss_limit_ticks:
            daily_loss_breaches += 1

        eod_high = max(eod_high, day_start)
        dd_floor = eod_high - guard.trailing_dd_limit_ticks
        max_dd_ticks = max(max_dd_ticks, float(eod_high - day_min))
        min_dd_buffer = min(min_dd_buffer, float(day_min - dd_floor))
        if day_min <= dd_floor:
            trailing_dd_breaches += 1

        eod_equity = day_end
        eod_high = max(eod_high, eod_equity)

    if min_dd_buffer == float("inf"):
        min_dd_buffer = 0.0
    return {
        "worst_day_ticks": float(worst_day_ticks),
        "max_dd_ticks": float(max_dd_ticks),
        "daily_loss_breaches": int(daily_loss_breaches),
        "trailing_dd_breaches": int(trailing_dd_breaches),
        "min_dd_buffer": float(min_dd_buffer),
    }


def _find_latest_run_dir(output_dir: Path) -> Path:
    run_dirs = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run_*/ dirs under {output_dir}")
    return sorted(run_dirs, key=lambda p: p.stat().st_mtime)[-1]


def _load_trades(run_dir: Path) -> pd.DataFrame:
    matches = list(run_dir.glob("trades_entry_alpha_v1_W*_gate_*_baseline_vs_gated.csv"))
    if not matches:
        raise FileNotFoundError(f"Missing trades csv in {run_dir}")
    df = pd.read_csv(matches[0])
    if "strategy" in df.columns:
        df = df[df["strategy"] == "gated"].copy()
    return df


def _check_artifacts(run_dir: Path) -> List[str]:
    required = [
        "summary_aggregate.csv",
        "pnl_day_entry_alpha_v1_W",
        "mfe_thresholds_root_entry_alpha_v1_W",
        "strategy_diagnostics.csv",
        "trades_entry_alpha_v1_W",
    ]
    missing = []
    for req in required:
        if req.endswith(".csv"):
            found = any(p.name == req for p in run_dir.rglob(req))
            if not found:
                missing.append(req)
        else:
            found = any(p.name.startswith(req) for p in run_dir.rglob("*"))
            if not found:
                missing.append(req)
    return missing


def _split_trades(trades: pd.DataFrame, train_days: List[str], lock_days: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades, trades
    ts_col = "entry_time" if "entry_time" in trades.columns else "exit_time"
    dates = pd.to_datetime(trades[ts_col], errors="coerce").dt.date.astype(str)
    trades = trades.copy()
    trades["date"] = dates
    train = trades[trades["date"].isin(train_days)].copy()
    lock = trades[trades["date"].isin(lock_days)].copy()
    return train, lock


def run_experiments(config_path: Path, outdir: Path, dry_run: bool = False, max_runs: int | None = None) -> pd.DataFrame:
    config = _parse_config(config_path)
    lock_path = _acquire_lock(outdir)
    guard = _validate_guardrails(config)

    fixed_env = _ensure_fixed_env(config)
    fixed_env["STRATEGY_MODE"] = str(config.get("strategy_mode", "entry_alpha_v1"))
    family_allowlist = config.get("family_allowlist", [])
    if not family_allowlist:
        raise ValueError("family_allowlist must be provided")
    fixed_env["ENTRY_ALPHA_FAMILY_ALLOWLIST"] = ",".join(map(str, family_allowlist))

    data_dir = Path(os.environ.get("DATA_DIR", "data/processed")).expanduser().resolve()
    instrument = os.environ.get("INSTRUMENT", "ES").strip()
    days = _discover_days(data_dir, instrument)
    days = _select_window(days, guard.train_start, guard.lock_end)
    if not days:
        raise ValueError("No days found in the specified train/lock range.")
    train_days = _select_window(days, guard.train_start, guard.train_end)
    lock_days = _select_window(days, guard.lock_start, guard.lock_end)
    if not train_days or not lock_days:
        raise ValueError("Train or lock days missing; check data availability.")

    cost_ticks = _compute_cost_ticks(fixed_env)

    grid = config.get("grid", [])
    if not isinstance(grid, list) or not grid:
        raise ValueError("grid must be a non-empty list")
    if max_runs is not None:
        grid = grid[: max_runs]

    outdir.mkdir(parents=True, exist_ok=True)
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    try:
        for entry in grid:
            name = entry.get("name")
            env_overrides = entry.get("env", {})
            if not name or not isinstance(env_overrides, dict):
                raise ValueError("Each grid entry must include name and env dict.")
            env_overrides = {str(k): str(v) for k, v in env_overrides.items()}
            _check_grid_env(env_overrides, fixed_env)

            run_env = os.environ.copy()
            run_env.update(fixed_env)
        run_env.update(env_overrides)
        run_env["TEST_DAYS"] = ",".join(days)

        run_root = runs_dir / name
        run_root.mkdir(parents=True, exist_ok=True)
        artifacts_dir = run_root / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        run_env["OUTPUT_DIR"] = str(artifacts_dir)

        (run_root / "env.json").write_text(json.dumps(run_env, indent=2), encoding="utf-8")
        log_path = run_root / "stdout.log"

        if dry_run:
            rows.append(
                _empty_result_row(
                    name,
                    "DRY_RUN",
                    stdout_log=str(log_path),
                    failure_reason="dry_run",
                )
            )
            continue

        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [sys.executable, str(BACKTEST_SCRIPT)],
                cwd=str(PROJECT_ROOT),
                env=run_env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode != 0:
            rows.append(
                _empty_result_row(
                    name,
                    "FAILED",
                    stdout_log=str(log_path),
                    exit_code=int(proc.returncode),
                    failure_reason=f"exit_code:{proc.returncode}",
                )
            )
            continue

        run_dir = _find_latest_run_dir(artifacts_dir)
        missing = _check_artifacts(run_dir)
        trades = _load_trades(run_dir)
        train_trades, lock_trades = _split_trades(trades, train_days, lock_days)

        train_metrics = _compute_metrics(train_trades, cost_ticks)
        lock_metrics = _compute_metrics(lock_trades, cost_ticks)
        lock_risk = _risk_metrics(lock_trades, cost_ticks, guard)

        trade_starve = len(train_trades) < guard.min_trades_train or len(lock_trades) < guard.min_trades_lock
        disqualified = lock_risk["daily_loss_breaches"] > 0 or lock_risk["trailing_dd_breaches"] > 0
        good = (
            lock_metrics["expectancy"] > 0
            and not trade_starve
            and not disqualified
        )
        status = "GOOD" if good else "OK"
        if trade_starve:
            status = "TRADE_STARVATION"
        if disqualified:
            status = "DISQUALIFIED"
        if missing:
            status = "INVALID"

            rows.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "stdout_log": str(log_path),
                "exit_code": int(proc.returncode),
                "trades_train": int(len(train_trades)),
                "trades_lock": int(len(lock_trades)),
                "train_expectancy_ticks": train_metrics["expectancy"],
                "lock_expectancy_ticks": lock_metrics["expectancy"],
                "lock_net_ticks": lock_metrics["net_ticks"],
                "lock_win_rate": lock_metrics["win_rate"],
                "lock_avg_win_ticks": lock_metrics["avg_win"],
                "lock_avg_loss_ticks": lock_metrics["avg_loss"],
                "lock_pct_mfe_ge_1": lock_metrics["pct_mfe_ge_1"],
                "lock_pct_mfe_ge_4": lock_metrics["pct_mfe_ge_4"],
                "lock_worst_day_ticks": lock_risk["worst_day_ticks"],
                "lock_max_dd_ticks": lock_risk["max_dd_ticks"],
                "daily_loss_breaches": lock_risk["daily_loss_breaches"],
                "trailing_dd_breaches": lock_risk["trailing_dd_breaches"],
                "min_dd_buffer": lock_risk["min_dd_buffer"],
                "status": status,
                "missing_artifacts": ";".join(missing),
                "failure_reason": "",
            }
            )
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if not df.empty:
        status_rank = {
            "GOOD": 0,
            "OK": 1,
            "TRADE_STARVATION": 2,
            "DISQUALIFIED": 3,
            "INVALID": 4,
            "FAILED": 5,
            "DRY_RUN": 6,
        }
        df["status_rank"] = df["status"].map(status_rank).fillna(99).astype(int)
        df = df.sort_values(
            ["status_rank", "lock_expectancy_ticks", "lock_net_ticks", "min_dd_buffer", "lock_worst_day_ticks"],
            ascending=[True, False, False, False, False],
        )
        df = df.drop(columns=["status_rank"])
    outdir.mkdir(parents=True, exist_ok=True)
    comparison_path = outdir / "comparison.csv"
    df.to_csv(comparison_path, index=False)

    report_path = outdir / "report.md"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("# Experiment Report\n\n")
        fh.write(f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fh.write(f"Train: {guard.train_start}..{guard.train_end}\n\n")
        fh.write(f"Lock: {guard.lock_start}..{guard.lock_end}\n\n")
        fh.write("## Top Results\n\n")
        if df.empty:
            fh.write("No runs completed.\n")
        else:
            try:
                fh.write(df.head(5).to_markdown(index=False))
            except ImportError:
                fh.write(df.head(5).to_string(index=False))
            fh.write("\n")

    manifest = {
        "run_id": config.get("run_id"),
        "guardrails": guard.__dict__,
        "days": {"all": days, "train": train_days, "lock": lock_days},
        "fixed_env": fixed_env,
        "grid": config.get("grid", []),
        "cost_ticks": cost_ticks,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return df


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config JSON.")
    parser.add_argument("--outdir", required=True, help="Output directory for this run.")
    parser.add_argument("--dry_run", action="store_true", help="Validate only; do not execute.")
    parser.add_argument("--max_runs", type=int, default=None, help="Optional max number of runs.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = _parse_args(argv)
    cfg_path = Path(args.config).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    df = run_experiments(cfg_path, outdir, dry_run=args.dry_run, max_runs=args.max_runs)
    if not df.empty:
        print(df.to_string(index=False))
    print(f"Wrote comparison.csv to {outdir / 'comparison.csv'}")
    print(f"Wrote report.md to {outdir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
