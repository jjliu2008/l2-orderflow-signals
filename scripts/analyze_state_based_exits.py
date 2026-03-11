"""
State-based exit analysis for frozen ES cluster branch.

Goal:
Compare the current 30-minute fixed exit against thesis-invalidation exits:
1) fail-fast invalidation (no early progress + adverse flow/book re-accel)
2) optional volatility-scaled stop
3) optional trailing giveback after sufficient favorable excursion

Usage:
  python scripts/analyze_state_based_exits.py
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts/signal_research"
TRADES_CSV = OUT_DIR / "es_cluster_signedvol_gate_validation_20260310_trades.csv"

sys.path.append(str(PROJECT_ROOT / "scripts"))
import run_es_exhaustion_suite as suite  # noqa: E402


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    fail_fast_enabled: bool = False
    fail_window_sec: int = 300
    min_progress_ticks: float = 1.0
    adverse_flow30_thresh: float = 300.0
    adverse_obi_thresh: float = 0.05
    vol_stop_enabled: bool = False
    vol_stop_mult: float = 2.5
    vol_stop_min_ticks: float = 2.0
    vol_stop_max_ticks: float = 8.0
    trail_enabled: bool = False
    trail_arm_mfe_ticks: float = 12.0
    trail_giveback_ticks: float = 6.0


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _policy_set() -> list[ExitPolicy]:
    return [
        ExitPolicy(name="hold_30m"),
        ExitPolicy(
            name="failfast_5m",
            fail_fast_enabled=True,
            fail_window_sec=300,
            min_progress_ticks=1.0,
            adverse_flow30_thresh=300.0,
            adverse_obi_thresh=0.05,
        ),
        ExitPolicy(
            name="failfast_3m",
            fail_fast_enabled=True,
            fail_window_sec=180,
            min_progress_ticks=1.0,
            adverse_flow30_thresh=200.0,
            adverse_obi_thresh=0.03,
        ),
        ExitPolicy(
            name="failfast_5m_volstop",
            fail_fast_enabled=True,
            fail_window_sec=300,
            min_progress_ticks=1.0,
            adverse_flow30_thresh=300.0,
            adverse_obi_thresh=0.05,
            vol_stop_enabled=True,
            vol_stop_mult=2.5,
            vol_stop_min_ticks=2.0,
            vol_stop_max_ticks=8.0,
        ),
        ExitPolicy(
            name="failfast_5m_trail",
            fail_fast_enabled=True,
            fail_window_sec=300,
            min_progress_ticks=1.0,
            adverse_flow30_thresh=300.0,
            adverse_obi_thresh=0.05,
            trail_enabled=True,
            trail_arm_mfe_ticks=12.0,
            trail_giveback_ticks=6.0,
        ),
    ]


def _vol_stop_ticks(entry_row: pd.Series, tick_size: float, policy: ExitPolicy) -> float:
    rv = _safe_float(entry_row.get("rv_5m", np.nan))
    mid = _safe_float(entry_row.get("mid", np.nan))
    if rv is None or mid is None or rv <= 0.0 or mid <= 0.0:
        return float(policy.vol_stop_min_ticks)
    # Convert pct-vol to rough tick-vol.
    vol_ticks = (rv * mid) / tick_size
    stop = policy.vol_stop_mult * vol_ticks
    stop = max(policy.vol_stop_min_ticks, min(policy.vol_stop_max_ticks, stop))
    return float(stop)


def _simulate_trade_exit(
    market: pd.DataFrame,
    trade_row: pd.Series,
    policy: ExitPolicy,
    tick_size: float,
) -> dict[str, Any] | None:
    try:
        entry_idx = int(trade_row["entry_idx_used"])
        natural_exit_idx = int(trade_row["exit_idx_used"])
        side = int(trade_row["side"])
    except Exception:
        return None

    if entry_idx < 0 or natural_exit_idx <= entry_idx or natural_exit_idx >= len(market):
        return None
    if str(market.iloc[entry_idx]["date_et"]) != str(trade_row["date_et"]):
        return None

    path = market.iloc[entry_idx : natural_exit_idx + 1].copy()
    if path.empty or len(path) < 2:
        return None

    entry_fill = suite._entry_fill(path.iloc[0], side)
    mid0 = float(path.iloc[0]["mid"])

    signed_volume = pd.to_numeric(path.get("signed_volume", 0.0), errors="coerce").fillna(0.0)
    flow30 = signed_volume.rolling(30, min_periods=5).sum().to_numpy()
    obi = pd.to_numeric(path.get("order_book_imbalance", 0.0), errors="coerce").fillna(0.0).to_numpy()
    mid = pd.to_numeric(path["mid"], errors="coerce").to_numpy()

    fail_window_bars = max(1, int(policy.fail_window_sec))
    stop_ticks = _vol_stop_ticks(path.iloc[0], tick_size=tick_size, policy=policy) if policy.vol_stop_enabled else None

    mfe_mid = 0.0
    chosen_rel_idx = len(path) - 1
    exit_reason = "TIME_30M"

    for j in range(1, len(path)):
        row = path.iloc[j]
        # Use executable fill for realized pnl.
        exit_fill = suite._exit_fill(row, side)
        pnl_ticks = side * (exit_fill - entry_fill) / tick_size
        # Use mid for state tracking.
        mid_pnl_ticks = side * (mid[j] - mid0) / tick_size
        mfe_mid = max(mfe_mid, mid_pnl_ticks)

        # 1) Thesis invalidation: no progress + adverse flow/book re-accel.
        if policy.fail_fast_enabled and j <= fail_window_bars:
            no_progress = mfe_mid < policy.min_progress_ticks
            aligned_flow30 = side * float(flow30[j])  # adverse if negative
            aligned_obi = side * float(obi[j])  # adverse if negative
            adverse_flow = aligned_flow30 < -policy.adverse_flow30_thresh
            adverse_obi = aligned_obi < -policy.adverse_obi_thresh
            if no_progress and adverse_flow and adverse_obi:
                chosen_rel_idx = j
                exit_reason = "FAIL_FAST"
                break

        # 2) Volatility-scaled risk boundary.
        if stop_ticks is not None and pnl_ticks <= -stop_ticks:
            chosen_rel_idx = j
            exit_reason = "VOL_STOP"
            break

        # 3) Trail after enough favorable excursion.
        if policy.trail_enabled and mfe_mid >= policy.trail_arm_mfe_ticks:
            giveback = mfe_mid - mid_pnl_ticks
            if giveback >= policy.trail_giveback_ticks:
                chosen_rel_idx = j
                exit_reason = "TRAIL_GIVEBACK"
                break

    chosen_idx = entry_idx + chosen_rel_idx
    exit_fill = suite._exit_fill(market.iloc[chosen_idx], side)
    pnl_gross_ticks = side * (exit_fill - entry_fill) / tick_size
    hold_secs = int(chosen_rel_idx)

    out = {
        "variant": trade_row["variant"],
        "policy": policy.name,
        "date_et": trade_row["date_et"],
        "month_et": trade_row["month_et"],
        "entry_session_bucket": trade_row["entry_session_bucket"],
        "is_event_day": bool(trade_row["is_event_day"]),
        "side": side,
        "entry_idx_used": entry_idx,
        "exit_idx_used_policy": chosen_idx,
        "exit_reason_policy": exit_reason,
        "hold_seconds_policy": hold_secs,
        "pnl_gross_ticks_policy": float(pnl_gross_ticks),
        "pnl_gross_ticks_orig": float(trade_row["pnl_gross_ticks"]),
    }
    return out


def _summarize(df: pd.DataFrame, commission_ticks: float) -> dict[str, Any]:
    if df.empty:
        return {
            "n": 0,
            "ev_gross_ticks": None,
            "ev_net_ticks": None,
            "win_rate_net": None,
            "avg_win_net_ticks": None,
            "avg_loss_net_ticks": None,
            "profit_factor_net": None,
            "top1_share_net": None,
            "top3_share_net": None,
            "top5_share_net": None,
            "exit_reason_counts": {},
        }

    net = pd.to_numeric(df["pnl_gross_ticks_policy"], errors="coerce").fillna(0.0) - commission_ticks
    wins = net[net > 0]
    losses = net[net <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    s = net.sort_values(ascending=False).reset_index(drop=True)
    total = float(s.sum())

    def share(n: int) -> float | None:
        if total <= 0:
            return None
        return float(s.head(n).sum() / total)

    return {
        "n": int(len(df)),
        "ev_gross_ticks": _safe_float(pd.to_numeric(df["pnl_gross_ticks_policy"], errors="coerce").mean()),
        "ev_net_ticks": _safe_float(net.mean()),
        "win_rate_net": _safe_float((net > 0).mean()),
        "avg_win_net_ticks": _safe_float(wins.mean()) if len(wins) else None,
        "avg_loss_net_ticks": _safe_float(losses.mean()) if len(losses) else None,
        "profit_factor_net": _safe_float(pf) if pf is not None else None,
        "top1_share_net": share(1),
        "top3_share_net": share(3),
        "top5_share_net": share(5),
        "exit_reason_counts": df["exit_reason_policy"].value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze state-based exits for frozen ES cluster branch.")
    ap.add_argument("--trades-csv", default=str(TRADES_CSV))
    ap.add_argument("--date-glob", default="date=2026-0[12]-*")
    ap.add_argument("--step-ms", type=int, default=1000)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--commission-ticks", type=float, default=0.36)
    ap.add_argument("--variants", default="baseline_no_sv_gate,sv_le_250")
    ap.add_argument("--output-json", default=str(OUT_DIR / "state_exit_analysis_20260310.json"))
    ap.add_argument("--output-csv", default=str(OUT_DIR / "state_exit_analysis_20260310_details.csv"))
    args = ap.parse_args()

    trades_path = Path(args.trades_csv)
    if not trades_path.exists():
        raise SystemExit(f"Trades CSV not found: {trades_path}")
    trades = pd.read_csv(trades_path)
    keep_variants = {v.strip() for v in str(args.variants).split(",") if v.strip()}
    trades = trades[trades["variant"].isin(keep_variants)].copy()
    if trades.empty:
        raise SystemExit(f"No trades left after variant filter: {keep_variants}")

    market = suite._load_resampled_data(
        root=PROJECT_ROOT / "data/processed_esh6/instrument=ES",
        date_glob=args.date_glob,
        step_ms=int(args.step_ms),
    )
    market = suite._add_features(market, tick_size=float(args.tick_size), step_ms=int(args.step_ms))

    policies = _policy_set()
    sim_rows: list[dict[str, Any]] = []
    for _, tr in trades.iterrows():
        for pol in policies:
            out = _simulate_trade_exit(
                market=market,
                trade_row=tr,
                policy=pol,
                tick_size=float(args.tick_size),
            )
            if out is not None:
                sim_rows.append(out)

    sim = pd.DataFrame(sim_rows)
    if sim.empty:
        raise SystemExit("Simulation produced no rows.")

    summary_rows: list[dict[str, Any]] = []
    for (variant, policy), g in sim.groupby(["variant", "policy"], sort=False):
        row = {"variant": variant, "policy": policy}
        row.update(_summarize(g, commission_ticks=float(args.commission_ticks)))
        # Month split.
        for month, gg in g.groupby("month_et", sort=True):
            mstats = _summarize(gg, commission_ticks=float(args.commission_ticks))
            row[f"ev_net_{month}"] = mstats["ev_net_ticks"]
            row[f"n_{month}"] = mstats["n"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["variant", "policy"]).reset_index(drop=True)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "trades_csv": str(trades_path),
            "date_glob": args.date_glob,
            "step_ms": int(args.step_ms),
            "tick_size": float(args.tick_size),
            "commission_ticks": float(args.commission_ticks),
            "variants": sorted(keep_variants),
            "policies": [p.__dict__ for p in policies],
        },
        "summary": summary.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sim.to_csv(out_csv, index=False)

    print(f"Wrote summary JSON: {out_json}")
    print(f"Wrote detailed rows: {out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

