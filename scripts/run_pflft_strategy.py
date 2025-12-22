"""
PFLFT fixed-rule strategy test (Phases 1-3).

This script freezes the premise knobs and evaluates a dumb, non-ML strategy:
  - Gate: PFLFT quantile thresholds (fixed)
  - Entry: immediate on gate
  - Exit: fixed horizon (H=10, H=20)
  - No overlap, one unit, cost model = spread/2 + sweep_cost
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_loader import load_all_raw_data

# Phase 1: frozen research settings (do not tune here)
HORIZONS = (10, 20)  # steps (for 100ms cadence -> 1s, 2s)
REPRICE_BPS = 4.0
FLOW_WINDOW = 20
STEP_MS = 100
Q_SPREAD = 0.3
Q_SWEEP = 0.3
Q_LAG = 0.7
Q_FLOW = 0.7
Q_DMID = 0.5
Q_DEPTH = 0.1
MID_MIN = 1000.0
MID_MAX = 10000.0
MAX_GAP_MULT = 2.0
MAX_STEP_BPS = 10.0


def _infer_step_ms(times: pd.Series, fallback: int) -> int:
    dt = times.to_series().diff().dropna()
    if dt.empty:
        return fallback
    ms = int(dt.dt.total_seconds().median() * 1000)
    return ms if ms > 0 else fallback


def _add_pflft_features(df: pd.DataFrame, step_ms: int, window: int) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(level=0)
    if "signed_volume" not in df.columns:
        if "trade_volume" in df.columns:
            df["signed_volume"] = pd.to_numeric(df["trade_volume"], errors="coerce").fillna(0.0)
        else:
            df["signed_volume"] = 0.0
    df["abs_signed_volume"] = df["signed_volume"].abs()
    if {"signed_volume", "trade_volume"}.issubset(df.columns):
        denom = pd.to_numeric(df["trade_volume"], errors="coerce").replace(0.0, np.nan)
        df["trade_imbalance_ratio"] = df["signed_volume"] / denom
    if "trade_volume" in df.columns:
        per_sec = 1000.0 / max(step_ms, 1)
        df["flow_intensity"] = pd.to_numeric(df["trade_volume"], errors="coerce").fillna(0.0) * per_sec
        df["flow_intensity_roll"] = grouped["flow_intensity"].transform(lambda s: s.rolling(window, min_periods=1).mean())
        df["flow_accel"] = grouped["flow_intensity_roll"].transform(lambda s: s.diff())
    df["signed_volume_roll"] = grouped["signed_volume"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    prev = grouped["mid"].transform(lambda s: s.shift(1))
    df["dmid_bps"] = ((df["mid"] - prev) / prev) * 1e4
    prev_w = grouped["mid"].transform(lambda s: s.shift(window))
    df["dmid_bps_roll"] = ((df["mid"] - prev_w) / prev_w) * 1e4
    eps = 1e-9
    df["impact_per_flow"] = df["dmid_bps"] / (df["abs_signed_volume"] + eps)
    df["lag_score"] = df["flow_intensity"].fillna(0.0) / (df["dmid_bps"].abs() + eps)
    if {"depth_bid_top5", "depth_ask_top5"}.issubset(df.columns):
        df["depth_total_top5"] = df["depth_bid_top5"] + df["depth_ask_top5"]
    return df


def _quantile(series: pd.Series, q: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


def _cost_bps(df: pd.DataFrame, sweep_cost_mag: pd.Series | None) -> pd.Series:
    spread_bps = (df["spread"] / df["mid"]) * 1e4
    if sweep_cost_mag is None:
        sweep_bps = pd.Series(0.0, index=df.index)
    else:
        sweep_bps = (sweep_cost_mag.reindex(df.index) * 1e4).fillna(0.0)
    return spread_bps / 2 + sweep_bps


def _build_gate(df: pd.DataFrame, sweep_cost_mag: pd.Series | None) -> pd.Series:
    spread_bps = (df["spread"] / df["mid"]) * 1e4
    sweep_bps = (sweep_cost_mag * 1e4) if sweep_cost_mag is not None else pd.Series(0.0, index=df.index)
    spread_cut = _quantile(spread_bps, Q_SPREAD)
    sweep_cut = _quantile(sweep_bps, Q_SWEEP)
    lag_cut = _quantile(df["lag_score"], Q_LAG)
    flow_cut = _quantile(df["flow_intensity_roll"], Q_FLOW)
    dmid_cut = _quantile(df["dmid_bps_roll"].abs(), Q_DMID)
    depth_cut = _quantile(df["depth_total_top5"], Q_DEPTH)

    gate = (
        (spread_bps <= spread_cut)
        & (sweep_bps <= sweep_cut)
        & (df["lag_score"] >= lag_cut)
        & (df["flow_intensity_roll"] >= flow_cut)
        & (df["dmid_bps_roll"].abs() <= dmid_cut)
        & (df["depth_total_top5"] >= depth_cut)
    )
    print(
        f"Gate thresholds: spread={spread_cut:.6g}, sweep={sweep_cut:.6g}, "
        f"lag={lag_cut:.6g}, flow={flow_cut:.6g}, dmid={dmid_cut:.6g}, depth={depth_cut:.6g}"
    )
    print(f"Gate rate: {gate.mean():.2%}")
    return gate


def _simulate_fixed_horizon(
    df: pd.DataFrame,
    gate: pd.Series,
    cost_bps: pd.Series,
    horizon: int,
    debug_trades: int = 0,
    max_gap_mult: float = MAX_GAP_MULT,
    max_step_bps: float = MAX_STEP_BPS,
    exit_mode: str = "fixed",
    reprice_bps: float = REPRICE_BPS,
) -> pd.DataFrame:
    trades: List[Dict[str, object]] = []
    debug_count = 0
    for symbol, group in df.groupby(level=0):
        for day, g in group.groupby(group.index.get_level_values(1).date):
            g_gate = gate.loc[g.index]
            g_cost = cost_bps.loc[g.index]
            times = g.index.get_level_values(1)
            mid = pd.to_numeric(g["mid"], errors="coerce").to_numpy()
            step_bps_arr = pd.to_numeric(g["dmid_bps"], errors="coerce").to_numpy()
            direction = np.sign(pd.to_numeric(g["signed_volume_roll"], errors="coerce").fillna(0.0).to_numpy())
            expected_gap = (horizon * STEP_MS) / 1000.0
            next_ok = 0
            for i in range(len(g) - horizon):
                if i < next_ok:
                    continue
                if not bool(g_gate.iloc[i]):
                    continue
                if direction[i] == 0:
                    continue
                if max_step_bps > 0:
                    step_bps = step_bps_arr[i]
                    if not np.isfinite(step_bps) or abs(step_bps) > max_step_bps:
                        continue
                    window_steps = step_bps_arr[i + 1 : i + horizon + 1]
                    if np.any(~np.isfinite(window_steps)) or np.any(np.abs(window_steps) > max_step_bps):
                        continue
                gap = (times[i + horizon] - times[i]).total_seconds()
                if gap > expected_gap * max_gap_mult:
                    continue
                entry = mid[i]
                exit_px = mid[i + horizon]
                exit_idx = i + horizon
                exit_reason = "timeout"
                tp_mid = float("nan")
                if exit_mode == "reprice":
                    thresh = reprice_bps / 1e4
                    if direction[i] > 0:
                        tp_mid = entry * (1.0 + thresh)
                    else:
                        tp_mid = entry * (1.0 - thresh)
                    for j in range(i + 1, i + horizon + 1):
                        if not np.isfinite(mid[j]):
                            continue
                        if direction[i] > 0 and mid[j] >= tp_mid:
                            exit_px = mid[j]
                            exit_idx = j
                            exit_reason = "tp_hit"
                            break
                        if direction[i] < 0 and mid[j] <= tp_mid:
                            exit_px = mid[j]
                            exit_idx = j
                            exit_reason = "tp_hit"
                            break
                if not np.isfinite(entry) or not np.isfinite(exit_px) or entry == 0:
                    continue
                gross_bps = direction[i] * ((exit_px - entry) / entry) * 1e4
                pnl_bps = gross_bps - float(g_cost.iloc[i])
                trades.append(
                    {
                        "symbol": symbol,
                        "time": times[i],
                        "horizon": horizon,
                        "pnl_bps": pnl_bps,
                        "gross_bps": gross_bps,
                        "cost_bps": float(g_cost.iloc[i]),
                        "entry_mid": entry,
                        "exit_mid": exit_px,
                        "exit_reason": exit_reason,
                        "tp_mid": tp_mid,
                        "tp_bps": reprice_bps,
                    }
                )
                if debug_trades > 0 and debug_count < debug_trades:
                    print(
                        f"debug H={horizon} {symbol} {day} entry={times[i]} exit={times[i + horizon]} "
                        f"exit_used={times[exit_idx]} mode={exit_mode} tp_mid={(tp_mid if exit_mode == 'reprice' else float('nan')):.6g} "
                        f"entry_mid={entry:.6g} exit_mid={exit_px:.6g} "
                        f"dir={direction[i]:.0f} signed_vol_roll={float(g['signed_volume_roll'].iloc[i]):.6g} "
                        f"gross_bps={gross_bps:.6g} cost_bps={float(g_cost.iloc[i]):.6g} "
                        f"pnl_bps={pnl_bps:.6g}"
                    )
                    debug_count += 1
                next_ok = i + horizon
    return pd.DataFrame(trades)


def _summarize(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades.")
        return
    trades = trades.sort_values("time")
    pnl = trades["pnl_bps"]
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    dd_min = float(drawdown.min())
    by_day = trades.groupby(trades["time"].dt.date).size()
    print(
        f"trades={len(trades)}, ev_mean={pnl.mean():.6g}, ev_median={pnl.median():.6g}, "
        f"p75={pnl.quantile(0.75):.6g}, p90={pnl.quantile(0.9):.6g}, "
        f"win_rate={(pnl>0).mean():.3f}, max_drawdown_bps={dd_min:.6g}"
    )
    print(
        f"trades/day: min={by_day.min()}, median={by_day.median():.1f}, max={by_day.max()}"
    )
    if "exit_reason" in trades.columns:
        counts = trades["exit_reason"].value_counts().to_dict()
        print(f"exit_reason counts: {counts}")
        by_reason = trades.groupby("exit_reason")["pnl_bps"]
        for reason, pnl_series in by_reason:
            print(
                f"exit_reason={reason}: n={len(pnl_series)}, "
                f"ev_mean={pnl_series.mean():.6g}, ev_median={pnl_series.median():.6g}, "
                f"p75={pnl_series.quantile(0.75):.6g}, p90={pnl_series.quantile(0.9):.6g}"
            )


def main():
    default_dirs = os.environ.get("TRAIN_DATA_DIRS", "data/processed")
    dir_list = [Path(d.strip()) for d in default_dirs.split(",") if d.strip()]

    frames = []
    for d in dir_list:
        if d.exists():
            print(f"Loading data from {d} ...")
            frames.append(load_all_raw_data(d, max_rows_per_file=None))
        else:
            print(f"Skipping missing data dir: {d}")
    if not frames:
        raise FileNotFoundError("No training data directories contained usable files.")
    df = pd.concat(frames).sort_index()

    # Mid filters to avoid corrupted price scales.
    df = df[(df["mid"] >= MID_MIN) & (df["mid"] <= MID_MAX)]
    if df.empty:
        raise ValueError("No rows left after mid filtering; check MID_MIN/MAX.")

    step_ms = _infer_step_ms(df.index.get_level_values(1), STEP_MS)
    df = _add_pflft_features(df, step_ms=step_ms, window=FLOW_WINDOW)

    sweep_cost_mag = None
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df.columns):
        sweep_cost_mag = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)

    gate = _build_gate(df, sweep_cost_mag)
    cost_bps = _cost_bps(df, sweep_cost_mag)
    debug_trades = int(os.environ.get("PFLFT_DEBUG_TRADES", "0"))
    exit_mode = os.environ.get("PFLFT_EXIT_MODE", "fixed").strip().lower()
    max_step_bps = float(os.environ.get("PFLFT_MAX_STEP_BPS", str(MAX_STEP_BPS)))
    tp_list_env = os.environ.get("PFLFT_TP_LIST", "").strip()
    if tp_list_env:
        tp_list = [float(x.strip()) for x in tp_list_env.split(",") if x.strip()]
    else:
        tp_list = [REPRICE_BPS]

    for h in HORIZONS:
        for tp_bps in tp_list:
            label = f"H={h}"
            if exit_mode == "reprice":
                label = f"H={h}, TP={tp_bps}bps"
            print(f"\nStrategy results ({label})")
            trades = _simulate_fixed_horizon(
                df,
                gate,
                cost_bps,
                horizon=h,
                debug_trades=debug_trades,
                max_step_bps=max_step_bps,
                exit_mode=exit_mode,
                reprice_bps=tp_bps,
            )
            _summarize(trades)


if __name__ == "__main__":
    main()
