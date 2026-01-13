"""
LRAMS gate backtest: compare baseline vs baseline + LRAMS gate.

Note: aggregate improvement stats are computed over all days, with improved/worsened medians reported separately.

Env vars:
  DATA_DIR                root data dir (default: data/processed)
  OUTPUT_DIR              artifacts output dir (default: artifacts/lrams_gate)
  TEST_DAYS               comma-separated YYYY-MM-DD (optional; wins)
  USE_ALL_AVAILABLE_DAYS  use all processed days on disk (optional)
  DAYS_BACK               most recent N available days (optional)
  START_DATE              inclusive YYYY-MM-DD (optional)
  END_DATE                inclusive YYYY-MM-DD (optional)
  EXCLUDE_DATES           comma-separated YYYY-MM-DD (optional)
  QUICK_EXCLUDE_MAJOR     exclude known major macro dates if present (optional)
  INSTRUMENT              instrument filter (default: ES)
  TICK_SIZE               tick size (default: 0.25)

Baseline:
  LOOKBACK_BARS           lookback L (default: 10)
  ENTRY_THRESHOLD_TICKS   entry threshold in ticks (default: 1)
  HOLD_BARS               holding horizon H (default: 20)
  BASELINE_MODE           flat|not_awful|mmas (default: flat)
  BASELINE_K_BARS         window for not_awful (default: 5)
  STRATEGY_MODE           baseline_flat|micro_momo_v1|impulse_confirm_v1|absorption_failure_v1|absorption_failure_v2|absorption_failure_v3 (default: micro_momo_v1)
  MICRO_K_BARS            micro momentum window (default: 5)
  MICRO_IMPULSE_TICKS     min impulse ticks (default: 1)
  MICRO_FLOW_MIN          min abs flow (default: 0)
  MAX_SPREAD_TICKS_FOR_ENTRY max spread ticks to allow entry (default: 2)
  MMAS_K_BARS             window for MMAS (default: 5)
  MMAS_MIN_DMID_TICKS     min dmid ticks for MMAS (default: 1)
  MMAS_MIN_FLOW_ABS       min abs flow for MMAS (default: 20)
  MMAS_REQUIRE_AGREE      require dmid/flow sign agreement (default: 1)
  DEBUG_FIRST_MMAS        print first MMAS decision (default: 0)
  TRADE_SESSION           all|rth (default: all)
  MIN_SPREAD_TICKS        min spread ticks to allow entry (default: 0)
  ENTRY_COOLDOWN_BARS     bars to wait after exit (default: 10)

Impulse confirm entry (strategy_mode=impulse_confirm_v1):
  IMPULSE_LOOKBACK_BARS   impulse lookback in bars (default: 1)
  IMPULSE_MIN_TICKS       min impulse in ticks (default: 1)
  CONFIRM_BARS            confirmation window bars (default: 2)
  CONFIRM_REQUIRE_NONZERO require nonzero impulse during confirm window (default: 1)
  DEBUG_FIRST_IMPULSE     print first impulse decision (default: 0)
  DEBUG_FLOW_STATS        print per-day signed_volume stats (default: 0)
  DISABLE_GATE            bypass LRAMS gate (default: 0)
  DEBUG_FIRST_AFR         print first AFR candidate decision (default: 0)
  DEBUG_FIRST_AFR2        print first AFR2 candidate decision (default: 0)
  AFR_K_BARS              AFR window in bars (default: 10)
  AFR_MIN_FLOW_ABS        AFR min abs flow in window (default: 40)
  AFR_STALL_TICKS         AFR max |stall| ticks (default: 0)
  AFR_BREAK_TICKS         AFR breakout ticks (default: 1)
  AFR_REQUIRE_FLOW_SIGN   require flow sign aligns with breakout (default: 1)
  AFR_USE_MID_FOR_STALL   use mid for stall calc (default: 1)
  AFR_USE_SIGNED_VOLUME   use signed_volume for flow (default: 1)
  AFR_FT_BARS             AFR v2 follow-through bars (default: 1)
  AFR_FT_MIN_TICKS        AFR v2 follow-through min ticks (default: 0)
  AFR_FT_NO_BACKTRACK     AFR v2 block if price backtracks (default: 1)
  AFR_ENTER_ON            AFR v2 entry timing: break|ft|both (default: break)
  AFR_BREAK_QUALITY_MIN_FLOW_ABS AFR v2 min abs flow for break quality (default: 80)
  AFR_BREAK_QUALITY_MAX_SPREAD_TICKS AFR v2 max spread for break quality (default: 1)
  AFR_SNAPBACK_BARS       AFR v2 snapback lookahead bars (default: 2)
  AFR_SNAPBACK_BAND_TICKS AFR v2 snapback band in ticks (default: 1)
  AFR3_BREAK_MIN_FLOW_ABS AFR v3 min abs flow on break bar (default: 100)
  AFR3_BREAK_MAX_SPREAD_TICKS AFR v3 max spread on break bar (default: 2)
  AFR3_SNAPBACK_CHECK     AFR v3 require no snapback on next bar (default: 1)
  DEBUG_FIRST_AFR3        print first AFR v3 candidate (default: 0)
  AFR_TP_TICKS            override TP ticks for AFR modes (default: unset)
  AFR_SL_TICKS            override SL ticks for AFR modes (default: unset)
  AFR_MAX_HOLD_BARS       override max hold bars for AFR modes (default: unset)
  AFR_BREAKEVEN_AFTER_TICKS move SL to breakeven after MFE ticks (default: unset)
  AFR_ENTER_MODE         break_first|ft_only (default: break_first)
  AFR_FLOW_ALIGN_BARS     flow alignment window (default: 5)
  AFR_MIN_FLOW_ABS_ALIGN  min abs flow for alignment (default: 40)
  AFR_REARM_BAND_TICKS    rearm band around absorption level (default: 1)
  AFR_REARM_MAX_BARS      max bars to keep rearm active (default: 5)
  AFR_REARM_STOP_MAX_BARS allow rearm if stop within bars (default: 3)
  AFR_MOMENTUM_DECAY_BARS consecutive bars of weak flow to exit (default: 3)
  AFR_MOMENTUM_DECAY_MIN_FLOW min aligned flow to avoid decay exit (default: 0)

Gate:
  GATE_LOOKBACK_BARS      event lookback W (default: 10)
  GATE_MODE               side_matched or any_side (default: side_matched)
  GATE_MODE_LIST          comma list; overrides GATE_MODE if set
  SWEEP_W                 comma list of W values (optional)
  WORST_Q                 worst-quantile threshold (default: 0.90)
  TP_TICKS                take profit in ticks (default: 1)
  SL_TICKS                stop loss in ticks (default: 2)

LRAMS event params (match falsification defaults):
  V_MIN                   min abs signed_volume (default: 1)
  V_MAX                   max abs signed_volume (default: 10)
  SPREAD_MIN_T            min spread in ticks (default: 1)
  SPREAD_MAX_T            max spread in ticks (default: 4)
  STABLE_BARS             price stability window N (default: 3)
  REFILL_BARS             response window R (default: 10)
"""

from __future__ import annotations

import json
import argparse
import os
import sys
import time
import datetime
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data


def _forward_roll_min(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if window <= 0 or n == 0:
        return out
    dq: deque[int] = deque()
    for i, val in enumerate(arr):
        while dq and dq[0] <= i - window:
            dq.popleft()
        while dq and arr[dq[-1]] >= val:
            dq.pop()
        dq.append(i)
        if i >= window - 1:
            out[i - window + 1] = arr[dq[0]]
    return out


def _forward_roll_max(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if window <= 0 or n == 0:
        return out
    dq: deque[int] = deque()
    for i, val in enumerate(arr):
        while dq and dq[0] <= i - window:
            dq.popleft()
        while dq and arr[dq[-1]] <= val:
            dq.pop()
        dq.append(i)
        if i >= window - 1:
            out[i - window + 1] = arr[dq[0]]
    return out


def _stable_mask(arr: np.ndarray, n_bars: int) -> np.ndarray:
    n = len(arr)
    if n_bars <= 0:
        return np.zeros(n, dtype=bool)
    fwd_min = _forward_roll_min(arr[1:], n_bars)
    fwd_max = _forward_roll_max(arr[1:], n_bars)
    out = np.zeros(n, dtype=bool)
    if len(fwd_min) > 0:
        pad_min = np.full(n, np.nan, dtype=float)
        pad_max = np.full(n, np.nan, dtype=float)
        pad_min[: len(fwd_min)] = fwd_min
        pad_max[: len(fwd_max)] = fwd_max
        out = (pad_min == arr) & (pad_max == arr)
    return out


def _auc_deficit(pre_depth: float, window: np.ndarray) -> float:
    if not np.isfinite(pre_depth) or pre_depth <= 0:
        return np.nan
    if window.size == 0:
        return np.nan
    deficit = np.maximum(0.0, (pre_depth - window) / pre_depth)
    return float(np.nanmean(deficit))


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _discover_available_days(data_dir: Path, instrument: str) -> List[str]:
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


def _select_days(data_dir: Path, instrument: str) -> List[str]:
    use_all = os.environ.get("USE_ALL_AVAILABLE_DAYS", "0").strip() == "1"
    if use_all:
        return _discover_available_days(data_dir, instrument)

    test_days = os.environ.get("TEST_DAYS", "").strip()
    if test_days:
        days = [d.strip() for d in test_days.split(",") if d.strip()]
        return sorted(set(days))

    days = _discover_available_days(data_dir, instrument)
    if not days:
        return []

    start_date = os.environ.get("START_DATE", "").strip()
    end_date = os.environ.get("END_DATE", "").strip()
    if start_date:
        days = [d for d in days if d >= start_date]
    if end_date:
        days = [d for d in days if d <= end_date]

    exclude_env = os.environ.get("EXCLUDE_DATES", "").strip()
    if exclude_env:
        exclude = {d.strip() for d in exclude_env.split(",") if d.strip()}
        days = [d for d in days if d not in exclude]

    if os.environ.get("QUICK_EXCLUDE_MAJOR", "0").strip() == "1":
        major_dates: List[str] = []
        if major_dates:
            days = [d for d in days if d not in set(major_dates)]
        else:
            print("QUICK_EXCLUDE_MAJOR requested but no major date list configured; skipping.", flush=True)

    days_back = os.environ.get("DAYS_BACK", "").strip()
    if days_back:
        n = int(days_back)
        if n > 0:
            days = days[-n:]
    return sorted(days)


def _load_data() -> pd.DataFrame:
    data_dir = Path(os.environ.get("DATA_DIR", "data/processed")).expanduser().resolve()
    df = load_all_raw_data(data_dir).reset_index()
    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df = df.sort_values(["Symbol", "Time"]).reset_index(drop=True)
    return df


def _build_events(
    df: pd.DataFrame,
    v_min: float,
    v_max: float,
    spread_min_t: int,
    spread_max_t: int,
    stable_bars: int,
    refill_bars: int,
    tick_size: float,
) -> pd.DataFrame:
    required = [
        "Symbol",
        "Time",
        "bid_price_1",
        "ask_price_1",
        "top_bid_depth",
        "top_ask_depth",
        "signed_volume",
        "trade_count",
    ]
    _ensure_columns(df, required)

    df = df.copy()
    df["date"] = df["Time"].dt.date.astype(str)
    records: List[Dict[str, float]] = []
    max_lookahead = max(stable_bars, refill_bars)

    for (symbol, day), group in df.groupby(["Symbol", "date"], sort=False):
        group = group.sort_values("Time")
        idx = group.index.to_numpy()
        if len(idx) <= max_lookahead:
            continue
        bid = pd.to_numeric(group["bid_price_1"], errors="coerce").to_numpy()
        ask = pd.to_numeric(group["ask_price_1"], errors="coerce").to_numpy()
        mid = 0.5 * (bid + ask)
        bid_ticks = np.rint(bid / tick_size).astype(np.int64)
        ask_ticks = np.rint(ask / tick_size).astype(np.int64)
        mid_ticks = np.rint(mid / tick_size).astype(np.int64)
        spread_ticks = np.rint((ask - bid) / tick_size).astype(np.int64)

        agg_trade = (group["trade_count"].to_numpy() > 0) & (group["signed_volume"].to_numpy() != 0)
        vol = np.abs(group["signed_volume"].to_numpy())
        vol_band = (vol >= v_min) & (vol <= v_max)
        spread_band = (spread_ticks >= spread_min_t) & (spread_ticks <= spread_max_t)

        stable_bid = _stable_mask(bid_ticks.astype(float), stable_bars)
        stable_ask = _stable_mask(ask_ticks.astype(float), stable_bars)
        stable_mid = _stable_mask(mid_ticks.astype(float), stable_bars)
        stable = stable_bid & stable_ask & stable_mid

        event_mask = agg_trade & vol_band & spread_band & stable
        top_bid = pd.to_numeric(group["top_bid_depth"], errors="coerce").to_numpy()
        top_ask = pd.to_numeric(group["top_ask_depth"], errors="coerce").to_numpy()
        signed_vol = pd.to_numeric(group["signed_volume"], errors="coerce").to_numpy()

        for i_pos in range(len(group)):
            if i_pos + max_lookahead >= len(group):
                break
            if not bool(event_mask[i_pos]):
                continue
            side = int(np.sign(signed_vol[i_pos]))
            if side == 0:
                continue
            bid0 = float(top_bid[i_pos])
            ask0 = float(top_ask[i_pos])
            if not np.isfinite(bid0) or not np.isfinite(ask0) or bid0 <= 0 or ask0 <= 0:
                continue
            depth_bid = top_bid[i_pos + 1 : i_pos + refill_bars + 1]
            depth_ask = top_ask[i_pos + 1 : i_pos + refill_bars + 1]
            auc_bid = _auc_deficit(bid0, depth_bid)
            auc_ask = _auc_deficit(ask0, depth_ask)
            if side > 0:
                asym = auc_ask - auc_bid
            else:
                asym = auc_bid - auc_ask
            records.append(
                {
                    "Symbol": symbol,
                    "date": day,
                    "event_idx": int(idx[i_pos]),
                    "event_pos": int(i_pos),
                    "Time": group["Time"].iloc[i_pos],
                    "side": "buy" if side > 0 else "sell",
                    "asym": float(asym),
                }
            )

    return pd.DataFrame(records)


def _compute_thresholds(events: pd.DataFrame, worst_q: float) -> pd.DataFrame:
    if events.empty:
        events["asym_threshold"] = np.nan
        return events
    out = []
    for (day, side), g in events.groupby(["date", "side"], sort=False):
        g = g.sort_values("event_pos").copy()
        g["asym_threshold"] = g["asym"].expanding().quantile(worst_q).shift(1)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else events


def _simulate_day(
    df_day: pd.DataFrame,
    events_day: pd.DataFrame,
    tick_size: float,
    lookback_bars: int,
    entry_threshold_ticks: int,
    hold_bars: int,
    tp_ticks: int,
    sl_ticks: int,
    baseline_mode: str,
    baseline_k_bars: int,
    strategy_mode: str,
    micro_k_bars: int,
    micro_impulse_ticks: int,
    micro_flow_min: float,
    impulse_lookback_bars: int,
    impulse_min_ticks: int,
    confirm_bars: int,
    confirm_require_nonzero: bool,
    debug_first_impulse: bool,
    max_spread_ticks_for_entry: int,
    mmas_k_bars: int,
    mmas_min_dmid_ticks: int,
    mmas_min_flow_abs: float,
    mmas_require_agree: bool,
    debug_first_mmas: bool,
    trade_session: str,
    min_spread_ticks: int,
    entry_cooldown_bars: int,
    gate_lookback_bars: int,
    gated: bool,
    disable_gate: bool,
    gate_mode: str,
    afr_k_bars: int,
    afr_min_flow_abs: float,
    afr_stall_ticks: int,
    afr_break_ticks: int,
    afr_require_flow_sign: bool,
    afr_use_mid_for_stall: bool,
    afr_use_signed_volume: bool,
    afr_ft_bars: int,
    afr_ft_min_ticks: int,
    afr_ft_no_backtrack: bool,
    afr_enter_on: str,
    afr_break_quality_min_flow_abs: float,
    afr_break_quality_max_spread_ticks: int,
    afr_snapback_bars: int,
    afr_snapback_band_ticks: int,
    debug_first_afr: bool,
    debug_first_afr2: bool,
    afr3_break_min_flow_abs: float,
    afr3_break_max_spread_ticks: int,
    afr3_snapback_check: bool,
    debug_first_afr3: bool,
    afr_tp_ticks: int | None,
    afr_sl_ticks: int | None,
    afr_max_hold_bars: int | None,
    afr_breakeven_after_ticks: int | None,
    afr_enter_mode: str,
    afr_flow_align_bars: int,
    afr_min_flow_abs_align: float,
    afr_rearm_band_ticks: int,
    afr_rearm_max_bars: int,
    afr_rearm_stop_max_bars: int,
    afr_momentum_decay_bars: int,
    afr_momentum_decay_min_flow: float,
    debug_entry_print: bool,
    debug_entry_limit: int,
    debug_entry_tag: str,
) -> Tuple[object, ...]:
    bid = pd.to_numeric(df_day["bid_price_1"], errors="coerce").to_numpy()
    ask = pd.to_numeric(df_day["ask_price_1"], errors="coerce").to_numpy()
    if "top_bid_depth" in df_day.columns:
        top_bid_depth = pd.to_numeric(df_day["top_bid_depth"], errors="coerce").to_numpy()
    else:
        top_bid_depth = np.full(len(df_day), np.nan, dtype=float)
    if "top_ask_depth" in df_day.columns:
        top_ask_depth = pd.to_numeric(df_day["top_ask_depth"], errors="coerce").to_numpy()
    else:
        top_ask_depth = np.full(len(df_day), np.nan, dtype=float)
    mid = 0.5 * (bid + ask)
    n = len(df_day)
    trades: List[Dict[str, float]] = []
    pnl_ticks_total = 0.0
    skipped = 0
    total_signals = 0
    signals_when_flat = 0
    signals_in_position = 0
    eligible_signals = 0
    blocked_signals = 0
    spread_suppressed = 0
    session_suppressed = 0
    cooldown_suppressed = 0
    entries_taken = 0
    strategy_long_signals = 0
    strategy_short_signals = 0
    entry_impulse_sum = 0.0
    entry_impulse_abs_sum = 0.0
    entry_impulse_sum_long = 0.0
    entry_impulse_sum_short = 0.0
    entry_flow_sum = 0.0
    entry_flow_abs_sum = 0.0
    entry_flow_sum_long = 0.0
    entry_flow_sum_short = 0.0
    entry_count = 0
    entry_count_long = 0
    entry_count_short = 0
    entry_break_sum = 0.0
    entry_break_abs_sum = 0.0
    entry_break_sum_long = 0.0
    entry_break_sum_short = 0.0
    entry_ft_sum = 0.0
    entry_ft_abs_sum = 0.0
    entry_spread_sum = 0.0
    afr_be_armed = 0
    afr_be_triggered = 0
    absorption_level_long = float("nan")
    absorption_level_short = float("nan")
    absorption_bar_long = -1
    absorption_bar_short = -1
    rearm_used_long = False
    rearm_used_short = False
    rearm_active_long = False
    rearm_active_short = False
    rearm_expiry_long = -1
    rearm_expiry_short = -1
    break_bar_long = -1
    break_bar_short = -1
    afr_checked = 0
    afr_absorption_pass = 0
    afr_break_pass = 0
    afr_entered = 0
    afr2_checked = 0
    afr2_absorption_pass = 0
    afr2_break_pass = 0
    afr2_ft_pass = 0
    afr2_entered = 0
    afr2_break_quality_pass = 0
    afr2_snapback_fail = 0
    afr2_break_quality_entries = 0
    afr3_checked = 0
    afr3_absorption_pass = 0
    afr3_break_pass = 0
    afr3_break_quality_pass = 0
    afr3_snapback_pass = 0
    afr3_entered = 0
    mmas_signals_checked = 0
    mmas_passed_filters = 0
    mmas_signaled = 0
    mmas_entered = 0
    impulse_signals_checked = 0
    impulse_passed_threshold = 0
    impulse_passed_confirm = 0
    impulse_entered = 0
    skip_reasons = {
        "gated_blocked": 0,
        "no_recent_event": 0,
        "no_threshold_yet": 0,
        "no_event_in_window": 0,
        "impulse_failed_threshold": 0,
        "impulse_failed_confirm": 0,
    }

    signed_vol = pd.to_numeric(df_day["signed_volume"], errors="coerce").fillna(0.0).to_numpy()
    if afr_use_signed_volume:
        afr_flow_series = signed_vol
    else:
        if "aggressor_count_imbalance" in df_day.columns:
            afr_flow_series = pd.to_numeric(df_day["aggressor_count_imbalance"], errors="coerce").fillna(0.0).to_numpy()
        else:
            afr_flow_series = signed_vol
    spread_ticks = np.rint((ask - bid) / tick_size).astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(signed_vol)])
    if tick_size <= 0:
        raise ValueError("tick_size must be positive for micro_momo_v1.")
    impulse_lb = max(1, int(impulse_lookback_bars))
    impulse_ticks_series = np.full(n, np.nan, dtype=float)
    if n > impulse_lb:
        impulse_ticks_series[impulse_lb:] = (mid[impulse_lb:] - mid[:-impulse_lb]) / tick_size

    def _flow_sum_at(idx: int, bars: int) -> float:
        if bars <= 0:
            return 0.0
        start = max(0, idx - bars + 1)
        if start > idx:
            return 0.0
        return float(np.sum(afr_flow_series[start : idx + 1]))

    def _arm_rearm(direction: str, abs_bar: int) -> None:
        nonlocal rearm_active_long, rearm_active_short, rearm_expiry_long, rearm_expiry_short
        if abs_bar < 0:
            return
        if direction == "long":
            if not rearm_used_long:
                rearm_active_long = True
                rearm_expiry_long = abs_bar + afr_rearm_max_bars
        else:
            if not rearm_used_short:
                rearm_active_short = True
                rearm_expiry_short = abs_bar + afr_rearm_max_bars

    def _print_impulse_debug(
        t_idx: int,
        impulse_val: float,
        spread_val: float,
        cooldown_ok_val: bool,
        gate_allowed_val: bool | None,
        gate_reason_val: str,
        desired_side_val: str | None,
        entry_action_val: str,
    ) -> None:
        window_start = max(0, t_idx - confirm_bars + 1)
        window = impulse_ticks_series[window_start : t_idx + 1]
        window_vals = [float(x) if np.isfinite(x) else float("nan") for x in window]
        window_signs = [float(np.sign(x)) if np.isfinite(x) else float("nan") for x in window]
        print(
            "IMPULSE_DEBUG",
            {
                "t": int(t_idx),
                "impulse_ticks": float(impulse_val),
                "confirm_window": window_vals,
                "confirm_signs": window_signs,
                "spread_ticks": float(spread_val),
                "cooldown_ok": bool(cooldown_ok_val),
                "gate_allowed": gate_allowed_val,
                "gate_reason": gate_reason_val,
                "desired_side": desired_side_val,
                "entry_action": entry_action_val,
            },
            flush=True,
        )

    def _print_afr_debug(
        t_idx: int,
        flow_val: float,
        stall_val: float,
        break_val: float,
        spread_val: float,
        cooldown_ok_val: bool,
        session_ok_val: bool,
        gate_allowed_val: bool | None,
        gate_reason_val: str,
        desired_side_val: str | None,
        absorption_pass_val: bool,
        break_pass_val: bool,
        entry_taken_val: bool,
        tp_ticks_val: int,
        sl_ticks_val: int,
        hold_bars_val: int,
        breakeven_ticks_val: int | None,
        entry_action_val: str,
    ) -> None:
        print(
            "AFR_DEBUG",
            {
                "t": int(t_idx),
                "flow": float(flow_val),
                "stall_ticks": float(stall_val),
                "break_ticks": float(break_val),
                "spread_ticks": float(spread_val),
                "cooldown_ok": bool(cooldown_ok_val),
                "session_ok": bool(session_ok_val),
                "gate_allowed": gate_allowed_val,
                "gate_reason": gate_reason_val,
                "desired_side": desired_side_val,
                "absorption_pass": bool(absorption_pass_val),
                "break_pass": bool(break_pass_val),
                "entry_taken": bool(entry_taken_val),
                "afr_tp_ticks": tp_ticks_val,
                "afr_sl_ticks": sl_ticks_val,
                "afr_max_hold_bars": hold_bars_val,
                "afr_breakeven_after_ticks": breakeven_ticks_val,
                "entry_action": entry_action_val,
            },
            flush=True,
        )

    def _print_afr2_debug(
        t_idx: int,
        tf_idx: int,
        flow_val: float,
        stall_val: float,
        break_val: float,
        ft_val: float,
        spread_val: float,
        cooldown_ok_val: bool,
        session_ok_val: bool,
        gate_allowed_val: bool | None,
        gate_reason_val: str,
        desired_side_val: str | None,
        entry_action_val: str,
    ) -> None:
        print(
            "AFR2_DEBUG",
            {
                "t": int(t_idx),
                "tf": int(tf_idx),
                "flow": float(flow_val),
                "stall_ticks": float(stall_val),
                "break_ticks": float(break_val),
                "ft_progress_ticks": float(ft_val),
                "spread_ticks": float(spread_val),
                "cooldown_ok": bool(cooldown_ok_val),
                "session_ok": bool(session_ok_val),
                "gate_allowed": gate_allowed_val,
                "gate_reason": gate_reason_val,
                "desired_side": desired_side_val,
                "entry_action": entry_action_val,
            },
            flush=True,
        )

    def _print_afr3_debug(
        t_idx: int,
        flow_val: float,
        spread_val: float,
        cooldown_ok_val: bool,
        session_ok_val: bool,
        gate_allowed_val: bool | None,
        gate_reason_val: str,
        desired_side_val: str | None,
        absorption_pass_val: bool,
        break_pass_val: bool,
        break_quality_pass_val: bool,
        snapback_pass_val: bool,
        entry_taken_val: bool,
        entry_action_val: str,
    ) -> None:
        print(
            "AFR3_DEBUG",
            {
                "t": int(t_idx),
                "flow_k": float(flow_val),
                "spread_ticks": float(spread_val),
                "cooldown_ok": bool(cooldown_ok_val),
                "session_ok": bool(session_ok_val),
                "gate_allowed": gate_allowed_val,
                "gate_reason": gate_reason_val,
                "desired_side": desired_side_val,
                "absorption_pass": bool(absorption_pass_val),
                "break_pass": bool(break_pass_val),
                "break_quality_pass": bool(break_quality_pass_val),
                "snapback_pass": bool(snapback_pass_val),
                "entry_taken": bool(entry_taken_val),
                "entry_action": entry_action_val,
            },
            flush=True,
        )
    if trade_session == "rth":
        times = pd.to_datetime(df_day["Time"], utc=True, errors="coerce")
        times_cst = times.dt.tz_convert("America/Chicago")
        session_ok = (times_cst.dt.time >= datetime.time(9, 30)) & (times_cst.dt.time < datetime.time(16, 0))
        session_ok = session_ok.to_numpy()
    else:
        session_ok = np.ones(n, dtype=bool)

    if baseline_mode == "not_awful":
        k = max(1, int(baseline_k_bars))
        def _signal(i: int) -> str | None:
            # Simple order-flow sign: take the side of recent signed volume.
            if i + 1 < k:
                return None
            window_sum = cs[i + 1] - cs[i + 1 - k]
            if window_sum >= 1.0:
                return "long"
            if window_sum <= -1.0:
                return "short"
            return None

    else:

        def _signal(i: int) -> str | None:
            if i < lookback_bars or not np.isfinite(mid[i]) or not np.isfinite(mid[i - lookback_bars]):
                return None
            ret_ticks = (mid[i] - mid[i - lookback_bars]) / tick_size
            if ret_ticks >= entry_threshold_ticks:
                return "long"
            if ret_ticks <= -entry_threshold_ticks:
                return "short"
            return None
    if baseline_mode == "mmas":
        k = max(1, int(mmas_k_bars))
        def _mmas_signal(i: int) -> Tuple[str | None, float, float, bool]:
            if i < k or not np.isfinite(mid[i]) or not np.isfinite(mid[i - k]):
                return None, float("nan"), 0.0, False
            dmid_ticks = (mid[i] - mid[i - k]) / tick_size
            flow = cs[i + 1] - cs[i + 1 - k]
            if abs(dmid_ticks) < mmas_min_dmid_ticks or abs(flow) < mmas_min_flow_abs:
                return None, float(dmid_ticks), float(flow), False
            if mmas_require_agree and np.sign(dmid_ticks) != np.sign(flow):
                return None, float(dmid_ticks), float(flow), False
            if dmid_ticks > 0:
                return "long", float(dmid_ticks), float(flow), True
            if dmid_ticks < 0:
                return "short", float(dmid_ticks), float(flow), True
            return None, float(dmid_ticks), float(flow), False

    if events_day.empty:
        event_pos_all = np.array([], dtype=int)
        event_asym_all = np.array([], dtype=float)
        event_thr_all = np.array([], dtype=float)
        event_pos_buy = event_pos_all
        event_asym_buy = event_asym_all
        event_thr_buy = event_thr_all
        event_pos_sell = event_pos_all
        event_asym_sell = event_asym_all
        event_thr_sell = event_thr_all
    else:
        event_pos_all = events_day["event_pos"].to_numpy(dtype=int)
        event_asym_all = events_day["asym"].to_numpy(dtype=float)
        event_thr_all = events_day["asym_threshold"].to_numpy(dtype=float)
        buy_mask = events_day["side"] == "buy"
        sell_mask = events_day["side"] == "sell"
        event_pos_buy = event_pos_all[buy_mask.to_numpy()]
        event_asym_buy = event_asym_all[buy_mask.to_numpy()]
        event_thr_buy = event_thr_all[buy_mask.to_numpy()]
        event_pos_sell = event_pos_all[sell_mask.to_numpy()]
        event_asym_sell = event_asym_all[sell_mask.to_numpy()]
        event_thr_sell = event_thr_all[sell_mask.to_numpy()]

    in_position = False
    pos: Dict[str, object] = {}
    debug_trigger_printed = False
    debug_mmas_printed = False
    debug_afr3_printed = False
    pnl_bound_printed = 0
    cooldown_until = -1
    i = lookback_bars
    max_i = n - 1
    entry_debug_printed = 0
    while i <= max_i:
        if in_position:
            entry_bar = int(pos["entry_bar"])
            side = str(pos["side"])
            entry_px = float(pos["entry_px"])
            tp_level = float(pos["tp_level"])
            sl_level = float(pos["sl_level"])
            exit_bar = int(pos["exit_bar"])
            exit_px = float("nan")
            pnl_ticks = float("nan")
            exit_reason = str(pos.get("exit_reason", "TIME"))
            sl_ticks_local = int(pos.get("sl_ticks", sl_ticks))
            desired_side = _signal(i)
            if desired_side is not None:
                total_signals += 1
                signals_in_position += 1
            if i >= entry_bar + 1:
                mark_px = bid[i] if side == "long" else ask[i]
                if np.isfinite(mark_px):
                    pnl_mark = (mark_px - entry_px) / tick_size if side == "long" else (entry_px - mark_px) / tick_size
                    if np.isfinite(pnl_mark):
                        pos["mfe_ticks"] = max(float(pos.get("mfe_ticks", 0.0)), float(pnl_mark))
                        pos["mae_ticks"] = min(float(pos.get("mae_ticks", 0.0)), float(pnl_mark))
                    breakeven_ticks = pos.get("breakeven_ticks")
                    if breakeven_ticks is not None and not pos.get("breakeven_set", False):
                        mfe_ticks = (mark_px - entry_px) / tick_size if side == "long" else (entry_px - mark_px) / tick_size
                        if mfe_ticks >= float(breakeven_ticks):
                            pos["sl_level"] = float(entry_px)
                            pos["breakeven_set"] = True
                            pos["breakeven_triggered"] = True
                            sl_level = float(pos["sl_level"])
                    if side == "long":
                        if mark_px >= tp_level:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "TP"
                        elif mark_px <= sl_level:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "SL"
                    else:
                        if mark_px <= tp_level:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "TP"
                        elif mark_px >= sl_level:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "SL"
                if strategy_mode.startswith("absorption_failure") and pos.get("exit_reason") == "TIME":
                    flow_sum = _flow_sum_at(i, afr_flow_align_bars)
                    decay_count = int(pos.get("decay_count", 0))
                    if side == "long":
                        if flow_sum <= afr_momentum_decay_min_flow:
                            decay_count += 1
                        else:
                            decay_count = 0
                    else:
                        if flow_sum >= -afr_momentum_decay_min_flow:
                            decay_count += 1
                        else:
                            decay_count = 0
                    pos["decay_count"] = decay_count
                    if decay_count >= afr_momentum_decay_bars:
                        pos["exit_bar"] = i
                        pos["exit_reason"] = "DECAY"
                        pos["exit_on_decay"] = True
            if i >= int(pos["exit_bar"]):
                exit_bar = int(pos["exit_bar"])
                if not np.isfinite(bid[exit_bar]) or not np.isfinite(ask[exit_bar]):
                    raise RuntimeError("Non-finite exit price; check data integrity.")
                exit_px = bid[exit_bar] if side == "long" else ask[exit_bar]
                pnl_ticks = (exit_px - entry_px) / tick_size if side == "long" else (entry_px - exit_px) / tick_size
                exit_reason = str(pos.get("exit_reason", "TIME"))
                if bool(pos.get("breakeven_triggered", False)):
                    afr_be_triggered += 1
                if strategy_mode.startswith("absorption_failure"):
                    if exit_reason == "TP":
                        exit_reason = "tp"
                    elif exit_reason == "SL":
                        if bool(pos.get("breakeven_set", False)):
                            exit_reason = "breakeven"
                            pos["exit_on_be"] = True
                        else:
                            exit_reason = "sl"
                    elif exit_reason == "TIME":
                        exit_reason = "max_hold"
                    elif exit_reason == "DECAY":
                        exit_reason = "momentum_decay"
                    if exit_bar - entry_bar <= afr_rearm_stop_max_bars:
                        _arm_rearm(side, int(pos.get("absorption_bar", -1)))
                eps = 1e-9
                entry_spread = float(pos.get("entry_spread_ticks", 0.0))
                floor_ticks = -(sl_ticks_local + entry_spread + eps)
                if not np.isfinite(pnl_ticks):
                    if pnl_bound_printed < 5:
                        try:
                            print(
                                "PnL nan:",
                                {
                                    "side": side,
                                    "reason": exit_reason,
                                    "entry_time": str(pos["entry_time"]),
                                    "exit_time": str(df_day["Time"].iloc[exit_bar]),
                                    "entry_px": float(entry_px),
                                    "exit_px": float(exit_px),
                                },
                                flush=True,
                            )
                        except OSError:
                            pass
                        pnl_bound_printed += 1
                    in_position = False
                    pos = {}
                    i += 1
                    continue
                if pnl_ticks < floor_ticks and pnl_bound_printed < 5:
                    dt_ms = (df_day["Time"].iloc[exit_bar] - pos["entry_time"]).total_seconds() * 1000.0
                    try:
                        print(
                            "PnL below bound:",
                            {
                                "pnl_ticks": float(pnl_ticks),
                                "floor_ticks": float(floor_ticks),
                                "side": side,
                                "reason": exit_reason,
                                "entry_time": str(pos["entry_time"]),
                                "exit_time": str(df_day["Time"].iloc[exit_bar]),
                                "entry_px": float(entry_px),
                                "exit_px": float(exit_px),
                                "entry_spread_ticks": float(entry_spread),
                                "dt_ms": float(dt_ms),
                            },
                            flush=True,
                        )
                    except OSError:
                        pass
                    pnl_bound_printed += 1
                if not debug_trigger_printed and exit_reason in {"TP", "SL"}:
                    j0 = entry_bar + 1
                    j1 = min(entry_bar + 5, exit_bar)
                    series = (bid[j0 : j1 + 1] if side == "long" else ask[j0 : j1 + 1]).tolist()
                    try:
                        print(
                            f"Exit trigger debug: side={side} entry_bar={entry_bar} exit_bar={exit_bar} "
                            f"entry_time={pos['entry_time']} exit_time={df_day['Time'].iloc[exit_bar]} "
                            f"entry_px={entry_px:.2f} exit_px={exit_px:.2f} tp_level={tp_level:.2f} sl_level={sl_level:.2f} "
                            f"series={series} first_cross_bar={exit_bar} reason={exit_reason}",
                            flush=True,
                        )
                    except OSError:
                        pass
                    debug_trigger_printed = True
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": df_day["Time"].iloc[exit_bar],
                        "side": side,
                        "entry_px": float(entry_px),
                        "exit_px": float(exit_px),
                        "pnl_ticks": float(pnl_ticks),
                        "entry_bar": int(pos["entry_bar"]),
                        "exit_bar": int(exit_bar),
                        "exit_reason": exit_reason,
                        "entry_reason": pos.get("entry_reason"),
                        "absorption_level": float(pos.get("absorption_level", float("nan"))),
                        "break_level": float(pos.get("break_level", float("nan"))),
                        "flow_align_sum_at_entry": float(pos.get("flow_align_sum_at_entry", 0.0)),
                        "bars_from_absorption_to_entry": int(pos.get("bars_from_absorption", -1)),
                        "mae_ticks": float(pos.get("mae_ticks", 0.0)),
                        "mfe_ticks": float(pos.get("mfe_ticks", 0.0)),
                        "exit_on_decay": bool(pos.get("exit_on_decay", False)),
                        "exit_on_be": bool(pos.get("exit_on_be", False)),
                    }
                )
                pnl_ticks_total += float(pnl_ticks)
                in_position = False
                pos = {}
                cooldown_until = exit_bar + entry_cooldown_bars
                i += 1
                continue

        desired_side = None
        dmid_ticks = float("nan")
        impulse_ticks = float("nan")
        flow = 0.0
        stall_ticks = float("nan")
        break_ticks = float("nan")
        ft_progress_ticks = float("nan")
        afr_t_idx = -1
        afr_tf_idx = -1
        afr3_candidate = False
        afr3_debug_info: Dict[str, object] = {}
        mmas_passed = False
        entry_bar = i + 1
        if strategy_mode == "absorption_failure_v2":
            entry_bar = i
        if entry_bar >= n:
            break
        if not np.isfinite(bid[entry_bar]) or not np.isfinite(ask[entry_bar]) or not np.isfinite(mid[entry_bar]):
            i += 1
            continue
        cooldown_ok = i >= cooldown_until
        if entry_bar < len(session_ok) and not session_ok[entry_bar]:
            session_suppressed += 1
            i += 1
            continue
        if spread_ticks[entry_bar] < min_spread_ticks:
            spread_suppressed += 1
            i += 1
            continue
        if max_spread_ticks_for_entry > 0 and spread_ticks[entry_bar] > max_spread_ticks_for_entry:
            spread_suppressed += 1
            i += 1
            continue
        if not cooldown_ok:
            cooldown_suppressed += 1
            i += 1
            continue

        if strategy_mode == "impulse_confirm_v1":
            impulse_signals_checked += 1
            impulse_ticks = impulse_ticks_series[i]
            if not np.isfinite(impulse_ticks) or abs(impulse_ticks) < impulse_min_ticks:
                skip_reasons["impulse_failed_threshold"] += 1
                skipped += 1
                i += 1
                continue
            if impulse_ticks >= impulse_min_ticks:
                desired_side = "long"
            elif impulse_ticks <= -impulse_min_ticks:
                desired_side = "short"
            else:
                skip_reasons["impulse_failed_threshold"] += 1
                skipped += 1
                i += 1
                continue
            impulse_passed_threshold += 1
            window_start = i - confirm_bars + 1
            confirm_ok = True
            if window_start < 0:
                confirm_ok = False
            else:
                window = impulse_ticks_series[window_start : i + 1]
                if not np.all(np.isfinite(window)):
                    confirm_ok = False
                if confirm_require_nonzero and np.any(window == 0):
                    confirm_ok = False
                if confirm_ok:
                    sign_now = np.sign(impulse_ticks)
                    if np.any(np.sign(window) != sign_now):
                        confirm_ok = False
            if not confirm_ok:
                skip_reasons["impulse_failed_confirm"] += 1
                skipped += 1
                i += 1
                continue
            impulse_passed_confirm += 1
            flow = float(signed_vol[i]) if i < len(signed_vol) else 0.0
            signals_when_flat += 1
            total_signals += 1
        elif strategy_mode == "absorption_failure_v1":
            afr_checked += 1
            k = max(1, int(afr_k_bars))
            if i < k or i - 1 < 0:
                i += 1
                continue
            if afr_use_mid_for_stall:
                stall_series = mid
            elif "last_price" in df_day.columns:
                stall_series = pd.to_numeric(df_day["last_price"], errors="coerce").fillna(np.nan).to_numpy()
            else:
                stall_series = mid
            if not np.isfinite(stall_series[i - 1]) or not np.isfinite(stall_series[i - k]):
                i += 1
                continue
            flow = float(np.sum(afr_flow_series[i - k : i]))
            stall_ticks = float((stall_series[i - 1] - stall_series[i - k]) / tick_size)
            if abs(flow) < afr_min_flow_abs or abs(stall_ticks) > afr_stall_ticks:
                i += 1
                continue
            afr_absorption_pass += 1
            if not np.isfinite(mid[i]) or not np.isfinite(mid[i - 1]):
                i += 1
                continue
            break_ticks = float((mid[i] - mid[i - 1]) / tick_size)
            if abs(break_ticks) < afr_break_ticks:
                i += 1
                continue
            afr_break_pass += 1
            if break_ticks > 0:
                desired_side = "long"
            elif break_ticks < 0:
                desired_side = "short"
            else:
                i += 1
                continue
            if afr_require_flow_sign and np.sign(flow) != np.sign(break_ticks):
                i += 1
                continue
            signals_when_flat += 1
            total_signals += 1
        elif strategy_mode == "absorption_failure_v2":
            afr2_checked += 1
            k = max(1, int(afr_k_bars))
            ft_bars = max(1, int(afr_ft_bars))
            enter_on = afr_enter_on
            allow_break = enter_on in {"break", "both"}
            allow_ft = enter_on in {"ft", "both"}
            entry_reason = None
            if afr_use_mid_for_stall:
                stall_series = mid
            elif "last_price" in df_day.columns:
                stall_series = pd.to_numeric(df_day["last_price"], errors="coerce").fillna(np.nan).to_numpy()
            else:
                stall_series = mid
            if i >= k and i - 1 >= 0 and np.isfinite(stall_series[i - 1]) and np.isfinite(stall_series[i - k]):
                flow_setup = float(np.sum(afr_flow_series[i - k : i]))
                stall_ticks = float((stall_series[i - 1] - stall_series[i - k]) / tick_size)
                if abs(flow_setup) >= afr_min_flow_abs and abs(stall_ticks) <= afr_stall_ticks:
                    afr2_absorption_pass += 1
                    if flow_setup > 0:
                        absorption_level_long = float(stall_series[i - 1])
                        absorption_bar_long = i - 1
                        rearm_used_long = False
                        rearm_active_long = False
                        rearm_expiry_long = -1
                    elif flow_setup < 0:
                        absorption_level_short = float(stall_series[i - 1])
                        absorption_bar_short = i - 1
                        rearm_used_short = False
                        rearm_active_short = False
                        rearm_expiry_short = -1

            def _check_break(direction: str) -> Tuple[bool, float, float, int]:
                if direction == "long":
                    level = absorption_level_long
                    abs_bar = absorption_bar_long
                    if not np.isfinite(level) or abs_bar < 0:
                        return False, float("nan"), float("nan"), -1
                    break_level = level + afr_break_ticks * tick_size
                    triggered = np.isfinite(mid[entry_bar]) and mid[entry_bar] >= break_level
                    return bool(triggered), level, break_level, abs_bar
                level = absorption_level_short
                abs_bar = absorption_bar_short
                if not np.isfinite(level) or abs_bar < 0:
                    return False, float("nan"), float("nan"), -1
                break_level = level - afr_break_ticks * tick_size
                triggered = np.isfinite(mid[entry_bar]) and mid[entry_bar] <= break_level
                return bool(triggered), level, break_level, abs_bar

            def _align_ok(direction: str, idx: int) -> Tuple[bool, float]:
                flow_sum = _flow_sum_at(idx, afr_flow_align_bars)
                if direction == "long":
                    return bool(flow_sum >= afr_min_flow_abs_align), flow_sum
                return bool(flow_sum <= -afr_min_flow_abs_align), flow_sum

            def _afr2_entry_intent() -> Dict[str, object] | None:
                nonlocal break_bar_long
                nonlocal break_bar_short
                nonlocal break_ticks
                nonlocal ft_progress_ticks
                nonlocal afr_t_idx
                nonlocal afr_tf_idx
                nonlocal afr2_absorption_pass
                nonlocal afr2_break_pass
                nonlocal afr2_ft_pass
                nonlocal afr2_break_quality_pass
                nonlocal afr2_snapback_fail
                nonlocal afr2_break_quality_entries
                nonlocal rearm_active_long
                nonlocal rearm_active_short
                nonlocal rearm_used_long
                nonlocal rearm_used_short

                missed_break = False
                entry_reason_local = None
                absorption_level_local = float("nan")
                break_level_local = float("nan")
                absorption_bar_local = -1
                flow_align_sum_local = 0.0
                desired_side_local = None
                rearm_used_local = False
                break_quality_ok = False
                snapback_fail = False

                if rearm_active_long:
                    if entry_bar > rearm_expiry_long or (
                        np.isfinite(absorption_level_long)
                        and np.isfinite(mid[entry_bar])
                        and abs(mid[entry_bar] - absorption_level_long) > afr_rearm_band_ticks * tick_size
                    ):
                        rearm_active_long = False
                if rearm_active_short:
                    if entry_bar > rearm_expiry_short or (
                        np.isfinite(absorption_level_short)
                        and np.isfinite(mid[entry_bar])
                        and abs(mid[entry_bar] - absorption_level_short) > afr_rearm_band_ticks * tick_size
                    ):
                        rearm_active_short = False

                long_break, long_abs_level, long_break_level, long_abs_bar = _check_break("long")
                short_break, short_abs_level, short_break_level, short_abs_bar = _check_break("short")
                if allow_ft:
                    if long_break:
                        break_bar_long = entry_bar
                    if short_break:
                        break_bar_short = entry_bar

                # Primary break entry (break or both).
                if allow_break:
                    if long_break:
                        break_ticks = float((mid[entry_bar] - mid[entry_bar - 1]) / tick_size)
                        afr2_break_pass += 1
                        afr_t_idx = entry_bar
                        afr_tf_idx = entry_bar + ft_bars
                        ok, flow_align_sum_local = _align_ok("long", entry_bar)
                        if ok:
                            break_quality_ok = (
                                abs(flow_align_sum_local) >= afr_break_quality_min_flow_abs
                                and spread_ticks[entry_bar] <= afr_break_quality_max_spread_ticks
                            )
                            if break_quality_ok:
                                afr2_break_quality_pass += 1
                                if np.isfinite(absorption_level_long) and afr_snapback_bars > 0:
                                    end_idx = min(entry_bar + afr_snapback_bars, n - 1)
                                    for j in range(entry_bar + 1, end_idx + 1):
                                        if np.isfinite(mid[j]) and abs(mid[j] - absorption_level_long) <= afr_snapback_band_ticks * tick_size:
                                            snapback_fail = True
                                            break
                                    if snapback_fail:
                                        afr2_snapback_fail += 1
                            if not break_quality_ok or snapback_fail:
                                missed_break = True
                            else:
                                desired_side_local = "long"
                                absorption_level_local = long_abs_level
                                break_level_local = long_break_level
                                absorption_bar_local = long_abs_bar
                                entry_reason_local = "break"
                                if rearm_active_long:
                                    rearm_used_local = True
                        else:
                            missed_break = True
                    elif short_break:
                        break_ticks = float((mid[entry_bar] - mid[entry_bar - 1]) / tick_size)
                        afr2_break_pass += 1
                        afr_t_idx = entry_bar
                        afr_tf_idx = entry_bar + ft_bars
                        ok, flow_align_sum_local = _align_ok("short", entry_bar)
                        if ok:
                            break_quality_ok = (
                                abs(flow_align_sum_local) >= afr_break_quality_min_flow_abs
                                and spread_ticks[entry_bar] <= afr_break_quality_max_spread_ticks
                            )
                            if break_quality_ok:
                                afr2_break_quality_pass += 1
                                if np.isfinite(absorption_level_short) and afr_snapback_bars > 0:
                                    end_idx = min(entry_bar + afr_snapback_bars, n - 1)
                                    for j in range(entry_bar + 1, end_idx + 1):
                                        if np.isfinite(mid[j]) and abs(mid[j] - absorption_level_short) <= afr_snapback_band_ticks * tick_size:
                                            snapback_fail = True
                                            break
                                    if snapback_fail:
                                        afr2_snapback_fail += 1
                            if not break_quality_ok or snapback_fail:
                                missed_break = True
                            else:
                                desired_side_local = "short"
                                absorption_level_local = short_abs_level
                                break_level_local = short_break_level
                                absorption_bar_local = short_abs_bar
                                entry_reason_local = "break"
                                if rearm_active_short:
                                    rearm_used_local = True
                        else:
                            missed_break = True
                else:
                    if long_break or short_break:
                        afr2_break_pass += 1

                if desired_side_local is None and allow_ft:
                    # Allow FT entry only if enabled and FT bar is reached.
                    if break_bar_long >= 0 and entry_bar == break_bar_long + ft_bars:
                        ft_progress_ticks = float((mid[entry_bar] - mid[break_bar_long]) / tick_size)
                        ok, flow_align_sum_local = _align_ok("long", entry_bar)
                        ft_ok = ok and ft_progress_ticks >= afr_ft_min_ticks
                        if ft_ok:
                            afr2_ft_pass += 1
                            desired_side_local = "long"
                            absorption_level_local = absorption_level_long
                            break_level_local = absorption_level_long + afr_break_ticks * tick_size
                            absorption_bar_local = absorption_bar_long
                            afr_t_idx = break_bar_long
                            afr_tf_idx = entry_bar
                            if break_bar_long - 1 >= 0 and np.isfinite(mid[break_bar_long]) and np.isfinite(mid[break_bar_long - 1]):
                                break_ticks = float((mid[break_bar_long] - mid[break_bar_long - 1]) / tick_size)
                            entry_reason_local = "ft"
                            if rearm_active_long:
                                rearm_used_local = True
                    elif break_bar_short >= 0 and entry_bar == break_bar_short + ft_bars:
                        ft_progress_ticks = float((mid[entry_bar] - mid[break_bar_short]) / tick_size)
                        ok, flow_align_sum_local = _align_ok("short", entry_bar)
                        ft_ok = ok and ft_progress_ticks <= -afr_ft_min_ticks
                        if ft_ok:
                            afr2_ft_pass += 1
                            desired_side_local = "short"
                            absorption_level_local = absorption_level_short
                            break_level_local = absorption_level_short - afr_break_ticks * tick_size
                            absorption_bar_local = absorption_bar_short
                            afr_t_idx = break_bar_short
                            afr_tf_idx = entry_bar
                            if break_bar_short - 1 >= 0 and np.isfinite(mid[break_bar_short]) and np.isfinite(mid[break_bar_short - 1]):
                                break_ticks = float((mid[break_bar_short] - mid[break_bar_short - 1]) / tick_size)
                            entry_reason_local = "ft"
                            if rearm_active_short:
                                rearm_used_local = True

                if desired_side_local is None and missed_break:
                    # Missed entry: allow one rearm per direction.
                    if long_break:
                        _arm_rearm("long", absorption_bar_long)
                    if short_break:
                        _arm_rearm("short", absorption_bar_short)
                    return None

                if desired_side_local is None:
                    return None

                # Flow alignment check for FT entry bar.
                if desired_side_local == "long":
                    align_ok, flow_align_sum_local = _align_ok("long", entry_bar)
                    if not align_ok:
                        missed_break = True
                else:
                    align_ok, flow_align_sum_local = _align_ok("short", entry_bar)
                    if not align_ok:
                        missed_break = True
                if missed_break:
                    if desired_side_local == "long":
                        _arm_rearm("long", absorption_bar_long)
                    if desired_side_local == "short":
                        _arm_rearm("short", absorption_bar_short)
                    return None

                if rearm_used_local:
                    if desired_side_local == "long":
                        rearm_used_long = True
                        rearm_active_long = False
                    else:
                        rearm_used_short = True
                        rearm_active_short = False
                if entry_reason_local == "break":
                    if desired_side_local == "long":
                        break_bar_long = -1
                    else:
                        break_bar_short = -1
                    if break_quality_ok and not snapback_fail:
                        afr2_break_quality_entries += 1

                return {
                    "desired_side": desired_side_local,
                    "entry_reason": entry_reason_local,
                    "absorption_level": absorption_level_local,
                    "break_level": break_level_local,
                    "absorption_bar": absorption_bar_local,
                    "flow_align_sum": flow_align_sum_local,
                    "rearm_used": rearm_used_local,
                }

            afr2_intent = _afr2_entry_intent()
            if afr2_intent is None:
                i += 1
                continue

            desired_side = afr2_intent["desired_side"]
            entry_reason = afr2_intent["entry_reason"]
            absorption_level = float(afr2_intent["absorption_level"])
            break_level = float(afr2_intent["break_level"])
            absorption_bar = int(afr2_intent["absorption_bar"])
            flow_align_sum = float(afr2_intent["flow_align_sum"])

            signals_when_flat += 1
            total_signals += 1
        elif strategy_mode == "absorption_failure_v3":
            afr3_checked += 1
            k = max(1, int(afr_k_bars))
            if i < k or i - 1 < 0:
                i += 1
                continue
            if afr_use_mid_for_stall:
                stall_series = mid
            elif "last_price" in df_day.columns:
                stall_series = pd.to_numeric(df_day["last_price"], errors="coerce").fillna(np.nan).to_numpy()
            else:
                stall_series = mid
            if (
                not np.isfinite(stall_series[i - 1])
                or not np.isfinite(stall_series[i - k])
                or not np.isfinite(mid[i])
                or not np.isfinite(mid[i - 1])
                or not np.isfinite(bid[i])
                or not np.isfinite(ask[i])
            ):
                i += 1
                continue
            flow = float(np.sum(afr_flow_series[i - k + 1 : i + 1]))
            stall_ticks = float((stall_series[i - 1] - stall_series[i - k]) / tick_size)
            if abs(flow) < afr_min_flow_abs or abs(stall_ticks) > afr_stall_ticks:
                i += 1
                continue
            afr3_absorption_pass += 1
            break_ticks = float((mid[i] - mid[i - 1]) / tick_size)
            if abs(break_ticks) < afr_break_ticks or break_ticks == 0:
                i += 1
                continue
            afr3_break_pass += 1
            if abs(flow) < afr3_break_min_flow_abs or spread_ticks[i] > afr3_break_max_spread_ticks:
                i += 1
                continue
            afr3_break_quality_pass += 1
            if break_ticks > 0:
                desired_side = "long"
            elif break_ticks < 0:
                desired_side = "short"
            else:
                i += 1
                continue
            if afr_require_flow_sign and np.sign(flow) != np.sign(break_ticks):
                i += 1
                continue
            snapback_ok = True
            if afr3_snapback_check:
                if i + 1 >= n or not np.isfinite(mid[i + 1]):
                    snapback_ok = False
                elif desired_side == "long":
                    if mid[i + 1] < mid[i]:
                        snapback_ok = False
                else:
                    if mid[i + 1] > mid[i]:
                        snapback_ok = False
            if not snapback_ok:
                i += 1
                continue
            afr3_snapback_pass += 1
            afr3_candidate = True
            afr3_debug_info = {
                "t": int(i),
                "absorption_pass": True,
                "break_pass": True,
                "break_quality_pass": True,
                "flow_k": float(flow),
                "spread_ticks": int(spread_ticks[i]),
                "snapback_pass": True,
                "side": desired_side,
            }
            signals_when_flat += 1
            total_signals += 1
        elif strategy_mode == "micro_momo_v1":
            k = max(1, int(micro_k_bars))
            if i >= k and np.isfinite(mid[i]) and np.isfinite(mid[i - k]):
                dmid = mid[i] - mid[i - k]
                impulse_ticks = dmid / tick_size
                flow = cs[i + 1] - cs[i + 1 - k]
                if impulse_ticks >= micro_impulse_ticks and flow >= micro_flow_min:
                    desired_side = "long"
                elif impulse_ticks <= -micro_impulse_ticks and flow <= -micro_flow_min:
                    desired_side = "short"
        elif baseline_mode == "mmas":
            mmas_signals_checked += 1
            desired_side, dmid_ticks, flow, mmas_passed = _mmas_signal(i)
            if mmas_passed:
                mmas_passed_filters += 1
            if desired_side is not None:
                mmas_signaled += 1
        else:
            desired_side = _signal(i)
            if desired_side is not None and i >= lookback_bars and np.isfinite(mid[i]) and np.isfinite(mid[i - lookback_bars]):
                dmid = mid[i] - mid[i - lookback_bars]
                dmid_ticks = dmid / tick_size
                impulse_ticks = dmid_ticks
                flow = cs[i + 1] - cs[i + 1 - lookback_bars]
        if desired_side is None:
            i += 1
            continue
        if strategy_mode != "impulse_confirm_v1":
            total_signals += 1
            signals_when_flat += 1
        if desired_side == "long":
            strategy_long_signals += 1
        else:
            strategy_short_signals += 1
        if strategy_mode == "micro_momo_v1" and micro_impulse_ticks > 0:
            if desired_side == "long" and impulse_ticks < (micro_impulse_ticks - 1e-9):
                raise RuntimeError(
                    f"micro_momo_v1 impulse below threshold for long: t={i} impulse={impulse_ticks} "
                    f"thr={micro_impulse_ticks}"
                )
            if desired_side == "short" and impulse_ticks > (-micro_impulse_ticks + 1e-9):
                raise RuntimeError(
                    f"micro_momo_v1 impulse above threshold for short: t={i} impulse={impulse_ticks} "
                    f"thr={-micro_impulse_ticks}"
                )

        gate_allowed = True
        gate_reason = "not_gated"
        gate_enabled = gated and not disable_gate
        if gate_enabled:
            gate_allowed = True
            gate_reason = "allowed"
            if gate_mode == "side_matched":
                if desired_side == "long":
                    event_pos = event_pos_buy
                    event_asym = event_asym_buy
                    event_thr = event_thr_buy
                else:
                    event_pos = event_pos_sell
                    event_asym = event_asym_sell
                    event_thr = event_thr_sell
            else:
                event_pos = event_pos_all
                event_asym = event_asym_all
                event_thr = event_thr_all

            if event_pos.size == 0:
                skip_reasons["no_recent_event"] += 1
                skipped += 1
                blocked_signals += 1
                gate_allowed = False
                gate_reason = "no_recent_event"
                if strategy_mode == "impulse_confirm_v1" and debug_first_impulse and entry_debug_printed == 0:
                    _print_impulse_debug(
                        t_idx=i,
                        impulse_val=impulse_ticks,
                        spread_val=spread_ticks[entry_bar],
                        cooldown_ok_val=cooldown_ok,
                        gate_allowed_val=False,
                        gate_reason_val=gate_reason,
                        desired_side_val=desired_side,
                        entry_action_val="blocked",
                    )
                    entry_debug_printed += 1
                if strategy_mode == "absorption_failure_v1" and debug_first_afr and entry_debug_printed == 0:
                    _print_afr_debug(
                        t_idx=i,
                        flow_val=flow,
                        stall_val=stall_ticks,
                        break_val=break_ticks,
                        spread_val=spread_ticks[entry_bar],
                        cooldown_ok_val=cooldown_ok,
                        session_ok_val=bool(session_ok[entry_bar]),
                        gate_allowed_val=False,
                        gate_reason_val=gate_reason,
                        desired_side_val=desired_side,
                        absorption_pass_val=True,
                        break_pass_val=True,
                        entry_taken_val=False,
                        tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                        sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                        hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                        breakeven_ticks_val=afr_breakeven_after_ticks,
                        entry_action_val="blocked",
                    )
                    entry_debug_printed += 1
                if strategy_mode == "absorption_failure_v2" and debug_first_afr2 and entry_debug_printed == 0:
                    _print_afr2_debug(
                        t_idx=afr_t_idx,
                        tf_idx=afr_tf_idx,
                        flow_val=flow,
                        stall_val=stall_ticks,
                        break_val=break_ticks,
                        ft_val=ft_progress_ticks,
                        spread_val=spread_ticks[entry_bar],
                        cooldown_ok_val=cooldown_ok,
                        session_ok_val=bool(session_ok[entry_bar]),
                        gate_allowed_val=False,
                        gate_reason_val=gate_reason,
                        desired_side_val=desired_side,
                        entry_action_val="blocked",
                    )
                    entry_debug_printed += 1
                if strategy_mode == "absorption_failure_v2" and debug_first_afr and entry_debug_printed == 0:
                    _print_afr_debug(
                        t_idx=entry_bar,
                        flow_val=flow,
                        stall_val=stall_ticks,
                        break_val=break_ticks,
                        spread_val=spread_ticks[entry_bar],
                        cooldown_ok_val=cooldown_ok,
                        session_ok_val=bool(session_ok[entry_bar]),
                        gate_allowed_val=False,
                        gate_reason_val=gate_reason,
                        desired_side_val=desired_side,
                        absorption_pass_val=True,
                        break_pass_val=True,
                        entry_taken_val=False,
                        tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                        sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                        hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                        breakeven_ticks_val=afr_breakeven_after_ticks,
                        entry_action_val="blocked",
                    )
                    entry_debug_printed += 1
                if strategy_mode == "absorption_failure_v3" and debug_first_afr3 and entry_debug_printed == 0 and afr3_candidate:
                    _print_afr3_debug(
                        t_idx=i,
                        flow_val=flow,
                        spread_val=spread_ticks[i],
                        cooldown_ok_val=cooldown_ok,
                        session_ok_val=bool(session_ok[entry_bar]),
                        gate_allowed_val=False,
                        gate_reason_val=gate_reason,
                        desired_side_val=desired_side,
                        absorption_pass_val=True,
                        break_pass_val=True,
                        break_quality_pass_val=True,
                        snapback_pass_val=True,
                        entry_taken_val=False,
                        entry_action_val="blocked",
                    )
                    entry_debug_printed += 1
                if baseline_mode == "mmas" and debug_first_mmas and (not debug_mmas_printed) and mmas_passed:
                    print(
                        "MMAS_DEBUG",
                        {
                            "t": int(i),
                            "dmid_ticks": float(dmid_ticks),
                            "flow": float(flow),
                            "spread_ticks": float(spread_ticks[entry_bar]),
                            "gate_allowed": False,
                            "gate_reason": gate_reason,
                            "side": desired_side,
                        },
                        flush=True,
                    )
                    debug_mmas_printed = True
                if strategy_mode.startswith("absorption_failure") and desired_side is not None:
                    _arm_rearm(desired_side, absorption_bar)
                i += 1
                continue
            else:
                idx_pos = np.searchsorted(event_pos, entry_bar - 1, side="right") - 1
                if idx_pos < 0:
                    skip_reasons["no_recent_event"] += 1
                    skipped += 1
                    blocked_signals += 1
                    gate_allowed = False
                    gate_reason = "no_recent_event"
                    if strategy_mode == "impulse_confirm_v1" and debug_first_impulse and entry_debug_printed == 0:
                        _print_impulse_debug(
                            t_idx=i,
                            impulse_val=impulse_ticks,
                            spread_val=spread_ticks[entry_bar],
                            cooldown_ok_val=cooldown_ok,
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if strategy_mode == "absorption_failure_v1" and debug_first_afr and entry_debug_printed == 0:
                        _print_afr_debug(
                            t_idx=i,
                            flow_val=flow,
                            stall_val=stall_ticks,
                            break_val=break_ticks,
                            spread_val=spread_ticks[entry_bar],
                            cooldown_ok_val=cooldown_ok,
                            session_ok_val=bool(session_ok[entry_bar]),
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            absorption_pass_val=True,
                            break_pass_val=True,
                            entry_taken_val=False,
                            tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                            sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                            hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                            breakeven_ticks_val=afr_breakeven_after_ticks,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if strategy_mode == "absorption_failure_v2" and debug_first_afr2 and entry_debug_printed == 0:
                        _print_afr2_debug(
                            t_idx=afr_t_idx,
                            tf_idx=afr_tf_idx,
                            flow_val=flow,
                            stall_val=stall_ticks,
                            break_val=break_ticks,
                            ft_val=ft_progress_ticks,
                            spread_val=spread_ticks[entry_bar],
                            cooldown_ok_val=cooldown_ok,
                            session_ok_val=bool(session_ok[entry_bar]),
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if strategy_mode == "absorption_failure_v2" and debug_first_afr and entry_debug_printed == 0:
                        _print_afr_debug(
                            t_idx=entry_bar,
                            flow_val=flow,
                            stall_val=stall_ticks,
                            break_val=break_ticks,
                            spread_val=spread_ticks[entry_bar],
                            cooldown_ok_val=cooldown_ok,
                            session_ok_val=bool(session_ok[entry_bar]),
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            absorption_pass_val=True,
                            break_pass_val=True,
                            entry_taken_val=False,
                            tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                            sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                            hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                            breakeven_ticks_val=afr_breakeven_after_ticks,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if strategy_mode == "absorption_failure_v3" and debug_first_afr3 and entry_debug_printed == 0 and afr3_candidate:
                        _print_afr3_debug(
                            t_idx=i,
                            flow_val=flow,
                            spread_val=spread_ticks[i],
                            cooldown_ok_val=cooldown_ok,
                            session_ok_val=bool(session_ok[entry_bar]),
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            absorption_pass_val=True,
                            break_pass_val=True,
                            break_quality_pass_val=True,
                            snapback_pass_val=True,
                            entry_taken_val=False,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if strategy_mode == "absorption_failure_v3" and debug_first_afr3 and entry_debug_printed == 0 and afr3_candidate:
                        _print_afr3_debug(
                            t_idx=i,
                            flow_val=flow,
                            spread_val=spread_ticks[i],
                            cooldown_ok_val=cooldown_ok,
                            session_ok_val=bool(session_ok[entry_bar]),
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            absorption_pass_val=True,
                            break_pass_val=True,
                            break_quality_pass_val=True,
                            snapback_pass_val=True,
                            entry_taken_val=False,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if baseline_mode == "mmas" and debug_first_mmas and (not debug_mmas_printed) and mmas_passed:
                        print(
                            "MMAS_DEBUG",
                            {
                                "t": int(i),
                                "dmid_ticks": float(dmid_ticks),
                                "flow": float(flow),
                                "spread_ticks": float(spread_ticks[entry_bar]),
                                "gate_allowed": False,
                                "gate_reason": gate_reason,
                                "side": desired_side,
                            },
                            flush=True,
                        )
                        debug_mmas_printed = True
                    if strategy_mode.startswith("absorption_failure") and desired_side is not None:
                        _arm_rearm(desired_side, absorption_bar)
                    i += 1
                    continue
                elif event_pos[idx_pos] < entry_bar - gate_lookback_bars:
                    skip_reasons["no_event_in_window"] += 1
                    skipped += 1
                    blocked_signals += 1
                    gate_allowed = False
                    gate_reason = "no_event_in_window"
                    if strategy_mode == "impulse_confirm_v1" and debug_first_impulse and entry_debug_printed == 0:
                        _print_impulse_debug(
                            t_idx=i,
                            impulse_val=impulse_ticks,
                            spread_val=spread_ticks[entry_bar],
                            cooldown_ok_val=cooldown_ok,
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if strategy_mode == "absorption_failure_v1" and debug_first_afr and entry_debug_printed == 0:
                        _print_afr_debug(
                            t_idx=i,
                            flow_val=flow,
                            stall_val=stall_ticks,
                            break_val=break_ticks,
                            spread_val=spread_ticks[entry_bar],
                            cooldown_ok_val=cooldown_ok,
                            session_ok_val=bool(session_ok[entry_bar]),
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            absorption_pass_val=True,
                            break_pass_val=True,
                            entry_taken_val=False,
                            tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                            sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                            hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                            breakeven_ticks_val=afr_breakeven_after_ticks,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if strategy_mode == "absorption_failure_v2" and debug_first_afr2 and entry_debug_printed == 0:
                        _print_afr2_debug(
                            t_idx=afr_t_idx,
                            tf_idx=afr_tf_idx,
                            flow_val=flow,
                            stall_val=stall_ticks,
                            break_val=break_ticks,
                            ft_val=ft_progress_ticks,
                            spread_val=spread_ticks[entry_bar],
                            cooldown_ok_val=cooldown_ok,
                            session_ok_val=bool(session_ok[entry_bar]),
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if strategy_mode == "absorption_failure_v2" and debug_first_afr and entry_debug_printed == 0:
                        _print_afr_debug(
                            t_idx=entry_bar,
                            flow_val=flow,
                            stall_val=stall_ticks,
                            break_val=break_ticks,
                            spread_val=spread_ticks[entry_bar],
                            cooldown_ok_val=cooldown_ok,
                            session_ok_val=bool(session_ok[entry_bar]),
                            gate_allowed_val=False,
                            gate_reason_val=gate_reason,
                            desired_side_val=desired_side,
                            absorption_pass_val=True,
                            break_pass_val=True,
                            entry_taken_val=False,
                            tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                            sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                            hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                            breakeven_ticks_val=afr_breakeven_after_ticks,
                            entry_action_val="blocked",
                        )
                        entry_debug_printed += 1
                    if baseline_mode == "mmas" and debug_first_mmas and (not debug_mmas_printed) and mmas_passed:
                        print(
                            "MMAS_DEBUG",
                            {
                                "t": int(i),
                                "dmid_ticks": float(dmid_ticks),
                                "flow": float(flow),
                                "spread_ticks": float(spread_ticks[entry_bar]),
                                "gate_allowed": False,
                                "gate_reason": gate_reason,
                                "side": desired_side,
                            },
                            flush=True,
                        )
                        debug_mmas_printed = True
                    if strategy_mode.startswith("absorption_failure") and desired_side is not None:
                        _arm_rearm(desired_side, absorption_bar)
                    i += 1
                    continue
                else:
                    thr = event_thr[idx_pos]
                    if not np.isfinite(thr):
                        skip_reasons["no_threshold_yet"] += 1
                        skipped += 1
                        blocked_signals += 1
                        gate_allowed = False
                        gate_reason = "no_threshold_yet"
                        if strategy_mode == "impulse_confirm_v1" and debug_first_impulse and entry_debug_printed == 0:
                            _print_impulse_debug(
                                t_idx=i,
                                impulse_val=impulse_ticks,
                                spread_val=spread_ticks[entry_bar],
                                cooldown_ok_val=cooldown_ok,
                                gate_allowed_val=False,
                                gate_reason_val=gate_reason,
                                desired_side_val=desired_side,
                                entry_action_val="blocked",
                            )
                            entry_debug_printed += 1
                        if strategy_mode == "absorption_failure_v1" and debug_first_afr and entry_debug_printed == 0:
                            _print_afr_debug(
                                t_idx=i,
                                flow_val=flow,
                                stall_val=stall_ticks,
                                break_val=break_ticks,
                                spread_val=spread_ticks[entry_bar],
                                cooldown_ok_val=cooldown_ok,
                                session_ok_val=bool(session_ok[entry_bar]),
                                gate_allowed_val=False,
                                gate_reason_val=gate_reason,
                                desired_side_val=desired_side,
                                absorption_pass_val=True,
                                break_pass_val=True,
                                entry_taken_val=False,
                                tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                                sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                                hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                                breakeven_ticks_val=afr_breakeven_after_ticks,
                                entry_action_val="blocked",
                            )
                            entry_debug_printed += 1
                        if strategy_mode == "absorption_failure_v2" and debug_first_afr2 and entry_debug_printed == 0:
                            _print_afr2_debug(
                                t_idx=afr_t_idx,
                                tf_idx=afr_tf_idx,
                                flow_val=flow,
                                stall_val=stall_ticks,
                                break_val=break_ticks,
                                ft_val=ft_progress_ticks,
                                spread_val=spread_ticks[entry_bar],
                                cooldown_ok_val=cooldown_ok,
                                session_ok_val=bool(session_ok[entry_bar]),
                                gate_allowed_val=False,
                                gate_reason_val=gate_reason,
                                desired_side_val=desired_side,
                                entry_action_val="blocked",
                            )
                            entry_debug_printed += 1
                        if strategy_mode == "absorption_failure_v2" and debug_first_afr and entry_debug_printed == 0:
                            _print_afr_debug(
                                t_idx=entry_bar,
                                flow_val=flow,
                                stall_val=stall_ticks,
                                break_val=break_ticks,
                                spread_val=spread_ticks[entry_bar],
                                cooldown_ok_val=cooldown_ok,
                                session_ok_val=bool(session_ok[entry_bar]),
                                gate_allowed_val=False,
                                gate_reason_val=gate_reason,
                                desired_side_val=desired_side,
                                absorption_pass_val=True,
                                break_pass_val=True,
                                entry_taken_val=False,
                                tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                                sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                                hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                                breakeven_ticks_val=afr_breakeven_after_ticks,
                                entry_action_val="blocked",
                            )
                            entry_debug_printed += 1
                        if strategy_mode == "absorption_failure_v3" and debug_first_afr3 and entry_debug_printed == 0 and afr3_candidate:
                            _print_afr3_debug(
                                t_idx=i,
                                flow_val=flow,
                                spread_val=spread_ticks[i],
                                cooldown_ok_val=cooldown_ok,
                                session_ok_val=bool(session_ok[entry_bar]),
                                gate_allowed_val=False,
                                gate_reason_val=gate_reason,
                                desired_side_val=desired_side,
                                absorption_pass_val=True,
                                break_pass_val=True,
                                break_quality_pass_val=True,
                                snapback_pass_val=True,
                                entry_taken_val=False,
                                entry_action_val="blocked",
                            )
                            entry_debug_printed += 1
                        if baseline_mode == "mmas" and debug_first_mmas and (not debug_mmas_printed) and mmas_passed:
                            print(
                                "MMAS_DEBUG",
                                {
                                    "t": int(i),
                                    "dmid_ticks": float(dmid_ticks),
                                    "flow": float(flow),
                                    "spread_ticks": float(spread_ticks[entry_bar]),
                                    "gate_allowed": False,
                                    "gate_reason": gate_reason,
                                    "side": desired_side,
                                },
                                flush=True,
                            )
                            debug_mmas_printed = True
                        if strategy_mode.startswith("absorption_failure") and desired_side is not None:
                            _arm_rearm(desired_side, absorption_bar)
                        i += 1
                        continue
                    else:
                        eligible_signals += 1
                        if event_asym[idx_pos] >= thr:
                            skip_reasons["gated_blocked"] += 1
                            skipped += 1
                            blocked_signals += 1
                            gate_allowed = False
                            gate_reason = "gated_blocked"
                            if strategy_mode == "impulse_confirm_v1" and debug_first_impulse and entry_debug_printed == 0:
                                _print_impulse_debug(
                                    t_idx=i,
                                    impulse_val=impulse_ticks,
                                    spread_val=spread_ticks[entry_bar],
                                    cooldown_ok_val=cooldown_ok,
                                    gate_allowed_val=False,
                                    gate_reason_val=gate_reason,
                                    desired_side_val=desired_side,
                                    entry_action_val="blocked",
                                )
                                entry_debug_printed += 1
                            if strategy_mode == "absorption_failure_v1" and debug_first_afr and entry_debug_printed == 0:
                                _print_afr_debug(
                                    t_idx=i,
                                    flow_val=flow,
                                    stall_val=stall_ticks,
                                    break_val=break_ticks,
                                    spread_val=spread_ticks[entry_bar],
                                    cooldown_ok_val=cooldown_ok,
                                    session_ok_val=bool(session_ok[entry_bar]),
                                    gate_allowed_val=False,
                                    gate_reason_val=gate_reason,
                                    desired_side_val=desired_side,
                                    absorption_pass_val=True,
                                    break_pass_val=True,
                                    entry_taken_val=False,
                                    tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                                    sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                                    hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                                    breakeven_ticks_val=afr_breakeven_after_ticks,
                                    entry_action_val="blocked",
                                )
                                entry_debug_printed += 1
                            if strategy_mode == "absorption_failure_v2" and debug_first_afr2 and entry_debug_printed == 0:
                                _print_afr2_debug(
                                    t_idx=afr_t_idx,
                                    tf_idx=afr_tf_idx,
                                    flow_val=flow,
                                    stall_val=stall_ticks,
                                    break_val=break_ticks,
                                    ft_val=ft_progress_ticks,
                                    spread_val=spread_ticks[entry_bar],
                                    cooldown_ok_val=cooldown_ok,
                                    session_ok_val=bool(session_ok[entry_bar]),
                                    gate_allowed_val=False,
                                    gate_reason_val=gate_reason,
                                    desired_side_val=desired_side,
                                    entry_action_val="blocked",
                                )
                                entry_debug_printed += 1
                            if strategy_mode == "absorption_failure_v2" and debug_first_afr and entry_debug_printed == 0:
                                _print_afr_debug(
                                    t_idx=entry_bar,
                                    flow_val=flow,
                                    stall_val=stall_ticks,
                                    break_val=break_ticks,
                                    spread_val=spread_ticks[entry_bar],
                                    cooldown_ok_val=cooldown_ok,
                                    session_ok_val=bool(session_ok[entry_bar]),
                                    gate_allowed_val=False,
                                    gate_reason_val=gate_reason,
                                    desired_side_val=desired_side,
                                    absorption_pass_val=True,
                                    break_pass_val=True,
                                    entry_taken_val=False,
                                    tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                                    sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                                    hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                                    breakeven_ticks_val=afr_breakeven_after_ticks,
                                    entry_action_val="blocked",
                                )
                                entry_debug_printed += 1
                            if strategy_mode == "absorption_failure_v3" and debug_first_afr3 and entry_debug_printed == 0 and afr3_candidate:
                                _print_afr3_debug(
                                    t_idx=i,
                                    flow_val=flow,
                                    spread_val=spread_ticks[i],
                                    cooldown_ok_val=cooldown_ok,
                                    session_ok_val=bool(session_ok[entry_bar]),
                                    gate_allowed_val=False,
                                    gate_reason_val=gate_reason,
                                    desired_side_val=desired_side,
                                    absorption_pass_val=True,
                                    break_pass_val=True,
                                    break_quality_pass_val=True,
                                    snapback_pass_val=True,
                                    entry_taken_val=False,
                                    entry_action_val="blocked",
                                )
                                entry_debug_printed += 1
                            if baseline_mode == "mmas" and debug_first_mmas and (not debug_mmas_printed) and mmas_passed:
                                print(
                                    "MMAS_DEBUG",
                                    {
                                        "t": int(i),
                                        "dmid_ticks": float(dmid_ticks),
                                        "flow": float(flow),
                                        "spread_ticks": float(spread_ticks[entry_bar]),
                                        "gate_allowed": False,
                                        "gate_reason": gate_reason,
                                        "side": desired_side,
                                    },
                                    flush=True,
                                )
                                debug_mmas_printed = True
                            if strategy_mode.startswith("absorption_failure") and desired_side is not None:
                                _arm_rearm(desired_side, absorption_bar)
                            i += 1
                            continue
            if (
                debug_first_mmas
                and (not debug_mmas_printed)
                and ((baseline_mode == "mmas" and mmas_passed) or (strategy_mode == "micro_momo_v1"))
            ):
                print(
                    "MMAS_DEBUG",
                    {
                        "t": int(i),
                        "dmid_ticks": float(dmid_ticks),
                        "flow": float(flow),
                        "spread_ticks": float(spread_ticks[entry_bar]),
                        "cooldown_ok": bool(cooldown_ok),
                        "gate_ok": bool(gate_allowed),
                        "gate_reason": gate_reason,
                        "side": desired_side,
                    },
                    flush=True,
                )
                debug_mmas_printed = True
        elif (
            debug_first_mmas
            and (not debug_mmas_printed)
            and ((baseline_mode == "mmas" and mmas_passed) or (strategy_mode == "micro_momo_v1"))
        ):
            print(
                "MMAS_DEBUG",
                {
                    "t": int(i),
                    "dmid_ticks": float(dmid_ticks),
                    "flow": float(flow),
                    "spread_ticks": float(spread_ticks[entry_bar]),
                    "cooldown_ok": bool(cooldown_ok),
                    "gate_ok": True,
                    "gate_reason": "not_gated",
                    "side": desired_side,
                },
                flush=True,
            )
            debug_mmas_printed = True

        if strategy_mode == "impulse_confirm_v1" and debug_first_impulse and entry_debug_printed == 0:
            _print_impulse_debug(
                t_idx=i,
                impulse_val=impulse_ticks,
                spread_val=spread_ticks[entry_bar],
                cooldown_ok_val=cooldown_ok,
                gate_allowed_val=True if disable_gate and gated else gate_allowed,
                gate_reason_val="disabled" if disable_gate and gated else gate_reason,
                desired_side_val=desired_side,
                entry_action_val="entered",
            )
            entry_debug_printed += 1
        if strategy_mode == "absorption_failure_v1" and debug_first_afr and entry_debug_printed == 0:
            _print_afr_debug(
                t_idx=i,
                flow_val=flow,
                stall_val=stall_ticks,
                break_val=break_ticks,
                spread_val=spread_ticks[entry_bar],
                cooldown_ok_val=cooldown_ok,
                session_ok_val=bool(session_ok[entry_bar]),
                gate_allowed_val=True if disable_gate and gated else gate_allowed,
                gate_reason_val="disabled" if disable_gate and gated else gate_reason,
                desired_side_val=desired_side,
                absorption_pass_val=True,
                break_pass_val=True,
                entry_taken_val=True,
                tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                breakeven_ticks_val=afr_breakeven_after_ticks,
                entry_action_val="entered",
            )
            entry_debug_printed += 1
        if strategy_mode == "absorption_failure_v2" and debug_first_afr2 and entry_debug_printed == 0:
            _print_afr2_debug(
                t_idx=afr_t_idx,
                tf_idx=afr_tf_idx,
                flow_val=flow,
                stall_val=stall_ticks,
                break_val=break_ticks,
                ft_val=ft_progress_ticks,
                spread_val=spread_ticks[entry_bar],
                cooldown_ok_val=cooldown_ok,
                session_ok_val=bool(session_ok[entry_bar]),
                gate_allowed_val=True if disable_gate and gated else gate_allowed,
                gate_reason_val="disabled" if disable_gate and gated else gate_reason,
                desired_side_val=desired_side,
                entry_action_val="entered",
            )
            entry_debug_printed += 1
        if strategy_mode == "absorption_failure_v3" and debug_first_afr3 and entry_debug_printed == 0 and afr3_candidate:
            _print_afr3_debug(
                t_idx=i,
                flow_val=flow,
                spread_val=spread_ticks[i],
                cooldown_ok_val=cooldown_ok,
                session_ok_val=bool(session_ok[entry_bar]),
                gate_allowed_val=True if disable_gate and gated else gate_allowed,
                gate_reason_val="disabled" if disable_gate and gated else gate_reason,
                desired_side_val=desired_side,
                absorption_pass_val=True,
                break_pass_val=True,
                break_quality_pass_val=True,
                snapback_pass_val=True,
                entry_taken_val=True,
                entry_action_val="entered",
            )
            entry_debug_printed += 1
        if strategy_mode == "absorption_failure_v2" and debug_first_afr and entry_debug_printed == 0:
            _print_afr_debug(
                t_idx=entry_bar,
                flow_val=flow,
                stall_val=stall_ticks,
                break_val=break_ticks,
                spread_val=spread_ticks[entry_bar],
                cooldown_ok_val=cooldown_ok,
                session_ok_val=bool(session_ok[entry_bar]),
                gate_allowed_val=True if disable_gate and gated else gate_allowed,
                gate_reason_val="disabled" if disable_gate and gated else gate_reason,
                desired_side_val=desired_side,
                absorption_pass_val=True,
                break_pass_val=True,
                entry_taken_val=True,
                tp_ticks_val=int(afr_tp_ticks) if afr_tp_ticks is not None else int(tp_ticks),
                sl_ticks_val=int(afr_sl_ticks) if afr_sl_ticks is not None else int(sl_ticks),
                hold_bars_val=int(afr_max_hold_bars) if afr_max_hold_bars is not None else int(hold_bars),
                breakeven_ticks_val=afr_breakeven_after_ticks,
                entry_action_val="entered",
            )
            entry_debug_printed += 1

        entry_px = ask[entry_bar] if desired_side == "long" else bid[entry_bar]
        entry_spread_ticks = float(spread_ticks[entry_bar])
        tp_ticks_local = tp_ticks
        sl_ticks_local = sl_ticks
        hold_bars_local = hold_bars
        breakeven_ticks_local = None
        if strategy_mode.startswith("absorption_failure"):
            if afr_tp_ticks is not None:
                tp_ticks_local = afr_tp_ticks
            if afr_sl_ticks is not None:
                sl_ticks_local = afr_sl_ticks
            if afr_max_hold_bars is not None:
                hold_bars_local = afr_max_hold_bars
            if afr_breakeven_after_ticks is not None:
                breakeven_ticks_local = afr_breakeven_after_ticks
        tp_level = entry_px + (tp_ticks_local * tick_size if desired_side == "long" else -tp_ticks_local * tick_size)
        sl_level = entry_px - (sl_ticks_local * tick_size if desired_side == "long" else -sl_ticks_local * tick_size)
        pos = {
            "entry_time": df_day["Time"].iloc[entry_bar],
            "entry_bar": entry_bar,
            "entry_px": float(entry_px),
            "exit_bar": int(min(entry_bar + hold_bars_local, n - 1)),
            "exit_reason": "TIME",
            "tp_level": float(tp_level),
            "sl_level": float(sl_level),
            "entry_spread_ticks": entry_spread_ticks,
            "sl_ticks": int(sl_ticks_local),
            "breakeven_ticks": breakeven_ticks_local,
            "breakeven_set": False,
            "breakeven_triggered": False,
            "entry_reason": entry_reason,
            "absorption_level": float(absorption_level) if np.isfinite(absorption_level) else float("nan"),
            "break_level": float(break_level) if np.isfinite(break_level) else float("nan"),
            "flow_align_sum_at_entry": float(flow_align_sum),
            "bars_from_absorption": int(entry_bar - absorption_bar) if absorption_bar >= 0 else -1,
            "absorption_bar": int(absorption_bar),
            "mae_ticks": 0.0,
            "mfe_ticks": 0.0,
            "exit_on_decay": False,
            "exit_on_be": False,
            "side": desired_side,
        }
        in_position = True
        entries_taken += 1
        if strategy_mode == "impulse_confirm_v1":
            impulse_entered += 1
        if strategy_mode == "absorption_failure_v1":
            afr_entered += 1
        if strategy_mode == "absorption_failure_v2":
            afr2_entered += 1
        if strategy_mode == "absorption_failure_v3":
            afr3_entered += 1
        impulse_for_entry = impulse_ticks if np.isfinite(impulse_ticks) else dmid_ticks
        if np.isfinite(impulse_for_entry):
            entry_impulse_sum += float(impulse_for_entry)
            entry_impulse_abs_sum += float(abs(impulse_for_entry))
            if desired_side == "long":
                entry_impulse_sum_long += float(impulse_for_entry)
            else:
                entry_impulse_sum_short += float(impulse_for_entry)
        if np.isfinite(break_ticks):
            entry_break_sum += float(break_ticks)
            entry_break_abs_sum += float(abs(break_ticks))
            if desired_side == "long":
                entry_break_sum_long += float(break_ticks)
            else:
                entry_break_sum_short += float(break_ticks)
        if np.isfinite(ft_progress_ticks):
            entry_ft_sum += float(ft_progress_ticks)
            entry_ft_abs_sum += float(abs(ft_progress_ticks))
        if np.isfinite(flow):
            entry_flow_sum += float(flow)
            entry_flow_abs_sum += float(abs(flow))
            if desired_side == "long":
                entry_flow_sum_long += float(flow)
            else:
                entry_flow_sum_short += float(flow)
        if np.isfinite(spread_ticks[entry_bar]):
            entry_spread_sum += float(spread_ticks[entry_bar])
        entry_count += 1
        if desired_side == "long":
            entry_count_long += 1
        else:
            entry_count_short += 1
        if strategy_mode.startswith("absorption_failure") and breakeven_ticks_local is not None:
            afr_be_armed += 1
        if baseline_mode == "mmas":
            mmas_entered += 1
        if debug_entry_print and entry_debug_printed < debug_entry_limit:
            try:
                print(
                    f"ENTRY_DEBUG {debug_entry_tag} t={i} side={desired_side} "
                    f"mid_t={mid[i]:.4f} mid_tk={mid[i - max(1, int(micro_k_bars))]:.4f} "
                    f"dmid={(mid[i] - mid[i - max(1, int(micro_k_bars))]):.4f} "
                    f"impulse_ticks={impulse_for_entry:.4f} flow={flow:.2f} "
                    f"spread_ticks={spread_ticks[entry_bar]}",
                    flush=True,
                )
            except OSError:
                pass
            entry_debug_printed += 1
        i = entry_bar + 1

    return (
        pd.DataFrame(trades),
        float(pnl_ticks_total),
        skipped,
        total_signals,
        signals_when_flat,
        signals_in_position,
        entries_taken,
        eligible_signals,
        blocked_signals,
        spread_suppressed,
        session_suppressed,
        cooldown_suppressed,
        strategy_long_signals,
        strategy_short_signals,
        float(entry_impulse_sum),
        float(entry_impulse_abs_sum),
        float(entry_impulse_sum_long),
        float(entry_impulse_sum_short),
        float(entry_flow_sum),
        float(entry_flow_abs_sum),
        float(entry_flow_sum_long),
        float(entry_flow_sum_short),
        int(entry_count),
        int(entry_count_long),
        int(entry_count_short),
        float(entry_break_sum),
        float(entry_break_abs_sum),
        float(entry_break_sum_long),
        float(entry_break_sum_short),
        float(entry_ft_sum),
        float(entry_ft_abs_sum),
        float(entry_spread_sum),
        afr_be_armed,
        afr_be_triggered,
        afr_checked,
        afr_absorption_pass,
        afr_break_pass,
        afr_entered,
        afr2_checked,
        afr2_absorption_pass,
        afr2_break_pass,
        afr2_ft_pass,
        afr2_entered,
        afr2_break_quality_pass,
        afr2_snapback_fail,
        afr2_break_quality_entries,
        afr3_checked,
        afr3_absorption_pass,
        afr3_break_pass,
        afr3_break_quality_pass,
        afr3_snapback_pass,
        afr3_entered,
        mmas_signals_checked,
        mmas_passed_filters,
        mmas_signaled,
        mmas_entered,
        impulse_signals_checked,
        impulse_passed_threshold,
        impulse_passed_confirm,
        impulse_entered,
        skip_reasons,
    )


def _equity_stats(pnl: np.ndarray) -> Dict[str, float]:
    if pnl.size == 0:
        return {"final": 0.0, "peak": 0.0, "max_dd": 0.0}
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(equity)
    drawdown = running_peak - equity
    return {
        "final": float(equity[-1]),
        "peak": float(running_peak.max()) if running_peak.size else 0.0,
        "max_dd": float(drawdown.max()) if drawdown.size else 0.0,
    }


def _metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0,
            "total_pnl_ticks": 0.0,
            "final_pnl_ticks": 0.0,
            "peak_equity_ticks": 0.0,
            "mean_pnl_ticks": 0.0,
            "win_rate": 0.0,
            "max_drawdown_ticks": 0.0,
            "p1": 0.0,
            "p5": 0.0,
            "p10": 0.0,
            "worst_trade_ticks": 0.0,
        }
    pnl = trades["pnl_ticks"].to_numpy()
    eq = _equity_stats(pnl)
    return {
        "trade_count": int(len(trades)),
        "total_pnl_ticks": float(np.sum(pnl)),
        "final_pnl_ticks": eq["final"],
        "peak_equity_ticks": eq["peak"],
        "mean_pnl_ticks": float(np.mean(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "max_drawdown_ticks": eq["max_dd"],
        "p1": float(np.quantile(pnl, 0.01)),
        "p5": float(np.quantile(pnl, 0.05)),
        "p10": float(np.quantile(pnl, 0.10)),
        "worst_trade_ticks": float(np.min(pnl)),
    }


def _plot_equity(trades_base: pd.DataFrame, trades_gated: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(10, 5))
    if not trades_base.empty:
        eq_base = np.cumsum(trades_base["pnl_ticks"].to_numpy())
        plt.plot(eq_base, label="baseline")
    if not trades_gated.empty:
        eq_gate = np.cumsum(trades_gated["pnl_ticks"].to_numpy())
        plt.plot(eq_gate, label="gated")
    plt.title("Equity curve (ticks)")
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative PnL (ticks)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _day_health_report(
    df_day: pd.DataFrame,
    events_day: pd.DataFrame,
    tick_size: float,
    lookback_bars: int,
    entry_threshold_ticks: int,
    hold_bars: int,
    spread_min_t: int,
    spread_max_t: int,
) -> Dict[str, object]:
    report: Dict[str, object] = {}
    n_rows = int(len(df_day))
    report["n_rows"] = n_rows
    report["first_time"] = str(df_day["Time"].iloc[0]) if n_rows else None
    report["last_time"] = str(df_day["Time"].iloc[-1]) if n_rows else None

    if n_rows >= 2:
        dt_ms = df_day["Time"].diff().dt.total_seconds().to_numpy() * 1000.0
        dt_ms = dt_ms[np.isfinite(dt_ms) & (dt_ms > 0)]
        report["median_dt_ms"] = float(np.median(dt_ms)) if dt_ms.size else None
    else:
        report["median_dt_ms"] = None

    bid = pd.to_numeric(df_day["bid_price_1"], errors="coerce").to_numpy()
    ask = pd.to_numeric(df_day["ask_price_1"], errors="coerce").to_numpy()
    mid = 0.5 * (bid + ask)
    spread = ask - bid
    spread_ticks = np.rint(spread / tick_size).astype(np.int64)

    def _nan_count(arr: np.ndarray) -> int:
        return int(np.sum(~np.isfinite(arr)))

    report["nan_counts"] = {
        "bid_price_1": _nan_count(bid),
        "ask_price_1": _nan_count(ask),
        "mid": _nan_count(mid),
        "spread": _nan_count(spread),
        "top_bid_depth": _nan_count(pd.to_numeric(df_day["top_bid_depth"], errors="coerce").to_numpy())
        if "top_bid_depth" in df_day.columns
        else None,
        "top_ask_depth": _nan_count(pd.to_numeric(df_day["top_ask_depth"], errors="coerce").to_numpy())
        if "top_ask_depth" in df_day.columns
        else None,
        "signed_volume": _nan_count(pd.to_numeric(df_day["signed_volume"], errors="coerce").to_numpy())
        if "signed_volume" in df_day.columns
        else None,
        "trade_count": _nan_count(pd.to_numeric(df_day["trade_count"], errors="coerce").to_numpy())
        if "trade_count" in df_day.columns
        else None,
    }

    report["unique_counts"] = {
        "bid_price_1": int(pd.Series(bid).nunique(dropna=True)),
        "ask_price_1": int(pd.Series(ask).nunique(dropna=True)),
        "mid": int(pd.Series(mid).nunique(dropna=True)),
    }
    mid_diff = np.diff(mid)
    mid_change_pct = float(np.mean(np.isfinite(mid_diff) & (mid_diff != 0))) if mid_diff.size else 0.0
    report["mid_change_pct"] = mid_change_pct
    spread_ok = (spread_ticks >= spread_min_t) & (spread_ticks <= spread_max_t)
    report["spread_ok_pct"] = float(np.mean(spread_ok)) if spread_ok.size else 0.0

    ret_ticks = (mid - np.roll(mid, lookback_bars)) / tick_size
    ret_ticks[:lookback_bars] = np.nan
    pos_signals = np.isfinite(ret_ticks) & (ret_ticks >= entry_threshold_ticks)
    neg_signals = np.isfinite(ret_ticks) & (ret_ticks <= -entry_threshold_ticks)
    max_i = n_rows - hold_bars - 2
    valid_idx = np.zeros(n_rows, dtype=bool)
    if max_i >= lookback_bars:
        valid_idx[lookback_bars : max_i + 1] = True
    report["signal_counts"] = {
        "pos_total": int(np.sum(pos_signals)),
        "neg_total": int(np.sum(neg_signals)),
        "pos_valid": int(np.sum(pos_signals & valid_idx)),
        "neg_valid": int(np.sum(neg_signals & valid_idx)),
    }

    report["event_counts"] = {
        "events_total": int(len(events_day)),
        "events_with_threshold": int(events_day["asym_threshold"].notna().sum()) if not events_day.empty else 0,
    }

    reasons = []
    if n_rows < (lookback_bars + hold_bars + 3):
        reasons.append("insufficient rows for lookback/hold horizon")
    if report["nan_counts"]["bid_price_1"] and report["nan_counts"]["bid_price_1"] > n_rows * 0.5:
        reasons.append("bid_price_1 missing for most rows")
    if report["nan_counts"]["ask_price_1"] and report["nan_counts"]["ask_price_1"] > n_rows * 0.5:
        reasons.append("ask_price_1 missing for most rows")
    if mid_change_pct < 0.01:
        reasons.append("mid rarely changes")
    if (report["signal_counts"]["pos_valid"] + report["signal_counts"]["neg_valid"]) == 0:
        reasons.append("no valid entry signals")
    if report["spread_ok_pct"] < 0.5:
        reasons.append("spread rarely within expected band")
    report["likely_reasons"] = reasons[:3]

    return report


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--strategy-mode", dest="strategy_mode")
    parser.add_argument("--impulse-lookback-bars", type=int, dest="impulse_lookback_bars")
    parser.add_argument("--impulse-min-ticks", type=int, dest="impulse_min_ticks")
    parser.add_argument("--confirm-bars", type=int, dest="confirm_bars")
    parser.add_argument("--confirm-require-nonzero", type=int, dest="confirm_require_nonzero")
    parser.add_argument("--disable-gate", action="store_true", dest="disable_gate")
    parser.add_argument("--afr-k-bars", type=int, dest="afr_k_bars")
    parser.add_argument("--afr-min-flow-abs", type=float, dest="afr_min_flow_abs")
    parser.add_argument("--afr-stall-ticks", type=int, dest="afr_stall_ticks")
    parser.add_argument("--afr-break-ticks", type=int, dest="afr_break_ticks")
    parser.add_argument("--afr-require-flow-sign", type=int, dest="afr_require_flow_sign")
    parser.add_argument("--afr-use-mid-for-stall", type=int, dest="afr_use_mid_for_stall")
    parser.add_argument("--afr-use-signed-volume", type=int, dest="afr_use_signed_volume")
    parser.add_argument("--afr-ft-bars", type=int, dest="afr_ft_bars")
    parser.add_argument("--afr-ft-min-ticks", type=int, dest="afr_ft_min_ticks")
    parser.add_argument("--afr-ft-no-backtrack", type=int, dest="afr_ft_no_backtrack")
    parser.add_argument("--afr-enter-on", dest="afr_enter_on")
    parser.add_argument("--afr-enter-mode", dest="afr_enter_mode")
    parser.add_argument("--afr-flow-align-bars", type=int, dest="afr_flow_align_bars")
    parser.add_argument("--afr-min-flow-abs-align", type=float, dest="afr_min_flow_abs_align")
    parser.add_argument("--afr-rearm-band-ticks", type=int, dest="afr_rearm_band_ticks")
    parser.add_argument("--afr-rearm-max-bars", type=int, dest="afr_rearm_max_bars")
    parser.add_argument("--afr-rearm-stop-max-bars", type=int, dest="afr_rearm_stop_max_bars")
    parser.add_argument("--afr-momentum-decay-bars", type=int, dest="afr_momentum_decay_bars")
    parser.add_argument("--afr-momentum-decay-min-flow", type=float, dest="afr_momentum_decay_min_flow")
    args, _ = parser.parse_known_args()

    def _arg_or_env(name: str, env_name: str, default: str) -> str:
        val = getattr(args, name, None)
        if val is not None:
            return str(val)
        return os.environ.get(env_name, default)

    tick_size = float(os.environ.get("TICK_SIZE", "0.25"))
    lookback_bars = int(os.environ.get("LOOKBACK_BARS", "10"))
    entry_threshold_ticks = int(os.environ.get("ENTRY_THRESHOLD_TICKS", "1"))
    hold_bars = int(os.environ.get("HOLD_BARS", "20"))
    strategy_mode = _arg_or_env("strategy_mode", "STRATEGY_MODE", "").strip().lower()
    baseline_mode = os.environ.get("BASELINE_MODE", "flat").strip().lower()
    if not strategy_mode:
        strategy_mode = "micro_momo_v1"
    baseline_k_bars = int(os.environ.get("BASELINE_K_BARS", "5"))
    micro_k_bars = int(os.environ.get("MICRO_K_BARS", "5"))
    micro_impulse_ticks = int(os.environ.get("MICRO_IMPULSE_TICKS", "1"))
    micro_flow_min = float(os.environ.get("MICRO_FLOW_MIN", "0"))
    impulse_lookback_bars = int(_arg_or_env("impulse_lookback_bars", "IMPULSE_LOOKBACK_BARS", "1"))
    impulse_min_ticks = int(_arg_or_env("impulse_min_ticks", "IMPULSE_MIN_TICKS", "1"))
    confirm_bars = int(_arg_or_env("confirm_bars", "CONFIRM_BARS", "2"))
    confirm_require_nonzero = _arg_or_env("confirm_require_nonzero", "CONFIRM_REQUIRE_NONZERO", "1").strip() == "1"
    debug_first_impulse = os.environ.get("DEBUG_FIRST_IMPULSE", "0").strip() == "1"
    debug_flow_stats = os.environ.get("DEBUG_FLOW_STATS", "0").strip() == "1"
    disable_gate = args.disable_gate or os.environ.get("DISABLE_GATE", "0").strip() == "1"
    debug_first_afr = os.environ.get("DEBUG_FIRST_AFR", "0").strip() == "1"
    debug_first_afr2 = os.environ.get("DEBUG_FIRST_AFR2", "0").strip() == "1"
    debug_first_afr3 = os.environ.get("DEBUG_FIRST_AFR3", "0").strip() == "1"
    afr_k_bars = int(_arg_or_env("afr_k_bars", "AFR_K_BARS", "10"))
    afr_min_flow_abs = float(_arg_or_env("afr_min_flow_abs", "AFR_MIN_FLOW_ABS", "40"))
    afr_stall_ticks = int(_arg_or_env("afr_stall_ticks", "AFR_STALL_TICKS", "0"))
    afr_break_ticks = int(_arg_or_env("afr_break_ticks", "AFR_BREAK_TICKS", "1"))
    afr_require_flow_sign = _arg_or_env("afr_require_flow_sign", "AFR_REQUIRE_FLOW_SIGN", "1").strip() == "1"
    afr_use_mid_for_stall = _arg_or_env("afr_use_mid_for_stall", "AFR_USE_MID_FOR_STALL", "1").strip() == "1"
    afr_use_signed_volume = _arg_or_env("afr_use_signed_volume", "AFR_USE_SIGNED_VOLUME", "1").strip() == "1"
    afr_ft_bars = int(_arg_or_env("afr_ft_bars", "AFR_FT_BARS", "1"))
    afr_ft_min_ticks = int(_arg_or_env("afr_ft_min_ticks", "AFR_FT_MIN_TICKS", "0"))
    afr_ft_no_backtrack = _arg_or_env("afr_ft_no_backtrack", "AFR_FT_NO_BACKTRACK", "1").strip() == "1"
    afr_enter_on = _arg_or_env("afr_enter_on", "AFR_ENTER_ON", "break").strip().lower()
    if afr_enter_on not in {"break", "ft", "both"}:
        raise ValueError(f"Invalid AFR_ENTER_ON: {afr_enter_on}")
    afr_break_quality_min_flow_abs = float(os.environ.get("AFR_BREAK_QUALITY_MIN_FLOW_ABS", "80"))
    afr_break_quality_max_spread_ticks = int(os.environ.get("AFR_BREAK_QUALITY_MAX_SPREAD_TICKS", "1"))
    afr_snapback_bars = int(os.environ.get("AFR_SNAPBACK_BARS", "2"))
    afr_snapback_band_ticks = int(os.environ.get("AFR_SNAPBACK_BAND_TICKS", "1"))
    afr_enter_mode = _arg_or_env("afr_enter_mode", "AFR_ENTER_MODE", "break_first").strip().lower()
    afr_flow_align_bars = int(_arg_or_env("afr_flow_align_bars", "AFR_FLOW_ALIGN_BARS", "5"))
    afr_min_flow_abs_align = float(_arg_or_env("afr_min_flow_abs_align", "AFR_MIN_FLOW_ABS_ALIGN", "40"))
    afr_rearm_band_ticks = int(_arg_or_env("afr_rearm_band_ticks", "AFR_REARM_BAND_TICKS", "1"))
    afr_rearm_max_bars = int(_arg_or_env("afr_rearm_max_bars", "AFR_REARM_MAX_BARS", "5"))
    afr_rearm_stop_max_bars = int(_arg_or_env("afr_rearm_stop_max_bars", "AFR_REARM_STOP_MAX_BARS", "3"))
    afr_momentum_decay_bars = int(_arg_or_env("afr_momentum_decay_bars", "AFR_MOMENTUM_DECAY_BARS", "3"))
    afr_momentum_decay_min_flow = float(_arg_or_env("afr_momentum_decay_min_flow", "AFR_MOMENTUM_DECAY_MIN_FLOW", "0"))
    afr3_break_min_flow_abs = float(os.environ.get("AFR3_BREAK_MIN_FLOW_ABS", "100"))
    afr3_break_max_spread_ticks = int(os.environ.get("AFR3_BREAK_MAX_SPREAD_TICKS", "2"))
    afr3_snapback_check = os.environ.get("AFR3_SNAPBACK_CHECK", "1").strip() == "1"
    afr_tp_env = os.environ.get("AFR_TP_TICKS", "").strip()
    afr_sl_env = os.environ.get("AFR_SL_TICKS", "").strip()
    afr_hold_env = os.environ.get("AFR_MAX_HOLD_BARS", "").strip()
    afr_be_env = os.environ.get("AFR_BREAKEVEN_AFTER_TICKS", "").strip()
    afr_tp_ticks = int(afr_tp_env) if afr_tp_env else None
    afr_sl_ticks = int(afr_sl_env) if afr_sl_env else None
    afr_max_hold_bars = int(afr_hold_env) if afr_hold_env else None
    afr_breakeven_after_ticks = int(afr_be_env) if afr_be_env else None
    run_mode = os.environ.get("RUN_MODE", "baseline_vs_gated").strip().lower()
    max_spread_ticks_for_entry = int(os.environ.get("MAX_SPREAD_TICKS_FOR_ENTRY", "2"))
    mmas_k_bars = int(os.environ.get("MMAS_K_BARS", "5"))
    mmas_min_dmid_ticks = int(os.environ.get("MMAS_MIN_DMID_TICKS", "1"))
    mmas_min_flow_abs = float(os.environ.get("MMAS_MIN_FLOW_ABS", "20"))
    mmas_require_agree = os.environ.get("MMAS_REQUIRE_AGREE", "1").strip() == "1"
    debug_first_mmas = os.environ.get("DEBUG_FIRST_MMAS", "0").strip() == "1"
    trade_session = os.environ.get("TRADE_SESSION", "all").strip().lower()
    min_spread_ticks = int(os.environ.get("MIN_SPREAD_TICKS", "0"))
    entry_cooldown_bars = int(os.environ.get("ENTRY_COOLDOWN_BARS", "10"))
    gate_lookback_bars = int(os.environ.get("GATE_LOOKBACK_BARS", "10"))
    gate_mode = os.environ.get("GATE_MODE", "side_matched").strip().lower()
    gate_mode_list_env = os.environ.get("GATE_MODE_LIST", "").strip()
    if gate_mode_list_env:
        gate_modes = [m.strip().lower() for m in gate_mode_list_env.split(",") if m.strip()]
    else:
        gate_modes = [gate_mode]
    worst_q = float(os.environ.get("WORST_Q", "0.90"))
    tp_ticks = int(os.environ.get("TP_TICKS", "1"))
    sl_ticks = int(os.environ.get("SL_TICKS", "2"))

    v_min = float(os.environ.get("V_MIN", "1"))
    v_max = float(os.environ.get("V_MAX", "10"))
    spread_min_t = int(os.environ.get("SPREAD_MIN_T", "1"))
    spread_max_t = int(os.environ.get("SPREAD_MAX_T", "4"))
    stable_bars = int(os.environ.get("STABLE_BARS", "3"))
    refill_bars = int(os.environ.get("REFILL_BARS", "10"))

    out_root = Path(os.environ.get("OUTPUT_DIR", "artifacts/lrams_gate")).expanduser().resolve()
    run_tag = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = out_root / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_env = os.environ.get("SWEEP_W", "").strip()
    if sweep_env:
        sweep_ws = [int(x.strip()) for x in sweep_env.split(",") if x.strip()]
    else:
        sweep_ws = [gate_lookback_bars]

    data_dir = Path(os.environ.get("DATA_DIR", "data/processed")).expanduser().resolve()
    instrument = os.environ.get("INSTRUMENT", "ES").strip()
    selected_days = _select_days(data_dir, instrument)
    if not selected_days:
        raise ValueError("No days selected; check DATA_DIR and date filters.")
    print(f"Discovered {len(selected_days)} processed days: {selected_days}", flush=True)
    if os.environ.get("USE_ALL_AVAILABLE_DAYS", "0").strip() == "1" and len(selected_days) < 5:
        print("Warning: USE_ALL_AVAILABLE_DAYS selected fewer than 5 days; low-confidence results.", flush=True)
    if len(selected_days) < 3:
        print("Warning: fewer than 3 days selected; continuing.", flush=True)
    print(
        "Run config:",
        {
            "strategy_mode": strategy_mode,
            "baseline_mode": baseline_mode,
            "baseline_k_bars": baseline_k_bars,
            "micro_k_bars": micro_k_bars,
            "micro_impulse_ticks": micro_impulse_ticks,
            "micro_flow_min": micro_flow_min,
            "max_spread_ticks_for_entry": max_spread_ticks_for_entry,
            "mmas_k_bars": mmas_k_bars,
            "mmas_min_dmid_ticks": mmas_min_dmid_ticks,
            "mmas_min_flow_abs": mmas_min_flow_abs,
            "mmas_require_agree": mmas_require_agree,
            "trade_session": trade_session,
            "min_spread_ticks": min_spread_ticks,
            "entry_cooldown_bars": entry_cooldown_bars,
            "tp_ticks": tp_ticks,
            "sl_ticks": sl_ticks,
            "afr3_break_min_flow_abs": afr3_break_min_flow_abs,
            "afr3_break_max_spread_ticks": afr3_break_max_spread_ticks,
            "afr3_snapback_check": afr3_snapback_check,
            "gate_mode_list": gate_modes,
            "sweep_ws": sweep_ws,
        },
        flush=True,
    )
    print(
        "Run header:",
        {
            "strategy_mode": strategy_mode,
            "gate_modes": gate_modes,
            "sweep_ws": sweep_ws,
            "disable_gate": disable_gate,
            "run_mode": run_mode,
            "impulse_lookback_bars": impulse_lookback_bars,
            "impulse_min_ticks": impulse_min_ticks,
            "confirm_bars": confirm_bars,
            "confirm_require_nonzero": confirm_require_nonzero,
            "afr_k_bars": afr_k_bars,
            "afr_min_flow_abs": afr_min_flow_abs,
            "afr_stall_ticks": afr_stall_ticks,
            "afr_break_ticks": afr_break_ticks,
            "afr_require_flow_sign": afr_require_flow_sign,
            "afr_enter_on": afr_enter_on,
            "afr_tp_ticks": afr_tp_ticks,
            "afr_sl_ticks": afr_sl_ticks,
            "afr_max_hold_bars": afr_max_hold_bars,
            "afr_breakeven_after_ticks": afr_breakeven_after_ticks,
            "afr_enter_mode": afr_enter_mode,
            "afr_flow_align_bars": afr_flow_align_bars,
            "afr_min_flow_abs_align": afr_min_flow_abs_align,
            "afr_rearm_band_ticks": afr_rearm_band_ticks,
            "afr_rearm_max_bars": afr_rearm_max_bars,
            "afr_rearm_stop_max_bars": afr_rearm_stop_max_bars,
            "afr_momentum_decay_bars": afr_momentum_decay_bars,
            "afr_momentum_decay_min_flow": afr_momentum_decay_min_flow,
            "trade_session": trade_session,
            "min_spread_ticks": min_spread_ticks,
            "entry_cooldown_bars": entry_cooldown_bars,
        },
        flush=True,
    )
    print(f"AFR_ENTER_ON={afr_enter_on} strategy_mode={strategy_mode}", flush=True)
    if run_mode == "gated_only":
        print("RUN_MODE: gated_only (no baseline)", flush=True)
    else:
        print("RUN_MODE: baseline_vs_gated (default)", flush=True)
    if strategy_mode == "impulse_confirm_v1":
        print(
            "Impulse config:",
            {
                "impulse_lookback_bars": impulse_lookback_bars,
                "impulse_min_ticks": impulse_min_ticks,
                "confirm_bars": confirm_bars,
                "confirm_require_nonzero": confirm_require_nonzero,
                "debug_first_impulse": debug_first_impulse,
            },
            flush=True,
        )
    if strategy_mode == "absorption_failure_v1":
        print(
            "AFR config:",
            {
                "afr_k_bars": afr_k_bars,
                "afr_min_flow_abs": afr_min_flow_abs,
                "afr_stall_ticks": afr_stall_ticks,
                "afr_break_ticks": afr_break_ticks,
                "afr_require_flow_sign": afr_require_flow_sign,
                "afr_use_mid_for_stall": afr_use_mid_for_stall,
                "afr_use_signed_volume": afr_use_signed_volume,
                "debug_first_afr": debug_first_afr,
            },
            flush=True,
        )
    if strategy_mode == "absorption_failure_v2":
        print(
            "AFR2 config:",
            {
                "afr_k_bars": afr_k_bars,
                "afr_min_flow_abs": afr_min_flow_abs,
                "afr_stall_ticks": afr_stall_ticks,
                "afr_break_ticks": afr_break_ticks,
                "afr_require_flow_sign": afr_require_flow_sign,
                "afr_use_mid_for_stall": afr_use_mid_for_stall,
                "afr_use_signed_volume": afr_use_signed_volume,
                "afr_ft_bars": afr_ft_bars,
                "afr_ft_min_ticks": afr_ft_min_ticks,
                "afr_ft_no_backtrack": afr_ft_no_backtrack,
                "afr_enter_on": afr_enter_on,
                "afr_break_quality_min_flow_abs": afr_break_quality_min_flow_abs,
                "afr_break_quality_max_spread_ticks": afr_break_quality_max_spread_ticks,
                "afr_snapback_bars": afr_snapback_bars,
                "afr_snapback_band_ticks": afr_snapback_band_ticks,
                "debug_first_afr2": debug_first_afr2,
                "afr_enter_mode": afr_enter_mode,
                "afr_flow_align_bars": afr_flow_align_bars,
                "afr_min_flow_abs_align": afr_min_flow_abs_align,
                "afr_rearm_band_ticks": afr_rearm_band_ticks,
                "afr_rearm_max_bars": afr_rearm_max_bars,
                "afr_rearm_stop_max_bars": afr_rearm_stop_max_bars,
                "afr_momentum_decay_bars": afr_momentum_decay_bars,
                "afr_momentum_decay_min_flow": afr_momentum_decay_min_flow,
            },
            flush=True,
        )
    if strategy_mode == "absorption_failure_v3":
        print(
            "AFR3 config:",
            {
                "afr_k_bars": afr_k_bars,
                "afr_min_flow_abs": afr_min_flow_abs,
                "afr_stall_ticks": afr_stall_ticks,
                "afr_break_ticks": afr_break_ticks,
                "afr_require_flow_sign": afr_require_flow_sign,
                "afr_use_mid_for_stall": afr_use_mid_for_stall,
                "afr_use_signed_volume": afr_use_signed_volume,
                "afr3_break_min_flow_abs": afr3_break_min_flow_abs,
                "afr3_break_max_spread_ticks": afr3_break_max_spread_ticks,
                "afr3_snapback_check": afr3_snapback_check,
                "debug_first_afr3": debug_first_afr3,
            },
            flush=True,
        )
    if strategy_mode.startswith("absorption_failure"):
        print(
            "AFR risk overrides:",
            {
                "afr_tp_ticks": afr_tp_ticks,
                "afr_sl_ticks": afr_sl_ticks,
                "afr_max_hold_bars": afr_max_hold_bars,
                "afr_breakeven_after_ticks": afr_breakeven_after_ticks,
            },
            flush=True,
        )
    df = _load_data()
    df["date"] = df["Time"].dt.date.astype(str)
    df = df[(df["Symbol"] == instrument) & (df["date"].isin(selected_days))].reset_index(drop=True)
    if df.empty:
        raise ValueError("No data after applying date filters.")
    events = _build_events(
        df,
        v_min=v_min,
        v_max=v_max,
        spread_min_t=spread_min_t,
        spread_max_t=spread_max_t,
        stable_bars=stable_bars,
        refill_bars=refill_bars,
        tick_size=tick_size,
    )
    events = _compute_thresholds(events, worst_q=worst_q)

    sweep_rows = []
    skipped_days: List[str] = []
    day_count = len(selected_days)
    flow_warned = False
    afr_flow_warned = False

    health_dir = out_dir / "health"
    health_dir.mkdir(parents=True, exist_ok=True)

    for gate_mode in gate_modes:
        for gate_lookback_bars in sweep_ws:
            out_dir_w = out_dir / f"W{gate_lookback_bars}" / f"mode={gate_mode}"
            out_dir_w.mkdir(parents=True, exist_ok=True)
            print(
                "Executing run:",
                {
                    "strategy_mode": strategy_mode,
                    "W": gate_lookback_bars,
                    "gate_mode": gate_mode,
                    "disable_gate": disable_gate,
                    "trade_session": trade_session,
                    "min_spread_ticks": min_spread_ticks,
                    "entry_cooldown_bars": entry_cooldown_bars,
                },
                flush=True,
            )

            summaries = []
            all_base = []
            all_gated = []
            all_events = []
            strategy_rows = []

            day_index = 0
            sample_printed = False
            for day in selected_days:
                day_index += 1
                try:
                    df_day = df[df["date"] == day].sort_values("Time").reset_index(drop=True)
                    if df_day.empty:
                        raise ValueError("No rows for day after filtering")
                    if "signed_volume" not in df_day.columns:
                        if not flow_warned:
                            print("Warning: signed_volume missing; flow_at_entry stats will be 0.0.", flush=True)
                            flow_warned = True
                        df_day["signed_volume"] = 0.0
                    if not afr_use_signed_volume and "aggressor_count_imbalance" not in df_day.columns:
                        if not afr_flow_warned:
                            print(
                                "Warning: aggressor_count_imbalance missing; AFR flow uses signed_volume.",
                                flush=True,
                            )
                            afr_flow_warned = True
                    elif debug_flow_stats:
                        sv = pd.to_numeric(df_day["signed_volume"], errors="coerce").fillna(0.0).to_numpy()
                        try:
                            print(
                                f"{instrument} {day} signed_volume stats "
                                f"min={float(np.min(sv)):.4f} mean={float(np.mean(sv)):.4f} "
                                f"max={float(np.max(sv)):.4f} nonzero={int(np.count_nonzero(sv))}",
                                flush=True,
                            )
                        except OSError:
                            pass
                    events_day = events[(events["date"] == day) & (events["Symbol"] == instrument)].sort_values(
                        "event_pos"
                    )
                    all_events.append(events_day)

                    health = _day_health_report(
                        df_day,
                        events_day,
                        tick_size=tick_size,
                        lookback_bars=lookback_bars,
                        entry_threshold_ticks=entry_threshold_ticks,
                        hold_bars=hold_bars,
                        spread_min_t=spread_min_t,
                        spread_max_t=spread_max_t,
                    )
                    health_path = health_dir / f"{instrument}_{day}_health.json"
                    with open(health_path, "w", encoding="utf-8") as f:
                        json.dump(health, f, indent=2)
                    print(
                        f"{instrument} {day} health n_rows={health['n_rows']} "
                        f"median_dt_ms={health['median_dt_ms']} "
                        f"mid_change_pct={health['mid_change_pct']:.2%} "
                        f"signals_valid={health['signal_counts']['pos_valid'] + health['signal_counts']['neg_valid']} "
                        f"events={health['event_counts']['events_total']}",
                        flush=True,
                    )

                    debug_entry = strategy_mode == "micro_momo_v1" and day == selected_days[0]
                    run_baseline = run_mode != "gated_only"
                    if run_baseline:
                        (
                            trades_base,
                            pnl_total_base,
                            skipped_base,
                            total_signals_base,
                            signals_flat_base,
                            signals_in_pos_base,
                            entries_taken_base,
                            eligible_base,
                            blocked_base,
                            spread_supp_base,
                            session_supp_base,
                            cooldown_supp_base,
                            strategy_long_base,
                            strategy_short_base,
                            entry_impulse_sum_base,
                            entry_impulse_abs_sum_base,
                            entry_impulse_sum_long_base,
                            entry_impulse_sum_short_base,
                            entry_flow_sum_base,
                            entry_flow_abs_sum_base,
                            entry_flow_sum_long_base,
                            entry_flow_sum_short_base,
                            entry_count_base,
                            entry_count_long_base,
                            entry_count_short_base,
                            entry_break_sum_base,
                            entry_break_abs_sum_base,
                            entry_break_sum_long_base,
                            entry_break_sum_short_base,
                            entry_ft_sum_base,
                            entry_ft_abs_sum_base,
                            entry_spread_sum_base,
                            afr_be_armed_base,
                            afr_be_triggered_base,
                            afr_checked_base,
                            afr_absorption_pass_base,
                            afr_break_pass_base,
                            afr_entered_base,
                            afr2_checked_base,
                            afr2_absorption_pass_base,
                            afr2_break_pass_base,
                            afr2_ft_pass_base,
                            afr2_entered_base,
                            afr2_break_quality_pass_base,
                            afr2_snapback_fail_base,
                            afr2_break_quality_entries_base,
                            afr3_checked_base,
                            afr3_absorption_pass_base,
                            afr3_break_pass_base,
                            afr3_break_quality_pass_base,
                            afr3_snapback_pass_base,
                            afr3_entered_base,
                            mmas_checked_base,
                            mmas_passed_base,
                            mmas_signaled_base,
                            mmas_entered_base,
                            impulse_checked_base,
                            impulse_passed_thr_base,
                            impulse_passed_confirm_base,
                            impulse_entered_base,
                            skip_base,
                        ) = _simulate_day(
                            df_day,
                            events_day,
                            tick_size=tick_size,
                            lookback_bars=lookback_bars,
                            entry_threshold_ticks=entry_threshold_ticks,
                            hold_bars=hold_bars,
                            tp_ticks=tp_ticks,
                            sl_ticks=sl_ticks,
                            baseline_mode=baseline_mode,
                            baseline_k_bars=baseline_k_bars,
                            strategy_mode=strategy_mode,
                            micro_k_bars=micro_k_bars,
                            micro_impulse_ticks=micro_impulse_ticks,
                            micro_flow_min=micro_flow_min,
                            impulse_lookback_bars=impulse_lookback_bars,
                            impulse_min_ticks=impulse_min_ticks,
                            confirm_bars=confirm_bars,
                            confirm_require_nonzero=confirm_require_nonzero,
                            debug_first_impulse=debug_first_impulse,
                            max_spread_ticks_for_entry=max_spread_ticks_for_entry,
                            mmas_k_bars=mmas_k_bars,
                            mmas_min_dmid_ticks=mmas_min_dmid_ticks,
                            mmas_min_flow_abs=mmas_min_flow_abs,
                            mmas_require_agree=mmas_require_agree,
                            debug_first_mmas=debug_first_mmas,
                            trade_session=trade_session,
                            min_spread_ticks=min_spread_ticks,
                            entry_cooldown_bars=entry_cooldown_bars,
                            gate_lookback_bars=gate_lookback_bars,
                            gated=False,
                            disable_gate=False,
                            gate_mode=gate_mode,
                            afr_k_bars=afr_k_bars,
                            afr_min_flow_abs=afr_min_flow_abs,
                            afr_stall_ticks=afr_stall_ticks,
                            afr_break_ticks=afr_break_ticks,
                            afr_require_flow_sign=afr_require_flow_sign,
                            afr_use_mid_for_stall=afr_use_mid_for_stall,
                            afr_use_signed_volume=afr_use_signed_volume,
                            afr_ft_bars=afr_ft_bars,
                            afr_ft_min_ticks=afr_ft_min_ticks,
                            afr_ft_no_backtrack=afr_ft_no_backtrack,
                            afr_enter_on=afr_enter_on,
                            afr_break_quality_min_flow_abs=afr_break_quality_min_flow_abs,
                            afr_break_quality_max_spread_ticks=afr_break_quality_max_spread_ticks,
                            afr_snapback_bars=afr_snapback_bars,
                            afr_snapback_band_ticks=afr_snapback_band_ticks,
                            debug_first_afr=debug_first_afr,
                            debug_first_afr2=debug_first_afr2,
                            afr3_break_min_flow_abs=afr3_break_min_flow_abs,
                            afr3_break_max_spread_ticks=afr3_break_max_spread_ticks,
                            afr3_snapback_check=afr3_snapback_check,
                            debug_first_afr3=debug_first_afr3,
                            afr_tp_ticks=afr_tp_ticks,
                            afr_sl_ticks=afr_sl_ticks,
                            afr_max_hold_bars=afr_max_hold_bars,
                            afr_breakeven_after_ticks=afr_breakeven_after_ticks,
                            afr_enter_mode=afr_enter_mode,
                            afr_flow_align_bars=afr_flow_align_bars,
                            afr_min_flow_abs_align=afr_min_flow_abs_align,
                            afr_rearm_band_ticks=afr_rearm_band_ticks,
                            afr_rearm_max_bars=afr_rearm_max_bars,
                            afr_rearm_stop_max_bars=afr_rearm_stop_max_bars,
                            afr_momentum_decay_bars=afr_momentum_decay_bars,
                            afr_momentum_decay_min_flow=afr_momentum_decay_min_flow,
                            debug_entry_print=debug_entry,
                            debug_entry_limit=5,
                            debug_entry_tag="baseline",
                        )
                    else:
                        trades_base = pd.DataFrame({"pnl_ticks": pd.Series(dtype=float)})
                        pnl_total_base = 0.0
                        skipped_base = 0
                        total_signals_base = 0
                        signals_flat_base = 0
                        signals_in_pos_base = 0
                        entries_taken_base = 0
                        eligible_base = 0
                        blocked_base = 0
                        spread_supp_base = 0
                        session_supp_base = 0
                        cooldown_supp_base = 0
                        strategy_long_base = 0
                        strategy_short_base = 0
                        entry_impulse_sum_base = 0.0
                        entry_impulse_abs_sum_base = 0.0
                        entry_impulse_sum_long_base = 0.0
                        entry_impulse_sum_short_base = 0.0
                        entry_flow_sum_base = 0.0
                        entry_flow_abs_sum_base = 0.0
                        entry_flow_sum_long_base = 0.0
                        entry_flow_sum_short_base = 0.0
                        entry_count_base = 0
                        entry_count_long_base = 0
                        entry_count_short_base = 0
                        entry_break_sum_base = 0.0
                        entry_break_abs_sum_base = 0.0
                        entry_break_sum_long_base = 0.0
                        entry_break_sum_short_base = 0.0
                        entry_ft_sum_base = 0.0
                        entry_ft_abs_sum_base = 0.0
                        entry_spread_sum_base = 0.0
                        afr_be_armed_base = 0
                        afr_be_triggered_base = 0
                        afr_checked_base = 0
                        afr_absorption_pass_base = 0
                        afr_break_pass_base = 0
                        afr_entered_base = 0
                        afr2_checked_base = 0
                        afr2_absorption_pass_base = 0
                        afr2_break_pass_base = 0
                        afr2_ft_pass_base = 0
                        afr2_entered_base = 0
                        afr2_break_quality_pass_base = 0
                        afr2_snapback_fail_base = 0
                        afr2_break_quality_entries_base = 0
                        afr3_checked_base = 0
                        afr3_absorption_pass_base = 0
                        afr3_break_pass_base = 0
                        afr3_break_quality_pass_base = 0
                        afr3_snapback_pass_base = 0
                        afr3_entered_base = 0
                        mmas_checked_base = 0
                        mmas_passed_base = 0
                        mmas_signaled_base = 0
                        mmas_entered_base = 0
                        impulse_checked_base = 0
                        impulse_passed_thr_base = 0
                        impulse_passed_confirm_base = 0
                        impulse_entered_base = 0
                        skip_base = {}
                    (
                        trades_gate,
                        pnl_total_gate,
                        skipped_gate,
                        total_signals_gate,
                        signals_flat_gate,
                        signals_in_pos_gate,
                        entries_taken_gate,
                        eligible_gate,
                        blocked_gate,
                        spread_supp_gate,
                        session_supp_gate,
                        cooldown_supp_gate,
                        strategy_long_gate,
                        strategy_short_gate,
                        entry_impulse_sum_gate,
                        entry_impulse_abs_sum_gate,
                        entry_impulse_sum_long_gate,
                        entry_impulse_sum_short_gate,
                        entry_flow_sum_gate,
                        entry_flow_abs_sum_gate,
                        entry_flow_sum_long_gate,
                        entry_flow_sum_short_gate,
                        entry_count_gate,
                        entry_count_long_gate,
                        entry_count_short_gate,
                        entry_break_sum_gate,
                        entry_break_abs_sum_gate,
                        entry_break_sum_long_gate,
                        entry_break_sum_short_gate,
                        entry_ft_sum_gate,
                        entry_ft_abs_sum_gate,
                        entry_spread_sum_gate,
                        afr_be_armed_gate,
                        afr_be_triggered_gate,
                        afr_checked_gate,
                        afr_absorption_pass_gate,
                        afr_break_pass_gate,
                        afr_entered_gate,
                        afr2_checked_gate,
                        afr2_absorption_pass_gate,
                        afr2_break_pass_gate,
                        afr2_ft_pass_gate,
                        afr2_entered_gate,
                        afr2_break_quality_pass_gate,
                        afr2_snapback_fail_gate,
                        afr2_break_quality_entries_gate,
                        afr3_checked_gate,
                        afr3_absorption_pass_gate,
                        afr3_break_pass_gate,
                        afr3_break_quality_pass_gate,
                        afr3_snapback_pass_gate,
                        afr3_entered_gate,
                        mmas_checked_gate,
                        mmas_passed_gate,
                        mmas_signaled_gate,
                        mmas_entered_gate,
                        impulse_checked_gate,
                        impulse_passed_thr_gate,
                        impulse_passed_confirm_gate,
                        impulse_entered_gate,
                        skip_gate,
                    ) = _simulate_day(
                        df_day,
                        events_day,
                        tick_size=tick_size,
                        lookback_bars=lookback_bars,
                        entry_threshold_ticks=entry_threshold_ticks,
                        hold_bars=hold_bars,
                        tp_ticks=tp_ticks,
                        sl_ticks=sl_ticks,
                        baseline_mode=baseline_mode,
                        baseline_k_bars=baseline_k_bars,
                        strategy_mode=strategy_mode,
                        micro_k_bars=micro_k_bars,
                        micro_impulse_ticks=micro_impulse_ticks,
                        micro_flow_min=micro_flow_min,
                        impulse_lookback_bars=impulse_lookback_bars,
                        impulse_min_ticks=impulse_min_ticks,
                        confirm_bars=confirm_bars,
                        confirm_require_nonzero=confirm_require_nonzero,
                        debug_first_impulse=debug_first_impulse,
                        max_spread_ticks_for_entry=max_spread_ticks_for_entry,
                        mmas_k_bars=mmas_k_bars,
                        mmas_min_dmid_ticks=mmas_min_dmid_ticks,
                        mmas_min_flow_abs=mmas_min_flow_abs,
                        mmas_require_agree=mmas_require_agree,
                        debug_first_mmas=debug_first_mmas,
                        trade_session=trade_session,
                        min_spread_ticks=min_spread_ticks,
                        entry_cooldown_bars=entry_cooldown_bars,
                        gate_lookback_bars=gate_lookback_bars,
                        gated=True,
                        disable_gate=disable_gate,
                        gate_mode=gate_mode,
                        afr_k_bars=afr_k_bars,
                        afr_min_flow_abs=afr_min_flow_abs,
                        afr_stall_ticks=afr_stall_ticks,
                        afr_break_ticks=afr_break_ticks,
                        afr_require_flow_sign=afr_require_flow_sign,
                        afr_use_mid_for_stall=afr_use_mid_for_stall,
                        afr_use_signed_volume=afr_use_signed_volume,
                        afr_ft_bars=afr_ft_bars,
                        afr_ft_min_ticks=afr_ft_min_ticks,
                        afr_ft_no_backtrack=afr_ft_no_backtrack,
                        afr_enter_on=afr_enter_on,
                        afr_break_quality_min_flow_abs=afr_break_quality_min_flow_abs,
                        afr_break_quality_max_spread_ticks=afr_break_quality_max_spread_ticks,
                        afr_snapback_bars=afr_snapback_bars,
                        afr_snapback_band_ticks=afr_snapback_band_ticks,
                        debug_first_afr=debug_first_afr,
                        debug_first_afr2=debug_first_afr2,
                        afr3_break_min_flow_abs=afr3_break_min_flow_abs,
                        afr3_break_max_spread_ticks=afr3_break_max_spread_ticks,
                        afr3_snapback_check=afr3_snapback_check,
                        debug_first_afr3=debug_first_afr3,
                        afr_tp_ticks=afr_tp_ticks,
                        afr_sl_ticks=afr_sl_ticks,
                        afr_max_hold_bars=afr_max_hold_bars,
                        afr_breakeven_after_ticks=afr_breakeven_after_ticks,
                        afr_enter_mode=afr_enter_mode,
                        afr_flow_align_bars=afr_flow_align_bars,
                        afr_min_flow_abs_align=afr_min_flow_abs_align,
                        afr_rearm_band_ticks=afr_rearm_band_ticks,
                        afr_rearm_max_bars=afr_rearm_max_bars,
                        afr_rearm_stop_max_bars=afr_rearm_stop_max_bars,
                        afr_momentum_decay_bars=afr_momentum_decay_bars,
                        afr_momentum_decay_min_flow=afr_momentum_decay_min_flow,
                        debug_entry_print=debug_entry,
                        debug_entry_limit=5,
                        debug_entry_tag="gated",
                    )
                    if disable_gate:
                        blocked_gate = 0
                        eligible_gate = signals_flat_gate

                    if not sample_printed and day == selected_days[0]:
                        gate_eq = _equity_stats(trades_gate["pnl_ticks"].to_numpy())
                        gate_stats_sample = _metrics(trades_gate)
                        if run_baseline:
                            base_eq = _equity_stats(trades_base["pnl_ticks"].to_numpy())
                            base_stats_sample = _metrics(trades_base)
                            print(
                                f"Sample day {instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                                f"baseline_final={base_eq['final']:.2f} baseline_peak={base_eq['peak']:.2f} "
                                f"baseline_max_dd={base_eq['max_dd']:.2f} baseline_worst={base_stats_sample['worst_trade_ticks']:.2f} "
                                f"baseline_p1={base_stats_sample['p1']:.2f} baseline_p5={base_stats_sample['p5']:.2f} | "
                                f"gated_final={gate_eq['final']:.2f} gated_peak={gate_eq['peak']:.2f} "
                                f"gated_max_dd={gate_eq['max_dd']:.2f} gated_worst={gate_stats_sample['worst_trade_ticks']:.2f} "
                                f"gated_p1={gate_stats_sample['p1']:.2f} gated_p5={gate_stats_sample['p5']:.2f}",
                                flush=True,
                            )
                            assert base_eq["max_dd"] >= 0 and gate_eq["max_dd"] >= 0
                            assert base_eq["peak"] >= base_eq["final"] or trades_base.empty
                        else:
                            print(
                                f"Sample day {instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                                f"gated_final={gate_eq['final']:.2f} gated_peak={gate_eq['peak']:.2f} "
                                f"gated_max_dd={gate_eq['max_dd']:.2f} gated_worst={gate_stats_sample['worst_trade_ticks']:.2f} "
                                f"gated_p1={gate_stats_sample['p1']:.2f} gated_p5={gate_stats_sample['p5']:.2f}",
                                flush=True,
                            )
                            assert gate_eq["max_dd"] >= 0
                        assert gate_eq["peak"] >= gate_eq["final"] or trades_gate.empty
                        sample_printed = True

                    if run_baseline and len(trades_base) == 0:
                        reasons = health.get("likely_reasons", [])
                        if reasons:
                            print(
                                f"{instrument} {day} baseline_trades=0 likely_reasons={reasons}",
                                flush=True,
                            )
                        else:
                            print(f"{instrument} {day} baseline_trades=0 reason=unknown", flush=True)

                    assert np.isfinite(pnl_total_base), "baseline pnl total is not finite"
                    assert np.isfinite(pnl_total_gate), "gated pnl total is not finite"
                    trades_base["date"] = day
                    trades_base["Symbol"] = instrument
                    trades_gate["date"] = day
                    trades_gate["Symbol"] = instrument
                    all_base.append(trades_base)
                    all_gated.append(trades_gate)

                    base_stats = _metrics(trades_base)
                    gate_stats = _metrics(trades_gate)
                    base_stats.update(
                        {
                            "strategy": "baseline",
                            "date": day,
                            "Symbol": instrument,
                            "W": gate_lookback_bars,
                            "gate_mode": gate_mode,
                            "baseline_mode": baseline_mode,
                            "strategy_mode": strategy_mode,
                            "trade_session": trade_session,
                            "min_spread_ticks": min_spread_ticks,
                            "entry_cooldown_bars": entry_cooldown_bars,
                            "skipped_trades": int(skipped_base),
                            "skip_rate": 0.0,
                            "total_signals": int(total_signals_base),
                            "signals_when_flat": int(signals_flat_base),
                            "signals_in_position": int(signals_in_pos_base),
                            "signals_spread_suppressed": int(spread_supp_base),
                            "signals_session_suppressed": int(session_supp_base),
                            "signals_cooldown_suppressed": int(cooldown_supp_base),
                            "eligible_signals": int(eligible_base),
                            "blocked_signals": int(blocked_base),
                            "entries_taken": int(entries_taken_base),
                            "entries_blocked_by_gate": 0,
                            "coverage": float(eligible_base / signals_flat_base) if signals_flat_base else 0.0,
                            "block_rate": 0.0,
                            "mmas_signals_checked": int(mmas_checked_base),
                            "mmas_passed_filters": int(mmas_passed_base),
                            "mmas_signaled": int(mmas_signaled_base),
                            "mmas_entered": int(mmas_entered_base),
                        }
                    )
                    coverage_gate = 0.0
                    if signals_flat_gate:
                        coverage_gate = 1.0 if disable_gate else float(eligible_gate / signals_flat_gate)
                    block_rate_gate = 0.0 if disable_gate else float(blocked_gate / signals_flat_gate) if signals_flat_gate else 0.0
                    skip_rate_gate = 0.0 if disable_gate else float(blocked_gate / signals_flat_gate) if signals_flat_gate else 0.0
                    gate_stats.update(
                        {
                            "strategy": "gated",
                            "date": day,
                            "Symbol": instrument,
                            "W": gate_lookback_bars,
                            "gate_mode": gate_mode,
                            "baseline_mode": baseline_mode,
                            "strategy_mode": strategy_mode,
                            "trade_session": trade_session,
                            "min_spread_ticks": min_spread_ticks,
                            "entry_cooldown_bars": entry_cooldown_bars,
                            "skipped_trades": int(skipped_gate),
                            "skip_rate": skip_rate_gate,
                            "total_signals": int(total_signals_gate),
                            "signals_when_flat": int(signals_flat_gate),
                            "signals_in_position": int(signals_in_pos_gate),
                            "signals_spread_suppressed": int(spread_supp_gate),
                            "signals_session_suppressed": int(session_supp_gate),
                            "signals_cooldown_suppressed": int(cooldown_supp_gate),
                            "eligible_signals": int(eligible_gate),
                            "blocked_signals": int(blocked_gate),
                            "entries_taken": int(entries_taken_gate),
                            "entries_blocked_by_gate": int(blocked_gate),
                            "coverage": coverage_gate,
                            "block_rate": block_rate_gate,
                            "mmas_signals_checked": int(mmas_checked_gate),
                            "mmas_passed_filters": int(mmas_passed_gate),
                            "mmas_signaled": int(mmas_signaled_gate),
                            "mmas_entered": int(mmas_entered_gate),
                        }
                    )
                    drop_reasons = []
                    if run_baseline:
                        if signals_flat_base == 0:
                            drop_reasons.append("zero_signals_when_flat")
                        if entries_taken_base == 0:
                            drop_reasons.append("zero_baseline_trades")
                    if entries_taken_gate == 0:
                        drop_reasons.append("zero_gated_trades")
                    if drop_reasons:
                        print(
                            f"DROP {instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                            f"reasons={drop_reasons} signals_when_flat={signals_flat_base} "
                            f"baseline_trades={entries_taken_base} gated_trades={entries_taken_gate} "
                            f"entries_blocked_by_gate={blocked_gate} coverage={gate_stats['coverage']:.2%}",
                            flush=True,
                        )
                        skipped_days.append(
                            f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode}: {','.join(drop_reasons)} "
                            f"signals_when_flat={signals_flat_base} baseline_trades={entries_taken_base} "
                            f"gated_trades={entries_taken_gate} entries_blocked_by_gate={blocked_gate} "
                            f"coverage={gate_stats['coverage']:.2%}"
                        )
                    summaries.extend([base_stats, gate_stats])
                    row = {
                        "Symbol": instrument,
                        "date": day,
                        "W": gate_lookback_bars,
                        "gate_mode": gate_mode,
                        "baseline_mode": baseline_mode,
                        "strategy_mode": strategy_mode,
                        "trade_session": trade_session,
                        "min_spread_ticks": min_spread_ticks,
                        "entry_cooldown_bars": entry_cooldown_bars,
                        "baseline_trades": int(len(trades_base)),
                        "gated_trades": int(len(trades_gate)),
                        "baseline_pnl_ticks": float(base_stats["total_pnl_ticks"]),
                        "gated_pnl_ticks": float(gate_stats["total_pnl_ticks"]),
                        "baseline_final_pnl_ticks": float(pnl_total_base),
                        "gated_final_pnl_ticks": float(pnl_total_gate),
                        "baseline_peak_equity_ticks": float(base_stats["peak_equity_ticks"]),
                        "gated_peak_equity_ticks": float(gate_stats["peak_equity_ticks"]),
                        "pnl_improvement_ticks": float(pnl_total_gate - pnl_total_base),
                        "blocked_signals": int(blocked_gate),
                        "improvement_per_blocked": float((pnl_total_gate - pnl_total_base) / max(int(blocked_gate), 1)),
                        "baseline_max_dd_ticks": float(base_stats["max_drawdown_ticks"]),
                        "gated_max_dd_ticks": float(gate_stats["max_drawdown_ticks"]),
                        "dd_improvement": float(base_stats["max_drawdown_ticks"] - gate_stats["max_drawdown_ticks"]),
                        "baseline_worst_trade_ticks": float(base_stats["worst_trade_ticks"]),
                        "gated_worst_trade_ticks": float(gate_stats["worst_trade_ticks"]),
                        "baseline_p1_trade_ticks": float(base_stats["p1"]),
                        "gated_p1_trade_ticks": float(gate_stats["p1"]),
                        "baseline_p5_trade_ticks": float(base_stats["p5"]),
                        "gated_p5_trade_ticks": float(gate_stats["p5"]),
                        "baseline_mean_trade_ticks": float(base_stats["mean_pnl_ticks"]),
                        "gated_mean_trade_ticks": float(gate_stats["mean_pnl_ticks"]),
                        "coverage": float(gate_stats["coverage"]),
                        "block_rate": float(gate_stats["block_rate"]),
                        "signals_total": int(total_signals_gate),
                        "signals_when_flat": int(signals_flat_gate),
                        "signals_in_position": int(signals_in_pos_gate),
                        "entries_blocked_by_gate": int(blocked_gate),
                        "entries_taken_gated": int(entries_taken_gate),
                        "signals_spread_suppressed": int(spread_supp_gate),
                        "signals_session_suppressed": int(session_supp_gate),
                        "signals_cooldown_suppressed": int(cooldown_supp_gate),
                        "mmas_signals_checked_base": int(mmas_checked_base),
                        "mmas_passed_filters_base": int(mmas_passed_base),
                        "mmas_signaled_base": int(mmas_signaled_base),
                        "mmas_entered_base": int(mmas_entered_base),
                        "mmas_signals_checked_gate": int(mmas_checked_gate),
                        "mmas_passed_filters_gate": int(mmas_passed_gate),
                        "mmas_signaled_gate": int(mmas_signaled_gate),
                        "mmas_entered_gate": int(mmas_entered_gate),
                    }
                    row["is_improved"] = row["pnl_improvement_ticks"] > 0
                    print(
                        "DAY_ROW_HAS",
                        "pnl_improvement_ticks" in row,
                        row.get("pnl_improvement_ticks"),
                        row.get("baseline_final_pnl_ticks"),
                        row.get("gated_final_pnl_ticks"),
                        flush=True,
                    )
                    print(f"DAY_ROW_KEYS={sorted(row.keys())}", flush=True)
                    print(
                        "ROW "
                        f"{row['Symbol']} {row['date']} W={row['W']} mode={row['gate_mode']} "
                        f"baseline_trades={row['baseline_trades']} gated_trades={row['gated_trades']} "
                        f"baseline_final={row['baseline_final_pnl_ticks']:.4f} gated_final={row['gated_final_pnl_ticks']:.4f} "
                        f"pnl_improvement={row['pnl_improvement_ticks']:.4f} "
                        f"baseline_max_dd={row['baseline_max_dd_ticks']:.4f} gated_max_dd={row['gated_max_dd_ticks']:.4f} "
                        f"dd_improvement={row['dd_improvement']:.4f} "
                        f"blocked={row['entries_blocked_by_gate']} "
                        f"improvement_per_blocked={row['improvement_per_blocked']:.4f} "
                        f"is_improved={row['is_improved']}",
                        flush=True,
                    )
                    print(
                        "DAY_ABS "
                        f"{row['Symbol']} {row['date']} W={row['W']} mode={row['gate_mode']} "
                        f"baseline_final_pnl_ticks={row['baseline_final_pnl_ticks']:.4f} "
                        f"gated_final_pnl_ticks={row['gated_final_pnl_ticks']:.4f} "
                        f"baseline_max_dd_ticks={row['baseline_max_dd_ticks']:.4f} "
                        f"gated_max_dd_ticks={row['gated_max_dd_ticks']:.4f}",
                        flush=True,
                    )
                    sweep_rows.append(row)
                    if drop_reasons:
                        continue

                    day_dir = out_dir_w / f"{instrument}_{day}"
                    day_dir.mkdir(parents=True, exist_ok=True)
                    trades_base.to_csv(day_dir / "trades_baseline.csv", index=False)
                    trades_gate.to_csv(day_dir / "trades_gated.csv", index=False)

                    print(
                        f"[{day_index}/{day_count}] {instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                        f"baseline trades={len(trades_base)} pnl={base_stats['total_pnl_ticks']:.2f} "
                        f"gated trades={len(trades_gate)} pnl={gate_stats['total_pnl_ticks']:.2f} "
                        f"skipped={skipped_gate} block_rate={gate_stats['block_rate']:.2%} coverage={gate_stats['coverage']:.2%} "
                        f"spread_supp={spread_supp_gate} cooldown_supp={cooldown_supp_gate}",
                        flush=True,
                    )
                    base_cooldown_reduction = cooldown_supp_base / max(1, (signals_flat_base + cooldown_supp_base))
                    gate_cooldown_reduction = cooldown_supp_gate / max(1, (signals_flat_gate + cooldown_supp_gate))
                    print(
                        f"{instrument} {day} cooldown_reduction baseline={base_cooldown_reduction:.2%} "
                        f"gated={gate_cooldown_reduction:.2%}",
                        flush=True,
                    )
                    print(
                        f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode} skip_reasons={skip_gate}",
                        flush=True,
                    )
                    print(
                        f"{instrument} {day} MMAS base checked={mmas_checked_base} passed={mmas_passed_base} "
                        f"signaled={mmas_signaled_base} entered={mmas_entered_base} | "
                        f"gate checked={mmas_checked_gate} passed={mmas_passed_gate} "
                        f"signaled={mmas_signaled_gate} entered={mmas_entered_gate}",
                        flush=True,
                    )
                    if strategy_mode == "impulse_confirm_v1":
                        print(
                            f"{instrument} {day} IMPULSE base checked={impulse_checked_base} "
                            f"thr={impulse_passed_thr_base} confirm={impulse_passed_confirm_base} "
                            f"entered={impulse_entered_base} | gate checked={impulse_checked_gate} "
                            f"thr={impulse_passed_thr_gate} confirm={impulse_passed_confirm_gate} "
                            f"entered={impulse_entered_gate}",
                            flush=True,
                        )
                    if strategy_mode == "absorption_failure_v1":
                        print(
                            f"{instrument} {day} AFR base checked={afr_checked_base} "
                            f"absorb={afr_absorption_pass_base} break={afr_break_pass_base} "
                            f"entered={afr_entered_base} | gate checked={afr_checked_gate} "
                            f"absorb={afr_absorption_pass_gate} break={afr_break_pass_gate} "
                            f"entered={afr_entered_gate}",
                            flush=True,
                        )
                    if strategy_mode == "absorption_failure_v2":
                        print(
                            f"{instrument} {day} AFR2 base checked={afr2_checked_base} "
                            f"absorb={afr2_absorption_pass_base} break={afr2_break_pass_base} ft={afr2_ft_pass_base} "
                            f"entered={afr2_entered_base} | gate checked={afr2_checked_gate} "
                            f"absorb={afr2_absorption_pass_gate} break={afr2_break_pass_gate} ft={afr2_ft_pass_gate} "
                            f"entered={afr2_entered_gate}",
                            flush=True,
                        )
                    if strategy_mode == "absorption_failure_v3":
                        print(
                            f"{instrument} {day} AFR3 base checked={afr3_checked_base} "
                            f"absorb={afr3_absorption_pass_base} break={afr3_break_pass_base} "
                            f"quality={afr3_break_quality_pass_base} snap={afr3_snapback_pass_base} "
                            f"entered={afr3_entered_base} | gate checked={afr3_checked_gate} "
                            f"absorb={afr3_absorption_pass_gate} break={afr3_break_pass_gate} "
                            f"quality={afr3_break_quality_pass_gate} snap={afr3_snapback_pass_gate} "
                            f"entered={afr3_entered_gate}",
                            flush=True,
                        )
                    base_entry_reason = trades_base["entry_reason"] if "entry_reason" in trades_base.columns else pd.Series(dtype=object)
                    gate_entry_reason = trades_gate["entry_reason"] if "entry_reason" in trades_gate.columns else pd.Series(dtype=object)
                    base_exit_reason = trades_base["exit_reason"] if "exit_reason" in trades_base.columns else pd.Series(dtype=object)
                    gate_exit_reason = trades_gate["exit_reason"] if "exit_reason" in trades_gate.columns else pd.Series(dtype=object)
                    base_mae = trades_base["mae_ticks"] if "mae_ticks" in trades_base.columns else pd.Series(dtype=float)
                    base_mfe = trades_base["mfe_ticks"] if "mfe_ticks" in trades_base.columns else pd.Series(dtype=float)
                    gate_mae = trades_gate["mae_ticks"] if "mae_ticks" in trades_gate.columns else pd.Series(dtype=float)
                    gate_mfe = trades_gate["mfe_ticks"] if "mfe_ticks" in trades_gate.columns else pd.Series(dtype=float)
                    count_break_entries_base = int(base_entry_reason.isin(["break", "rearm_break"]).sum())
                    count_ft_entries_base = int(base_entry_reason.isin(["ft", "rearm_ft"]).sum())
                    count_rearms_base = int(base_entry_reason.str.startswith("rearm").sum()) if not base_entry_reason.empty else 0
                    count_break_entries_gate = int(gate_entry_reason.isin(["break", "rearm_break"]).sum())
                    count_ft_entries_gate = int(gate_entry_reason.isin(["ft", "rearm_ft"]).sum())
                    count_rearms_gate = int(gate_entry_reason.str.startswith("rearm").sum()) if not gate_entry_reason.empty else 0
                    if strategy_mode == "absorption_failure_v2":
                        if afr_enter_on == "break" and (count_ft_entries_base > 0 or count_ft_entries_gate > 0):
                            raise RuntimeError("AFR_ENTER_ON=break but FT entries were recorded.")
                        if afr_enter_on == "ft" and (count_break_entries_base > 0 or count_break_entries_gate > 0):
                            raise RuntimeError("AFR_ENTER_ON=ft but break entries were recorded.")
                    pct_exit_on_decay_base = float((base_exit_reason == "momentum_decay").mean()) if len(base_exit_reason) else 0.0
                    pct_exit_on_decay_gate = float((gate_exit_reason == "momentum_decay").mean()) if len(gate_exit_reason) else 0.0
                    pct_exit_on_be_base = float((base_exit_reason == "breakeven").mean()) if len(base_exit_reason) else 0.0
                    pct_exit_on_be_gate = float((gate_exit_reason == "breakeven").mean()) if len(gate_exit_reason) else 0.0
                    median_mfe_base = float(np.nanmedian(base_mfe)) if len(base_mfe) else 0.0
                    median_mae_base = float(np.nanmedian(base_mae)) if len(base_mae) else 0.0
                    median_mfe_gate = float(np.nanmedian(gate_mfe)) if len(gate_mfe) else 0.0
                    median_mae_gate = float(np.nanmedian(gate_mae)) if len(gate_mae) else 0.0
                    strategy_rows.append(
                        {
                            "Symbol": instrument,
                            "date": day,
                            "W": gate_lookback_bars,
                            "gate_mode": gate_mode,
                            "strategy_mode": strategy_mode,
                            "base_entry_candidates_when_flat": int(signals_flat_base),
                            "base_long_candidates": int(strategy_long_base),
                            "base_short_candidates": int(strategy_short_base),
                            "impulse_checked_base": int(impulse_checked_base),
                            "impulse_passed_threshold_base": int(impulse_passed_thr_base),
                            "impulse_passed_confirm_base": int(impulse_passed_confirm_base),
                            "impulse_entered_base": int(impulse_entered_base),
                            "impulse_checked_gate": int(impulse_checked_gate),
                            "impulse_passed_threshold_gate": int(impulse_passed_thr_gate),
                            "impulse_passed_confirm_gate": int(impulse_passed_confirm_gate),
                            "impulse_entered_gate": int(impulse_entered_gate),
                            "afr_checked_base": int(afr_checked_base),
                            "afr_absorption_pass_base": int(afr_absorption_pass_base),
                            "afr_break_pass_base": int(afr_break_pass_base),
                            "afr_entered_base": int(afr_entered_base),
                            "afr_checked_gate": int(afr_checked_gate),
                            "afr_absorption_pass_gate": int(afr_absorption_pass_gate),
                            "afr_break_pass_gate": int(afr_break_pass_gate),
                            "afr_entered_gate": int(afr_entered_gate),
                            "afr2_checked_base": int(afr2_checked_base),
                            "afr2_absorption_pass_base": int(afr2_absorption_pass_base),
                            "afr2_break_pass_base": int(afr2_break_pass_base),
                            "afr2_ft_pass_base": int(afr2_ft_pass_base),
                            "afr2_entered_base": int(afr2_entered_base),
                            "afr2_break_quality_pass_base": int(afr2_break_quality_pass_base),
                            "afr2_snapback_fail_base": int(afr2_snapback_fail_base),
                            "afr2_break_quality_entries_base": int(afr2_break_quality_entries_base),
                            "afr2_checked_gate": int(afr2_checked_gate),
                            "afr2_absorption_pass_gate": int(afr2_absorption_pass_gate),
                            "afr2_break_pass_gate": int(afr2_break_pass_gate),
                            "afr2_ft_pass_gate": int(afr2_ft_pass_gate),
                            "afr2_entered_gate": int(afr2_entered_gate),
                            "afr2_break_quality_pass_gate": int(afr2_break_quality_pass_gate),
                            "afr2_snapback_fail_gate": int(afr2_snapback_fail_gate),
                            "afr2_break_quality_entries_gate": int(afr2_break_quality_entries_gate),
                            "afr3_checked_base": int(afr3_checked_base),
                            "afr3_absorption_pass_base": int(afr3_absorption_pass_base),
                            "afr3_break_pass_base": int(afr3_break_pass_base),
                            "afr3_break_quality_pass_base": int(afr3_break_quality_pass_base),
                            "afr3_snapback_pass_base": int(afr3_snapback_pass_base),
                            "afr3_entered_base": int(afr3_entered_base),
                            "afr3_checked_gate": int(afr3_checked_gate),
                            "afr3_absorption_pass_gate": int(afr3_absorption_pass_gate),
                            "afr3_break_pass_gate": int(afr3_break_pass_gate),
                            "afr3_break_quality_pass_gate": int(afr3_break_quality_pass_gate),
                            "afr3_snapback_pass_gate": int(afr3_snapback_pass_gate),
                            "afr3_entered_gate": int(afr3_entered_gate),
                            "afr_be_armed_base": int(afr_be_armed_base),
                            "afr_be_triggered_base": int(afr_be_triggered_base),
                            "afr_be_armed_gate": int(afr_be_armed_gate),
                            "afr_be_triggered_gate": int(afr_be_triggered_gate),
                            "count_break_entries_base": int(count_break_entries_base),
                            "count_ft_entries_base": int(count_ft_entries_base),
                            "count_rearms_base": int(count_rearms_base),
                            "count_break_entries_gate": int(count_break_entries_gate),
                            "count_ft_entries_gate": int(count_ft_entries_gate),
                            "count_rearms_gate": int(count_rearms_gate),
                            "afr_enter_on_used": afr_enter_on,
                            "pct_exit_on_decay_base": float(pct_exit_on_decay_base),
                            "pct_exit_on_decay_gate": float(pct_exit_on_decay_gate),
                            "pct_exit_on_be_base": float(pct_exit_on_be_base),
                            "pct_exit_on_be_gate": float(pct_exit_on_be_gate),
                            "median_mfe_ticks_base": float(median_mfe_base),
                            "median_mae_ticks_base": float(median_mae_base),
                            "median_mfe_ticks_gate": float(median_mfe_gate),
                            "median_mae_ticks_gate": float(median_mae_gate),
                            "avg_impulse_ticks_entry_base": float(entry_impulse_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_break_ticks_entry_base": float(entry_break_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_abs_break_ticks_entry_base": float(entry_break_abs_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_ft_progress_ticks_entry_base": float(entry_ft_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_abs_ft_progress_ticks_entry_base": float(entry_ft_abs_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_break_ticks_entry_long_base": float(entry_break_sum_long_base / entry_count_long_base)
                            if entry_count_long_base
                            else 0.0,
                            "avg_break_ticks_entry_short_base": float(entry_break_sum_short_base / entry_count_short_base)
                            if entry_count_short_base
                            else 0.0,
                            "avg_abs_impulse_ticks_entry_base": float(entry_impulse_abs_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_impulse_ticks_entry_long_base": float(entry_impulse_sum_long_base / entry_count_long_base)
                            if entry_count_long_base
                            else 0.0,
                            "avg_impulse_ticks_entry_short_base": float(entry_impulse_sum_short_base / entry_count_short_base)
                            if entry_count_short_base
                            else 0.0,
                            "avg_flow_entry_base": float(entry_flow_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_abs_flow_entry_base": float(entry_flow_abs_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_spread_ticks_entry_base": float(entry_spread_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_flow_entry_long_base": float(entry_flow_sum_long_base / entry_count_long_base)
                            if entry_count_long_base
                            else 0.0,
                            "avg_flow_entry_short_base": float(entry_flow_sum_short_base / entry_count_short_base)
                            if entry_count_short_base
                            else 0.0,
                            "avg_impulse_ticks_entry_gate": float(entry_impulse_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_break_ticks_entry_gate": float(entry_break_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_abs_break_ticks_entry_gate": float(entry_break_abs_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_ft_progress_ticks_entry_gate": float(entry_ft_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_abs_ft_progress_ticks_entry_gate": float(entry_ft_abs_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_break_ticks_entry_long_gate": float(entry_break_sum_long_gate / entry_count_long_gate)
                            if entry_count_long_gate
                            else 0.0,
                            "avg_break_ticks_entry_short_gate": float(entry_break_sum_short_gate / entry_count_short_gate)
                            if entry_count_short_gate
                            else 0.0,
                            "avg_abs_impulse_ticks_entry_gate": float(entry_impulse_abs_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_impulse_ticks_entry_long_gate": float(entry_impulse_sum_long_gate / entry_count_long_gate)
                            if entry_count_long_gate
                            else 0.0,
                            "avg_impulse_ticks_entry_short_gate": float(entry_impulse_sum_short_gate / entry_count_short_gate)
                            if entry_count_short_gate
                            else 0.0,
                            "avg_flow_entry_gate": float(entry_flow_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_abs_flow_entry_gate": float(entry_flow_abs_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_spread_ticks_entry_gate": float(entry_spread_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_flow_entry_long_gate": float(entry_flow_sum_long_gate / entry_count_long_gate)
                            if entry_count_long_gate
                            else 0.0,
                            "avg_flow_entry_short_gate": float(entry_flow_sum_short_gate / entry_count_short_gate)
                            if entry_count_short_gate
                            else 0.0,
                        }
                    )
                    if sum(skip_gate.values()) != skipped_gate:
                        raise RuntimeError(
                            f"Skip reasons mismatch: sum={sum(skip_gate.values())} skipped={skipped_gate} "
                            f"day={day} W={gate_lookback_bars} mode={gate_mode}"
                        )
                    if strategy_mode != "impulse_confirm_v1" and entries_taken_base != signals_flat_base:
                        raise RuntimeError(
                            f"Baseline entries mismatch: entries_taken={entries_taken_base} "
                            f"signals_when_flat={signals_flat_base} day={day}"
                        )
                    expected_gate = entries_taken_gate + blocked_gate
                    if expected_gate != signals_flat_gate:
                        print(
                            f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode} gate_mismatch "
                            f"entries_taken={entries_taken_gate} blocked={blocked_gate} signals_when_flat={signals_flat_gate}",
                            flush=True,
                        )
                        raise RuntimeError(
                            f"Gated entries mismatch: entries_taken+blocked={expected_gate} "
                            f"signals_when_flat={signals_flat_gate} day={day}"
                        )
                except Exception as exc:
                    skipped_days.append(f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode}: {exc}")
                    print(f"[{day_index}/{day_count}] {instrument} {day} W={gate_lookback_bars} mode={gate_mode} FAILED: {exc}", flush=True)
                    continue

            events_out = pd.concat(all_events, ignore_index=True) if all_events else events
            events_out.to_csv(out_dir_w / "events.csv", index=False)
            pd.DataFrame(summaries).to_csv(out_dir_w / "summary.csv", index=False)

            trades_base_all = pd.concat(all_base, ignore_index=True) if all_base else pd.DataFrame()
            trades_gate_all = pd.concat(all_gated, ignore_index=True) if all_gated else pd.DataFrame()
            trades_base_all.to_csv(out_dir_w / "trades_baseline.csv", index=False)
            trades_gate_all.to_csv(out_dir_w / "trades_gated.csv", index=False)

            _plot_equity(trades_base_all, trades_gate_all, out_dir_w / "equity_curve.png")

            with open(out_dir_w / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summaries, f, indent=2)
            if strategy_rows:
                strategy_df = pd.DataFrame(strategy_rows)
                strategy_df.to_csv(out_dir_w / "strategy_diagnostics.csv", index=False)
                print("Strategy diagnostics by day:", flush=True)
                print(strategy_df.to_string(index=False), flush=True)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "summary_sweep.csv", index=False)
    if not sweep_df.empty:
        numeric_cols = [
            "baseline_pnl_ticks",
            "gated_pnl_ticks",
            "baseline_final_pnl_ticks",
            "gated_final_pnl_ticks",
            "baseline_peak_equity_ticks",
            "gated_peak_equity_ticks",
            "pnl_improvement_ticks",
            "improvement_per_blocked",
            "baseline_max_dd_ticks",
            "gated_max_dd_ticks",
            "dd_improvement",
            "baseline_worst_trade_ticks",
            "gated_worst_trade_ticks",
            "baseline_p1_trade_ticks",
            "gated_p1_trade_ticks",
            "baseline_p5_trade_ticks",
            "gated_p5_trade_ticks",
            "baseline_mean_trade_ticks",
            "gated_mean_trade_ticks",
        ]
        for col in numeric_cols:
            if col in sweep_df.columns:
                sweep_df[col] = pd.to_numeric(sweep_df[col], errors="coerce")
        sweep_df.to_csv(out_dir / "summary_sweep.csv", index=False)
    if not sweep_df.empty:
        coverage_cols = [
            "Symbol",
            "date",
            "W",
            "gate_mode",
            "baseline_trades",
            "gated_trades",
            "signals_when_flat",
            "entries_blocked_by_gate",
            "entries_taken_gated",
            "coverage",
            "block_rate",
        ]
        coverage_df = sweep_df[coverage_cols].copy()
        coverage_df = coverage_df.rename(columns={"signals_when_flat": "gated_entry_candidates_when_flat"})
        coverage_df.to_csv(out_dir / "coverage_activity.csv", index=False)
        print("Coverage/activity by day:", flush=True)
        print(coverage_df.to_string(index=False), flush=True)
    if not sweep_df.empty:
        dd_equal = np.isclose(
            sweep_df["baseline_max_dd_ticks"].astype(float),
            sweep_df["baseline_final_pnl_ticks"].abs().astype(float),
        )
        if bool(dd_equal.all()):
            print(
                "Warning: baseline_max_dd_ticks equals |baseline_final_pnl_ticks| for all rows; "
                "check drawdown computation.",
                flush=True,
            )
        abs_cols = [
            "W",
            "gate_mode",
            "baseline_final_pnl_ticks",
            "gated_final_pnl_ticks",
            "baseline_max_dd_ticks",
            "gated_max_dd_ticks",
        ]
        abs_df = sweep_df[abs_cols].copy()
        abs_df.to_csv(out_dir / "summary_absolute.csv", index=False)
        print("Absolute summary by day:", flush=True)
        print(abs_df.to_string(index=False), flush=True)
        bad_tail = (sweep_df["baseline_p1_trade_ticks"] > sweep_df["baseline_p5_trade_ticks"]).sum()
        if bad_tail:
            print("Warning: baseline p1 > p5 detected; check tail computation.", flush=True)
    agg_rows = []
    if not sweep_df.empty:
        for (w, mode), g in sweep_df.groupby(["W", "gate_mode"], sort=False):
            days_count = int(g.shape[0])
            pnl_imp = pd.to_numeric(g["pnl_improvement_ticks"], errors="coerce")
            imp_per_block = pd.to_numeric(g["improvement_per_blocked"], errors="coerce")
            dd_imp = pd.to_numeric(g["dd_improvement"], errors="coerce")
            worst_imp = g["baseline_worst_trade_ticks"] - g["gated_worst_trade_ticks"]
            p1_imp = g["baseline_p1_trade_ticks"] - g["gated_p1_trade_ticks"]
            p5_imp = g["baseline_p5_trade_ticks"] - g["gated_p5_trade_ticks"]
            pnl_values = pnl_imp.dropna().to_numpy()
            pnl_values = pnl_values[np.isfinite(pnl_values)]
            if pnl_values.size == 0:
                print(
                    f"Aggregate warning: empty pnl_improvement_ticks for W={w} mode={mode} "
                    f"rows={days_count}",
                    flush=True,
                )
                print(
                    g[
                        [
                            "date",
                            "baseline_final_pnl_ticks",
                            "gated_final_pnl_ticks",
                            "pnl_improvement_ticks",
                        ]
                    ]
                    .head()
                    .to_string(index=False),
                    flush=True,
                )
            improved = pnl_values[pnl_values > 0]
            worsened = pnl_values[pnl_values < 0]
            median_all = float(np.median(pnl_values)) if pnl_values.size else None
            mean_all = float(np.mean(pnl_values)) if pnl_values.size else None
            median_improved = float(np.median(improved)) if improved.size else None
            median_worsened = float(np.median(worsened)) if worsened.size else None
            agg_rows.append(
                {
                    "W": w,
                    "gate_mode": mode,
                    "days_count": days_count,
                    "pct_days_improved": float(improved.size / days_count) if days_count else 0.0,
                    "median_improvement_all_days": median_all,
                    "mean_improvement_all_days": mean_all,
                    "median_improvement_improved_days": median_improved,
                    "median_improvement_worsened_days": median_worsened,
                    "median_pnl_improvement_ticks": median_all,
                    "mean_pnl_improvement_ticks": mean_all,
                    "median_improvement_per_blocked": float(np.median(imp_per_block)) if days_count else np.nan,
                    "mean_improvement_per_blocked": float(np.mean(imp_per_block)) if days_count else np.nan,
                    "median_dd_improvement": float(np.median(dd_imp)) if days_count else np.nan,
                    "mean_dd_improvement": float(np.mean(dd_imp)) if days_count else np.nan,
                    "median_worst_trade_improvement": float(np.median(worst_imp)) if days_count else np.nan,
                    "mean_worst_trade_improvement": float(np.mean(worst_imp)) if days_count else np.nan,
                    "median_p1_improvement": float(np.median(p1_imp)) if days_count else np.nan,
                    "mean_p1_improvement": float(np.mean(p1_imp)) if days_count else np.nan,
                    "median_p5_improvement": float(np.median(p5_imp)) if days_count else np.nan,
                    "mean_p5_improvement": float(np.mean(p5_imp)) if days_count else np.nan,
                }
            )
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "summary_aggregate.csv", index=False)
    if not agg_df.empty:
        print("Aggregate summary by W, gate_mode:", flush=True)
        print(agg_df.to_string(index=False), flush=True)
        for metric, label in [
            ("median_improvement_all_days", "median pnl improvement"),
            ("median_improvement_per_blocked", "median improvement per blocked"),
            ("median_dd_improvement", "median dd improvement"),
        ]:
            top = agg_df.sort_values(metric, ascending=False, na_position="last").head(3)
            print(f"Top 3 by {label}:", flush=True)
            print(top[["W", "gate_mode", "days_count", "pct_days_improved", metric]].to_string(index=False), flush=True)
        if bool((agg_df["pct_days_improved"] > 0).any()):
            top = agg_df.sort_values("median_improvement_improved_days", ascending=False, na_position="last").head(3)
            print("Top 3 by median improvement (improved days):", flush=True)
            print(
                top[
                    ["W", "gate_mode", "days_count", "pct_days_improved", "median_improvement_improved_days"]
                ].to_string(index=False),
                flush=True,
            )

    if skipped_days:
        skipped_path = out_dir / "skipped_days.txt"
        with open(skipped_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_days))


if __name__ == "__main__":
    main()
