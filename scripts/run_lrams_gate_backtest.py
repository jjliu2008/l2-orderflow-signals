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
  STRATEGY_MODE           baseline_flat|micro_momo_v1|impulse_confirm_v1|absorption_failure_v1|absorption_failure_v2|absorption_failure_v3|lrams_breakout_v1|srf_entry_v1 (default: micro_momo_v1)
  MICRO_K_BARS            micro momentum window (default: 5)
  MICRO_IMPULSE_TICKS     min impulse ticks (default: 1)
  MICRO_FLOW_MIN          min abs flow (default: 0)
  MAX_SPREAD_TICKS_FOR_ENTRY max spread ticks to allow entry (default: 2)
  MMAS_K_BARS             window for MMAS (default: 5)
  MMAS_MIN_DMID_TICKS     min dmid ticks for MMAS (default: 1)
  MMAS_MIN_FLOW_ABS       min abs flow for MMAS (default: 20)
  MMAS_REQUIRE_AGREE      require dmid/flow sign agreement (default: 1)
  DEBUG_FIRST_MMAS        print first MMAS decision (default: 0)
  SRF_MIN_DISP_TICKS      SRF min displacement ticks (default: 3)
  SRF_DISP_WINDOW_BARS    SRF displacement window bars (default: 5)
  SRF_MIN_SPEED_TICKS_PER_BAR SRF min speed ticks/bar (default: 0.6)
  SRF_REFILL_WINDOW_BARS  SRF refill window bars (default: 5)
  SRF_BASELINE_BARS       SRF baseline size window bars (default: 20)
  SRF_REFILL_RATIO_MAX    SRF refill ratio max (default: 0.6)
  SRF_MIN_SPREAD_TICKS    SRF fallback min spread ticks (default: 1)
  SRF_FLOW_CONFIRM        SRF require flow confirm (default: 0)
  SRF_FLOW_WINDOW_BARS    SRF flow confirm window bars (default: 5)
  SRF_MIN_FLOW            SRF min flow for confirm (default: 0)
  SRF_SIDE_MODE           follow|fade (default: follow)
  SRF_DEBUG               SRF debug prints (default: 0)
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
  AFR_MIN_FLOW_ABS_SWEEP  comma list for AFR min abs flow sweep (default: unset)
  AFR_STALL_TICKS         AFR max |stall| ticks (default: 0)
  AFR_BREAK_TICKS         AFR breakout ticks (default: 1)
  AFR_REQUIRE_FLOW_SIGN   require flow sign aligns with breakout (default: 1)
  AFR_USE_MID_FOR_STALL   use mid for stall calc (default: 1)
  AFR_USE_SIGNED_VOLUME   use signed_volume for flow (default: 1)
  AFR_FT_BARS             AFR v2 follow-through bars (default: 2)
  AFR_FT_BARS_SWEEP       comma list for AFR FT bars sweep (default: unset)
  RUN_AFR2_SWEEP          force AFR2 sweep logging/artifacts (default: 0)
  AFR_FT_MIN_TICKS        AFR v2 follow-through min ticks (default: 1)
  AFR_FT_NO_BACKTRACK     AFR v2 block if price backtracks (default: 1)
  AFR_ENTER_ON            AFR v2 entry timing: break|ft|both (default: ft)
  AFR_BREAK_QUALITY_MIN_FLOW_ABS AFR v2 min abs flow for break quality (default: 80)
  AFR_BREAK_QUALITY_MAX_SPREAD_TICKS AFR v2 max spread for break quality (default: 1)
  AFR_SNAPBACK_BARS       AFR v2 snapback lookahead bars (default: 2)
  AFR_SNAPBACK_BAND_TICKS AFR v2 snapback band in ticks (default: 1)
  AFR_SCRATCH_BARS        AFR v2 scratch after N bars (default: 3)
  AFR_SCRATCH_MIN_PROGRESS_TICKS min progress before scratch (default: 1)
  LBO_BREAK_TICKS         LRAMS breakout break ticks (default: 1)
  LBO_CONFIRM_BARS        LRAMS breakout confirm window (default: 2)
  LBO_MIN_FLOW_ABS        LRAMS breakout min abs flow for confirm (default: 20)
  LBO_MAX_SPREAD_TICKS    LRAMS breakout max spread for confirm (default: 2)
  LBO_FT_BARS             LRAMS breakout FT window bars (default: 2)
  LBO_FT_MIN_TICKS        LRAMS breakout FT min ticks beyond break (default: 1)
  LBO_REARM_ENABLED       LRAMS breakout allow rearm (default: 0)
  LBO_REARM_BAND_TICKS    LRAMS breakout rearm band ticks (default: 1)
  LBO_REARM_MAX_BARS      LRAMS breakout rearm max bars (default: 5)
  LBO_TP_TICKS            LRAMS breakout TP ticks (default: 4)
  LBO_SL_TICKS            LRAMS breakout SL ticks (default: 3)
  LBO_MAX_HOLD_BARS       LRAMS breakout max hold bars (default: 15)
  LBO_SCRATCH_BARS        LRAMS breakout scratch after N bars (default: 3)
  LBO_SCRATCH_MIN_PROGRESS_TICKS min progress before scratch (default: 1)
  LBO_DECAY_BARS          LRAMS breakout decay bars (default: 3)
  LBO_BREAKEVEN_AFTER_TICKS LRAMS breakout breakeven ticks (default: unset)
  LBO_IGNORE_THR          LRAMS breakout ignore asym <= thr (default: 0)
  LBO_IGNORE_THR_OVERRIDE override LBO_IGNORE_THR if set (default: unset)
  LBO_FLIP_DIRECTION      LRAMS breakout invert weak_side mapping (default: 0)
  LBO_FLIP_MAPPING        LRAMS breakout invert weak_side mapping (default: 0)
  LBO_CONFIRM_MODE        LRAMS breakout confirm mode none|confirm_ticks|pullback (default: none)
  LBO_CONFIRM_TICKS       LRAMS breakout confirm ticks (default: 1)
  LBO_CONFIRM_MAX_BARS    LRAMS breakout confirm max bars (default: 5)
  LBO_CONFIRM_USE_MID     LRAMS breakout confirm uses mid (default: 1)
  LBO_CONFIRM_REQUIRE_FLOW_ALIGN require flow alignment for confirm (default: 0)
  LBO_CONFIRM_FLOW_ALIGN_BARS flow align window for confirm (default: 5)
  LBO_CONFIRM_MIN_FLOW_ABS_ALIGN min abs flow for confirm align (default: 40)
  LBO_PULLBACK_TICKS      LRAMS breakout pullback ticks (default: 1)
  LBO_PULLBACK_MAX_BARS   LRAMS breakout pullback max bars (default: 10)
  LBO_RESUME_TICKS        LRAMS breakout resume ticks (default: 1)
  LBO_RESUME_MAX_BARS     LRAMS breakout resume max bars (default: 5)
  ALLOW_GATE_ON_NONE      allow gate when weak_side is None (default: 0)
  GATE_DEBUG              print first 5 gate decisions per day (default: 0)
  GATE_BYPASS_ON_NONE     bypass event/threshold gating when weak_side is None (default: 0)
  WEAK_SIDE_LOOKBACK_BARS lookback for weak_side detection (default: max(gate_lookback_bars, 100))
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
  FAIL_FAST_ENABLED     enable FAIL_FAST exits (default: 1)
  FAIL_FAST_BARS         fail fast after N bars (default: 3)
  FAIL_FAST_MAX_ADVERSE_TICKS max adverse ticks for fail fast (default: 2)
  ENTRY_VIABILITY_MODE   entry viability gate: base|strict (default: base)
  ENTRY_VIABILITY_FLOW_CONFIRM require flow alignment for viability gate (default: 0)
  SRF_MIN_DISP_TICKS     SRF min displacement ticks (default: 3)
  SRF_DISP_WINDOW_BARS   SRF displacement window bars (default: 5)
  SRF_MIN_SPEED_TICKS_PER_BAR SRF min speed ticks/bar (default: 0.6)
  SRF_REFILL_WINDOW_BARS SRF refill window bars (default: 5)
  SRF_BASELINE_BARS      SRF baseline window bars (default: 20)
  SRF_REFILL_RATIO_MAX   SRF refill ratio max (default: 0.6)
  SRF_MIN_SPREAD_TICKS   SRF spread min when no size (default: 1)
  SRF_FLOW_CONFIRM       SRF flow confirm enable (default: 0)
  SRF_FLOW_WINDOW_BARS   SRF flow window bars (default: 5)
  SRF_MIN_FLOW           SRF min flow (default: 0)

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


def _mfe_threshold_table(trades_df: pd.DataFrame, reason_col: str) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["date"] = df["entry_time"].dt.date.astype(str)
    df[reason_col] = df[reason_col].fillna("unknown").astype(str)
    df["strategy"] = df["strategy"].fillna("unknown").astype(str)
    mfe = pd.to_numeric(df["mfe_ticks"], errors="coerce")
    df["mfe_ticks"] = mfe
    rows = []
    for (day, strategy, reason), g in df.groupby(["date", "strategy", reason_col], sort=False):
        mf = g["mfe_ticks"].dropna()
        if mf.empty:
            continue
        rows.append(
            {
                "date": day,
                "strategy": strategy,
                reason_col: reason,
                "count": int(mf.shape[0]),
                "pct_mfe_ge_1": float((mf >= 1.0).mean()),
                "pct_mfe_ge_2": float((mf >= 2.0).mean()),
                "pct_mfe_ge_3": float((mf >= 3.0).mean()),
            }
        )
    return pd.DataFrame(rows)


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
    weak_side_lookback_bars: int,
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
    lbo_break_ticks: int,
    lbo_confirm_bars: int,
    lbo_min_flow_abs: float,
    lbo_max_spread_ticks: int,
    lbo_ft_bars: int,
    lbo_ft_min_ticks: int,
    lbo_rearm_enabled: bool,
    lbo_rearm_band_ticks: int,
    lbo_rearm_max_bars: int,
    lbo_tp_ticks: int,
    lbo_sl_ticks: int,
    lbo_max_hold_bars: int,
    lbo_scratch_bars: int,
    lbo_scratch_min_progress_ticks: int,
    lbo_decay_bars: int,
    lbo_breakeven_after_ticks: int | None,
    lbo_ignore_thr: bool,
    lbo_flip_direction: bool,
    lbo_confirm_mode: str,
    lbo_confirm_ticks: int,
    lbo_confirm_max_bars: int,
    lbo_confirm_use_mid: bool,
    lbo_confirm_require_flow_align: bool,
    lbo_confirm_flow_align_bars: int,
    lbo_confirm_min_flow_abs_align: float,
    lbo_pullback_max_bars: int,
    lbo_pullback_ticks: int,
    lbo_resume_ticks: int,
    lbo_resume_max_bars: int,
    allow_gate_on_none: bool,
    gate_debug: bool,
    debug_first_afr: bool,
    debug_first_afr2: bool,
    afr_scratch_bars: int,
    afr_scratch_min_progress_ticks: int,
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
    validate_debug: bool,
    entry_confirm_bars: int,
    entry_min_progress_ticks: int,
    entry_confirm_style: str,
    exit_debug: bool,
    exit_mode: str,
    validate_bars: int,
    min_progress_ticks: int,
    be_arm_ticks: int,
    be_offset_ticks: int | None,
    decay_bars: int,
    scratch_bars: int,
    scratch_min_progress_ticks: int,
    be_after_ticks: int,
    decay_min_flow: float,
    be_grace_bars: int,
    scratch_require_mfe_ticks: int,
    scratch_grace_bars: int,
    fail_fast_bars: int,
    fail_fast_max_adverse_ticks: int,
    fail_fast_enabled: bool,
    entry_viability_mode: str,
    entry_viability_flow_confirm: bool,
    srf_min_disp_ticks: int,
    srf_disp_window_bars: int,
    srf_min_speed_ticks_per_bar: float,
    srf_refill_window_bars: int,
    srf_baseline_bars: int,
    srf_refill_ratio_max: float,
    srf_min_spread_ticks: int,
    srf_flow_confirm: bool,
    srf_flow_window_bars: int,
    srf_min_flow: float,
    srf_side_mode: str,
    srf_debug: bool,
    runner_trail_start_ticks: int,
    runner_trail_giveback_ticks: int,
    passive_exit_enabled: bool,
    passive_exit_bars: int,
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
    if lbo_confirm_use_mid:
        lbo_confirm_price = mid
    elif "last_price" in df_day.columns:
        lbo_confirm_price = pd.to_numeric(df_day["last_price"], errors="coerce").fillna(np.nan).to_numpy()
    else:
        lbo_confirm_price = mid
    day_str = str(df_day["date"].iloc[0]) if "date" in df_day.columns and not df_day.empty else "unknown"
    symbol_str = str(df_day["Symbol"].iloc[0]) if "Symbol" in df_day.columns and not df_day.empty else "NA"
    entry_confirm_style = entry_confirm_style.strip().lower()
    if entry_confirm_style not in {"off", "price_only"}:
        raise ValueError(f"Invalid ENTRY_CONFIRM_STYLE: {entry_confirm_style}")
    if entry_confirm_bars <= 0 or entry_confirm_style in {"", "off", "none"}:
        entry_confirm_bars = 0
        entry_confirm_style = "off"
    if validate_debug:
        print(
            f"ENTRY_CONFIRM_DEBUG {symbol_str} {day_str} style={entry_confirm_style} "
            f"bars={entry_confirm_bars} min_prog={entry_min_progress_ticks}",
            flush=True,
        )
    n = len(df_day)
    trades: List[Dict[str, float]] = []
    pnl_ticks_total = 0.0
    skipped = 0
    srf_checked = 0
    srf_triggered = 0
    srf_entered = 0
    srf_raw_long = 0
    srf_raw_short = 0
    srf_exec_long = 0
    srf_exec_short = 0
    srf_event_stats: List[Dict[str, float]] = []
    total_signals = 0
    # signals_when_flat: realized entries (trade records)
    # entry_candidates_when_flat: candidate evaluations while flat (pre-gate)
    signals_when_flat = 0
    entry_candidates_when_flat = 0
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
    entry_confirm_checked = 0
    entry_confirm_passed = 0
    entry_confirm_failed = 0
    pending_entry_active = False
    pending_entry_side = ""
    pending_entry_price = float("nan")
    pending_entry_created = -1
    pending_entry_expiry = -1
    pending_entry: Dict[str, object] = {}
    passive_filled_count = 0
    passive_fallback_count = 0
    be_offset_logged = False
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
    lbo_absorption_level_long = float("nan")
    lbo_absorption_level_short = float("nan")
    lbo_absorption_bar_long = -1
    lbo_absorption_bar_short = -1
    lbo_break_blocked_bar_long = -1
    lbo_break_blocked_bar_short = -1
    lbo_break_blocked_abs_level_long = float("nan")
    lbo_break_blocked_abs_level_short = float("nan")
    lbo_break_blocked_break_level_long = float("nan")
    lbo_break_blocked_break_level_short = float("nan")
    lbo_break_blocked_abs_bar_long = -1
    lbo_break_blocked_abs_bar_short = -1
    lbo_rearm_used_long = False
    lbo_rearm_used_short = False
    lbo_rearm_active_long = False
    lbo_rearm_active_short = False
    lbo_rearm_expiry_long = -1
    lbo_rearm_expiry_short = -1
    lbo_weak_none = 0
    lbo_weak_ask = 0
    lbo_weak_bid = 0
    lbo_weak_none_blocked = 0
    lbo_rej_empty = 0
    lbo_rej_before_lookback = 0
    lbo_rej_thr_nan = 0
    lbo_rej_asym_below_thr = 0
    lbo_missing_both = 0
    lbo_only_buy = 0
    lbo_only_sell = 0
    lbo_both_fail_thr = 0
    lbo_pending_started = 0
    lbo_confirm_entered = 0
    lbo_confirm_expired = 0
    lbo_pull_triggered = 0
    lbo_pull_pulled_back = 0
    lbo_pull_resumed_entered = 0
    lbo_pull_expired = 0
    lbo_confirm_pending = False
    lbo_confirm_dir = 0
    lbo_confirm_t0 = -1
    lbo_confirm_trigger_level = float("nan")
    lbo_confirm_expiry = -1
    lbo_confirm_absorption_level = float("nan")
    lbo_confirm_absorption_bar = -1
    lbo_pull_triggered_state = False
    lbo_pull_trigger_bar = -1
    lbo_pull_trigger_price = float("nan")
    lbo_pull_pulled_back_state = False
    lbo_pullback_bar = -1
    lbo_pullback_price = float("nan")
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
    exit_sm_called = 0
    exit_sm_exit_taken = 0
    exit_sm_hold = 0
    legacy_exit_taken = 0
    fail_fast_exit_taken = 0
    fail_fast_exit_suppressed = 0
    skip_reasons = {
        "gated_blocked": 0,
        "no_recent_event": 0,
        "no_threshold_yet": 0,
        "no_event_in_window": 0,
        "impulse_failed_threshold": 0,
        "impulse_failed_confirm": 0,
    }
    gate_diag = {
        "weak_side_ask": 0,
        "weak_side_bid": 0,
        "weak_side_none": 0,
        "blocked_none": 0,
        "blocked_mismatch": 0,
        "allowed_count": 0,
        "allow_on_none": 0,
        "bypass_on_none": 0,
    }
    gate_avail = {
        "weak_side_checks": 0,
        "any_buy": 0,
        "any_sell": 0,
        "qual_buy": 0,
        "qual_sell": 0,
    }
    gate_event_idx = -1
    gate_event_age = -1
    gate_asym_val = float("nan")
    gate_thr_val = float("nan")
    gate_weak_side = None

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

    def _check_srf_event(bar_idx: int) -> Tuple[bool, str | None, Dict[str, float]]:
        if bar_idx < srf_disp_window_bars or bar_idx + srf_refill_window_bars >= n:
            return False, None, {}
        disp_ticks = (mid[bar_idx] - mid[bar_idx - srf_disp_window_bars]) / tick_size
        speed = abs(disp_ticks) / float(max(1, srf_disp_window_bars))
        if disp_ticks >= srf_min_disp_ticks:
            side = "long"
        elif disp_ticks <= -srf_min_disp_ticks:
            side = "short"
        else:
            return False, None, {}
        if speed < srf_min_speed_ticks_per_bar:
            return False, None, {}
        size_arr = top_ask_depth if side == "long" else top_bid_depth
        baseline_slice = size_arr[max(0, bar_idx - srf_baseline_bars) : bar_idx]
        post_slice = size_arr[bar_idx + 1 : bar_idx + 1 + srf_refill_window_bars]
        baseline_size = float(np.nanmedian(baseline_slice)) if baseline_slice.size else float("nan")
        post_size = float(np.nanmedian(post_slice)) if post_slice.size else float("nan")
        refill_ratio = float("nan")
        refill_ok = False
        if np.isfinite(baseline_size) and baseline_size > 0 and np.isfinite(post_size):
            refill_ratio = post_size / baseline_size
            refill_ok = refill_ratio <= srf_refill_ratio_max
        else:
            if spread_ticks[bar_idx] >= srf_min_spread_ticks:
                future = mid[bar_idx + 1 : bar_idx + 1 + srf_refill_window_bars]
                if future.size:
                    if side == "long":
                        refill_ok = np.nanmax(future) - mid[bar_idx] < tick_size
                    else:
                        refill_ok = mid[bar_idx] - np.nanmin(future) < tick_size
        if not refill_ok:
            return False, None, {}
        flow_sum = (
            float(np.sum(afr_flow_series[bar_idx + 1 : bar_idx + 1 + srf_flow_window_bars]))
            if srf_flow_window_bars > 0
            else 0.0
        )
        if srf_flow_confirm:
            if side == "long" and flow_sum < srf_min_flow:
                return False, None, {}
            if side == "short" and flow_sum > -srf_min_flow:
                return False, None, {}
        return True, side, {
            "srf_disp_ticks": float(disp_ticks),
            "srf_speed": float(speed),
            "srf_refill_ratio": float(refill_ratio),
            "srf_flow_sum": float(flow_sum),
        }

    srf_horizons = [2, 5, 10, 20, 50]

    def _srf_perfect_exec_stats(entry_idx: int, side: str) -> Dict[str, float]:
        s = 1.0 if side == "long" else -1.0
        entry_mid = mid[entry_idx]
        out: Dict[str, float] = {
            "entry_bar": float(entry_idx),
            "side": 1.0 if side == "long" else -1.0,
        }
        if not np.isfinite(entry_mid):
            for h in srf_horizons:
                out[f"R_{h}"] = float("nan")
            out["MFE_20"] = float("nan")
            out["MAE_20"] = float("nan")
            out["MFE_50"] = float("nan")
            out["MAE_50"] = float("nan")
            return out
        for h in srf_horizons:
            idx = entry_idx + h
            if idx < n and np.isfinite(mid[idx]):
                out[f"R_{h}"] = float(s * (mid[idx] - entry_mid) / tick_size)
            else:
                out[f"R_{h}"] = float("nan")
        for h in (20, 50):
            end = min(entry_idx + h, n - 1)
            if end <= entry_idx:
                out[f"MFE_{h}"] = float("nan")
                out[f"MAE_{h}"] = float("nan")
                continue
            window = mid[entry_idx + 1 : end + 1]
            diff = s * (window - entry_mid) / tick_size
            out[f"MFE_{h}"] = float(np.nanmax(diff)) if diff.size else float("nan")
            out[f"MAE_{h}"] = float(np.nanmax(-diff)) if diff.size else float("nan")
        return out

    def _exit_sm_v1_action(
        pos_state: Dict[str, object],
        bar_idx: int,
        mark_px: float,
        tick_size_val: float,
    ) -> Dict[str, object] | None:
        required = ("entry_bar", "entry_px", "side", "exit_bar", "tp_level", "sl_level")
        missing = [name for name in required if name not in pos_state]
        if missing:
            raise RuntimeError(f"EXIT_MODE=exit_sm_v1 missing pos fields: {missing}")
        if not np.isfinite(mark_px):
            return None
        side_val = str(pos_state["side"])
        entry_bar_val = int(pos_state["entry_bar"])
        entry_px_val = float(pos_state["entry_px"])
        exit_bar_val = int(pos_state["exit_bar"])
        tp_level_val = float(pos_state["tp_level"])
        sl_level_val = float(pos_state["sl_level"])
        age_bars = int(bar_idx - entry_bar_val)
        mfe_ticks_val = float(pos_state.get("mfe_ticks", 0.0))
        if side_val == "long":
            sl_hit = mark_px <= sl_level_val
            tp_hit = mark_px >= tp_level_val
        else:
            sl_hit = mark_px >= sl_level_val
            tp_hit = mark_px <= tp_level_val
        if sl_hit:
            return {"exit_reason": "SL", "exit_bar": bar_idx}
        if tp_hit:
            return {"exit_reason": "TP", "exit_bar": bar_idx}
        if age_bars >= int(validate_bars) and mfe_ticks_val < float(min_progress_ticks):
            return {"exit_reason": "SCRATCH", "exit_bar": bar_idx}
        be_arm_threshold = float(be_arm_ticks)
        if be_after_ticks > 0:
            be_arm_threshold = max(be_arm_threshold, float(be_after_ticks))
        if mfe_ticks_val >= be_arm_threshold:
            if not pos_state.get("be_armed", False):
                pos_state["be_armed"] = True
                pos_state["be_arm_bar"] = int(bar_idx)
            be_arm_bar = int(pos_state.get("be_arm_bar", entry_bar_val))
            if int(be_grace_bars) > 0 and (bar_idx - be_arm_bar) <= int(be_grace_bars):
                if (side_val == "long" and mark_px >= entry_px_val) or (
                    side_val == "short" and mark_px <= entry_px_val
                ):
                    pos_state["breakeven_set"] = True
                    pos_state["exit_on_be"] = True
                    return {"exit_reason": "BE_LIMIT", "exit_bar": bar_idx}
            be_offset = pos_state.get("be_offset_ticks")
            be_offset = int(be_offset) if be_offset is not None else (int(be_offset_ticks) if be_offset_ticks is not None else 0)
            be_level = float(
                entry_px_val + (be_offset * tick_size_val)
                if side_val == "long"
                else entry_px_val - (be_offset * tick_size_val)
            )
            if (side_val == "long" and mark_px <= be_level) or (side_val == "short" and mark_px >= be_level):
                pos_state["breakeven_set"] = True
                pos_state["exit_on_be"] = True
                return {"exit_reason": "BE", "exit_bar": bar_idx}
        if age_bars >= int(scratch_bars) and mfe_ticks_val < float(scratch_min_progress_ticks):
            if mfe_ticks_val >= float(scratch_require_mfe_ticks):
                if pos_state.get("scratch_start_bar") is None:
                    pos_state["scratch_start_bar"] = int(bar_idx)
                scratch_start = int(pos_state.get("scratch_start_bar", bar_idx))
                if (bar_idx - scratch_start) >= int(scratch_grace_bars):
                    return {"exit_reason": "SCRATCH", "exit_bar": bar_idx}
        if int(decay_bars) > 0:
            flow_val = float(afr_flow_series[bar_idx]) if bar_idx < len(afr_flow_series) else 0.0
            against = flow_val <= -float(decay_min_flow) if side_val == "long" else flow_val >= float(decay_min_flow)
            decay_count = int(pos_state.get("decay_flow_count", 0))
            decay_count = decay_count + 1 if against else 0
            pos_state["decay_flow_count"] = decay_count
            if decay_count >= int(decay_bars):
                return {"exit_reason": "DECAY", "exit_bar": bar_idx}
        if bar_idx >= exit_bar_val:
            return {"exit_reason": "TIME", "exit_bar": bar_idx}
        return None

    def _exit_sm_v2_action(
        pos_state: Dict[str, object],
        bar_idx: int,
        mark_px: float,
        tick_size_val: float,
    ) -> Dict[str, object] | None:
        nonlocal fail_fast_exit_taken, fail_fast_exit_suppressed
        required = ("entry_bar", "entry_px", "side", "exit_bar", "tp_level", "sl_level")
        missing = [name for name in required if name not in pos_state]
        if missing:
            raise RuntimeError(f"EXIT_MODE=exit_sm_v2 missing pos fields: {missing}")
        if not np.isfinite(mark_px):
            return None
        side_val = str(pos_state["side"])
        entry_bar_val = int(pos_state["entry_bar"])
        entry_px_val = float(pos_state["entry_px"])
        exit_bar_val = int(pos_state["exit_bar"])
        age_bars = int(bar_idx - entry_bar_val)
        pnl_ticks = (
            (mark_px - entry_px_val) / tick_size_val
            if side_val == "long"
            else (entry_px_val - mark_px) / tick_size_val
        )
        peak_mfe = float(pos_state.get("mfe_ticks", 0.0))
        mae_ticks = float(pos_state.get("mae_ticks", 0.0))
        if pnl_ticks >= peak_mfe:
            pos_state["last_best_bar"] = int(bar_idx)
        last_best_bar = int(pos_state.get("last_best_bar", entry_bar_val))
        sl_ticks_local = int(pos_state.get("sl_ticks", sl_ticks))
        tp_ticks_local = int(pos_state.get("tp_ticks", tp_ticks))
        if pnl_ticks <= -float(sl_ticks_local):
            return {"exit_reason": "SL", "exit_bar": bar_idx}
        if pnl_ticks >= float(tp_ticks_local):
            return {"exit_reason": "TP", "exit_bar": bar_idx}
        if passive_exit_enabled and pos_state.get("passive_exit_active", False):
            limit_px = float(pos_state.get("passive_exit_price", float("nan")))
            expiry_bar = int(pos_state.get("passive_exit_expiry", -1))
            pending_reason = str(pos_state.get("passive_exit_reason", "FAIL_FAST"))
            if bar_idx <= expiry_bar:
                filled = (mark_px >= limit_px) if side_val == "long" else (mark_px <= limit_px)
                if filled:
                    pos_state["exit_exec_style"] = "passive_filled"
                    pos_state["exit_px_override"] = limit_px
                    pos_state["passive_exit_active"] = False
                    if pending_reason == "FAIL_FAST":
                        fail_fast_exit_taken += 1
                    return {"exit_reason": pending_reason, "exit_bar": bar_idx}
                return None
            pos_state["exit_exec_style"] = "market_fallback"
            pos_state["passive_exit_active"] = False
            if pending_reason == "FAIL_FAST":
                fail_fast_exit_taken += 1
            return {"exit_reason": pending_reason, "exit_bar": bar_idx}
        fail_fast_active = fail_fast_enabled and int(fail_fast_max_adverse_ticks) < 999
        if age_bars <= int(fail_fast_bars):
            if mae_ticks <= -float(fail_fast_max_adverse_ticks):
                if not fail_fast_active:
                    fail_fast_exit_suppressed += 1
                elif passive_exit_enabled:
                    be_offset = pos_state.get("be_offset_ticks")
                    be_offset = int(be_offset) if be_offset is not None else 0
                    limit_px = (
                        entry_px_val + (be_offset * tick_size_val)
                        if side_val == "long"
                        else entry_px_val - (be_offset * tick_size_val)
                    )
                    pos_state["passive_exit_active"] = True
                    pos_state["passive_exit_reason"] = "FAIL_FAST"
                    pos_state["passive_exit_price"] = float(limit_px)
                    pos_state["passive_exit_expiry"] = int(bar_idx + passive_exit_bars)
                    return None
                elif fail_fast_active:
                    fail_fast_exit_taken += 1
                    return {"exit_reason": "FAIL_FAST", "exit_bar": bar_idx}
            if age_bars >= int(fail_fast_bars) and peak_mfe < float(min_progress_ticks):
                if not fail_fast_active:
                    fail_fast_exit_suppressed += 1
                elif passive_exit_enabled:
                    be_offset = pos_state.get("be_offset_ticks")
                    be_offset = int(be_offset) if be_offset is not None else 0
                    limit_px = (
                        entry_px_val + (be_offset * tick_size_val)
                        if side_val == "long"
                        else entry_px_val - (be_offset * tick_size_val)
                    )
                    pos_state["passive_exit_active"] = True
                    pos_state["passive_exit_reason"] = "FAIL_FAST"
                    pos_state["passive_exit_price"] = float(limit_px)
                    pos_state["passive_exit_expiry"] = int(bar_idx + passive_exit_bars)
                    return None
                elif fail_fast_active:
                    fail_fast_exit_taken += 1
                    return {"exit_reason": "FAIL_FAST", "exit_bar": bar_idx}
        if peak_mfe >= float(be_arm_ticks):
            if not pos_state.get("be_armed", False):
                pos_state["be_armed"] = True
                pos_state["be_arm_bar"] = int(bar_idx)
            be_offset = pos_state.get("be_offset_ticks")
            be_offset = int(be_offset) if be_offset is not None else 0
            be_level = float(
                entry_px_val + (be_offset * tick_size_val)
                if side_val == "long"
                else entry_px_val - (be_offset * tick_size_val)
            )
            if (side_val == "long" and mark_px <= be_level) or (side_val == "short" and mark_px >= be_level):
                if passive_exit_enabled:
                    pos_state["breakeven_set"] = True
                    pos_state["exit_on_be"] = True
                    pos_state["passive_exit_active"] = True
                    pos_state["passive_exit_reason"] = "BE_LIMIT"
                    pos_state["passive_exit_price"] = float(be_level)
                    pos_state["passive_exit_expiry"] = int(bar_idx + passive_exit_bars)
                    return None
                pos_state["breakeven_set"] = True
                pos_state["exit_on_be"] = True
                return {"exit_reason": "BE_LIMIT", "exit_bar": bar_idx}
        if peak_mfe >= float(runner_trail_start_ticks):
            pos_state["runner_active"] = True
        if pos_state.get("runner_active", False):
            giveback = float(peak_mfe - pnl_ticks)
            if giveback >= float(runner_trail_giveback_ticks):
                return {"exit_reason": "TRAIL", "exit_bar": bar_idx}
        if int(decay_bars) > 0:
            if (bar_idx - last_best_bar) >= int(decay_bars):
                flow_val = float(afr_flow_series[bar_idx]) if bar_idx < len(afr_flow_series) else 0.0
                if side_val == "long":
                    decay_ok = flow_val <= float(decay_min_flow)
                else:
                    decay_ok = flow_val >= -float(decay_min_flow)
                if decay_ok:
                    return {"exit_reason": "DECAY", "exit_bar": bar_idx}
        if bar_idx >= exit_bar_val:
            return {"exit_reason": "TIME", "exit_bar": bar_idx}
        return None

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

    def _mark_break_for_ft_on_block(entry_reason_val: str | None, side_val: str | None, entry_bar_val: int) -> None:
        nonlocal break_bar_long, break_bar_short
        if strategy_mode != "absorption_failure_v2":
            return
        if entry_reason_val != "break":
            return
        if afr_enter_on not in {"ft", "both"}:
            return
        if side_val == "long" and not rearm_used_long:
            break_bar_long = entry_bar_val
        if side_val == "short" and not rearm_used_short:
            break_bar_short = entry_bar_val

    def _lbo_latest_event_idx(
        event_pos: np.ndarray,
        event_asym: np.ndarray,
        event_thr: np.ndarray,
        entry_bar_val: int,
        lookback_bars_val: int,
    ) -> Tuple[int | None, str | None]:
        nonlocal lbo_rej_empty
        nonlocal lbo_rej_before_lookback
        nonlocal lbo_rej_thr_nan
        nonlocal lbo_rej_asym_below_thr
        if event_pos.size == 0:
            lbo_rej_empty += 1
            return None, "empty"
        idx_pos = np.searchsorted(event_pos, entry_bar_val, side="right") - 1
        if idx_pos < 0:
            lbo_rej_empty += 1
            return None, "empty"
        if event_pos[idx_pos] < entry_bar_val - lookback_bars_val:
            lbo_rej_before_lookback += 1
            return None, "before_lookback"
        thr_val = event_thr[idx_pos]
        if not np.isfinite(thr_val):
            lbo_rej_thr_nan += 1
            return None, "thr_nan"
        if not lbo_ignore_thr and event_asym[idx_pos] <= thr_val:
            lbo_rej_asym_below_thr += 1
            return None, "asym_below_thr"
        return int(idx_pos), None

    def _lbo_latest_event_idx_any(
        event_pos: np.ndarray,
        entry_bar_val: int,
        lookback_bars_val: int,
    ) -> int | None:
        if event_pos.size == 0:
            return None
        idx_pos = np.searchsorted(event_pos, entry_bar_val, side="right") - 1
        if idx_pos < 0:
            return None
        if event_pos[idx_pos] < entry_bar_val - lookback_bars_val:
            return None
        return int(idx_pos)

    def _lbo_weak_side(entry_bar_val: int) -> str | None:
        nonlocal lbo_missing_both
        nonlocal lbo_only_buy
        nonlocal lbo_only_sell
        nonlocal lbo_both_fail_thr
        if gate_mode == "side_matched":
            idx_buy = _lbo_latest_event_idx_any(event_pos_buy, entry_bar_val, weak_side_lookback_bars)
            idx_sell = _lbo_latest_event_idx_any(event_pos_sell, entry_bar_val, weak_side_lookback_bars)
            if idx_buy is None and idx_sell is None:
                lbo_missing_both += 1
                return None
            if idx_sell is None:
                lbo_only_buy += 1
                return "ask_weak"
            if idx_buy is None:
                lbo_only_sell += 1
                return "bid_weak"
            pos_buy = event_pos_buy[idx_buy]
            pos_sell = event_pos_sell[idx_sell]
            return "ask_weak" if pos_buy >= pos_sell else "bid_weak"
        idx_all, _ = _lbo_latest_event_idx(
            event_pos_all, event_asym_all, event_thr_all, entry_bar_val, weak_side_lookback_bars
        )
        if idx_all is None:
            return None
        if event_side_all[idx_all] == "buy":
            return "ask_weak"
        return "bid_weak"

    def _mark_lbo_break_for_ft_on_block(
        entry_reason_val: str | None,
        side_val: str | None,
        entry_bar_val: int,
        absorption_level_val: float,
        absorption_bar_val: int,
        break_level_val: float,
    ) -> None:
        nonlocal lbo_break_blocked_bar_long
        nonlocal lbo_break_blocked_bar_short
        nonlocal lbo_break_blocked_abs_level_long
        nonlocal lbo_break_blocked_abs_level_short
        nonlocal lbo_break_blocked_break_level_long
        nonlocal lbo_break_blocked_break_level_short
        nonlocal lbo_break_blocked_abs_bar_long
        nonlocal lbo_break_blocked_abs_bar_short
        if strategy_mode != "lrams_breakout_v1":
            return
        if entry_reason_val != "break":
            return
        if side_val == "long":
            lbo_break_blocked_bar_long = entry_bar_val
            lbo_break_blocked_abs_level_long = float(absorption_level_val)
            lbo_break_blocked_break_level_long = float(break_level_val)
            lbo_break_blocked_abs_bar_long = int(absorption_bar_val)
        if side_val == "short":
            lbo_break_blocked_bar_short = entry_bar_val
            lbo_break_blocked_abs_level_short = float(absorption_level_val)
            lbo_break_blocked_break_level_short = float(break_level_val)
            lbo_break_blocked_abs_bar_short = int(absorption_bar_val)

    def _latest_event_side(entry_bar_val: int) -> str | None:
        idx_buy = np.searchsorted(event_pos_buy, entry_bar_val - 1, side="right") - 1
        idx_sell = np.searchsorted(event_pos_sell, entry_bar_val - 1, side="right") - 1
        if idx_buy < 0 and idx_sell < 0:
            return None
        if idx_buy < 0:
            return "sell"
        if idx_sell < 0:
            return "buy"
        return "buy" if event_pos_buy[idx_buy] >= event_pos_sell[idx_sell] else "sell"

    gate_diag_printed = False
    gate_debug_printed = 0
    gate_allow_filtered_printed = 0

    def _log_gate_decision(
        gate_allowed_val: bool,
        gate_reason_val: str,
        weak_side_val: str | None,
        weak_side_dir_val: str | None,
        desired_side_val: str | None,
        event_stream_val: str | None,
    ) -> None:
        nonlocal gate_diag_printed, gate_debug_printed
        if gate_mode != "side_matched":
            return
        if not gate_allowed_val:
            if weak_side_val is None:
                gate_diag["blocked_none"] += 1
            elif weak_side_val is not None:
                gate_diag["blocked_mismatch"] += 1
        if gate_debug and not gate_diag_printed:
            print(
                f"GATE_DIAG_DECISION {symbol_str} {day_str} weak_side={weak_side_val} "
                f"weak_side_dir={weak_side_dir_val} desired_side={desired_side_val} "
                f"allowed={int(gate_allowed_val)} reason={gate_reason_val} "
                f"weak_side_bid={gate_diag.get('weak_side_bid', 0)} "
                f"weak_side_ask={gate_diag.get('weak_side_ask', 0)} "
                f"weak_side_none={gate_diag.get('weak_side_none', 0)} "
                f"blocked_none={gate_diag.get('blocked_none', 0)} "
                f"blocked_mismatch={gate_diag.get('blocked_mismatch', 0)} "
                f"allowed_count={gate_diag.get('allowed_count', 0)} "
                f"allow_on_none={gate_diag.get('allow_on_none', 0)}",
                flush=True,
            )
            gate_diag_printed = True
        if gate_debug and gate_debug_printed < 5:
            print(
                f"GATE_DEBUG {symbol_str} {day_str} entry_bar={entry_bar} desired_side={desired_side_val} "
                f"weak_side={weak_side_val} stream={event_stream_val} "
                f"gate_decision={'allow' if gate_allowed_val else gate_reason_val}",
                flush=True,
            )
            gate_debug_printed += 1
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
        event_side_all = np.array([], dtype=object)
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
        event_side_all = events_day["side"].astype(str).to_numpy()
        entry_bar_min = int(lookback_bars)
        entry_bar_max = int(max(n - 1, 0))
        event_pos_mismatch = False
        if event_pos_all.size:
            event_pos_min = int(event_pos_all.min())
            event_pos_max = int(event_pos_all.max())
            if event_pos_min < 0 or event_pos_max > entry_bar_max:
                event_pos_mismatch = True
                if validate_debug:
                    print(
                        f"EVENT_POS_WARN {symbol_str} {day_str} "
                        f"event_pos_min={event_pos_min} event_pos_max={event_pos_max} "
                        f"entry_bar_range={entry_bar_min}-{entry_bar_max}",
                        flush=True,
                    )
        if event_pos_mismatch:
            if "Time" in events_day.columns:
                day_times = pd.to_datetime(df_day["Time"], errors="coerce").to_numpy()
                event_times = pd.to_datetime(events_day["Time"], errors="coerce")
                valid_mask = event_times.notna().to_numpy()
                if valid_mask.any():
                    event_pos_all = np.searchsorted(day_times, event_times[valid_mask].to_numpy(), side="left")
                    event_pos_all = np.clip(event_pos_all, 0, n - 1).astype(int)
                    event_asym_all = event_asym_all[valid_mask]
                    event_thr_all = event_thr_all[valid_mask]
                    event_side_all = event_side_all[valid_mask]
                    if validate_debug:
                        print(
                            f"EVENT_POS_CONVERT {symbol_str} {day_str} converted_to_bar_index count={int(valid_mask.sum())}",
                            flush=True,
                        )
                else:
                    event_pos_all = np.array([], dtype=int)
                    event_asym_all = np.array([], dtype=float)
                    event_thr_all = np.array([], dtype=float)
                    event_side_all = np.array([], dtype=object)
                    if validate_debug:
                        print(
                            f"EVENT_POS_WARN {symbol_str} {day_str} no valid event times to convert",
                            flush=True,
                        )
            elif validate_debug:
                print(
                    f"EVENT_POS_WARN {symbol_str} {day_str} no Time column to convert event_pos",
                    flush=True,
                )
        order = np.argsort(event_pos_all, kind="mergesort")
        event_pos_all = event_pos_all[order]
        event_asym_all = event_asym_all[order]
        event_thr_all = event_thr_all[order]
        event_side_all = event_side_all[order]
        if event_pos_all.size:
            if event_pos_all.min() < 0 or event_pos_all.max() >= n:
                raise RuntimeError(
                    f"event_pos out of range after normalization: min={int(event_pos_all.min())} "
                    f"max={int(event_pos_all.max())} n={n}"
                )
            if validate_debug and np.any(np.diff(event_pos_all) < 0):
                raise RuntimeError("event_pos is not monotonic after normalization.")
        buy_mask = event_side_all == "buy"
        sell_mask = event_side_all == "sell"
        event_pos_buy = event_pos_all[buy_mask]
        event_asym_buy = event_asym_all[buy_mask]
        event_thr_buy = event_thr_all[buy_mask]
        event_pos_sell = event_pos_all[sell_mask]
        event_asym_sell = event_asym_all[sell_mask]
        event_thr_sell = event_thr_all[sell_mask]
        if validate_debug:
            buy_range = "NA"
            sell_range = "NA"
            all_range = "NA"
            if event_pos_buy.size:
                buy_range = f"{int(event_pos_buy.min())}-{int(event_pos_buy.max())}"
            if event_pos_sell.size:
                sell_range = f"{int(event_pos_sell.min())}-{int(event_pos_sell.max())}"
            if event_pos_all.size:
                all_range = f"{int(event_pos_all.min())}-{int(event_pos_all.max())}"
            print(
                f"EVENT_POS_RANGE {symbol_str} {day_str} entry_bar_range={entry_bar_min}-{entry_bar_max} "
                f"event_pos_all_range={all_range} event_pos_buy_range={buy_range} event_pos_sell_range={sell_range}",
                flush=True,
            )

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
    flat_candidate_printed = 0
    flat_suppress_printed = 0
    event_debug_printed = False
    srf_mode_printed = False
    srf_trigger_printed = 0
    exit_mode_norm = (exit_mode or "").strip().lower()
    use_exit_sm_v1 = exit_mode_norm == "exit_sm_v1"
    use_exit_sm_v2 = exit_mode_norm == "exit_sm_v2"
    use_exit_exec_v2 = exit_mode_norm == "exit_exec_v2"
    if (not event_debug_printed) and event_pos_all.size:
        print(
            f"EVENT_POS_DEBUG {symbol_str} {day_str} "
            f"min={int(event_pos_all.min())} max={int(event_pos_all.max())} "
            f"first5={event_pos_all[:5].tolist()} last5={event_pos_all[-5:].tolist()} "
            f"gate_lookback_bars={gate_lookback_bars} n={n}",
            flush=True,
        )
        event_debug_printed = True
    while i <= max_i:
        entry_time = None
        exit_bar = -1
        side = ""
        entry_px = float("nan")
        entry_bar = -1
        tp_level = float("nan")
        sl_level = float("nan")
        # Evaluate signals each bar; otherwise a flat day would never increment counters.
        if strategy_mode == "srf_entry_v1":
            desired_side_top = None
        else:
            desired_side_top = _signal(i)
        if desired_side_top is not None:
            total_signals += 1
            if in_position:
                signals_in_position += 1
            else:
                if not pending_entry_active:
                    entry_candidates_when_flat += 1
        # DEBUG: prove we have flat candidates in this exact code path
        if (not in_position) and (desired_side_top is not None) and (not pending_entry_active):
            if debug_entry_print and flat_candidate_printed < 5:
                print(
                    f"FLAT_CANDIDATE {symbol_str} {day_str} i={i} side={desired_side_top} "
                    f"spread_ticks={spread_ticks if 'spread_ticks' in locals() else 'NA'}",
                    flush=True,
                )
                flat_candidate_printed += 1
        if in_position:
            exit_bar_val = pos.get("exit_bar")
            if exit_bar_val is None:
                exit_bar_val = int(pos.get("time_exit_bar", entry_bar))
                pos["exit_bar"] = exit_bar_val
            if use_exit_sm_v1:
                missing = [name for name in ("entry_bar", "entry_px", "side", "exit_bar", "tp_level", "sl_level") if name not in pos]
                if missing:
                    raise RuntimeError(f"EXIT_MODE=exit_sm_v1 missing pos fields: {missing}")
            if use_exit_sm_v2:
                missing = [name for name in ("entry_bar", "entry_px", "side", "exit_bar", "tp_level", "sl_level") if name not in pos]
                if missing:
                    raise RuntimeError(f"EXIT_MODE=exit_sm_v2 missing pos fields: {missing}")
            entry_bar = int(pos["entry_bar"])
            side = str(pos["side"])
            entry_px = float(pos["entry_px"])
            tp_level = float(pos["tp_level"])
            sl_level = float(pos["sl_level"])
            exit_bar = int(pos.get("exit_bar", -1))
            exit_px = float("nan")
            pnl_ticks = float("nan")
            exit_reason_raw = str(pos.get("exit_reason", "TIME"))
            exit_reason = exit_reason_raw
            sl_ticks_local = int(pos.get("sl_ticks", sl_ticks))
            # Exit routing handled below; avoid silent fallbacks when EXIT_MODE is set.
            if i >= entry_bar + 1:
                mark_px = bid[i] if side == "long" else ask[i]
                if np.isfinite(mark_px):
                    pnl_mark = (mark_px - entry_px) / tick_size if side == "long" else (entry_px - mark_px) / tick_size
                    if np.isfinite(pnl_mark):
                        pos["mfe_ticks"] = max(float(pos.get("mfe_ticks", 0.0)), float(pnl_mark))
                        pos["mae_ticks"] = min(float(pos.get("mae_ticks", 0.0)), float(pnl_mark))
                        if float(pos.get("bars_to_plus1", -1)) < 0 and float(pos.get("mfe_ticks", 0.0)) >= 1.0:
                            pos["bars_to_plus1"] = int(i - entry_bar)
                        if (
                            strategy_mode == "absorption_failure_v2"
                            and afr_scratch_min_progress_ticks > 0
                            and float(pos.get("bars_to_progress", -1)) < 0
                            and float(pos.get("mfe_ticks", 0.0)) >= float(afr_scratch_min_progress_ticks)
                        ):
                            pos["bars_to_progress"] = int(i - entry_bar)
                        if (
                            strategy_mode == "lrams_breakout_v1"
                            and lbo_scratch_min_progress_ticks > 0
                            and float(pos.get("bars_to_progress", -1)) < 0
                            and float(pos.get("mfe_ticks", 0.0)) >= float(lbo_scratch_min_progress_ticks)
                        ):
                            pos["bars_to_progress"] = int(i - entry_bar)
                    breakeven_ticks = pos.get("breakeven_ticks")
                    if breakeven_ticks is not None and not pos.get("breakeven_set", False):
                        mfe_ticks = (mark_px - entry_px) / tick_size if side == "long" else (entry_px - mark_px) / tick_size
                        if mfe_ticks >= float(breakeven_ticks):
                            pos["sl_level"] = float(entry_px)
                            pos["breakeven_set"] = True
                            pos["breakeven_triggered"] = True
                            sl_level = float(pos["sl_level"])
                    if pos.get("exit_locked"):
                        pass
                    elif use_exit_sm_v1:
                        # EXIT_MODE routing: exit_sm_v1 owns exit decisions; legacy exits are bypassed.
                        exit_sm_called += 1
                        sm_action = _exit_sm_v1_action(
                            pos_state=pos,
                            bar_idx=i,
                            mark_px=float(mark_px),
                            tick_size_val=tick_size,
                        )
                        if sm_action is not None:
                            exit_sm_exit_taken += 1
                            pos["exit_bar"] = int(sm_action["exit_bar"])
                            pos["exit_reason"] = str(sm_action["exit_reason"])
                            pos["exit_engine"] = "exit_sm_v1"
                            pos["exit_locked"] = True
                        else:
                            exit_sm_hold += 1
                    elif use_exit_sm_v2:
                        # EXIT_MODE routing: exit_sm_v2 owns exit decisions; legacy exits are bypassed.
                        exit_sm_called += 1
                        sm_action = _exit_sm_v2_action(
                            pos_state=pos,
                            bar_idx=i,
                            mark_px=float(mark_px),
                            tick_size_val=tick_size,
                        )
                        if sm_action is not None:
                            exit_sm_exit_taken += 1
                            pos["exit_bar"] = int(sm_action["exit_bar"])
                            pos["exit_reason"] = str(sm_action["exit_reason"])
                            pos["exit_engine"] = "exit_sm_v2"
                            pos["exit_locked"] = True
                        else:
                            exit_sm_hold += 1
                    elif use_exit_exec_v2:
                        exit_sm_called += 1
                        u_ticks = float(pnl_mark)
                        prev_mfe = float(pos.get("mfe_ticks", 0.0))
                        if u_ticks > prev_mfe:
                            pos["last_best_bar"] = int(i)
                        age_bars = int(i - entry_bar)
                        sl_ticks_local = int(pos.get("sl_ticks", sl_ticks))
                        tp_ticks_local = int(pos.get("tp_ticks", tp_ticks))
                        tp_level_exec = float(entry_px + tp_ticks_local * tick_size) if side == "long" else float(
                            entry_px - tp_ticks_local * tick_size
                        )
                        sl_level_exec = float(entry_px - sl_ticks_local * tick_size) if side == "long" else float(
                            entry_px + sl_ticks_local * tick_size
                        )
                        if side == "long":
                            sl_hit = mark_px <= sl_level_exec
                            tp_hit = mark_px >= tp_level_exec
                        else:
                            sl_hit = mark_px >= sl_level_exec
                            tp_hit = mark_px <= tp_level_exec
                        if float(pos.get("mfe_ticks", 0.0)) >= float(be_arm_ticks):
                            if not pos.get("be_armed", False):
                                pos["be_armed"] = True
                                pos["be_arm_bar"] = int(i)
                        against = (afr_flow_series[i] <= -decay_min_flow) if side == "long" else (
                            afr_flow_series[i] >= decay_min_flow
                        )
                        decay_count = int(pos.get("decay_flow_count", 0))
                        decay_count = decay_count + 1 if against else 0
                        pos["decay_flow_count"] = decay_count
                        be_armed = bool(pos.get("be_armed", False))
                        be_arm_bar = int(pos.get("be_arm_bar", entry_bar))
                        be_limit_active = be_armed and (i - be_arm_bar) <= int(be_grace_bars)
                        be_limit_hit = be_limit_active and (
                            (side == "long" and mark_px >= entry_px) or (side == "short" and mark_px <= entry_px)
                        )
                        scratch_condition = age_bars >= int(scratch_bars) and float(pos.get("mfe_ticks", 0.0)) < float(
                            scratch_min_progress_ticks
                        )
                        scratch_hit = False
                        if scratch_condition and float(pos.get("mfe_ticks", 0.0)) >= float(scratch_require_mfe_ticks):
                            if pos.get("scratch_start_bar") is None:
                                pos["scratch_start_bar"] = int(i)
                            scratch_start = int(pos.get("scratch_start_bar", i))
                            scratch_hit = (i - scratch_start) >= int(scratch_grace_bars)
                        time_hit = i >= int(pos.get("time_exit_bar", entry_bar))
                        decay_hit = decay_count >= int(decay_bars)
                        if sl_hit:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "SL"
                        elif tp_hit:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "TP"
                        elif be_limit_hit:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "BE_LIMIT"
                            pos["breakeven_set"] = True
                            pos["exit_on_be"] = True
                        elif decay_hit:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "DECAY"
                        elif scratch_hit:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "SCRATCH"
                        elif time_hit:
                            pos["exit_bar"] = i
                            pos["exit_reason"] = "TIME"
                        if pos.get("exit_reason") is not None:
                            pos["exit_engine"] = "exit_exec_v2"
                            pos["exit_locked"] = True
                            exit_sm_exit_taken += 1
                        else:
                            exit_sm_hold += 1
                    else:
                        if side == "long":
                            if mark_px >= tp_level:
                                pos["exit_bar"] = i
                                pos["exit_reason"] = "TP"
                                pos["exit_engine"] = "legacy"
                                pos["exit_locked"] = True
                            elif mark_px <= sl_level:
                                pos["exit_bar"] = i
                                pos["exit_reason"] = "SL"
                                pos["exit_engine"] = "legacy"
                                pos["exit_locked"] = True
                        else:
                            if mark_px <= tp_level:
                                pos["exit_bar"] = i
                                pos["exit_reason"] = "TP"
                                pos["exit_engine"] = "legacy"
                                pos["exit_locked"] = True
                            elif mark_px >= sl_level:
                                pos["exit_bar"] = i
                                pos["exit_reason"] = "SL"
                                pos["exit_engine"] = "legacy"
                                pos["exit_locked"] = True
                if not use_exit_sm_v2 and not use_exit_exec_v2 and not use_exit_sm_v1 and (
                    strategy_mode == "absorption_failure_v2"
                    and pos.get("exit_reason") == "TIME"
                    and afr_scratch_bars > 0
                    and (i - entry_bar) >= int(afr_scratch_bars)
                    and float(pos.get("mfe_ticks", 0.0)) < float(afr_scratch_min_progress_ticks)
                ):
                    pos["exit_bar"] = i
                    pos["exit_reason"] = "SCRATCH"
                    pos["exit_engine"] = "legacy"
                    pos["exit_locked"] = True
                if not use_exit_sm_v2 and not use_exit_exec_v2 and not use_exit_sm_v1 and (
                    strategy_mode == "lrams_breakout_v1"
                    and pos.get("exit_reason") == "TIME"
                    and lbo_scratch_bars > 0
                    and (i - entry_bar) >= int(lbo_scratch_bars)
                    and float(pos.get("mfe_ticks", 0.0)) < float(lbo_scratch_min_progress_ticks)
                ):
                    pos["exit_bar"] = i
                    pos["exit_reason"] = "SCRATCH"
                    pos["exit_engine"] = "legacy"
                    pos["exit_locked"] = True
                if (
                    not use_exit_sm_v2
                    and not use_exit_exec_v2
                    and not use_exit_sm_v1
                    and strategy_mode.startswith("absorption_failure")
                    and pos.get("exit_reason") == "TIME"
                ):
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
                        pos["exit_engine"] = "legacy"
                        pos["exit_locked"] = True
                if (
                    not use_exit_sm_v2
                    and not use_exit_exec_v2
                    and not use_exit_sm_v1
                    and strategy_mode == "lrams_breakout_v1"
                    and pos.get("exit_reason") == "TIME"
                ):
                    flow_sum = _flow_sum_at(i, lbo_confirm_bars)
                    decay_count = int(pos.get("decay_count", 0))
                    if side == "long":
                        if flow_sum < lbo_min_flow_abs:
                            decay_count += 1
                        else:
                            decay_count = 0
                    else:
                        if flow_sum > -lbo_min_flow_abs:
                            decay_count += 1
                        else:
                            decay_count = 0
                    pos["decay_count"] = decay_count
                    if decay_count >= lbo_decay_bars:
                        pos["exit_bar"] = i
                        pos["exit_reason"] = "DECAY"
                        pos["exit_on_decay"] = True
                        pos["exit_engine"] = "legacy"
                        pos["exit_locked"] = True
                if not use_exit_sm_v2 and not use_exit_exec_v2 and not use_exit_sm_v1 and pos.get("exit_reason") is None:
                    if i >= int(pos.get("time_exit_bar", entry_bar)):
                        pos["exit_bar"] = i
                        pos["exit_reason"] = "TIME"
                        pos["exit_engine"] = "legacy"
                        pos["exit_locked"] = True
            if pos.get("exit_bar") is not None and i >= int(pos["exit_bar"]):
                exit_bar = int(pos["exit_bar"])
                if not np.isfinite(bid[exit_bar]) or not np.isfinite(ask[exit_bar]):
                    raise RuntimeError("Non-finite exit price; check data integrity.")
                if not pos.get("exit_engine"):
                    pos["exit_engine"] = "legacy"
                exit_reason_raw = str(pos.get("exit_reason", "TIME"))
                exit_reason = exit_reason_raw
                tp_level = float(pos.get("tp_level", float("nan")))
                sl_level = float(pos.get("sl_level", float("nan")))
                if not use_exit_exec_v2 and not use_exit_sm_v2 and not use_exit_sm_v1:
                    legacy_exit_taken += 1
                # All exits fill at executable prices (or passive limit when specified).
                exit_px_override = pos.get("exit_px_override")
                if exit_px_override is not None and np.isfinite(exit_px_override):
                    exit_px = float(exit_px_override)
                else:
                    exit_px = bid[exit_bar] if side == "long" else ask[exit_bar]
                pnl_ticks = (exit_px - entry_px) / tick_size if side == "long" else (entry_px - exit_px) / tick_size
                tp_ticks_local = float(pos.get("tp_ticks", tp_ticks))
                sl_ticks_local = float(pos.get("sl_ticks", sl_ticks))
                if use_exit_exec_v2 and exit_reason == "SL":
                    max_pnl = -float(sl_ticks_local) + 1e-6
                    if pnl_ticks > max_pnl:
                        print(
                            f"PNL_BOUND_FAIL side={side} entry_px={entry_px:.4f} exit_px={exit_px:.4f} "
                            f"pnl_ticks={pnl_ticks:.4f} sl_ticks={sl_ticks_local}",
                            flush=True,
                        )
                        raise RuntimeError("PnL above SL bound (should be <= -sl_ticks).")
                if use_exit_exec_v2 and exit_reason == "TP":
                    min_pnl = float(tp_ticks_local) - 1e-6
                    if pnl_ticks < min_pnl:
                        print(
                            f"PNL_BOUND_FAIL side={side} entry_px={entry_px:.4f} exit_px={exit_px:.4f} "
                            f"pnl_ticks={pnl_ticks:.4f} tp_ticks={tp_ticks_local}",
                            flush=True,
                        )
                        raise RuntimeError("PnL below TP bound (should be >= tp_ticks).")
                if exit_debug:
                    trigger_px = bid[exit_bar] if side == "long" else ask[exit_bar]
                    print(
                        f"EXIT_DEBUG side={side} entry_bar={entry_bar} exit_bar={exit_bar} "
                        f"entry_px={entry_px:.4f} exit_px={exit_px:.4f} tp_level={tp_level:.4f} "
                        f"sl_level={sl_level:.4f} trigger_px={trigger_px:.4f} reason={exit_reason}",
                        flush=True,
                    )
                if bool(pos.get("breakeven_triggered", False)):
                    afr_be_triggered += 1
                exit_exec_style = str(pos.get("exit_exec_style", "market"))
                if exit_exec_style == "passive_filled":
                    passive_filled_count += 1
                elif exit_exec_style == "market_fallback":
                    passive_fallback_count += 1
                if strategy_mode.startswith("absorption_failure"):
                    if exit_reason == "TP":
                        exit_reason = "tp"
                    elif exit_reason == "SL":
                        if bool(pos.get("breakeven_set", False)):
                            exit_reason = "be"
                            pos["exit_on_be"] = True
                        else:
                            exit_reason = "sl"
                    elif exit_reason == "BE_LIMIT":
                        exit_reason = "be_limit"
                    elif exit_reason == "BE":
                        exit_reason = "be"
                        pos["exit_on_be"] = True
                    elif exit_reason == "SCRATCH":
                        exit_reason = "scratch"
                    elif exit_reason == "TIME":
                        exit_reason = "max_hold"
                    elif exit_reason == "DECAY":
                        exit_reason = "decay"
                    if strategy_mode == "absorption_failure_v2":
                        abs_bar = int(pos.get("absorption_bar", -1))
                        if exit_reason in {"scratch", "be"}:
                            if side == "long":
                                if not rearm_used_long:
                                    rearm_active_long = True
                                    rearm_expiry_long = abs_bar + afr_rearm_max_bars
                            else:
                                if not rearm_used_short:
                                    rearm_active_short = True
                                    rearm_expiry_short = abs_bar + afr_rearm_max_bars
                        elif exit_reason in {"tp", "sl"}:
                            if side == "long":
                                absorption_level_long = float("nan")
                                absorption_bar_long = -1
                                rearm_active_long = False
                                rearm_used_long = True
                                break_bar_long = -1
                            else:
                                absorption_level_short = float("nan")
                                absorption_bar_short = -1
                                rearm_active_short = False
                                rearm_used_short = True
                                break_bar_short = -1
                    else:
                        if exit_bar - entry_bar <= afr_rearm_stop_max_bars:
                            _arm_rearm(side, int(pos.get("absorption_bar", -1)))
                elif strategy_mode == "lrams_breakout_v1":
                    if exit_reason == "TP":
                        exit_reason = "tp"
                    elif exit_reason == "SL":
                        if bool(pos.get("breakeven_set", False)):
                            exit_reason = "be"
                            pos["exit_on_be"] = True
                        else:
                            exit_reason = "sl"
                    elif exit_reason == "BE_LIMIT":
                        exit_reason = "be_limit"
                    elif exit_reason == "BE":
                        exit_reason = "be"
                        pos["exit_on_be"] = True
                    elif exit_reason == "SCRATCH":
                        exit_reason = "scratch"
                    elif exit_reason == "TIME":
                        exit_reason = "max_hold"
                    elif exit_reason == "DECAY":
                        exit_reason = "decay"
                    abs_bar = int(pos.get("absorption_bar", -1))
                    if lbo_rearm_enabled and exit_reason in {"scratch", "be"}:
                        if side == "long":
                            if not lbo_rearm_used_long:
                                lbo_rearm_active_long = True
                                lbo_rearm_expiry_long = abs_bar + lbo_rearm_max_bars
                        else:
                            if not lbo_rearm_used_short:
                                lbo_rearm_active_short = True
                                lbo_rearm_expiry_short = abs_bar + lbo_rearm_max_bars
                    if exit_reason in {"tp", "sl"}:
                        if side == "long":
                            lbo_rearm_used_long = True
                        else:
                            lbo_rearm_used_short = True
                entry_time = pos.get("entry_time")
                if entry_time is None:
                    entry_time = df_day["Time"].iloc[entry_bar]
                    pos["entry_time"] = entry_time
                if not use_exit_exec_v2:
                    eps = 1e-9
                    entry_spread = float(pos.get("entry_spread_ticks", 0.0))
                    floor_ticks = -(sl_ticks_local + entry_spread + eps)
                entry_time = pos.get("entry_time")
                if entry_time is None:
                    entry_time = df_day["Time"].iloc[entry_bar]
                    pos["entry_time"] = entry_time
                if not np.isfinite(pnl_ticks):
                    if pnl_bound_printed < 5:
                        try:
                            print(
                                "PnL nan:",
                                {
                                    "side": side,
                                    "reason": exit_reason,
                                    "entry_time": str(entry_time),
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
                if (not use_exit_exec_v2) and pnl_ticks < floor_ticks and pnl_bound_printed < 5:
                    dt_ms = (df_day["Time"].iloc[exit_bar] - entry_time).total_seconds() * 1000.0
                    try:
                        print(
                            "PnL below bound:",
                            {
                                "pnl_ticks": float(pnl_ticks),
                                "floor_ticks": float(floor_ticks),
                                "side": side,
                                "reason": exit_reason,
                                "entry_time": str(entry_time),
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
                if exit_debug and (not debug_trigger_printed) and exit_reason in {"TP", "SL"}:
                    j0 = entry_bar + 1
                    j1 = min(entry_bar + 5, exit_bar)
                    series = (bid[j0 : j1 + 1] if side == "long" else ask[j0 : j1 + 1]).tolist()
                    try:
                        print(
                            f"Exit trigger debug: side={side} entry_bar={entry_bar} exit_bar={exit_bar} "
                            f"entry_time={entry_time} exit_time={df_day['Time'].iloc[exit_bar]} "
                            f"entry_px={entry_px:.2f} exit_px={exit_px:.2f} tp_level={tp_level:.2f} sl_level={sl_level:.2f} "
                            f"series={series} first_cross_bar={exit_bar} reason={exit_reason}",
                            flush=True,
                        )
                    except OSError:
                        pass
                    debug_trigger_printed = True
                entry_reason_out = pos.get("entry_reason")
                entry_root_reason_out = pos.get("entry_root_reason")
                entry_reason_raw_out = pos.get("entry_reason_raw")
                entry_reason_label_out = pos.get("entry_reason_label")
                if strategy_mode == "lrams_breakout_v1" and not entry_reason_out:
                    entry_reason_out = "break"
                exit_exec_style = str(pos.get("exit_exec_style", "market"))
                if exit_exec_style == "passive_filled":
                    passive_filled_count += 1
                elif exit_exec_style == "market_fallback":
                    passive_fallback_count += 1
                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": df_day["Time"].iloc[exit_bar],
                        "side": side,
                        "entry_px": float(entry_px),
                        "exit_px": float(exit_px),
                        "pnl_ticks": float(pnl_ticks),
                        "realized_pnl_ticks": float(pnl_ticks),
                        "entry_bar": int(pos["entry_bar"]),
                        "exit_bar": int(exit_bar),
                        "hold_bars_realized": int(exit_bar - int(pos["entry_bar"])),
                        "hold_bars": int(exit_bar - int(pos["entry_bar"])),
                        "exit_reason": exit_reason,
                        "exit_reason_raw": exit_reason_raw,
                        "exit_reason_std": _standard_exit_reason(exit_reason),
                        "exit_exec_style": exit_exec_style,
                        "exit_exec_style": exit_exec_style,
                        "exit_engine": str(pos.get("exit_engine", "legacy")),
                        "exit_mode_norm": exit_mode_norm,
                        "exit_mode": "exit_exec_v2"
                        if use_exit_exec_v2
                        else ("exit_sm_v2" if use_exit_sm_v2 else ("exit_sm_v1" if use_exit_sm_v1 else "legacy")),
                        "entry_reason": entry_reason_out if entry_reason_out else "other",
                        "entry_root_reason": entry_root_reason_out if entry_root_reason_out else "other",
                        "entry_reason_raw": entry_reason_raw_out if entry_reason_raw_out else "",
                        "entry_reason_label": entry_reason_label_out if entry_reason_label_out else "",
                        "absorption_level": float(pos.get("absorption_level", float("nan"))),
                        "break_level": float(pos.get("break_level", float("nan"))),
                        "flow_align_sum_at_entry": float(pos.get("flow_align_sum_at_entry", 0.0)),
                        "bars_from_absorption_to_entry": int(pos.get("bars_from_absorption", -1)),
                        "bars_from_absorption": int(pos.get("bars_from_absorption", -1)),
                        "bars_to_progress": int(pos.get("bars_to_progress", -1)),
                        "bars_to_plus1": int(pos.get("bars_to_plus1", -1)),
                        "confirm_bars_waited": int(pos.get("confirm_bars_waited", 0)),
                        "confirm_price_ref": float(pos.get("confirm_price_ref", float("nan"))),
                        "mae_ticks": float(pos.get("mae_ticks", 0.0)),
                        "mfe_ticks": float(pos.get("mfe_ticks", 0.0)),
                        "be_armed": bool(pos.get("be_armed", False)),
                        "entry_spread_ticks": float(pos.get("entry_spread_ticks", float("nan"))),
                        "gate_event_age": int(pos.get("gate_event_age", -1)),
                        "gate_asym_val": float(pos.get("gate_asym_val", float("nan"))),
                        "gate_thr_val": float(pos.get("gate_thr_val", float("nan"))),
                        "exit_on_decay": bool(pos.get("exit_on_decay", False)),
                        "exit_on_be": bool(pos.get("exit_on_be", False)),
                        "lbo_direction": str(pos.get("lbo_direction", "")),
                        "lbo_weak_side": str(pos.get("lbo_weak_side", "")),
                        "lbo_break_ticks": float(pos.get("lbo_break_ticks", float("nan"))),
                        "lbo_flow_align_sum": float(pos.get("lbo_flow_align_sum", 0.0)),
                        "srf_disp_ticks": float(pos.get("srf_disp_ticks", float("nan"))),
                        "srf_speed": float(pos.get("srf_speed", float("nan"))),
                        "srf_refill_ratio": float(pos.get("srf_refill_ratio", float("nan"))),
                        "srf_flow_sum": float(pos.get("srf_flow_sum", float("nan"))),
                    }
                )
                pnl_ticks_total += float(pnl_ticks)
                in_position = False
                pos = {}
                cooldown_until = exit_bar + entry_cooldown_bars
                i += 1
                continue

        if in_position:
            # Still in a position after exit handling; skip entry logic.
            i += 1
            continue

        if strategy_mode == "srf_entry_v1" and srf_debug and not srf_mode_printed:
            print(f"{symbol_str} {day_str} SRF_SIDE_MODE={srf_side_mode}", flush=True)
            srf_mode_printed = True

        pending_confirmed = False
        pending_payload: Dict[str, object] | None = None
        if pending_entry_active:
            if i > pending_entry_expiry:
                entry_confirm_failed += 1
                pending_entry_active = False
                pending_entry = {}
            else:
                if i <= pending_entry_created:
                    i += 1
                    continue
                confirm_px = bid[i] if pending_entry_side == "long" else ask[i]
                if np.isfinite(confirm_px):
                    if pending_entry_side == "long":
                        progress_ticks = (confirm_px - pending_entry_price) / tick_size
                    else:
                        progress_ticks = (pending_entry_price - confirm_px) / tick_size
                else:
                    progress_ticks = float("nan")
                if np.isfinite(progress_ticks) and progress_ticks >= float(entry_min_progress_ticks):
                    entry_confirm_passed += 1
                    pending_confirmed = True
                    pending_payload = dict(pending_entry)
                    pending_entry_active = False
                else:
                    i += 1
                    continue
        if pending_confirmed and pending_payload is not None:
            desired_side_top = str(pending_payload.get("side") or pending_payload.get("desired_side"))

        def _print_flat_suppress(reason: str, extra: str = "") -> None:
            nonlocal flat_suppress_printed
            if desired_side_top is None:
                return
            if flat_suppress_printed < 10:
                print(
                    f"FLAT_SUPPRESS {symbol_str} {day_str} i={i} reason={reason} {extra}".strip(),
                    flush=True,
                )
                flat_suppress_printed += 1

        desired_side = desired_side_top
        dmid_ticks = float("nan")
        impulse_ticks = float("nan")
        flow = 0.0
        flow_align_sum = 0.0
        stall_ticks = float("nan")
        break_ticks = float("nan")
        ft_progress_ticks = float("nan")
        break_level = float("nan")
        absorption_level = float("nan")
        absorption_bar = -1
        srf_disp_ticks = float("nan")
        srf_speed = float("nan")
        srf_refill_ratio = float("nan")
        srf_flow_sum = float("nan")
        lbo_break_ticks_entry = float("nan")
        lbo_flow_align_sum = 0.0
        lbo_direction = ""
        lbo_weak_side = ""
        entry_reason = None
        entry_reason_label = None
        confirm_bars_waited = 0
        confirm_price_ref = float("nan")
        lbo_entry_mode = ""
        afr_t_idx = -1
        afr_tf_idx = -1
        afr3_candidate = False
        afr3_debug_info: Dict[str, object] = {}
        mmas_passed = False
        entry_bar = i + 1
        if strategy_mode == "absorption_failure_v2":
            entry_bar = i
        if strategy_mode == "lrams_breakout_v1":
            entry_bar = i
        if strategy_mode == "srf_entry_v1":
            entry_bar = i
        if pending_confirmed:
            entry_bar = i
        if entry_bar >= n:
            _print_flat_suppress("entry_bar_oob", f"entry_bar={entry_bar} n={n}")
            break
        if not np.isfinite(bid[entry_bar]) or not np.isfinite(ask[entry_bar]) or not np.isfinite(mid[entry_bar]):
            _print_flat_suppress("nonfinite_quote", f"entry_bar={entry_bar}")
            i += 1
            continue
        cooldown_ok = i >= cooldown_until
        filter_blocked = False
        if entry_bar < len(session_ok) and not session_ok[entry_bar]:
            session_suppressed += 1
            filter_blocked = True
        if spread_ticks[entry_bar] < min_spread_ticks:
            spread_suppressed += 1
            filter_blocked = True
        if max_spread_ticks_for_entry > 0 and spread_ticks[entry_bar] > max_spread_ticks_for_entry:
            spread_suppressed += 1
            filter_blocked = True
        if not cooldown_ok:
            cooldown_suppressed += 1
            filter_blocked = True
        if filter_blocked and strategy_mode != "absorption_failure_v2":
            _print_flat_suppress(
                "filter_blocked",
                f"session_ok={bool(session_ok[entry_bar])} spread_ticks={int(spread_ticks[entry_bar])} cooldown_ok={cooldown_ok}",
            )
            i += 1
            continue

        if pending_confirmed and pending_payload is not None:
            entry_reason = pending_payload.get("entry_reason")
            entry_reason_label = pending_payload.get("entry_reason_label")
            absorption_level = float(pending_payload.get("absorption_level", float("nan")))
            break_level = float(pending_payload.get("break_level", float("nan")))
            flow_align_sum = float(pending_payload.get("flow_align_sum", 0.0))
            absorption_bar = int(pending_payload.get("absorption_bar", -1))
            confirm_bars_waited = int(pending_payload.get("confirm_bars_waited", i - pending_entry_created))
            confirm_price_ref = float(pending_payload.get("confirm_price_ref", pending_entry_price))
            lbo_direction = str(pending_payload.get("lbo_direction", ""))
            lbo_weak_side = str(pending_payload.get("lbo_weak_side", ""))
            lbo_break_ticks_entry = float(pending_payload.get("lbo_break_ticks_entry", float("nan")))
            lbo_flow_align_sum = float(pending_payload.get("lbo_flow_align_sum", 0.0))
            lbo_entry_mode = str(pending_payload.get("lbo_entry_mode", ""))
            flow = float(pending_payload.get("flow", 0.0))
            break_ticks = float(pending_payload.get("break_ticks", float("nan")))
            ft_progress_ticks = float(pending_payload.get("ft_progress_ticks", float("nan")))
            impulse_ticks = float(pending_payload.get("impulse_ticks", float("nan")))
            dmid_ticks = float(pending_payload.get("dmid_ticks", float("nan")))

        use_srf_entry = strategy_mode == "srf_entry_v1"
        srf_trigger = False
        raw_side = None
        if desired_side is None and use_srf_entry:
            srf_checked += 1
            srf_ok, srf_side, srf_feats = _check_srf_event(i)
            if srf_ok and srf_side is not None:
                srf_triggered += 1
                srf_trigger = True
                raw_side = srf_side
                if raw_side == "long":
                    srf_raw_long += 1
                else:
                    srf_raw_short += 1
                if srf_side_mode == "fade":
                    desired_side = "short" if raw_side == "long" else "long"
                else:
                    desired_side = raw_side
                if srf_debug and srf_trigger_printed < 10:
                    print(
                        f"SRF trig bar={i} raw={raw_side} mode={srf_side_mode} desired={desired_side}",
                        flush=True,
                    )
                    srf_trigger_printed += 1
                entry_reason = "srf"
                entry_reason_label = "srf_disp_refill"
                entry_root_reason = "srf"
                entry_exec_reason = "srf"
                srf_disp_ticks = float(srf_feats.get("srf_disp_ticks", float("nan")))
                srf_speed = float(srf_feats.get("srf_speed", float("nan")))
                srf_refill_ratio = float(srf_feats.get("srf_refill_ratio", float("nan")))
                srf_flow_sum = float(srf_feats.get("srf_flow_sum", float("nan")))
            if desired_side is None:
                i += 1
                continue
        if desired_side is None and use_srf_entry:
            i += 1
            continue
        if desired_side is None and strategy_mode == "impulse_confirm_v1":
            impulse_signals_checked += 1
            impulse_ticks = impulse_ticks_series[i]
            if not np.isfinite(impulse_ticks) or abs(impulse_ticks) < impulse_min_ticks:
                skip_reasons["impulse_failed_threshold"] += 1
                skipped += 1
                _print_flat_suppress("impulse_failed_threshold", f"impulse_ticks={impulse_ticks}")
                i += 1
                continue
            if impulse_ticks >= impulse_min_ticks:
                desired_side = "long"
            elif impulse_ticks <= -impulse_min_ticks:
                desired_side = "short"
            else:
                skip_reasons["impulse_failed_threshold"] += 1
                skipped += 1
                _print_flat_suppress("impulse_failed_threshold", f"impulse_ticks={impulse_ticks}")
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
                _print_flat_suppress("impulse_failed_confirm", f"confirm_bars={confirm_bars}")
                i += 1
                continue
            impulse_passed_confirm += 1
            flow = float(signed_vol[i]) if i < len(signed_vol) else 0.0
            entry_candidates_when_flat += 1
            total_signals += 1
        elif desired_side is None and pending_confirmed:
            # pending entry uses stored payload; no additional signal logic
            pass
        elif desired_side is None and strategy_mode == "absorption_failure_v1":
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
            entry_candidates_when_flat += 1
            total_signals += 1
        elif desired_side is None and strategy_mode == "absorption_failure_v2":
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

                rearm_allowed_long = False
                rearm_allowed_short = False
                if rearm_active_long:
                    if rearm_used_long or entry_bar > rearm_expiry_long or not np.isfinite(absorption_level_long):
                        rearm_active_long = False
                    else:
                        price_ok = np.isfinite(mid[entry_bar]) and abs(mid[entry_bar] - absorption_level_long) <= (
                            afr_rearm_band_ticks * tick_size
                        )
                        align_ok_long, _ = _align_ok("long", entry_bar)
                        rearm_allowed_long = bool(price_ok and align_ok_long)
                        if not rearm_allowed_long:
                            rearm_active_long = False
                if rearm_active_short:
                    if rearm_used_short or entry_bar > rearm_expiry_short or not np.isfinite(absorption_level_short):
                        rearm_active_short = False
                    else:
                        price_ok = np.isfinite(mid[entry_bar]) and abs(mid[entry_bar] - absorption_level_short) <= (
                            afr_rearm_band_ticks * tick_size
                        )
                        align_ok_short, _ = _align_ok("short", entry_bar)
                        rearm_allowed_short = bool(price_ok and align_ok_short)
                        if not rearm_allowed_short:
                            rearm_active_short = False

                long_break, long_abs_level, long_break_level, long_abs_bar = _check_break("long")
                short_break, short_abs_level, short_break_level, short_abs_bar = _check_break("short")
                if rearm_used_long or (rearm_active_long and not rearm_allowed_long):
                    long_break = False
                if rearm_used_short or (rearm_active_short and not rearm_allowed_short):
                    short_break = False
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
                            if (not allow_break) and allow_ft and break_quality_ok and not snapback_fail:
                                break_bar_long = entry_bar
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
                            if (not allow_break) and allow_ft and break_quality_ok and not snapback_fail:
                                break_bar_short = entry_bar
                        else:
                            missed_break = True
                else:
                    if long_break or short_break:
                        afr2_break_pass += 1
                        if long_break:
                            break_ticks = float((mid[entry_bar] - mid[entry_bar - 1]) / tick_size)
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
                            if allow_ft and break_quality_ok and not snapback_fail:
                                break_bar_long = entry_bar
                        if short_break:
                            break_ticks = float((mid[entry_bar] - mid[entry_bar - 1]) / tick_size)
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
                            if allow_ft and break_quality_ok and not snapback_fail:
                                break_bar_short = entry_bar

                if desired_side_local is None and allow_ft:
                    # Allow FT entry only if enabled and FT bar is reached.
                    if rearm_used_long or (rearm_active_long and not rearm_allowed_long):
                        break_bar_long = -1
                    if rearm_used_short or (rearm_active_short and not rearm_allowed_short):
                        break_bar_short = -1
                    if break_bar_long >= 0 and entry_bar > break_bar_long + ft_bars:
                        break_bar_long = -1
                    if break_bar_short >= 0 and entry_bar > break_bar_short + ft_bars:
                        break_bar_short = -1
                    if break_bar_long >= 0 and entry_bar > break_bar_long and entry_bar <= break_bar_long + ft_bars:
                        break_level_long = absorption_level_long + afr_break_ticks * tick_size
                        ft_progress_ticks = float((mid[entry_bar] - break_level_long) / tick_size)
                        ok, flow_align_sum_local = _align_ok("long", entry_bar)
                        ft_ok = ok and ft_progress_ticks >= afr_ft_min_ticks
                        if ft_ok:
                            afr2_ft_pass += 1
                            desired_side_local = "long"
                            absorption_level_local = absorption_level_long
                            break_level_local = break_level_long
                            absorption_bar_local = absorption_bar_long
                            afr_t_idx = break_bar_long
                            afr_tf_idx = entry_bar
                            if break_bar_long - 1 >= 0 and np.isfinite(mid[break_bar_long]) and np.isfinite(mid[break_bar_long - 1]):
                                break_ticks = float((mid[break_bar_long] - mid[break_bar_long - 1]) / tick_size)
                            entry_reason_local = "ft"
                            if rearm_active_long:
                                rearm_used_local = True
                            break_bar_long = -1
                    elif break_bar_short >= 0 and entry_bar > break_bar_short and entry_bar <= break_bar_short + ft_bars:
                        break_level_short = absorption_level_short - afr_break_ticks * tick_size
                        ft_progress_ticks = float((break_level_short - mid[entry_bar]) / tick_size)
                        ok, flow_align_sum_local = _align_ok("short", entry_bar)
                        ft_ok = ok and ft_progress_ticks <= -afr_ft_min_ticks
                        if ft_ok:
                            afr2_ft_pass += 1
                            desired_side_local = "short"
                            absorption_level_local = absorption_level_short
                            break_level_local = break_level_short
                            absorption_bar_local = absorption_bar_short
                            afr_t_idx = break_bar_short
                            afr_tf_idx = entry_bar
                            if break_bar_short - 1 >= 0 and np.isfinite(mid[break_bar_short]) and np.isfinite(mid[break_bar_short - 1]):
                                break_ticks = float((mid[break_bar_short] - mid[break_bar_short - 1]) / tick_size)
                            entry_reason_local = "ft"
                            if rearm_active_short:
                                rearm_used_local = True
                            break_bar_short = -1

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
                if rearm_used_local:
                    entry_reason_local = "rearm"
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

            if filter_blocked:
                def _probe_break_for_ft() -> None:
                    nonlocal break_bar_long
                    nonlocal break_bar_short
                    if not allow_ft:
                        return
                    long_break, _, _, _ = _check_break("long")
                    short_break, _, _, _ = _check_break("short")
                    if long_break and not rearm_used_long:
                        ok_long, flow_align_sum_local = _align_ok("long", entry_bar)
                        if ok_long:
                            break_quality_ok = (
                                abs(flow_align_sum_local) >= afr_break_quality_min_flow_abs
                                and spread_ticks[entry_bar] <= afr_break_quality_max_spread_ticks
                            )
                            snapback_fail = False
                            if break_quality_ok and np.isfinite(absorption_level_long) and afr_snapback_bars > 0:
                                end_idx = min(entry_bar + afr_snapback_bars, n - 1)
                                for j in range(entry_bar + 1, end_idx + 1):
                                    if np.isfinite(mid[j]) and abs(mid[j] - absorption_level_long) <= afr_snapback_band_ticks * tick_size:
                                        snapback_fail = True
                                        break
                            if break_quality_ok and not snapback_fail:
                                break_bar_long = entry_bar
                    if short_break and not rearm_used_short:
                        ok_short, flow_align_sum_local = _align_ok("short", entry_bar)
                        if ok_short:
                            break_quality_ok = (
                                abs(flow_align_sum_local) >= afr_break_quality_min_flow_abs
                                and spread_ticks[entry_bar] <= afr_break_quality_max_spread_ticks
                            )
                            snapback_fail = False
                            if break_quality_ok and np.isfinite(absorption_level_short) and afr_snapback_bars > 0:
                                end_idx = min(entry_bar + afr_snapback_bars, n - 1)
                                for j in range(entry_bar + 1, end_idx + 1):
                                    if np.isfinite(mid[j]) and abs(mid[j] - absorption_level_short) <= afr_snapback_band_ticks * tick_size:
                                        snapback_fail = True
                                        break
                            if break_quality_ok and not snapback_fail:
                                break_bar_short = entry_bar

                _probe_break_for_ft()
                i += 1
                continue

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

            entry_candidates_when_flat += 1
            total_signals += 1
        elif desired_side is None and strategy_mode == "absorption_failure_v3":
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
            entry_candidates_when_flat += 1
            total_signals += 1
        elif desired_side is None and strategy_mode == "lrams_breakout_v1":
            k = max(1, int(afr_k_bars))
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
                    if flow_setup > 0:
                        lbo_absorption_level_long = float(stall_series[i - 1])
                        lbo_absorption_bar_long = i - 1
                        if lbo_rearm_enabled:
                            lbo_rearm_used_long = False
                            lbo_rearm_active_long = False
                            lbo_rearm_expiry_long = -1
                    elif flow_setup < 0:
                        lbo_absorption_level_short = float(stall_series[i - 1])
                        lbo_absorption_bar_short = i - 1
                        if lbo_rearm_enabled:
                            lbo_rearm_used_short = False
                            lbo_rearm_active_short = False
                            lbo_rearm_expiry_short = -1

            if lbo_rearm_enabled:
                if lbo_rearm_active_long:
                    if entry_bar > lbo_rearm_expiry_long or (
                        np.isfinite(lbo_absorption_level_long)
                        and np.isfinite(mid[entry_bar])
                        and abs(mid[entry_bar] - lbo_absorption_level_long) > lbo_rearm_band_ticks * tick_size
                    ):
                        lbo_rearm_active_long = False
                if lbo_rearm_active_short:
                    if entry_bar > lbo_rearm_expiry_short or (
                        np.isfinite(lbo_absorption_level_short)
                        and np.isfinite(mid[entry_bar])
                        and abs(mid[entry_bar] - lbo_absorption_level_short) > lbo_rearm_band_ticks * tick_size
                    ):
                        lbo_rearm_active_short = False

            if lbo_break_blocked_bar_long >= 0 and entry_bar > lbo_break_blocked_bar_long + lbo_ft_bars:
                lbo_break_blocked_bar_long = -1
            if lbo_break_blocked_bar_short >= 0 and entry_bar > lbo_break_blocked_bar_short + lbo_ft_bars:
                lbo_break_blocked_bar_short = -1

            lbo_weak_side = _lbo_weak_side(entry_bar)
            if lbo_weak_side == "ask_weak":
                lbo_weak_ask += 1
                desired_side = "short" if lbo_flip_direction else "long"
                lbo_direction = desired_side
            elif lbo_weak_side == "bid_weak":
                lbo_weak_bid += 1
                desired_side = "long" if lbo_flip_direction else "short"
                lbo_direction = desired_side
            else:
                lbo_weak_none += 1
                lbo_weak_none_blocked += 1
                desired_side = None

            if desired_side is not None:
                if desired_side == "long":
                    absorption_level = lbo_absorption_level_long
                    absorption_bar = lbo_absorption_bar_long
                else:
                    absorption_level = lbo_absorption_level_short
                    absorption_bar = lbo_absorption_bar_short
                if not np.isfinite(absorption_level) or absorption_bar < 0:
                    desired_side = None

            weak_side_dir = desired_side
            if lbo_confirm_mode != "none":
                desired_side = None
                if lbo_confirm_pending:
                    confirm_flow_sum = (
                        _flow_sum_at(entry_bar, lbo_confirm_flow_align_bars)
                        if lbo_confirm_flow_align_bars > 0
                        else float(afr_flow_series[entry_bar])
                    )
                    flow_align_ok = True
                    if lbo_confirm_require_flow_align:
                        if lbo_confirm_dir > 0:
                            flow_align_ok = confirm_flow_sum >= lbo_confirm_min_flow_abs_align
                        else:
                            flow_align_ok = confirm_flow_sum <= -lbo_confirm_min_flow_abs_align
                    if entry_bar > lbo_confirm_t0 and entry_bar <= lbo_confirm_expiry:
                        progress_ticks = lbo_confirm_dir * (
                            lbo_confirm_price[entry_bar] - lbo_confirm_trigger_level
                        ) / tick_size
                        if not lbo_pull_triggered_state and progress_ticks >= lbo_confirm_ticks and flow_align_ok:
                            if lbo_confirm_mode == "confirm_ticks":
                                entry_reason = "confirm"
                                entry_reason_label = "lbo_confirm"
                                desired_side = "long" if lbo_confirm_dir > 0 else "short"
                                confirm_bars_waited = entry_bar - lbo_confirm_t0
                                confirm_price_ref = lbo_confirm_trigger_level
                                lbo_flow_align_sum = confirm_flow_sum
                                absorption_level = lbo_confirm_absorption_level
                                absorption_bar = lbo_confirm_absorption_bar
                                if np.isfinite(absorption_level):
                                    lbo_break_ticks_entry = (mid[entry_bar] - absorption_level) / tick_size
                                    break_level = (
                                        absorption_level + lbo_break_ticks * tick_size
                                        if desired_side == "long"
                                        else absorption_level - lbo_break_ticks * tick_size
                                    )
                                lbo_entry_mode = "confirm"
                                lbo_confirm_pending = False
                            else:
                                lbo_pull_triggered_state = True
                                lbo_pull_trigger_bar = entry_bar
                                lbo_pull_trigger_price = float(lbo_confirm_price[entry_bar])
                                lbo_pullback_expiry = entry_bar + lbo_pullback_max_bars
                                lbo_pull_triggered += 1
                        if lbo_confirm_mode == "confirm_ticks" and entry_reason is None:
                            desired_side = None
                    if lbo_confirm_mode == "pullback" and lbo_pull_triggered_state:
                        if entry_bar > lbo_pullback_expiry:
                            lbo_pull_expired += 1
                            lbo_confirm_pending = False
                        elif not lbo_pull_pulled_back_state:
                            retrace_ticks = lbo_confirm_dir * (
                                lbo_confirm_price[entry_bar] - lbo_pull_trigger_price
                            ) / tick_size
                            if retrace_ticks <= -lbo_pullback_ticks:
                                lbo_pull_pulled_back_state = True
                                lbo_pullback_bar = entry_bar
                                lbo_pullback_price = float(lbo_confirm_price[entry_bar])
                                lbo_pull_pulled_back += 1
                                lbo_resume_expiry = entry_bar + lbo_resume_max_bars
                        else:
                            if entry_bar > lbo_resume_expiry:
                                lbo_pull_expired += 1
                                lbo_confirm_pending = False
                            else:
                                resume_ticks = lbo_confirm_dir * (
                                    lbo_confirm_price[entry_bar] - lbo_pull_trigger_price
                                ) / tick_size
                                if resume_ticks >= lbo_resume_ticks:
                                    entry_reason = "pullback"
                                    entry_reason_label = "lbo_pullback"
                                    desired_side = "long" if lbo_confirm_dir > 0 else "short"
                                    confirm_bars_waited = entry_bar - lbo_confirm_t0
                                    confirm_price_ref = lbo_confirm_trigger_level
                                    lbo_flow_align_sum = confirm_flow_sum
                                    absorption_level = lbo_confirm_absorption_level
                                    absorption_bar = lbo_confirm_absorption_bar
                                    break_level = lbo_pull_trigger_price
                                    if np.isfinite(absorption_level):
                                        lbo_break_ticks_entry = (mid[entry_bar] - absorption_level) / tick_size
                                    lbo_entry_mode = "pullback"
                                    lbo_confirm_pending = False
                                else:
                                    desired_side = None
                    if (
                        lbo_confirm_mode == "confirm_ticks"
                        and entry_bar > lbo_confirm_expiry
                        and lbo_confirm_pending
                        and not lbo_pull_triggered_state
                    ):
                        lbo_confirm_expired += 1
                        lbo_confirm_pending = False
                    if (
                        lbo_confirm_mode == "pullback"
                        and entry_bar > lbo_confirm_expiry
                        and lbo_confirm_pending
                        and not lbo_pull_triggered_state
                    ):
                        lbo_pull_expired += 1
                        lbo_confirm_pending = False
                if not lbo_confirm_pending and entry_reason is None and weak_side_dir is not None:
                    lbo_confirm_pending = True
                    lbo_confirm_dir = 1 if weak_side_dir == "long" else -1
                    lbo_confirm_t0 = entry_bar
                    lbo_confirm_trigger_level = float(lbo_confirm_price[entry_bar])
                    lbo_confirm_absorption_level = float(absorption_level)
                    lbo_confirm_absorption_bar = int(absorption_bar)
                    lbo_confirm_expiry = entry_bar + lbo_confirm_max_bars
                    lbo_pending_started += 1
                    lbo_pull_triggered_state = False
                    lbo_pull_pulled_back_state = False
                    lbo_pull_trigger_bar = -1
                    lbo_pullback_bar = -1
                    lbo_pull_trigger_price = float("nan")
                    lbo_pullback_price = float("nan")
                    lbo_pullback_expiry = -1
                    lbo_resume_expiry = -1
                if entry_reason is None:
                    desired_side = None
            else:
                if desired_side is None:
                    if lbo_break_blocked_bar_long < 0 and lbo_break_blocked_bar_short < 0:
                        i += 1
                        continue

                break_level = (
                    absorption_level + lbo_break_ticks * tick_size
                    if desired_side == "long"
                    else absorption_level - lbo_break_ticks * tick_size
                )
                break_trigger = False
                if np.isfinite(mid[entry_bar]):
                    if desired_side == "long":
                        break_trigger = mid[entry_bar] >= break_level
                    else:
                        break_trigger = mid[entry_bar] <= break_level

                flow_align_sum = (
                    _flow_sum_at(entry_bar, lbo_confirm_bars) if lbo_confirm_bars > 0 else float(afr_flow_series[entry_bar])
                )
                confirm_ok = True
                if lbo_confirm_bars > 0:
                    if desired_side == "long":
                        confirm_ok = flow_align_sum >= lbo_min_flow_abs
                    else:
                        confirm_ok = flow_align_sum <= -lbo_min_flow_abs
                if spread_ticks[entry_bar] > lbo_max_spread_ticks:
                    confirm_ok = False

                if break_trigger:
                    entry_reason = "rearm" if (lbo_rearm_enabled and (lbo_rearm_active_long if desired_side == "long" else lbo_rearm_active_short)) else "break"
                    entry_reason_label = "lbo_now"
                    entry_root_reason = entry_reason
                    entry_exec_reason = "break"
                    lbo_break_ticks_entry = (mid[entry_bar] - absorption_level) / tick_size
                    lbo_flow_align_sum = flow_align_sum
                    lbo_entry_mode = "now"
                    if entry_reason == "break":
                        _mark_lbo_break_for_ft_on_block(
                            entry_reason,
                            desired_side,
                            entry_bar,
                            absorption_level,
                            absorption_bar,
                            break_level,
                        )
                        entry_exec_reason = "ft"
                        entry_reason = "ft"
                        entry_reason_label = "lbo_ft_wait"
                        lbo_entry_mode = "ft_wait"
                    if entry_reason == "rearm":
                        if desired_side == "long":
                            lbo_rearm_used_long = True
                            lbo_rearm_active_long = False
                        else:
                            lbo_rearm_used_short = True
                            lbo_rearm_active_short = False
                    if desired_side == "long":
                        lbo_break_blocked_bar_long = -1
                    else:
                        lbo_break_blocked_bar_short = -1
                if lbo_ft_bars > 0 and entry_reason is None:
                    ft_ok = False
                    if lbo_break_blocked_bar_long >= 0:
                        if entry_bar <= lbo_break_blocked_bar_long + lbo_ft_bars:
                            ft_level = lbo_break_blocked_break_level_long + lbo_ft_min_ticks * tick_size
                            abs_level = lbo_break_blocked_abs_level_long
                            if np.isfinite(mid[entry_bar]) and mid[entry_bar] >= ft_level and mid[entry_bar] >= abs_level:
                                if lbo_confirm_bars > 0:
                                    ft_ok = flow_align_sum >= lbo_min_flow_abs
                                else:
                                    ft_ok = flow_align_sum > 0.0
                                if spread_ticks[entry_bar] > lbo_max_spread_ticks:
                                    ft_ok = False
                                if ft_ok:
                                    entry_reason = "ft"
                                    entry_reason_label = "lbo_now"
                                    desired_side = "long"
                                    absorption_level = lbo_break_blocked_abs_level_long
                                    absorption_bar = lbo_break_blocked_abs_bar_long
                                    break_level = lbo_break_blocked_break_level_long
                                    lbo_break_ticks_entry = (mid[entry_bar] - absorption_level) / tick_size
                                    lbo_flow_align_sum = flow_align_sum
                                    lbo_entry_mode = "now"
                                    lbo_break_blocked_bar_long = -1
                        else:
                            lbo_break_blocked_bar_long = -1
                    elif lbo_break_blocked_bar_short >= 0:
                        if entry_bar <= lbo_break_blocked_bar_short + lbo_ft_bars:
                            ft_level = lbo_break_blocked_break_level_short - lbo_ft_min_ticks * tick_size
                            abs_level = lbo_break_blocked_abs_level_short
                            if np.isfinite(mid[entry_bar]) and mid[entry_bar] <= ft_level and mid[entry_bar] <= abs_level:
                                if lbo_confirm_bars > 0:
                                    ft_ok = flow_align_sum <= -lbo_min_flow_abs
                                else:
                                    ft_ok = flow_align_sum < 0.0
                                if spread_ticks[entry_bar] > lbo_max_spread_ticks:
                                    ft_ok = False
                                if ft_ok:
                                    entry_reason = "ft"
                                    entry_reason_label = "lbo_now"
                                    desired_side = "short"
                                    absorption_level = lbo_break_blocked_abs_level_short
                                    absorption_bar = lbo_break_blocked_abs_bar_short
                                    break_level = lbo_break_blocked_break_level_short
                                    lbo_break_ticks_entry = (mid[entry_bar] - absorption_level) / tick_size
                                    lbo_flow_align_sum = flow_align_sum
                                    lbo_entry_mode = "now"
                                    lbo_break_blocked_bar_short = -1
                        else:
                            lbo_break_blocked_bar_short = -1

            if entry_reason is not None and np.isfinite(lbo_break_ticks_entry):
                break_ticks = float(lbo_break_ticks_entry)
                flow = float(lbo_flow_align_sum)
                flow_align_sum = float(lbo_flow_align_sum)
            if entry_reason is None:
                desired_side = None
        elif desired_side is None and strategy_mode == "micro_momo_v1":
            k = max(1, int(micro_k_bars))
            if i >= k and np.isfinite(mid[i]) and np.isfinite(mid[i - k]):
                dmid = mid[i] - mid[i - k]
                impulse_ticks = dmid / tick_size
                flow = cs[i + 1] - cs[i + 1 - k]
                if impulse_ticks >= micro_impulse_ticks and flow >= micro_flow_min:
                    desired_side = "long"
                elif impulse_ticks <= -micro_impulse_ticks and flow <= -micro_flow_min:
                    desired_side = "short"
        elif desired_side is None and baseline_mode == "mmas":
            mmas_signals_checked += 1
            desired_side, dmid_ticks, flow, mmas_passed = _mmas_signal(i)
            if mmas_passed:
                mmas_passed_filters += 1
            if desired_side is not None:
                mmas_signaled += 1
        elif desired_side is None:
            desired_side = _signal(i)
            if desired_side is not None and i >= lookback_bars and np.isfinite(mid[i]) and np.isfinite(mid[i - lookback_bars]):
                dmid = mid[i] - mid[i - lookback_bars]
                dmid_ticks = dmid / tick_size
                impulse_ticks = dmid_ticks
                flow = cs[i + 1] - cs[i + 1 - lookback_bars]
        if desired_side is None:
            i += 1
            continue
        if debug_entry_print and flat_candidate_printed < 5:
            print(
                f"FLAT_CANDIDATE {symbol_str} {day_str} t={i} side={desired_side}",
                flush=True,
            )
            flat_candidate_printed += 1
        if strategy_mode != "impulse_confirm_v1" and not pending_confirmed:
            total_signals += 1
            entry_candidates_when_flat += 1
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
        weak_side = None
        gate_event_idx = -1
        gate_event_age = -1
        gate_asym_val = float("nan")
        gate_thr_val = float("nan")
        gate_weak_side = None
        weak_side_dir = None
        event_stream = None
        gate_bypass = False
        if gate_enabled:
            gate_allowed = True
            gate_reason = "allowed"
            if gate_mode == "side_matched":
                event_stream = "buy" if desired_side == "long" else "sell"
                if desired_side == "long":
                    event_pos = event_pos_buy
                    event_asym = event_asym_buy
                    event_thr = event_thr_buy
                else:
                    event_pos = event_pos_sell
                    event_asym = event_asym_sell
                    event_thr = event_thr_sell
                gate_avail["weak_side_checks"] += 1
                idx_buy_any = _lbo_latest_event_idx_any(event_pos_buy, entry_bar, weak_side_lookback_bars)
                idx_sell_any = _lbo_latest_event_idx_any(event_pos_sell, entry_bar, weak_side_lookback_bars)
                if idx_buy_any is not None:
                    gate_avail["any_buy"] += 1
                if idx_sell_any is not None:
                    gate_avail["any_sell"] += 1
                idx_buy_qual, _ = _lbo_latest_event_idx(
                    event_pos_buy, event_asym_buy, event_thr_buy, entry_bar, weak_side_lookback_bars
                )
                idx_sell_qual, _ = _lbo_latest_event_idx(
                    event_pos_sell, event_asym_sell, event_thr_sell, entry_bar, weak_side_lookback_bars
                )
                if idx_buy_qual is not None:
                    gate_avail["qual_buy"] += 1
                if idx_sell_qual is not None:
                    gate_avail["qual_sell"] += 1
                weak_side = _lbo_weak_side(entry_bar)
                weak_side_dir = None
                if weak_side is None:
                    gate_diag["weak_side_none"] += 1
                    if allow_gate_on_none:
                        gate_diag["allow_on_none"] += 1
                        weak_side_dir = desired_side
                        gate_reason = "allowed_none"
                        gate_allowed = True
                        gate_bypass = True
                        gate_diag["bypass_on_none"] += 1
                elif weak_side == "ask_weak":
                    gate_diag["weak_side_ask"] += 1
                    weak_side_dir = "long"
                    event_pos = event_pos_buy
                    event_asym = event_asym_buy
                    event_thr = event_thr_buy
                    event_stream = "buy"
                else:
                    gate_diag["weak_side_bid"] += 1
                    weak_side_dir = "short"
                    event_pos = event_pos_sell
                    event_asym = event_asym_sell
                    event_thr = event_thr_sell
                    event_stream = "sell"
                gate_weak_side = weak_side
                if weak_side_dir is None:
                    gate_allowed = False
                    gate_reason = "blocked_none"
                elif desired_side != weak_side_dir:
                    gate_allowed = False
                    gate_reason = "blocked_mismatch"
                _log_gate_decision(
                    gate_allowed_val=gate_allowed,
                    gate_reason_val=gate_reason,
                    weak_side_val=weak_side,
                    weak_side_dir_val=weak_side_dir,
                    desired_side_val=desired_side,
                    event_stream_val=event_stream,
                )
                if not gate_allowed:
                    skip_reasons["gated_blocked"] += 1
                    skipped += 1
                    blocked_signals += 1
                    if strategy_mode == "lrams_breakout_v1":
                        _mark_lbo_break_for_ft_on_block(
                            entry_reason,
                            desired_side,
                            entry_bar,
                            absorption_level,
                            absorption_bar,
                            break_level,
                        )
                    _mark_break_for_ft_on_block(entry_reason, desired_side, entry_bar)
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
                event_pos = event_pos_all
                event_asym = event_asym_all
                event_thr = event_thr_all
            if gate_bypass:
                eligible_signals += 1
            elif event_pos.size == 0:
                skip_reasons["no_recent_event"] += 1
                skipped += 1
                blocked_signals += 1
                gate_allowed = False
                gate_reason = "no_recent_event"
                _log_gate_decision(
                    gate_allowed_val=gate_allowed,
                    gate_reason_val=gate_reason,
                    weak_side_val=weak_side,
                    weak_side_dir_val=weak_side_dir,
                    desired_side_val=desired_side,
                    event_stream_val=event_stream,
                )
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
                if strategy_mode == "lrams_breakout_v1":
                    _mark_lbo_break_for_ft_on_block(
                        entry_reason,
                        desired_side,
                        entry_bar,
                        absorption_level,
                        absorption_bar,
                        break_level,
                    )
                _mark_break_for_ft_on_block(entry_reason, desired_side, entry_bar)
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
                idx_pos, rej = _lbo_latest_event_idx(
                    event_pos, event_asym, event_thr, entry_bar, gate_lookback_bars
                )
                if idx_pos is None:
                    if rej in {"empty", "before_lookback"}:
                        skip_reasons["no_recent_event"] += 1
                        gate_reason = "no_recent_event"
                    elif rej == "thr_nan":
                        skip_reasons["no_threshold_yet"] += 1
                        gate_reason = "no_threshold_yet"
                    elif rej == "asym_below_thr":
                        skip_reasons["gated_blocked"] += 1
                        gate_reason = "asym_below_thr"
                    else:
                        skip_reasons["no_recent_event"] += 1
                        gate_reason = "no_recent_event"
                    skipped += 1
                    blocked_signals += 1
                    gate_allowed = False
                    _log_gate_decision(
                        gate_allowed_val=gate_allowed,
                        gate_reason_val=gate_reason,
                        weak_side_val=weak_side,
                        weak_side_dir_val=weak_side_dir,
                        desired_side_val=desired_side,
                        event_stream_val=event_stream,
                    )
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
                    if strategy_mode == "lrams_breakout_v1":
                        _mark_lbo_break_for_ft_on_block(
                            entry_reason,
                            desired_side,
                            entry_bar,
                            absorption_level,
                            absorption_bar,
                            break_level,
                        )
                    _mark_break_for_ft_on_block(entry_reason, desired_side, entry_bar)
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
                eligible_signals += 1
                gate_event_idx = int(idx_pos)
                gate_event_age = int(entry_bar - int(event_pos[idx_pos]))
                gate_asym_val = float(event_asym[idx_pos])
                gate_thr_val = float(event_thr[idx_pos])
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

        # allowed_count is incremented only when a gated entry actually opens.

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

        if gate_enabled and gate_allowed:
            gate_diag["allowed_count"] += 1
        if gate_debug and gate_enabled and gate_allowed and filter_blocked and gate_allow_filtered_printed < 10:
            print(
                f"GATE_ALLOW_BUT_FILTERED {symbol_str} {day_str} entry_bar={entry_bar} side={desired_side} "
                f"weak_side={gate_weak_side} stream={event_stream} "
                f"session_ok={bool(session_ok[entry_bar])} spread_ticks={int(spread_ticks[entry_bar])} "
                f"cooldown_ok={cooldown_ok}",
                flush=True,
            )
            gate_allow_filtered_printed += 1
        if entry_confirm_style != "off" and entry_confirm_bars > 0 and not pending_confirmed:
            pending_entry_active = True
            pending_entry_side = desired_side
            pending_entry_price = float(mid[entry_bar])
            pending_entry_created = int(entry_bar)
            pending_entry_expiry = int(entry_bar + entry_confirm_bars)
            entry_confirm_checked += 1
            pending_entry = {
                "side": desired_side,
                "created_bar": int(entry_bar),
                "expiry_bar": int(entry_bar + entry_confirm_bars),
                "entry_price_ref": float(pending_entry_price),
                "entry_reason": entry_reason,
                "entry_reason_label": entry_reason_label,
                "absorption_level": float(absorption_level) if np.isfinite(absorption_level) else float("nan"),
                "break_level": float(break_level) if np.isfinite(break_level) else float("nan"),
                "flow_align_sum": float(flow_align_sum),
                "absorption_bar": int(absorption_bar),
                "confirm_price_ref": float(pending_entry_price),
                "lbo_direction": lbo_direction,
                "lbo_weak_side": lbo_weak_side,
                "lbo_break_ticks_entry": float(lbo_break_ticks_entry)
                if np.isfinite(lbo_break_ticks_entry)
                else float("nan"),
                "lbo_flow_align_sum": float(lbo_flow_align_sum),
                "lbo_entry_mode": lbo_entry_mode,
                "flow": float(flow),
                "break_ticks": float(break_ticks),
                "ft_progress_ticks": float(ft_progress_ticks),
                "impulse_ticks": float(impulse_ticks),
                "dmid_ticks": float(dmid_ticks),
            }
            i = entry_bar + 1
            continue
        entry_px = ask[entry_bar] if desired_side == "long" else bid[entry_bar]
        entry_spread_ticks = float(spread_ticks[entry_bar])
        # Minimal cost-aware viability gate: require some progress metric to exceed spread.
        entry_viability_mode = (entry_viability_mode or "base").strip().lower()
        if entry_viability_mode not in {"base", "strict"}:
            raise ValueError(f"Invalid ENTRY_VIABILITY_MODE: {entry_viability_mode}")
        min_viable_ticks = float(entry_spread_ticks) + (2.0 if entry_viability_mode == "strict" else 1.0)
        progress_vals = []
        if np.isfinite(ft_progress_ticks):
            progress_vals.append(abs(float(ft_progress_ticks)))
        if np.isfinite(break_ticks):
            progress_vals.append(abs(float(break_ticks)))
        if np.isfinite(impulse_ticks):
            progress_vals.append(abs(float(impulse_ticks)))
        progress_metric = max(progress_vals) if progress_vals else 0.0
        if entry_viability_flow_confirm:
            if not np.isfinite(flow_align_sum):
                i += 1
                continue
            if desired_side == "long" and flow_align_sum < 0.0:
                i += 1
                continue
            if desired_side == "short" and flow_align_sum > 0.0:
                i += 1
                continue
        if progress_metric < min_viable_ticks:
            i += 1
            continue
        if be_offset_ticks is None:
            be_offset_ticks_local = max(1, int(round(entry_spread_ticks)))
            if not be_offset_logged:
                print(
                    f"AUTO_BE_OFFSET {symbol_str} {day_str} spread_ticks_entry={entry_spread_ticks:.2f} "
                    f"be_offset_ticks={be_offset_ticks_local}",
                    flush=True,
                )
                be_offset_logged = True
        else:
            be_offset_ticks_local = int(be_offset_ticks)
            if not be_offset_logged:
                print(
                    f"BE_OFFSET {symbol_str} {day_str} spread_ticks_entry={entry_spread_ticks:.2f} "
                    f"be_offset_ticks={be_offset_ticks_local}",
                    flush=True,
                )
                be_offset_logged = True
        if strategy_mode == "lrams_breakout_v1" and entry_reason_label is None and entry_reason is None:
            entry_reason_label = "lbo_now"
        entry_reason_std = "other"
        entry_reason_raw = entry_reason_label if entry_reason_label is not None else entry_reason
        entry_root_reason = entry_root_reason if "entry_root_reason" in locals() else None
        entry_exec_reason = entry_exec_reason if "entry_exec_reason" in locals() else None
        if strategy_mode == "lrams_breakout_v1":
            entry_reason_std = "ft" if entry_reason == "ft" else "break"
        elif strategy_mode == "impulse_confirm_v1":
            entry_reason_std = "impulse"
        elif strategy_mode == "micro_momo_v1":
            entry_reason_std = "impulse"
        elif strategy_mode == "absorption_failure_v1":
            entry_reason_std = "afr"
        elif strategy_mode == "absorption_failure_v2":
            entry_reason_std = "afr2"
        elif strategy_mode == "absorption_failure_v3":
            entry_reason_std = "afr3"
        elif strategy_mode == "srf_entry_v1":
            entry_reason_std = "srf"
        if entry_exec_reason is None:
            entry_exec_reason = entry_reason_std
        if entry_root_reason is None:
            entry_root_reason = entry_reason_std
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
        if strategy_mode == "lrams_breakout_v1":
            tp_ticks_local = lbo_tp_ticks
            sl_ticks_local = lbo_sl_ticks
            hold_bars_local = lbo_max_hold_bars
            if lbo_breakeven_after_ticks is not None:
                breakeven_ticks_local = lbo_breakeven_after_ticks
        tp_level = entry_px + (tp_ticks_local * tick_size if desired_side == "long" else -tp_ticks_local * tick_size)
        sl_level = entry_px - (sl_ticks_local * tick_size if desired_side == "long" else -sl_ticks_local * tick_size)
        pos = {
            "entry_time": df_day["Time"].iloc[entry_bar],
            "entry_bar": entry_bar,
            "entry_px": float(entry_px),
            "exit_bar": None,
            "exit_reason": None,
            "time_exit_bar": int(min(entry_bar + hold_bars_local, n - 1)),
            "tp_level": float(tp_level),
            "sl_level": float(sl_level),
            "entry_spread_ticks": entry_spread_ticks,
            "sl_ticks": int(sl_ticks_local),
            "tp_ticks": int(tp_ticks_local),
            "breakeven_ticks": breakeven_ticks_local,
            "breakeven_set": False,
            "breakeven_triggered": False,
            "be_offset_ticks": int(be_offset_ticks_local),
            "last_best_bar": int(entry_bar),
            "runner_active": False,
            "entry_reason": entry_exec_reason,
            "entry_root_reason": entry_root_reason,
            "entry_reason_raw": entry_reason_raw,
            "entry_reason_label": entry_reason_label if entry_reason_label is not None else "",
            "lbo_entry_mode": lbo_entry_mode,
            "absorption_level": float(absorption_level) if np.isfinite(absorption_level) else float("nan"),
            "break_level": float(break_level) if np.isfinite(break_level) else float("nan"),
            "flow_align_sum_at_entry": float(flow_align_sum),
            "bars_from_absorption": int(entry_bar - absorption_bar) if absorption_bar >= 0 else -1,
            "bars_to_progress": -1,
            "absorption_bar": int(absorption_bar),
            "confirm_bars_waited": int(confirm_bars_waited),
            "confirm_price_ref": float(confirm_price_ref) if np.isfinite(confirm_price_ref) else float("nan"),
            "mae_ticks": 0.0,
            "mfe_ticks": 0.0,
            "exit_on_decay": False,
            "exit_on_be": False,
            "lbo_direction": lbo_direction,
            "lbo_weak_side": lbo_weak_side,
            "lbo_break_ticks": float(lbo_break_ticks_entry) if np.isfinite(lbo_break_ticks_entry) else float("nan"),
            "lbo_flow_align_sum": float(lbo_flow_align_sum),
            "srf_disp_ticks": float(srf_disp_ticks) if np.isfinite(srf_disp_ticks) else float("nan"),
            "srf_speed": float(srf_speed) if np.isfinite(srf_speed) else float("nan"),
            "srf_refill_ratio": float(srf_refill_ratio) if np.isfinite(srf_refill_ratio) else float("nan"),
            "srf_flow_sum": float(srf_flow_sum) if np.isfinite(srf_flow_sum) else float("nan"),
            "side": desired_side,
            "gate_event_age": int(gate_event_age),
            "gate_asym_val": float(gate_asym_val),
            "gate_thr_val": float(gate_thr_val),
        }
        if strategy_mode == "srf_entry_v1":
            if not srf_trigger:
                raise RuntimeError("SRF entry without trigger.")
            if entry_reason_std != "srf" or entry_root_reason != "srf":
                raise RuntimeError("SRF entry missing entry_reason labels.")
        if gate_debug and gated and gate_enabled:
            try:
                print(
                    f"GATE_ENTRY {symbol_str} {day_str} mode={gate_mode} "
                    f"desired_side={desired_side} weak_side={gate_weak_side} "
                    f"event_idx={gate_event_idx} event_age_bars={gate_event_age} "
                    f"asym={gate_asym_val:.6f} thr={gate_thr_val:.6f}",
                    flush=True,
                )
            except OSError:
                pass
        in_position = True
        entries_taken += 1
        signals_when_flat += 1
        if strategy_mode == "impulse_confirm_v1":
            impulse_entered += 1
        if strategy_mode == "absorption_failure_v1":
            afr_entered += 1
        if strategy_mode == "absorption_failure_v2":
            afr2_entered += 1
        if strategy_mode == "absorption_failure_v3":
            afr3_entered += 1
        if strategy_mode == "srf_entry_v1":
            srf_entered += 1
            if desired_side == "long":
                srf_exec_long += 1
            else:
                srf_exec_short += 1
            srf_stat = _srf_perfect_exec_stats(entry_bar, desired_side)
            srf_stat["srf_disp_ticks"] = float(srf_disp_ticks) if np.isfinite(srf_disp_ticks) else float("nan")
            srf_stat["srf_speed"] = float(srf_speed) if np.isfinite(srf_speed) else float("nan")
            srf_stat["srf_refill_ratio"] = float(srf_refill_ratio) if np.isfinite(srf_refill_ratio) else float("nan")
            srf_stat["srf_flow_sum"] = float(srf_flow_sum) if np.isfinite(srf_flow_sum) else float("nan")
            srf_stat["spread_ticks_entry"] = float(spread_ticks[entry_bar])
            srf_event_stats.append(srf_stat)
        if strategy_mode == "lrams_breakout_v1":
            if lbo_entry_mode == "confirm":
                lbo_confirm_entered += 1
            elif lbo_entry_mode == "pullback":
                lbo_pull_resumed_entered += 1
        if gate_debug and gated and gate_enabled and flat_candidate_printed < 5:
            print(
                f"GATE_DEBUG {symbol_str} {day_str} entry_bar={entry_bar} desired_side={desired_side} "
                f"weak_side={gate_weak_side} stream={event_stream} "
                f"gate_decision={'allow' if gate_allowed else gate_reason}",
                flush=True,
            )
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

    print(
        f"DAY_STATE {symbol_str} {day_str} signals_total={total_signals} "
        f"signals_when_flat={signals_when_flat} signals_in_position={signals_in_position} "
        f"in_position={in_position} entry_bar={int(pos.get('entry_bar', -1)) if pos else -1} "
        f"exit_bar={int(pos.get('exit_bar', -1)) if pos else -1}",
        flush=True,
    )
    skip_total = sum(skip_reasons.values())
    if skip_total != skipped:
        print(
            f"SKIP_MISMATCH {symbol_str} {day_str} skipped={skipped} sum={skip_total} reasons={skip_reasons}",
            flush=True,
        )
        raise RuntimeError(
            f"Skip reasons mismatch: sum={skip_total} skipped={skipped} day={day_str} mode={gate_mode}"
        )
    if validate_debug:
        pending_flag = 1 if pending_entry_active else 0
        print(
            f"ENTRY_CONFIRM_STATS {symbol_str} {day_str} checked={entry_confirm_checked} "
            f"passed={entry_confirm_passed} failed={entry_confirm_failed} pending={pending_flag} "
            f"candidates={entry_candidates_when_flat}",
            flush=True,
        )
        if entry_confirm_checked != entry_confirm_passed + entry_confirm_failed + pending_flag:
            raise RuntimeError(
                f"Entry confirm invariant failed: checked={entry_confirm_checked} "
                f"passed={entry_confirm_passed} failed={entry_confirm_failed} pending={pending_flag}"
            )
        if entry_confirm_checked > entry_candidates_when_flat:
            raise RuntimeError(
                f"Entry confirm checked exceeds candidates: checked={entry_confirm_checked} "
                f"candidates={entry_candidates_when_flat}"
            )
        trade_count = int(len(trades))
        mean_pnl = float(pnl_ticks_total / trade_count) if trade_count else 0.0
        print(
            f"FAIL_FAST_SUMMARY {symbol_str} {day_str} trades={trade_count} "
            f"fail_fast_exits={fail_fast_exit_taken} fail_fast_suppressed={fail_fast_exit_suppressed} "
            f"mean_pnl_per_trade={mean_pnl:.4f}",
            flush=True,
        )
    if strategy_mode == "srf_entry_v1":
        # In SRF mode, candidate counts reflect post-viability gateable candidates.
        if gated:
            entry_candidates_when_flat = int(blocked_signals + gate_diag.get("allowed_count", 0))
        else:
            entry_candidates_when_flat = int(entries_taken)
        if srf_debug:
            print(
                f"{symbol_str} {day_str} SRF raw long={srf_raw_long} short={srf_raw_short} | "
                f"exec long={srf_exec_long} short={srf_exec_short}",
                flush=True,
            )

    return (
        pd.DataFrame(trades),
        float(pnl_ticks_total),
        skipped,
        total_signals,
        entry_candidates_when_flat,
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
        lbo_weak_none,
        lbo_weak_ask,
        lbo_weak_bid,
        lbo_weak_none_blocked,
        lbo_rej_empty,
        lbo_rej_before_lookback,
        lbo_rej_thr_nan,
        lbo_rej_asym_below_thr,
        lbo_missing_both,
        lbo_only_buy,
        lbo_only_sell,
        lbo_both_fail_thr,
        lbo_pending_started,
        lbo_confirm_entered,
        lbo_confirm_expired,
        lbo_pull_triggered,
        lbo_pull_pulled_back,
        lbo_pull_resumed_entered,
        lbo_pull_expired,
        mmas_signals_checked,
        mmas_passed_filters,
        mmas_signaled,
        mmas_entered,
        srf_checked,
        srf_triggered,
        srf_entered,
        srf_event_stats,
        impulse_signals_checked,
        impulse_passed_threshold,
        impulse_passed_confirm,
        impulse_entered,
        entry_confirm_checked,
        entry_confirm_passed,
        entry_confirm_failed,
        passive_filled_count,
        passive_fallback_count,
        exit_sm_called,
        exit_sm_exit_taken,
        exit_sm_hold,
        legacy_exit_taken,
        fail_fast_exit_taken,
        fail_fast_exit_suppressed,
        skip_reasons,
        gate_avail,
        gate_diag,
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
    if trades.empty or "pnl_ticks" not in trades.columns:
        return {
            "trade_count": 0,
            "total_pnl_ticks": 0.0,
            "final_pnl_ticks": 0.0,
            "peak_equity_ticks": 0.0,
            "mean_pnl_ticks": 0.0,
            "median_pnl_ticks": 0.0,
            "win_rate": 0.0,
            "mean_win_ticks": 0.0,
            "mean_loss_ticks": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_ticks": 0.0,
            "p1": 0.0,
            "p5": 0.0,
            "p10": 0.0,
            "worst_trade_ticks": 0.0,
        }
    pnl = trades["pnl_ticks"].to_numpy()
    eq = _equity_stats(pnl)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    sum_wins = float(np.sum(wins)) if wins.size else 0.0
    sum_losses = float(np.sum(losses)) if losses.size else 0.0
    profit_factor = (sum_wins / abs(sum_losses)) if sum_losses != 0.0 else 0.0
    return {
        "trade_count": int(len(trades)),
        "total_pnl_ticks": float(np.sum(pnl)),
        "final_pnl_ticks": eq["final"],
        "peak_equity_ticks": eq["peak"],
        "mean_pnl_ticks": float(np.mean(pnl)),
        "median_pnl_ticks": float(np.median(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "mean_win_ticks": float(np.mean(wins)) if wins.size else 0.0,
        "mean_loss_ticks": float(np.mean(losses)) if losses.size else 0.0,
        "profit_factor": float(profit_factor),
        "max_drawdown_ticks": eq["max_dd"],
        "p1": float(np.quantile(pnl, 0.01)),
        "p5": float(np.quantile(pnl, 0.05)),
        "p10": float(np.quantile(pnl, 0.10)),
        "worst_trade_ticks": float(np.min(pnl)),
    }


def _standard_exit_reason(reason: str) -> str:
    r = str(reason).upper()
    mapping = {
        "FAIL_FAST": "FAIL_FAST",
        "TRAIL": "TRAIL",
        "TP": "TP",
        "SL": "SL",
        "BE_LIMIT": "BE_LIMIT",
        "BE": "BE",
        "SCRATCH": "SCRATCH",
        "DECAY": "DECAY",
        "TIME": "TIME",
        "MAX_HOLD": "TIME",
    }
    return mapping.get(r, "OTHER")


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


def _exit_breakdown_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    if "exit_reason_std" not in trades.columns:
        trades = trades.copy()
        trades["exit_reason_std"] = trades["exit_reason"].apply(_standard_exit_reason)
    def _agg(g: pd.DataFrame) -> pd.Series:
        pnl = pd.to_numeric(g["pnl_ticks"], errors="coerce").to_numpy()
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        sum_wins = float(np.sum(wins)) if wins.size else 0.0
        sum_losses = float(np.sum(losses)) if losses.size else 0.0
        pf = (sum_wins / abs(sum_losses)) if sum_losses != 0.0 else 0.0
        return pd.Series(
            {
                "count_trades": int(len(pnl)),
                "sum_pnl_ticks": float(np.sum(pnl)),
                "mean_pnl_ticks": float(np.mean(pnl)) if pnl.size else 0.0,
                "median_pnl_ticks": float(np.median(pnl)) if pnl.size else 0.0,
                "win_rate": float(np.mean(pnl > 0)) if pnl.size else 0.0,
                "profit_factor": float(pf),
                "median_mfe_ticks": float(np.nanmedian(g["mfe_ticks"])) if "mfe_ticks" in g.columns else 0.0,
                "median_mae_ticks": float(np.nanmedian(g["mae_ticks"])) if "mae_ticks" in g.columns else 0.0,
                "median_hold_bars_realized": float(np.nanmedian(g["hold_bars_realized"]))
                if "hold_bars_realized" in g.columns
                else 0.0,
                "p5_pnl_ticks": float(np.quantile(pnl, 0.05)) if pnl.size else 0.0,
                "p1_pnl_ticks": float(np.quantile(pnl, 0.01)) if pnl.size else 0.0,
            }
        )
    return trades.groupby(["date", "strategy", "exit_reason_std"], sort=False).apply(_agg).reset_index()


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
    raw_strategy_env = os.environ.get("STRATEGY_MODE", "")
    if raw_strategy_env.strip():
        strategy_mode = raw_strategy_env
    else:
        strategy_mode = _arg_or_env("strategy_mode", "STRATEGY_MODE", "")
    strategy_mode = strategy_mode.strip().lower()
    baseline_mode = os.environ.get("BASELINE_MODE", "flat").strip().lower()
    if not strategy_mode:
        strategy_mode = "micro_momo_v1"
    print(
        f"STRATEGY_MODE raw_env={raw_strategy_env!r} resolved={strategy_mode}",
        flush=True,
    )
    if raw_strategy_env.strip():
        env_norm = raw_strategy_env.strip().lower()
        if strategy_mode != env_norm:
            raise RuntimeError(
                f"STRATEGY_MODE mismatch: env={env_norm} resolved={strategy_mode}"
            )
    run_afr2_sweep = strategy_mode.startswith("absorption_failure") or os.environ.get("RUN_AFR2_SWEEP", "0").strip() == "1"
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
    afr_ft_bars = int(_arg_or_env("afr_ft_bars", "AFR_FT_BARS", "2"))
    afr_ft_min_ticks = int(_arg_or_env("afr_ft_min_ticks", "AFR_FT_MIN_TICKS", "1"))
    afr_ft_no_backtrack = _arg_or_env("afr_ft_no_backtrack", "AFR_FT_NO_BACKTRACK", "1").strip() == "1"
    afr_enter_on = _arg_or_env("afr_enter_on", "AFR_ENTER_ON", "ft").strip().lower()
    afr_scratch_bars = int(os.environ.get("AFR_SCRATCH_BARS", "3"))
    afr_scratch_min_progress_ticks = int(os.environ.get("AFR_SCRATCH_MIN_PROGRESS_TICKS", "1"))
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
    print("ENV EXIT_MODE =", os.getenv("EXIT_MODE"), flush=True)
    exit_mode = os.environ.get("EXIT_MODE", "").strip().lower()
    exit_debug = os.environ.get("EXIT_DEBUG", "0").strip() == "1"
    validate_debug = os.environ.get("VALIDATE_DEBUG", "0").strip() == "1"
    if exit_mode and exit_mode not in {"exit_sm_v1", "exit_sm_v2", "exit_exec_v2"}:
        raise ValueError(f"Invalid EXIT_MODE: {exit_mode}")
    entry_confirm_bars = int(os.environ.get("ENTRY_CONFIRM_BARS", "0"))
    entry_min_progress_ticks = int(os.environ.get("ENTRY_MIN_PROGRESS_TICKS", "0"))
    entry_confirm_style = os.environ.get("ENTRY_CONFIRM_STYLE", "off").strip().lower()
    print(
        f"ENTRY_CONFIRM: bars={entry_confirm_bars} min_prog={entry_min_progress_ticks} style={entry_confirm_style}",
        flush=True,
    )
    validate_bars = int(os.environ.get("VALIDATE_BARS", "5"))
    min_progress_ticks = int(os.environ.get("MIN_PROGRESS_TICKS", "1"))
    fail_fast_bars = int(os.environ.get("FAIL_FAST_BARS", "3"))
    fail_fast_max_adverse_ticks = int(os.environ.get("FAIL_FAST_MAX_ADVERSE_TICKS", "2"))
    fail_fast_enabled = os.environ.get("FAIL_FAST_ENABLED", "1").strip() == "1"
    entry_viability_mode = os.environ.get("ENTRY_VIABILITY_MODE", "base").strip().lower()
    entry_viability_flow_confirm = os.environ.get("ENTRY_VIABILITY_FLOW_CONFIRM", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    be_arm_ticks = int(os.environ.get("BE_ARM_TICKS", "1"))
    be_offset_env = os.environ.get("BE_OFFSET_TICKS", "").strip()
    be_offset_ticks = int(be_offset_env) if be_offset_env else None
    runner_trail_start_ticks = int(os.environ.get("RUNNER_TRAIL_START_TICKS", "4"))
    runner_trail_giveback_ticks = int(os.environ.get("RUNNER_TRAIL_GIVEBACK_TICKS", "2"))
    decay_bars = int(os.environ.get("DECAY_BARS", "10"))
    scratch_bars = int(os.environ.get("SCRATCH_BARS", "3"))
    scratch_min_progress_ticks = int(os.environ.get("SCRATCH_MIN_PROGRESS_TICKS", "1"))
    be_after_ticks = int(os.environ.get("BE_AFTER_TICKS", "2"))
    decay_min_flow = float(os.environ.get("DECAY_MIN_FLOW", "20"))
    be_grace_bars = int(os.environ.get("BE_GRACE_BARS", "5"))
    scratch_require_mfe_ticks = int(os.environ.get("SCRATCH_REQUIRE_MFE_TICKS", str(be_arm_ticks)))
    scratch_grace_bars = int(os.environ.get("SCRATCH_GRACE_BARS", "5"))
    passive_exit_enabled = os.environ.get("PASSIVE_EXIT_ENABLED", "1").strip() != "0"
    passive_exit_bars = int(os.environ.get("PASSIVE_EXIT_BARS", "2"))
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
    lbo_break_ticks = int(os.environ.get("LBO_BREAK_TICKS", "1"))
    lbo_confirm_bars = int(os.environ.get("LBO_CONFIRM_BARS", "2"))
    lbo_min_flow_abs = float(os.environ.get("LBO_MIN_FLOW_ABS", "20"))
    lbo_max_spread_ticks = int(os.environ.get("LBO_MAX_SPREAD_TICKS", "2"))
    lbo_ft_bars = int(os.environ.get("LBO_FT_BARS", "2"))
    lbo_ft_min_ticks = int(os.environ.get("LBO_FT_MIN_TICKS", "1"))
    lbo_rearm_enabled = os.environ.get("LBO_REARM_ENABLED", "0").strip() == "1"
    lbo_rearm_band_ticks = int(os.environ.get("LBO_REARM_BAND_TICKS", "1"))
    lbo_rearm_max_bars = int(os.environ.get("LBO_REARM_MAX_BARS", "5"))
    lbo_tp_ticks = int(os.environ.get("LBO_TP_TICKS", "4"))
    lbo_sl_ticks = int(os.environ.get("LBO_SL_TICKS", "3"))
    lbo_max_hold_bars = int(os.environ.get("LBO_MAX_HOLD_BARS", "15"))
    lbo_scratch_bars = int(os.environ.get("LBO_SCRATCH_BARS", "3"))
    lbo_scratch_min_progress_ticks = int(os.environ.get("LBO_SCRATCH_MIN_PROGRESS_TICKS", "1"))
    lbo_decay_bars = int(os.environ.get("LBO_DECAY_BARS", "3"))
    lbo_be_env = os.environ.get("LBO_BREAKEVEN_AFTER_TICKS", "").strip()
    lbo_breakeven_after_ticks = int(lbo_be_env) if lbo_be_env else None
    lbo_ignore_thr = os.environ.get("LBO_IGNORE_THR", "0").strip() == "1"
    lbo_ignore_thr_override = os.environ.get("LBO_IGNORE_THR_OVERRIDE", "").strip()
    if lbo_ignore_thr_override:
        lbo_ignore_thr = lbo_ignore_thr_override == "1"
    lbo_flip_direction = os.environ.get("LBO_FLIP_DIRECTION", "0").strip() == "1"
    lbo_flip_mapping_env = os.environ.get("LBO_FLIP_MAPPING", "").strip()
    if lbo_flip_mapping_env:
        lbo_flip_direction = lbo_flip_mapping_env == "1"
    lbo_confirm_mode = os.environ.get("LBO_CONFIRM_MODE", "none").strip().lower()
    if lbo_confirm_mode not in {"none", "confirm_ticks", "pullback"}:
        raise ValueError(f"Invalid LBO_CONFIRM_MODE: {lbo_confirm_mode}")
    lbo_confirm_ticks = int(os.environ.get("LBO_CONFIRM_TICKS", "1"))
    lbo_confirm_max_bars = int(os.environ.get("LBO_CONFIRM_MAX_BARS", "5"))
    lbo_confirm_use_mid = os.environ.get("LBO_CONFIRM_USE_MID", "1").strip() == "1"
    lbo_confirm_require_flow_align = os.environ.get("LBO_CONFIRM_REQUIRE_FLOW_ALIGN", "0").strip() == "1"
    lbo_confirm_flow_align_bars = int(os.environ.get("LBO_CONFIRM_FLOW_ALIGN_BARS", "5"))
    lbo_confirm_min_flow_abs_align = float(os.environ.get("LBO_CONFIRM_MIN_FLOW_ABS_ALIGN", "40"))
    lbo_pullback_ticks = int(os.environ.get("LBO_PULLBACK_TICKS", "1"))
    lbo_pullback_max_bars = int(os.environ.get("LBO_PULLBACK_MAX_BARS", "10"))
    lbo_resume_ticks = int(os.environ.get("LBO_RESUME_TICKS", "1"))
    lbo_resume_max_bars = int(os.environ.get("LBO_RESUME_MAX_BARS", "5"))
    gate_bypass_on_none_env = os.environ.get("GATE_BYPASS_ON_NONE", "").strip().lower()
    if gate_bypass_on_none_env:
        gate_bypass_on_none = gate_bypass_on_none_env in {"1", "true", "yes", "y"}
    else:
        gate_bypass_on_none = os.environ.get("ALLOW_GATE_ON_NONE", "0").strip().lower() in {"1", "true", "yes", "y"}
    allow_gate_on_none = gate_bypass_on_none
    gate_debug = os.environ.get("GATE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}
    if strategy_mode == "absorption_failure_v2":
        if afr_tp_ticks is None:
            afr_tp_ticks = 3
        if afr_sl_ticks is None:
            afr_sl_ticks = 2
    run_mode = os.environ.get("RUN_MODE", "baseline_vs_gated").strip().lower()
    max_spread_ticks_for_entry = int(os.environ.get("MAX_SPREAD_TICKS_FOR_ENTRY", "2"))
    mmas_k_bars = int(os.environ.get("MMAS_K_BARS", "5"))
    mmas_min_dmid_ticks = int(os.environ.get("MMAS_MIN_DMID_TICKS", "1"))
    mmas_min_flow_abs = float(os.environ.get("MMAS_MIN_FLOW_ABS", "20"))
    mmas_require_agree = os.environ.get("MMAS_REQUIRE_AGREE", "1").strip() == "1"
    debug_first_mmas = os.environ.get("DEBUG_FIRST_MMAS", "0").strip() == "1"
    srf_min_disp_ticks = int(os.environ.get("SRF_MIN_DISP_TICKS", "3"))
    srf_disp_window_bars = int(os.environ.get("SRF_DISP_WINDOW_BARS", "5"))
    srf_min_speed_ticks_per_bar = float(os.environ.get("SRF_MIN_SPEED_TICKS_PER_BAR", "0.6"))
    srf_refill_window_bars = int(os.environ.get("SRF_REFILL_WINDOW_BARS", "5"))
    srf_baseline_bars = int(os.environ.get("SRF_BASELINE_BARS", "20"))
    srf_refill_ratio_max = float(os.environ.get("SRF_REFILL_RATIO_MAX", "0.6"))
    srf_min_spread_ticks = int(os.environ.get("SRF_MIN_SPREAD_TICKS", "1"))
    srf_flow_confirm = os.environ.get("SRF_FLOW_CONFIRM", "0").strip() == "1"
    srf_flow_window_bars = int(os.environ.get("SRF_FLOW_WINDOW_BARS", "5"))
    srf_min_flow = float(os.environ.get("SRF_MIN_FLOW", "0"))
    srf_side_mode = os.environ.get("SRF_SIDE_MODE", "follow").strip().lower()
    if srf_side_mode not in {"follow", "fade"}:
        raise ValueError(f"Invalid SRF_SIDE_MODE: {srf_side_mode}")
    srf_debug = os.environ.get("SRF_DEBUG", "0").strip().lower() in {"1", "true", "yes", "y"}
    trade_session = os.environ.get("TRADE_SESSION", "all").strip().lower()
    min_spread_ticks = int(os.environ.get("MIN_SPREAD_TICKS", "0"))
    entry_cooldown_bars = int(os.environ.get("ENTRY_COOLDOWN_BARS", "10"))
    gate_lookback_bars = int(os.environ.get("GATE_LOOKBACK_BARS", "10"))
    weak_side_default = max(gate_lookback_bars, 100)
    weak_side_lookback_bars = int(os.environ.get("WEAK_SIDE_LOOKBACK_BARS", str(weak_side_default)))
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

    afr_min_flow_abs_sweep_env = os.environ.get("AFR_MIN_FLOW_ABS_SWEEP", "").strip()
    if run_afr2_sweep and afr_min_flow_abs_sweep_env:
        afr_min_flow_abs_sweep = [float(x.strip()) for x in afr_min_flow_abs_sweep_env.split(",") if x.strip()]
    else:
        afr_min_flow_abs_sweep = [afr_min_flow_abs]
    afr_ft_bars_sweep_env = os.environ.get("AFR_FT_BARS_SWEEP", "").strip()
    if run_afr2_sweep and afr_ft_bars_sweep_env:
        afr_ft_bars_sweep = [int(x.strip()) for x in afr_ft_bars_sweep_env.split(",") if x.strip()]
    else:
        afr_ft_bars_sweep = [afr_ft_bars]

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
            "srf_min_disp_ticks": srf_min_disp_ticks,
            "srf_disp_window_bars": srf_disp_window_bars,
            "srf_min_speed_ticks_per_bar": srf_min_speed_ticks_per_bar,
            "srf_refill_window_bars": srf_refill_window_bars,
            "srf_baseline_bars": srf_baseline_bars,
            "srf_refill_ratio_max": srf_refill_ratio_max,
            "srf_min_spread_ticks": srf_min_spread_ticks,
            "srf_flow_confirm": srf_flow_confirm,
            "srf_flow_window_bars": srf_flow_window_bars,
            "srf_min_flow": srf_min_flow,
            "srf_side_mode": srf_side_mode,
            "srf_debug": srf_debug,
            "trade_session": trade_session,
            "min_spread_ticks": min_spread_ticks,
            "entry_cooldown_bars": entry_cooldown_bars,
            "tp_ticks": tp_ticks,
            "sl_ticks": sl_ticks,
            "entry_viability_mode": entry_viability_mode,
            "entry_viability_flow_confirm": entry_viability_flow_confirm,
            "afr3_break_min_flow_abs": afr3_break_min_flow_abs,
            "afr3_break_max_spread_ticks": afr3_break_max_spread_ticks,
            "afr3_snapback_check": afr3_snapback_check,
            "afr_scratch_bars": afr_scratch_bars,
            "afr_scratch_min_progress_ticks": afr_scratch_min_progress_ticks,
            "gate_mode_list": gate_modes,
            "sweep_ws": sweep_ws,
            "afr_min_flow_abs_sweep": afr_min_flow_abs_sweep,
            "afr_ft_bars_sweep": afr_ft_bars_sweep,
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
            "srf_min_disp_ticks": srf_min_disp_ticks,
            "srf_disp_window_bars": srf_disp_window_bars,
            "srf_min_speed_ticks_per_bar": srf_min_speed_ticks_per_bar,
            "srf_refill_window_bars": srf_refill_window_bars,
            "srf_baseline_bars": srf_baseline_bars,
            "srf_refill_ratio_max": srf_refill_ratio_max,
            "srf_min_spread_ticks": srf_min_spread_ticks,
            "srf_flow_confirm": srf_flow_confirm,
            "srf_flow_window_bars": srf_flow_window_bars,
            "srf_min_flow": srf_min_flow,
            "srf_side_mode": srf_side_mode,
            "srf_debug": srf_debug,
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
            "entry_viability_mode": entry_viability_mode,
            "entry_viability_flow_confirm": entry_viability_flow_confirm,
            "lbo_flip_direction": lbo_flip_direction,
            "lbo_confirm_mode": lbo_confirm_mode,
            "lbo_confirm_ticks": lbo_confirm_ticks,
            "lbo_confirm_max_bars": lbo_confirm_max_bars,
            "lbo_confirm_use_mid": lbo_confirm_use_mid,
            "lbo_confirm_require_flow_align": lbo_confirm_require_flow_align,
            "lbo_confirm_flow_align_bars": lbo_confirm_flow_align_bars,
            "lbo_confirm_min_flow_abs_align": lbo_confirm_min_flow_abs_align,
            "lbo_pullback_ticks": lbo_pullback_ticks,
            "lbo_pullback_max_bars": lbo_pullback_max_bars,
            "lbo_resume_ticks": lbo_resume_ticks,
            "lbo_resume_max_bars": lbo_resume_max_bars,
        },
        flush=True,
    )
    print(f"AFR_ENTER_ON={afr_enter_on} strategy_mode={strategy_mode}", flush=True)
    if run_mode == "gated_only":
        print("RUN_MODE: gated_only (no baseline)", flush=True)
    else:
        print("RUN_MODE: baseline_vs_gated (default)", flush=True)
    # Testing:
    #  - Strict gate:
    #    set EXIT_MODE=exit_sm_v1
    #    set ALLOW_GATE_ON_NONE=0
    #    python scripts\run_lrams_gate_backtest.py
    #  - Debug gate:
    #    set ALLOW_GATE_ON_NONE=1
    #  - Optional:
    #    set GATE_DEBUG=1
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
                "afr_scratch_bars": afr_scratch_bars,
                "afr_scratch_min_progress_ticks": afr_scratch_min_progress_ticks,
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
    if strategy_mode == "lrams_breakout_v1":
        print(
            "LBO config:",
            {
                "lbo_break_ticks": lbo_break_ticks,
                "lbo_confirm_bars": lbo_confirm_bars,
                "lbo_min_flow_abs": lbo_min_flow_abs,
                "lbo_max_spread_ticks": lbo_max_spread_ticks,
                "lbo_ft_bars": lbo_ft_bars,
                "lbo_ft_min_ticks": lbo_ft_min_ticks,
                "lbo_rearm_enabled": lbo_rearm_enabled,
                "lbo_rearm_band_ticks": lbo_rearm_band_ticks,
                "lbo_rearm_max_bars": lbo_rearm_max_bars,
                "lbo_tp_ticks": lbo_tp_ticks,
                "lbo_sl_ticks": lbo_sl_ticks,
                "lbo_max_hold_bars": lbo_max_hold_bars,
                "lbo_scratch_bars": lbo_scratch_bars,
                "lbo_scratch_min_progress_ticks": lbo_scratch_min_progress_ticks,
                "lbo_decay_bars": lbo_decay_bars,
                "lbo_breakeven_after_ticks": lbo_breakeven_after_ticks,
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

    afr2_sweep_rows = []
    print(f"=== STRATEGY: {strategy_mode} ({run_mode}) ===", flush=True)
    if run_afr2_sweep:
        print(f"=== STRATEGY: {strategy_mode} AFR2_SWEEP ===", flush=True)
    for afr_min_flow_abs_cur in afr_min_flow_abs_sweep:
        for afr_ft_bars_cur in afr_ft_bars_sweep:
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
                            "exit_mode": os.getenv("EXIT_MODE"),
                            "afr_min_flow_abs": afr_min_flow_abs_cur,
                            "afr_ft_bars": afr_ft_bars_cur,
                            "disable_gate": disable_gate,
                            "trade_session": trade_session,
                            "min_spread_ticks": min_spread_ticks,
                            "entry_cooldown_bars": entry_cooldown_bars,
                            "lbo_ignore_thr": lbo_ignore_thr,
                            "lbo_flip_direction": lbo_flip_direction,
                        },
                        flush=True,
                    )

                    summaries = []
                    all_base = []
                    all_gated = []
                    srf_events_base_all: List[Dict[str, float]] = []
                    srf_events_gate_all: List[Dict[str, float]] = []
                    all_events = []
                    strategy_rows = []
                    gate_max_dds: List[float] = []
                    total_pnl_gate = 0.0
                    total_pnl_base = 0.0
                    total_blocked = 0

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
                                    entry_candidates_flat_base,
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
                            lbo_weak_none_base,
                            lbo_weak_ask_base,
                            lbo_weak_bid_base,
                            lbo_weak_none_blocked_base,
                            lbo_rej_empty_base,
                            lbo_rej_before_lookback_base,
                            lbo_rej_thr_nan_base,
                            lbo_rej_asym_below_thr_base,
                            lbo_missing_both_base,
                            lbo_only_buy_base,
                            lbo_only_sell_base,
                            lbo_both_fail_thr_base,
                            lbo_pending_started_base,
                            lbo_confirm_entered_base,
                            lbo_confirm_expired_base,
                            lbo_pull_triggered_base,
                            lbo_pull_pulled_back_base,
                            lbo_pull_resumed_entered_base,
                            lbo_pull_expired_base,
                            mmas_checked_base,
                            mmas_passed_base,
                            mmas_signaled_base,
                            mmas_entered_base,
                            srf_checked_base,
                            srf_triggered_base,
                            srf_entered_base,
                            srf_event_stats_base,
                            impulse_checked_base,
                                    impulse_passed_thr_base,
                                    impulse_passed_confirm_base,
                                    impulse_entered_base,
                                    entry_confirm_checked_base,
                                    entry_confirm_passed_base,
                                    entry_confirm_failed_base,
                                    passive_filled_base,
                                    passive_fallback_base,
                                    exit_sm_called_base,
                                    exit_sm_exit_taken_base,
                                    exit_sm_hold_base,
                                    legacy_exit_taken_base,
                                    fail_fast_exit_taken_base,
                                    fail_fast_exit_suppressed_base,
                                    skip_base,
                                    gate_avail_base,
                                    gate_diag_base,
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
                                    srf_min_disp_ticks=srf_min_disp_ticks,
                                    srf_disp_window_bars=srf_disp_window_bars,
                                    srf_min_speed_ticks_per_bar=srf_min_speed_ticks_per_bar,
                                    srf_refill_window_bars=srf_refill_window_bars,
                                    srf_baseline_bars=srf_baseline_bars,
                                    srf_refill_ratio_max=srf_refill_ratio_max,
                                    srf_min_spread_ticks=srf_min_spread_ticks,
                                    srf_flow_confirm=srf_flow_confirm,
                                    srf_flow_window_bars=srf_flow_window_bars,
                                    srf_min_flow=srf_min_flow,
                                    srf_side_mode=srf_side_mode,
                                    srf_debug=srf_debug,
                                    trade_session=trade_session,
                                    min_spread_ticks=min_spread_ticks,
                                    entry_cooldown_bars=entry_cooldown_bars,
                                    gate_lookback_bars=gate_lookback_bars,
                                    weak_side_lookback_bars=weak_side_lookback_bars,
                                    gated=False,
                                    disable_gate=False,
                                    gate_mode=gate_mode,
                                    afr_k_bars=afr_k_bars,
                            afr_min_flow_abs=afr_min_flow_abs_cur,
                                    afr_stall_ticks=afr_stall_ticks,
                                    afr_break_ticks=afr_break_ticks,
                                    afr_require_flow_sign=afr_require_flow_sign,
                                    afr_use_mid_for_stall=afr_use_mid_for_stall,
                                    afr_use_signed_volume=afr_use_signed_volume,
                            afr_ft_bars=afr_ft_bars_cur,
                                    afr_ft_min_ticks=afr_ft_min_ticks,
                                    afr_ft_no_backtrack=afr_ft_no_backtrack,
                                    afr_enter_on=afr_enter_on,
                                    afr_break_quality_min_flow_abs=afr_break_quality_min_flow_abs,
                                    afr_break_quality_max_spread_ticks=afr_break_quality_max_spread_ticks,
                            afr_snapback_bars=afr_snapback_bars,
                            afr_snapback_band_ticks=afr_snapback_band_ticks,
                            lbo_break_ticks=lbo_break_ticks,
                            lbo_confirm_bars=lbo_confirm_bars,
                            lbo_min_flow_abs=lbo_min_flow_abs,
                            lbo_max_spread_ticks=lbo_max_spread_ticks,
                            lbo_ft_bars=lbo_ft_bars,
                            lbo_ft_min_ticks=lbo_ft_min_ticks,
                            lbo_rearm_enabled=lbo_rearm_enabled,
                            lbo_rearm_band_ticks=lbo_rearm_band_ticks,
                            lbo_rearm_max_bars=lbo_rearm_max_bars,
                            lbo_tp_ticks=lbo_tp_ticks,
                            lbo_sl_ticks=lbo_sl_ticks,
                            lbo_max_hold_bars=lbo_max_hold_bars,
                            lbo_scratch_bars=lbo_scratch_bars,
                            lbo_scratch_min_progress_ticks=lbo_scratch_min_progress_ticks,
                            lbo_decay_bars=lbo_decay_bars,
                            lbo_breakeven_after_ticks=lbo_breakeven_after_ticks,
                            lbo_ignore_thr=lbo_ignore_thr,
                            lbo_flip_direction=lbo_flip_direction,
                            lbo_confirm_mode=lbo_confirm_mode,
                            lbo_confirm_ticks=lbo_confirm_ticks,
                            lbo_confirm_max_bars=lbo_confirm_max_bars,
                            lbo_confirm_use_mid=lbo_confirm_use_mid,
                            lbo_confirm_require_flow_align=lbo_confirm_require_flow_align,
                            lbo_confirm_flow_align_bars=lbo_confirm_flow_align_bars,
                            lbo_confirm_min_flow_abs_align=lbo_confirm_min_flow_abs_align,
                            lbo_pullback_max_bars=lbo_pullback_max_bars,
                            lbo_pullback_ticks=lbo_pullback_ticks,
                            lbo_resume_ticks=lbo_resume_ticks,
                            lbo_resume_max_bars=lbo_resume_max_bars,
                            allow_gate_on_none=allow_gate_on_none,
                            gate_debug=gate_debug,
                            debug_first_afr=debug_first_afr,
                            debug_first_afr2=debug_first_afr2,
                                    afr_scratch_bars=afr_scratch_bars,
                                    afr_scratch_min_progress_ticks=afr_scratch_min_progress_ticks,
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
                                    validate_debug=validate_debug,
                                    entry_confirm_bars=entry_confirm_bars,
                                    entry_min_progress_ticks=entry_min_progress_ticks,
                                    entry_confirm_style=entry_confirm_style,
                                    exit_debug=exit_debug,
                                    exit_mode=exit_mode,
                                    validate_bars=validate_bars,
                                    min_progress_ticks=min_progress_ticks,
                                    be_arm_ticks=be_arm_ticks,
                                    be_offset_ticks=be_offset_ticks,
                                    decay_bars=decay_bars,
                                    scratch_bars=scratch_bars,
                                    scratch_min_progress_ticks=scratch_min_progress_ticks,
                                    be_after_ticks=be_after_ticks,
                                    decay_min_flow=decay_min_flow,
                                    be_grace_bars=be_grace_bars,
                                    scratch_require_mfe_ticks=scratch_require_mfe_ticks,
                                    scratch_grace_bars=scratch_grace_bars,
                                    fail_fast_bars=fail_fast_bars,
                                    fail_fast_max_adverse_ticks=fail_fast_max_adverse_ticks,
                                    fail_fast_enabled=fail_fast_enabled,
                                    entry_viability_mode=entry_viability_mode,
                                    entry_viability_flow_confirm=entry_viability_flow_confirm,
                                    runner_trail_start_ticks=runner_trail_start_ticks,
                                    runner_trail_giveback_ticks=runner_trail_giveback_ticks,
                                    passive_exit_enabled=passive_exit_enabled,
                                    passive_exit_bars=passive_exit_bars,
                                )
                            else:
                                trades_base = pd.DataFrame({"pnl_ticks": pd.Series(dtype=float)})
                                pnl_total_base = 0.0
                                skipped_base = 0
                                total_signals_base = 0
                                entry_candidates_flat_base = 0
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
                                lbo_weak_none_base = 0
                                lbo_weak_ask_base = 0
                                lbo_weak_bid_base = 0
                                lbo_weak_none_blocked_base = 0
                                lbo_rej_empty_base = 0
                                lbo_rej_before_lookback_base = 0
                                lbo_rej_thr_nan_base = 0
                                lbo_rej_asym_below_thr_base = 0
                                lbo_missing_both_base = 0
                                lbo_only_buy_base = 0
                                lbo_only_sell_base = 0
                                lbo_both_fail_thr_base = 0
                                lbo_pending_started_base = 0
                                lbo_confirm_entered_base = 0
                                lbo_confirm_expired_base = 0
                                lbo_pull_triggered_base = 0
                                lbo_pull_pulled_back_base = 0
                                lbo_pull_resumed_entered_base = 0
                                lbo_pull_expired_base = 0
                                mmas_checked_base = 0
                                mmas_passed_base = 0
                                mmas_signaled_base = 0
                                mmas_entered_base = 0
                                impulse_checked_base = 0
                                impulse_passed_thr_base = 0
                                impulse_passed_confirm_base = 0
                                impulse_entered_base = 0
                                entry_confirm_checked_base = 0
                                entry_confirm_passed_base = 0
                                entry_confirm_failed_base = 0
                                passive_filled_base = 0
                                passive_fallback_base = 0
                                exit_sm_called_base = 0
                                exit_sm_exit_taken_base = 0
                                exit_sm_hold_base = 0
                                legacy_exit_taken_base = 0
                                fail_fast_exit_taken_base = 0
                                fail_fast_exit_suppressed_base = 0
                                skip_base = {}
                            (
                                trades_gate,
                                pnl_total_gate,
                                skipped_gate,
                                total_signals_gate,
                                entry_candidates_flat_gate,
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
                        lbo_weak_none_gate,
                        lbo_weak_ask_gate,
                        lbo_weak_bid_gate,
                        lbo_weak_none_blocked_gate,
                        lbo_rej_empty_gate,
                        lbo_rej_before_lookback_gate,
                        lbo_rej_thr_nan_gate,
                        lbo_rej_asym_below_thr_gate,
                        lbo_missing_both_gate,
                        lbo_only_buy_gate,
                        lbo_only_sell_gate,
                        lbo_both_fail_thr_gate,
                        lbo_pending_started_gate,
                        lbo_confirm_entered_gate,
                        lbo_confirm_expired_gate,
                        lbo_pull_triggered_gate,
                        lbo_pull_pulled_back_gate,
                        lbo_pull_resumed_entered_gate,
                        lbo_pull_expired_gate,
                        mmas_checked_gate,
                        mmas_passed_gate,
                        mmas_signaled_gate,
                        mmas_entered_gate,
                        srf_checked_gate,
                        srf_triggered_gate,
                        srf_entered_gate,
                        srf_event_stats_gate,
                        impulse_checked_gate,
                                impulse_passed_thr_gate,
                                impulse_passed_confirm_gate,
                                impulse_entered_gate,
                                entry_confirm_checked_gate,
                                entry_confirm_passed_gate,
                                entry_confirm_failed_gate,
                                passive_filled_gate,
                                passive_fallback_gate,
                                exit_sm_called_gate,
                                exit_sm_exit_taken_gate,
                                exit_sm_hold_gate,
                                legacy_exit_taken_gate,
                                fail_fast_exit_taken_gate,
                                fail_fast_exit_suppressed_gate,
                                skip_gate,
                                gate_avail_gate,
                                gate_diag,
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
                                    srf_min_disp_ticks=srf_min_disp_ticks,
                                    srf_disp_window_bars=srf_disp_window_bars,
                                    srf_min_speed_ticks_per_bar=srf_min_speed_ticks_per_bar,
                                    srf_refill_window_bars=srf_refill_window_bars,
                                    srf_baseline_bars=srf_baseline_bars,
                                    srf_refill_ratio_max=srf_refill_ratio_max,
                                    srf_min_spread_ticks=srf_min_spread_ticks,
                                    srf_flow_confirm=srf_flow_confirm,
                                    srf_flow_window_bars=srf_flow_window_bars,
                                    srf_min_flow=srf_min_flow,
                                    srf_side_mode=srf_side_mode,
                                    srf_debug=srf_debug,
                                    trade_session=trade_session,
                                min_spread_ticks=min_spread_ticks,
                                entry_cooldown_bars=entry_cooldown_bars,
                                gate_lookback_bars=gate_lookback_bars,
                                weak_side_lookback_bars=weak_side_lookback_bars,
                                gated=True,
                                disable_gate=disable_gate,
                                gate_mode=gate_mode,
                                afr_k_bars=afr_k_bars,
                            afr_min_flow_abs=afr_min_flow_abs_cur,
                                afr_stall_ticks=afr_stall_ticks,
                                afr_break_ticks=afr_break_ticks,
                                afr_require_flow_sign=afr_require_flow_sign,
                                afr_use_mid_for_stall=afr_use_mid_for_stall,
                                afr_use_signed_volume=afr_use_signed_volume,
                            afr_ft_bars=afr_ft_bars_cur,
                                afr_ft_min_ticks=afr_ft_min_ticks,
                                afr_ft_no_backtrack=afr_ft_no_backtrack,
                                afr_enter_on=afr_enter_on,
                                afr_break_quality_min_flow_abs=afr_break_quality_min_flow_abs,
                                afr_break_quality_max_spread_ticks=afr_break_quality_max_spread_ticks,
                        afr_snapback_bars=afr_snapback_bars,
                        afr_snapback_band_ticks=afr_snapback_band_ticks,
                        lbo_break_ticks=lbo_break_ticks,
                        lbo_confirm_bars=lbo_confirm_bars,
                        lbo_min_flow_abs=lbo_min_flow_abs,
                        lbo_max_spread_ticks=lbo_max_spread_ticks,
                        lbo_ft_bars=lbo_ft_bars,
                        lbo_ft_min_ticks=lbo_ft_min_ticks,
                        lbo_rearm_enabled=lbo_rearm_enabled,
                        lbo_rearm_band_ticks=lbo_rearm_band_ticks,
                        lbo_rearm_max_bars=lbo_rearm_max_bars,
                        lbo_tp_ticks=lbo_tp_ticks,
                        lbo_sl_ticks=lbo_sl_ticks,
                        lbo_max_hold_bars=lbo_max_hold_bars,
                        lbo_scratch_bars=lbo_scratch_bars,
                        lbo_scratch_min_progress_ticks=lbo_scratch_min_progress_ticks,
                        lbo_decay_bars=lbo_decay_bars,
                        lbo_breakeven_after_ticks=lbo_breakeven_after_ticks,
                        lbo_ignore_thr=lbo_ignore_thr,
                        lbo_flip_direction=lbo_flip_direction,
                        lbo_confirm_mode=lbo_confirm_mode,
                        lbo_confirm_ticks=lbo_confirm_ticks,
                        lbo_confirm_max_bars=lbo_confirm_max_bars,
                        lbo_confirm_use_mid=lbo_confirm_use_mid,
                        lbo_confirm_require_flow_align=lbo_confirm_require_flow_align,
                        lbo_confirm_flow_align_bars=lbo_confirm_flow_align_bars,
                        lbo_confirm_min_flow_abs_align=lbo_confirm_min_flow_abs_align,
                        lbo_pullback_max_bars=lbo_pullback_max_bars,
                        lbo_pullback_ticks=lbo_pullback_ticks,
                        lbo_resume_ticks=lbo_resume_ticks,
                        lbo_resume_max_bars=lbo_resume_max_bars,
                        allow_gate_on_none=allow_gate_on_none,
                        gate_debug=gate_debug,
                        debug_first_afr=debug_first_afr,
                        debug_first_afr2=debug_first_afr2,
                                afr_scratch_bars=afr_scratch_bars,
                                afr_scratch_min_progress_ticks=afr_scratch_min_progress_ticks,
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
                                validate_debug=validate_debug,
                                entry_confirm_bars=entry_confirm_bars,
                                entry_min_progress_ticks=entry_min_progress_ticks,
                                entry_confirm_style=entry_confirm_style,
                                exit_debug=exit_debug,
                                exit_mode=exit_mode,
                                validate_bars=validate_bars,
                                min_progress_ticks=min_progress_ticks,
                                be_arm_ticks=be_arm_ticks,
                                be_offset_ticks=be_offset_ticks,
                                decay_bars=decay_bars,
                                scratch_bars=scratch_bars,
                                scratch_min_progress_ticks=scratch_min_progress_ticks,
                                be_after_ticks=be_after_ticks,
                                decay_min_flow=decay_min_flow,
                                be_grace_bars=be_grace_bars,
                                scratch_require_mfe_ticks=scratch_require_mfe_ticks,
                                scratch_grace_bars=scratch_grace_bars,
                                fail_fast_bars=fail_fast_bars,
                                fail_fast_max_adverse_ticks=fail_fast_max_adverse_ticks,
                                fail_fast_enabled=fail_fast_enabled,
                                entry_viability_mode=entry_viability_mode,
                                entry_viability_flow_confirm=entry_viability_flow_confirm,
                                runner_trail_start_ticks=runner_trail_start_ticks,
                                runner_trail_giveback_ticks=runner_trail_giveback_ticks,
                                passive_exit_enabled=passive_exit_enabled,
                                passive_exit_bars=passive_exit_bars,
                            )
                            if disable_gate:
                                blocked_gate = 0
                                eligible_gate = entry_candidates_flat_gate

                            if not sample_printed and day == selected_days[0]:
                                gate_pnl = trades_gate.get("pnl_ticks", pd.Series(dtype=float)).to_numpy()
                                gate_eq = _equity_stats(gate_pnl)
                                gate_stats_sample = _metrics(trades_gate)
                                if run_baseline:
                                    base_pnl = trades_base.get("pnl_ticks", pd.Series(dtype=float)).to_numpy()
                                    base_eq = _equity_stats(base_pnl)
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
                            total_pnl_gate += float(pnl_total_gate)
                            total_pnl_base += float(pnl_total_base)
                            total_blocked += int(blocked_gate)
                            trades_base["date"] = day
                            trades_base["Symbol"] = instrument
                            trades_base["strategy"] = "baseline"
                            trades_base["W"] = gate_lookback_bars
                            trades_base["gate_mode"] = gate_mode
                            trades_base["strategy_mode"] = strategy_mode
                            trades_gate["date"] = day
                            trades_gate["Symbol"] = instrument
                            trades_gate["strategy"] = "gated"
                            trades_gate["W"] = gate_lookback_bars
                            trades_gate["gate_mode"] = gate_mode
                            trades_gate["strategy_mode"] = strategy_mode
                            all_base.append(trades_base)
                            all_gated.append(trades_gate)
                            for srf_stat in srf_event_stats_base:
                                srf_stat["date"] = day
                                srf_stat["Symbol"] = instrument
                                srf_stat["strategy"] = "baseline"
                                srf_events_base_all.append(srf_stat)
                            for srf_stat in srf_event_stats_gate:
                                srf_stat["date"] = day
                                srf_stat["Symbol"] = instrument
                                srf_stat["strategy"] = "gated"
                                srf_events_gate_all.append(srf_stat)
                            if validate_debug and run_baseline:
                                base_trade_count = int(len(trades_base))
                                print(
                                    f"VALIDATE_DEBUG {instrument} {day} "
                                    f"base_entry_candidates_when_flat={entry_candidates_flat_base} "
                                    f"signals_when_flat={signals_flat_base} entries_taken={entries_taken_base} "
                                    f"baseline_trade_count={base_trade_count}",
                                    flush=True,
                                )

                            base_stats = _metrics(trades_base)
                            gate_stats = _metrics(trades_gate)
                            if (not fail_fast_enabled) or (fail_fast_max_adverse_ticks >= 999):
                                base_dd = float(base_stats["max_drawdown_ticks"])
                                gate_dd = float(gate_stats["max_drawdown_ticks"])
                                if base_dd > 0 and gate_dd > (5.0 * base_dd):
                                    print(
                                        f"Warning: FAIL_FAST disabled and gated max_dd {gate_dd:.2f} exceeds "
                                        f"5x baseline max_dd {base_dd:.2f} on {instrument} {day}.",
                                        flush=True,
                                    )
                            gate_max_dds.append(float(gate_stats["max_drawdown_ticks"]))
                            base_stats.update(
                                {
                                    "strategy": "baseline",
                                    "date": day,
                                    "Symbol": instrument,
                            "W": gate_lookback_bars,
                            "gate_mode": gate_mode,
                            "afr_min_flow_abs": float(afr_min_flow_abs_cur),
                            "afr_ft_bars": int(afr_ft_bars_cur),
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
                                    "coverage": float(eligible_base / entry_candidates_flat_base) if entry_candidates_flat_base else 0.0,
                                    "block_rate": 0.0,
                                    "mmas_signals_checked": int(mmas_checked_base),
                                    "mmas_passed_filters": int(mmas_passed_base),
                                    "mmas_signaled": int(mmas_signaled_base),
                                    "mmas_entered": int(mmas_entered_base),
                                }
                            )
                            coverage_gate = 0.0
                            if entry_candidates_flat_gate:
                                coverage_gate = 1.0 if disable_gate else float(eligible_gate / entry_candidates_flat_gate)
                            block_rate_gate = (
                                0.0 if disable_gate else float(blocked_gate / entry_candidates_flat_gate) if entry_candidates_flat_gate else 0.0
                            )
                            skip_rate_gate = (
                                0.0 if disable_gate else float(blocked_gate / entry_candidates_flat_gate) if entry_candidates_flat_gate else 0.0
                            )
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
                            print(
                                f"BASE trades={base_stats['trade_count']} mean={base_stats['mean_pnl_ticks']:.4f} "
                                f"med={base_stats['median_pnl_ticks']:.4f} win%={base_stats['win_rate']:.2%} "
                                f"p5={base_stats['p5']:.2f}",
                                flush=True,
                            )
                            print(
                                f"GATE trades={gate_stats['trade_count']} mean={gate_stats['mean_pnl_ticks']:.4f} "
                                f"med={gate_stats['median_pnl_ticks']:.4f} win%={gate_stats['win_rate']:.2%} "
                                f"p5={gate_stats['p5']:.2f}",
                                flush=True,
                            )
                            drop_reasons = []
                            if run_baseline:
                                if entry_candidates_flat_base == 0:
                                    drop_reasons.append("zero_entry_candidates_when_flat")
                                if entries_taken_base == 0:
                                    drop_reasons.append("zero_baseline_trades")
                            if entries_taken_gate == 0:
                                drop_reasons.append("zero_gated_trades")
                            if drop_reasons:
                                print(
                                    f"DROP {instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                                    f"reasons={drop_reasons} entry_candidates_when_flat={entry_candidates_flat_base} "
                                    f"baseline_trades={entries_taken_base} gated_trades={entries_taken_gate} "
                                    f"entries_blocked_by_gate={blocked_gate} coverage={gate_stats['coverage']:.2%}",
                                    flush=True,
                                )
                                if gate_mode == "side_matched":
                                    weak_side_checks = max(1, int(gate_avail_gate.get("weak_side_checks", 0)))
                                    any_buy_rate = gate_avail_gate.get("any_buy", 0) / weak_side_checks
                                    any_sell_rate = gate_avail_gate.get("any_sell", 0) / weak_side_checks
                                    qual_buy_rate = gate_avail_gate.get("qual_buy", 0) / weak_side_checks
                                    qual_sell_rate = gate_avail_gate.get("qual_sell", 0) / weak_side_checks
                                    print(
                                        f"GATE_DIAG {instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                                        f"weak_side_bid={gate_diag.get('weak_side_bid', 0)} "
                                        f"weak_side_ask={gate_diag.get('weak_side_ask', 0)} "
                                        f"weak_side_none={gate_diag.get('weak_side_none', 0)} "
                                        f"blocked_none={gate_diag.get('blocked_none', 0)} "
                                        f"blocked_mismatch={gate_diag.get('blocked_mismatch', 0)} "
                                        f"allowed_count={gate_diag.get('allowed_count', 0)} "
                                        f"allow_on_none={gate_diag.get('allow_on_none', 0)} "
                                        f"bypass_on_none={gate_diag.get('bypass_on_none', 0)} "
                                        f"any_buy_rate={any_buy_rate:.3f} any_sell_rate={any_sell_rate:.3f} "
                                        f"qual_buy_rate={qual_buy_rate:.3f} qual_sell_rate={qual_sell_rate:.3f}",
                                        flush=True,
                                    )
                                skipped_days.append(
                                    f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode}: {','.join(drop_reasons)} "
                                    f"entry_candidates_when_flat={entry_candidates_flat_base} baseline_trades={entries_taken_base} "
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
                                "baseline_entry_candidates_when_flat": int(entry_candidates_flat_base),
                                "gated_entry_candidates_when_flat": int(entry_candidates_flat_gate),
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
                                "baseline_median_trade_ticks": float(base_stats["median_pnl_ticks"]),
                                "gated_median_trade_ticks": float(gate_stats["median_pnl_ticks"]),
                                "baseline_win_rate": float(base_stats["win_rate"]),
                                "gated_win_rate": float(gate_stats["win_rate"]),
                                "baseline_mean_win_ticks": float(base_stats["mean_win_ticks"]),
                                "gated_mean_win_ticks": float(gate_stats["mean_win_ticks"]),
                                "baseline_mean_loss_ticks": float(base_stats["mean_loss_ticks"]),
                                "gated_mean_loss_ticks": float(gate_stats["mean_loss_ticks"]),
                                "baseline_profit_factor": float(base_stats["profit_factor"]),
                                "gated_profit_factor": float(gate_stats["profit_factor"]),
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
                            if gate_mode == "side_matched":
                                print(
                                    f"{instrument} {day} gate_diag weak_side_bid={gate_diag.get('weak_side_bid', 0)} "
                                    f"weak_side_ask={gate_diag.get('weak_side_ask', 0)} "
                                    f"weak_side_none={gate_diag.get('weak_side_none', 0)} "
                                    f"blocked_none={gate_diag.get('blocked_none', 0)} "
                                    f"blocked_mismatch={gate_diag.get('blocked_mismatch', 0)} "
                                    f"allowed_count={gate_diag.get('allowed_count', 0)} "
                                    f"allow_on_none={gate_diag.get('allow_on_none', 0)}",
                                    flush=True,
                                )
                            if not trades_gate.empty and "mfe_ticks" in trades_gate.columns:
                                gate_mfe = trades_gate["mfe_ticks"].to_numpy()
                                gate_pnl = trades_gate["pnl_ticks"].to_numpy() if "pnl_ticks" in trades_gate.columns else np.array([])
                                touched = gate_mfe >= 1.0
                                touch_plus1_count = int(np.sum(touched))
                                touch_plus1_nonpos = int(np.sum(touched & (gate_pnl <= 0.0))) if gate_pnl.size else 0
                                touch_plus1_pos = int(np.sum(touched & (gate_pnl > 0.0))) if gate_pnl.size else 0
                                if "bars_to_plus1" in trades_gate.columns:
                                    bars_to_plus1 = trades_gate.loc[touched, "bars_to_plus1"]
                                    bars_to_plus1 = bars_to_plus1[bars_to_plus1 >= 0].to_numpy()
                                else:
                                    bars_to_plus1 = np.array([])
                                avg_bars_to_plus1 = float(np.mean(bars_to_plus1)) if bars_to_plus1.size else 0.0
                            else:
                                touch_plus1_count = 0
                                touch_plus1_nonpos = 0
                                touch_plus1_pos = 0
                                avg_bars_to_plus1 = 0.0
                            print(
                                f"EXIT_DIAG {instrument} {day} gated_trades={entries_taken_gate} "
                                f"touch+1={touch_plus1_count} touch+1_nonpos={touch_plus1_nonpos} "
                                f"touch+1_pos={touch_plus1_pos} avg_bars_to+1={avg_bars_to_plus1:.2f}",
                                flush=True,
                            )
                            if not trades_gate.empty:
                                debug_rows = trades_gate.head(5)
                                for _, row in debug_rows.iterrows():
                                    mfe_ticks = float(row.get("mfe_ticks", 0.0))
                                    entry_px = float(row.get("entry_px", float("nan")))
                                    best_fav_px = (
                                        entry_px + mfe_ticks * tick_size
                                        if row.get("side") == "long"
                                        else entry_px - mfe_ticks * tick_size
                                    )
                                    bars_held = int(row.get("exit_bar", 0)) - int(row.get("entry_bar", 0))
                                    print(
                                        f"TRADE_DEBUG {instrument} {day} side={row.get('side')} "
                                        f"entry_px={entry_px:.2f} best_fav_px={best_fav_px:.2f} "
                                        f"mfe_ticks={mfe_ticks:.2f} exit_reason={row.get('exit_reason')} "
                                        f"bars_held={bars_held}",
                                        flush=True,
                                    )
                                for _, row in trades_gate.iterrows():
                                    pnl = float(row.get("pnl_ticks", 0.0))
                                    if pnl in (0.0, -1.0):
                                        print(
                                            f"EXIT_SANITY {instrument} {day} side={row.get('side')} "
                                            f"pnl_ticks={pnl:.2f} exit_reason={row.get('exit_reason')} "
                                            f"exit_mode={row.get('exit_mode', '')}",
                                            flush=True,
                                        )
                            base_cooldown_reduction = cooldown_supp_base / max(
                                1, (entry_candidates_flat_base + cooldown_supp_base)
                            )
                            gate_cooldown_reduction = cooldown_supp_gate / max(
                                1, (entry_candidates_flat_gate + cooldown_supp_gate)
                            )
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
                                    f"entered={afr2_entered_gate} "
                                    f"break_entries_base={count_break_entries_base} ft_entries_base={count_ft_entries_base} "
                                    f"rearms_base={count_rearms_base} break_entries_gate={count_break_entries_gate} "
                                    f"ft_entries_gate={count_ft_entries_gate} rearms_gate={count_rearms_gate} "
                                    f"scratch_base={pct_exit_on_scratch_base:.2%} scratch_gate={pct_exit_on_scratch_gate:.2%}",
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
                            if strategy_mode == "lrams_breakout_v1":
                                base_side = trades_base["side"] if "side" in trades_base.columns else pd.Series(dtype=object)
                                gate_side = trades_gate["side"] if "side" in trades_gate.columns else pd.Series(dtype=object)
                                base_long_trades = int((base_side == "long").sum())
                                base_short_trades = int((base_side == "short").sum())
                                gate_long_trades = int((gate_side == "long").sum())
                                gate_short_trades = int((gate_side == "short").sum())
                                print(
                                    f"{instrument} {day} LBO weak_side base none={lbo_weak_none_base} "
                                    f"ask={lbo_weak_ask_base} bid={lbo_weak_bid_base} "
                                    f"none_blocked={lbo_weak_none_blocked_base} "
                                    f"rej_empty={lbo_rej_empty_base} rej_before_lookback={lbo_rej_before_lookback_base} "
                                    f"rej_thr_nan={lbo_rej_thr_nan_base} rej_asym_below_thr={lbo_rej_asym_below_thr_base} "
                                    f"missing_both={lbo_missing_both_base} only_buy={lbo_only_buy_base} "
                                    f"only_sell={lbo_only_sell_base} both_fail_thr={lbo_both_fail_thr_base} | "
                                    f"gate none={lbo_weak_none_gate} ask={lbo_weak_ask_gate} bid={lbo_weak_bid_gate} "
                                    f"none_blocked={lbo_weak_none_blocked_gate} "
                                    f"rej_empty={lbo_rej_empty_gate} rej_before_lookback={lbo_rej_before_lookback_gate} "
                                    f"rej_thr_nan={lbo_rej_thr_nan_gate} rej_asym_below_thr={lbo_rej_asym_below_thr_gate} "
                                    f"missing_both={lbo_missing_both_gate} only_buy={lbo_only_buy_gate} "
                                    f"only_sell={lbo_only_sell_gate} both_fail_thr={lbo_both_fail_thr_gate} "
                                    f"trades base long={base_long_trades} short={base_short_trades} | "
                                    f"gate long={gate_long_trades} short={gate_short_trades}",
                                    flush=True,
                                )
                                print(
                                    f"{instrument} {day} LBO confirm mode={lbo_confirm_mode} "
                                    f"pending_started base={lbo_pending_started_base} confirm_entered={lbo_confirm_entered_base} "
                                    f"confirm_expired={lbo_confirm_expired_base} pull_triggered={lbo_pull_triggered_base} "
                                    f"pull_pulled_back={lbo_pull_pulled_back_base} "
                                    f"pull_resumed_entered={lbo_pull_resumed_entered_base} pull_expired={lbo_pull_expired_base} | "
                                    f"gate pending_started={lbo_pending_started_gate} confirm_entered={lbo_confirm_entered_gate} "
                                    f"confirm_expired={lbo_confirm_expired_gate} pull_triggered={lbo_pull_triggered_gate} "
                                    f"pull_pulled_back={lbo_pull_pulled_back_gate} "
                                    f"pull_resumed_entered={lbo_pull_resumed_entered_gate} pull_expired={lbo_pull_expired_gate}",
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
                            count_break_entries_base = int((base_entry_reason == "break").sum())
                            count_ft_entries_base = int((base_entry_reason == "ft").sum())
                            count_rearms_base = int((base_entry_reason == "rearm").sum())
                            count_break_entries_gate = int((gate_entry_reason == "break").sum())
                            count_ft_entries_gate = int((gate_entry_reason == "ft").sum())
                            count_rearms_gate = int((gate_entry_reason == "rearm").sum())
                            if strategy_mode == "lrams_breakout_v1":
                                base_entry_reason_raw = trades_base.get("entry_reason_raw", pd.Series(dtype=object)).fillna("")
                                gate_entry_reason_raw = trades_gate.get("entry_reason_raw", pd.Series(dtype=object)).fillna("")
                                count_lbo_now_base = int((base_entry_reason_raw == "lbo_now").sum())
                                count_lbo_confirm_base = int((base_entry_reason_raw == "lbo_confirm").sum())
                                count_lbo_pullback_base = int((base_entry_reason_raw == "lbo_pullback").sum())
                                count_lbo_now_gate = int((gate_entry_reason_raw == "lbo_now").sum())
                                count_lbo_confirm_gate = int((gate_entry_reason_raw == "lbo_confirm").sum())
                                count_lbo_pullback_gate = int((gate_entry_reason_raw == "lbo_pullback").sum())
                                lbo_base_entered_count = (
                                    count_lbo_now_base + count_lbo_confirm_base + count_lbo_pullback_base
                                )
                                lbo_gate_entered_count = (
                                    count_lbo_now_gate + count_lbo_confirm_gate + count_lbo_pullback_gate
                                )
                                print(
                                    f"LBO_POS_OPEN {instrument} {day} "
                                    f"baseline_pos_opens={entries_taken_base} baseline_trades={len(trades_base)} "
                                    f"gated_pos_opens={entries_taken_gate} gated_trades={len(trades_gate)}",
                                    flush=True,
                                )
                                if validate_debug:
                                    print(
                                        f"VALIDATE_DEBUG {instrument} {day} "
                                        f"baseline_trade_count={len(trades_base)} "
                                        f"lbo_base_entered_count={lbo_base_entered_count} "
                                        f"lbo_gate_entered_count={lbo_gate_entered_count}",
                                        flush=True,
                                    )
                                if len(trades_base) != entries_taken_base:
                                    raise RuntimeError("LBO base pos-open count does not match trade count.")
                                if len(trades_gate) != entries_taken_gate:
                                    raise RuntimeError("LBO gate pos-open count does not match trade count.")
                            if strategy_mode == "srf_entry_v1":
                                if len(trades_base) != srf_entered_base:
                                    raise RuntimeError("SRF base entered count does not match trade count.")
                                if len(trades_gate) != srf_entered_gate:
                                    raise RuntimeError("SRF gate entered count does not match trade count.")
                                if srf_entered_base > srf_triggered_base:
                                    raise RuntimeError("SRF base entered exceeds SRF triggered count.")
                                if srf_entered_gate > srf_triggered_gate:
                                    raise RuntimeError("SRF gate entered exceeds SRF triggered count.")
                            if strategy_mode == "absorption_failure_v2":
                                if afr_enter_on == "break" and (count_ft_entries_base > 0 or count_ft_entries_gate > 0):
                                    raise RuntimeError("AFR_ENTER_ON=break but FT entries were recorded.")
                                if afr_enter_on == "ft" and (count_break_entries_base > 0 or count_break_entries_gate > 0):
                                    raise RuntimeError("AFR_ENTER_ON=ft but break entries were recorded.")
                            pct_exit_on_decay_base = float((base_exit_reason == "decay").mean()) if len(base_exit_reason) else 0.0
                            pct_exit_on_decay_gate = float((gate_exit_reason == "decay").mean()) if len(gate_exit_reason) else 0.0
                            pct_exit_on_be_base = float((base_exit_reason == "be").mean()) if len(base_exit_reason) else 0.0
                            pct_exit_on_be_gate = float((gate_exit_reason == "be").mean()) if len(gate_exit_reason) else 0.0
                            pct_exit_on_scratch_base = float((base_exit_reason == "scratch").mean()) if len(base_exit_reason) else 0.0
                            pct_exit_on_scratch_gate = float((gate_exit_reason == "scratch").mean()) if len(gate_exit_reason) else 0.0
                            base_exit_counts = base_exit_reason.value_counts(dropna=False).to_dict()
                            gate_exit_counts = gate_exit_reason.value_counts(dropna=False).to_dict()
                            print(
                                f"EXIT_COUNTS {instrument} {day} base={base_exit_counts} gate={gate_exit_counts}",
                                flush=True,
                            )
                            base_exec_style = trades_base.get("exit_exec_style", pd.Series(dtype=object)).fillna("market")
                            gate_exec_style = trades_gate.get("exit_exec_style", pd.Series(dtype=object)).fillna("market")
                            passive_filled_base = int((base_exec_style == "passive_filled").sum())
                            passive_fallback_base = int((base_exec_style == "market_fallback").sum())
                            passive_filled_gate = int((gate_exec_style == "passive_filled").sum())
                            passive_fallback_gate = int((gate_exec_style == "market_fallback").sum())
                            passive_den_base = max(passive_filled_base + passive_fallback_base, 1)
                            passive_den_gate = max(passive_filled_gate + passive_fallback_gate, 1)
                            passive_fill_rate_base = float(passive_filled_base / passive_den_base) if passive_den_base else 0.0
                            passive_fill_rate_gate = float(passive_filled_gate / passive_den_gate) if passive_den_gate else 0.0
                            print(
                                f"PASSIVE_EXIT {instrument} {day} base filled={passive_filled_base} "
                                f"fallback={passive_fallback_base} rate={passive_fill_rate_base:.2%} | "
                                f"gate filled={passive_filled_gate} fallback={passive_fallback_gate} "
                                f"rate={passive_fill_rate_gate:.2%}",
                                flush=True,
                            )
                            if exit_mode == "exit_sm_v2":
                                print(
                                    f"{instrument} {day} EXIT PATH = exit_sm_v2 "
                                    f"base_exit_taken={exit_sm_exit_taken_base} gate_exit_taken={exit_sm_exit_taken_gate} "
                                    f"legacy_base={legacy_exit_taken_base} legacy_gate={legacy_exit_taken_gate}",
                                    flush=True,
                                )
                                if legacy_exit_taken_base or legacy_exit_taken_gate:
                                    raise RuntimeError("EXIT_MODE=exit_sm_v2 but legacy exits were taken.")
                                if exit_sm_exit_taken_base != len(trades_base):
                                    raise RuntimeError("EXIT_MODE=exit_sm_v2 base exits do not match trade count.")
                                if exit_sm_exit_taken_gate != len(trades_gate):
                                    raise RuntimeError("EXIT_MODE=exit_sm_v2 gate exits do not match trade count.")
                            if validate_debug:
                                base_reason_std = (
                                    trades_base.get("exit_reason_std", pd.Series(dtype=object))
                                    .fillna("unknown")
                                    .astype(str)
                                )
                                gate_reason_std = (
                                    trades_gate.get("exit_reason_std", pd.Series(dtype=object))
                                    .fillna("unknown")
                                    .astype(str)
                                )
                                if int(base_reason_std.value_counts(dropna=False).sum()) != len(trades_base):
                                    raise RuntimeError("Base exit_reason_std counts do not match trade count.")
                                if int(gate_reason_std.value_counts(dropna=False).sum()) != len(trades_gate):
                                    raise RuntimeError("Gate exit_reason_std counts do not match trade count.")
                            base_be_armed = float(trades_base.get("be_armed", pd.Series(dtype=bool)).mean()) if len(trades_base) else 0.0
                            gate_be_armed = float(trades_gate.get("be_armed", pd.Series(dtype=bool)).mean()) if len(trades_gate) else 0.0
                            base_reason_std = trades_base.get("exit_reason_std", pd.Series(dtype=str)).astype(str).str.upper()
                            gate_reason_std = trades_gate.get("exit_reason_std", pd.Series(dtype=str)).astype(str).str.upper()
                            base_be_limit = float((base_reason_std == "BE_LIMIT").mean()) if len(base_reason_std) else 0.0
                            gate_be_limit = float((gate_reason_std == "BE_LIMIT").mean()) if len(gate_reason_std) else 0.0
                            print(
                                f"BE_STATS {instrument} {day} base_armed={base_be_armed:.2%} "
                                f"gate_armed={gate_be_armed:.2%} base_be_limit={base_be_limit:.2%} "
                                f"gate_be_limit={gate_be_limit:.2%}",
                                flush=True,
                            )
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
                                    "base_entry_candidates_when_flat": int(entry_candidates_flat_base),
                                    "gate_entry_candidates_when_flat": int(entry_candidates_flat_gate),
                                    "base_long_candidates": int(strategy_long_base),
                                    "base_short_candidates": int(strategy_short_base),
                                    "srf_checked_base": int(srf_checked_base),
                                    "srf_triggered_base": int(srf_triggered_base),
                                    "srf_entered_base": int(srf_entered_base),
                                    "srf_checked_gate": int(srf_checked_gate),
                                    "srf_triggered_gate": int(srf_triggered_gate),
                                    "srf_entered_gate": int(srf_entered_gate),
                                    "impulse_checked_base": int(impulse_checked_base),
                                    "impulse_passed_threshold_base": int(impulse_passed_thr_base),
                                    "impulse_passed_confirm_base": int(impulse_passed_confirm_base),
                                    "impulse_entered_base": int(impulse_entered_base),
                                    "exit_sm_called_base": int(exit_sm_called_base),
                                    "exit_sm_exit_taken_base": int(exit_sm_exit_taken_base),
                                    "exit_sm_hold_base": int(exit_sm_hold_base),
                                    "legacy_exit_taken_base": int(legacy_exit_taken_base),
                                    "passive_filled_count_base": int(passive_filled_base),
                                    "passive_fallback_count_base": int(passive_fallback_base),
                                    "passive_fill_rate_base": float(passive_fill_rate_base),
                                    "entry_confirm_checked_base": int(entry_confirm_checked_base),
                                    "entry_confirm_passed_base": int(entry_confirm_passed_base),
                                    "entry_confirm_failed_base": int(entry_confirm_failed_base),
                                    "impulse_checked_gate": int(impulse_checked_gate),
                                    "impulse_passed_threshold_gate": int(impulse_passed_thr_gate),
                                    "impulse_passed_confirm_gate": int(impulse_passed_confirm_gate),
                                    "impulse_entered_gate": int(impulse_entered_gate),
                                    "exit_sm_called_gate": int(exit_sm_called_gate),
                                    "exit_sm_exit_taken_gate": int(exit_sm_exit_taken_gate),
                                    "exit_sm_hold_gate": int(exit_sm_hold_gate),
                                    "legacy_exit_taken_gate": int(legacy_exit_taken_gate),
                                    "passive_filled_count_gate": int(passive_filled_gate),
                                    "passive_fallback_count_gate": int(passive_fallback_gate),
                                    "passive_fill_rate_gate": float(passive_fill_rate_gate),
                                    "entry_confirm_checked_gate": int(entry_confirm_checked_gate),
                                    "entry_confirm_passed_gate": int(entry_confirm_passed_gate),
                                    "entry_confirm_failed_gate": int(entry_confirm_failed_gate),
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
                                    "lbo_pending_started_base": int(lbo_pending_started_base),
                                    "lbo_confirm_entered_base": int(lbo_confirm_entered_base),
                                    "lbo_confirm_expired_base": int(lbo_confirm_expired_base),
                                    "lbo_pull_triggered_base": int(lbo_pull_triggered_base),
                                    "lbo_pull_pulled_back_base": int(lbo_pull_pulled_back_base),
                                    "lbo_pull_resumed_entered_base": int(lbo_pull_resumed_entered_base),
                                    "lbo_pull_expired_base": int(lbo_pull_expired_base),
                                    "lbo_pending_started_gate": int(lbo_pending_started_gate),
                                    "lbo_confirm_entered_gate": int(lbo_confirm_entered_gate),
                                    "lbo_confirm_expired_gate": int(lbo_confirm_expired_gate),
                                    "lbo_pull_triggered_gate": int(lbo_pull_triggered_gate),
                                    "lbo_pull_pulled_back_gate": int(lbo_pull_pulled_back_gate),
                                    "lbo_pull_resumed_entered_gate": int(lbo_pull_resumed_entered_gate),
                                    "lbo_pull_expired_gate": int(lbo_pull_expired_gate),
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
                                    "pct_exit_on_scratch_base": float(pct_exit_on_scratch_base),
                                    "pct_exit_on_scratch_gate": float(pct_exit_on_scratch_gate),
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
                            skip_reason_keys = list(skip_gate.keys())
                            if sum(skip_gate[k] for k in skip_reason_keys) != skipped_gate:
                                raise RuntimeError(
                                    f"Skip reasons mismatch: sum={sum(skip_gate[k] for k in skip_reason_keys)} "
                                    f"skipped={skipped_gate} day={day} W={gate_lookback_bars} mode={gate_mode}"
                                )
                            if gate_mode == "side_matched":
                                blocked_diag = int(gate_diag.get("blocked_none", 0)) + int(
                                    gate_diag.get("blocked_mismatch", 0)
                                )
                                if blocked_diag != blocked_gate:
                                    raise RuntimeError(
                                        f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                                        f"blocked_none+blocked_mismatch={blocked_diag} blocked_total={blocked_gate}"
                                    )
                                allowed_count = int(gate_diag.get("allowed_count", 0))
                                if blocked_diag + allowed_count != entry_candidates_flat_gate:
                                    print(
                                        f"GATE_COUNTS {instrument} {day} "
                                        f"blocked_diag={blocked_diag} allowed_count={allowed_count} "
                                        f"entry_candidates_when_flat_gate={entry_candidates_flat_gate}",
                                        flush=True,
                                    )
                                    raise RuntimeError(
                                        f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                                        f"blocked+allowed={blocked_diag + allowed_count} entry_candidates_when_flat={entry_candidates_flat_gate}"
                                    )
                            if strategy_mode != "impulse_confirm_v1" and entries_taken_base != signals_flat_base:
                                print(
                                    f"BASE_COUNTS {instrument} {day} "
                                    f"entries_taken={entries_taken_base} signals_when_flat={signals_flat_base} "
                                    f"baseline_trades={len(trades_base)}",
                                    flush=True,
                                )
                                raise RuntimeError(
                                    f"Baseline entries mismatch: entries_taken={entries_taken_base} "
                                    f"signals_when_flat={signals_flat_base} day={day}"
                                )
                            expected_gate = entries_taken_gate + blocked_gate
                            if expected_gate != entry_candidates_flat_gate:
                                print(
                                    f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode} gate_mismatch "
                                    f"entries_taken={entries_taken_gate} blocked={blocked_gate} entry_candidates_when_flat={entry_candidates_flat_gate}",
                                    flush=True,
                                )
                                raise RuntimeError(
                                    f"Gated entries mismatch: entries_taken+blocked={expected_gate} "
                                    f"entry_candidates_when_flat={entry_candidates_flat_gate} day={day}"
                                )
                        except Exception as exc:
                            skipped_days.append(f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode}: {exc}")
                            try:
                                print(
                                    f"[{day_index}/{day_count}] {instrument} {day} W={gate_lookback_bars} "
                                    f"mode={gate_mode} FAILED: {exc}",
                                    flush=True,
                                )
                            except OSError:
                                pass
                            msg = str(exc)
                            if "mismatch" in msg or "entries" in msg or "gate_mismatch" in msg:
                                raise
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
            print(f"=== STRATEGY: {strategy_mode} DIAGNOSTICS ===", flush=True)
            print("Strategy diagnostics by day:", flush=True)
            print(strategy_df.to_string(index=False), flush=True)

        if all_gated:
            all_gated_df = pd.concat(all_gated, ignore_index=True)
        else:
            all_gated_df = pd.DataFrame()
        total_gated_trades = int(all_gated_df.shape[0])
        total_gated_pnl = float(total_pnl_gate)
        pnl_per_trade = float(total_gated_pnl / total_gated_trades) if total_gated_trades else 0.0
        total_improvement = float(total_pnl_gate - total_pnl_base)
        pnl_per_blocked = float(total_improvement / max(total_blocked, 1))
        max_dd_ticks = float(np.max(gate_max_dds)) if gate_max_dds else 0.0
        if total_gated_trades:
            median_mfe = float(np.nanmedian(all_gated_df["mfe_ticks"].to_numpy()))
            median_mae = float(np.nanmedian(all_gated_df["mae_ticks"].to_numpy()))
            exit_reason = all_gated_df["exit_reason"].astype(str)
            pct_scratch = float((exit_reason == "scratch").mean())
            pct_be = float((exit_reason == "be").mean())
            pct_decay = float((exit_reason == "decay").mean())
        else:
            median_mfe = 0.0
            median_mae = 0.0
            pct_scratch = 0.0
            pct_be = 0.0
            pct_decay = 0.0

        if run_afr2_sweep:
            afr2_sweep_rows.append(
                {
                    "afr_min_flow_abs": float(afr_min_flow_abs_cur),
                    "afr_ft_bars": int(afr_ft_bars_cur),
                    "gate_mode": gate_mode,
                    "W": gate_lookback_bars,
                    "total_gated_trades": total_gated_trades,
                    "total_gated_pnl_ticks": total_gated_pnl,
                    "max_dd_ticks": max_dd_ticks,
                    "pnl_per_trade": pnl_per_trade,
                    "pnl_per_blocked_entry": pnl_per_blocked,
                    "median_mfe_ticks": median_mfe,
                    "median_mae_ticks": median_mae,
                    "pct_exit_on_scratch": pct_scratch,
                    "pct_exit_on_be": pct_be,
                    "pct_exit_on_decay": pct_decay,
                }
            )
            print(
                f"AFR2_SWEEP afr_min_flow_abs={afr_min_flow_abs_cur} afr_ft_bars={afr_ft_bars_cur} "
                f"mode={gate_mode} W={gate_lookback_bars} gated_trades={total_gated_trades} "
                f"gated_pnl={total_gated_pnl:.2f} max_dd={max_dd_ticks:.2f} "
                f"pnl_per_trade={pnl_per_trade:.4f} pnl_per_blocked={pnl_per_blocked:.4f} "
                f"median_mfe={median_mfe:.2f} median_mae={median_mae:.2f} "
                f"scratch={pct_scratch:.2%} be={pct_be:.2%} decay={pct_decay:.2%}",
                flush=True,
            )

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
            "baseline_median_trade_ticks",
            "gated_median_trade_ticks",
            "baseline_win_rate",
            "gated_win_rate",
            "baseline_mean_win_ticks",
            "gated_mean_win_ticks",
            "baseline_mean_loss_ticks",
            "gated_mean_loss_ticks",
            "baseline_profit_factor",
            "gated_profit_factor",
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
            "baseline_entry_candidates_when_flat",
            "gated_entry_candidates_when_flat",
            "baseline_trades",
            "gated_trades",
            "entries_blocked_by_gate",
            "entries_taken_gated",
            "coverage",
            "block_rate",
        ]
        coverage_df = sweep_df[coverage_cols].copy()
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
    if all_base or all_gated:
        all_trades = pd.concat(all_base + all_gated, ignore_index=True)
        w_tag = "multi" if len(sweep_ws) > 1 else str(sweep_ws[0])
        gate_tag = "multi" if len(gate_modes) > 1 else gate_modes[0]
        trades_path = out_dir / f"trades_{strategy_mode}_W{w_tag}_gate_{gate_tag}_baseline_vs_gated.csv"
        all_trades.to_csv(trades_path, index=False)
        mfe_table = _mfe_threshold_table(all_trades, "entry_reason")
        if not mfe_table.empty:
            mfe_path = out_dir / f"mfe_thresholds_{strategy_mode}_W{w_tag}_gate_{gate_tag}.csv"
            mfe_table.to_csv(mfe_path, index=False)
            print("MFE thresholds by entry_reason (per day):", flush=True)
            print(mfe_table.to_string(index=False), flush=True)
            agg_rows = []
            for (strategy, reason), g in mfe_table.groupby(["strategy", "entry_reason"], sort=False):
                total = float(g["count"].sum())
                if total <= 0:
                    continue
                agg_rows.append(
                    {
                        "strategy": strategy,
                        "entry_reason": reason,
                        "count": int(total),
                        "pct_mfe_ge_1": float((g["pct_mfe_ge_1"] * g["count"]).sum() / total),
                        "pct_mfe_ge_2": float((g["pct_mfe_ge_2"] * g["count"]).sum() / total),
                        "pct_mfe_ge_3": float((g["pct_mfe_ge_3"] * g["count"]).sum() / total),
                    }
                )
            mfe_agg = pd.DataFrame(agg_rows)
            if not mfe_agg.empty:
                print("MFE thresholds aggregate:", flush=True)
                print(mfe_agg.to_string(index=False), flush=True)
                viable = mfe_agg[
                    (mfe_agg["strategy"] == "baseline")
                    & (mfe_agg["pct_mfe_ge_2"] >= 0.05)
                    & (mfe_agg["count"] >= 500)
                ]
                if not viable.empty:
                    reasons = ", ".join(sorted(viable["entry_reason"].unique().tolist()))
                    print(f"ENTRY_EDGE_VERDICT: VIABLE ENTRY SUBSET EXISTS (baseline reasons: {reasons})", flush=True)
                else:
                    print("ENTRY_EDGE_VERDICT: NO STANDALONE ENTRY EDGE (demote to gate-only)", flush=True)
        mfe_root = _mfe_threshold_table(all_trades, "entry_root_reason")
        if not mfe_root.empty:
            mfe_root_path = out_dir / f"mfe_thresholds_root_{strategy_mode}_W{w_tag}_gate_{gate_tag}.csv"
            mfe_root.to_csv(mfe_root_path, index=False)
            print("MFE thresholds by entry_root_reason (per day):", flush=True)
            print(mfe_root.to_string(index=False), flush=True)
            agg_rows_root = []
            for (strategy, reason), g in mfe_root.groupby(["strategy", "entry_root_reason"], sort=False):
                total = float(g["count"].sum())
                if total <= 0:
                    continue
                agg_rows_root.append(
                    {
                        "strategy": strategy,
                        "entry_root_reason": reason,
                        "count": int(total),
                        "pct_mfe_ge_1": float((g["pct_mfe_ge_1"] * g["count"]).sum() / total),
                        "pct_mfe_ge_2": float((g["pct_mfe_ge_2"] * g["count"]).sum() / total),
                        "pct_mfe_ge_3": float((g["pct_mfe_ge_3"] * g["count"]).sum() / total),
                    }
                )
            mfe_root_agg = pd.DataFrame(agg_rows_root)
            if not mfe_root_agg.empty:
                print("MFE thresholds aggregate (entry_root_reason):", flush=True)
                print(mfe_root_agg.to_string(index=False), flush=True)
        if srf_events_base_all or srf_events_gate_all:
            srf_all = pd.DataFrame(srf_events_base_all + srf_events_gate_all)
            if not srf_all.empty:
                srf_horizons = [2, 5, 10, 20, 50]
                for strat in ["baseline", "gated"]:
                    sub = srf_all[srf_all["strategy"] == strat].copy()
                    if sub.empty:
                        continue
                    rows = []
                    for h in srf_horizons:
                        vals = pd.to_numeric(sub.get(f"R_{h}", pd.Series(dtype=float)), errors="coerce").dropna()
                        rows.append(
                            {
                                "h_bars": h,
                                "count": int(vals.size),
                                "mean_R": float(vals.mean()) if not vals.empty else 0.0,
                                "median_R": float(vals.median()) if not vals.empty else 0.0,
                            }
                        )
                    print(f"SRF perfect-exec forward returns ({strat}):", flush=True)
                    print(pd.DataFrame(rows).to_string(index=False), flush=True)
                    for h in [20, 50]:
                        mfe = pd.to_numeric(sub.get(f"MFE_{h}", pd.Series(dtype=float)), errors="coerce").dropna()
                        mae = pd.to_numeric(sub.get(f"MAE_{h}", pd.Series(dtype=float)), errors="coerce").dropna()
                        pct_mfe_ge_2 = float((mfe >= 2).mean()) if not mfe.empty else 0.0
                        pct_mfe_ge_4 = float((mfe >= 4).mean()) if not mfe.empty else 0.0
                        pct_mfe_ge_6 = float((mfe >= 6).mean()) if not mfe.empty else 0.0
                        mae_q = mae.quantile([0.5, 0.9, 0.95, 0.99]).to_dict() if not mae.empty else {}
                        print(
                            f"SRF MFE/MAE H={h} ({strat}) count={int(len(mfe))} "
                            f"pct_mfe_ge_2={pct_mfe_ge_2:.3f} pct_mfe_ge_4={pct_mfe_ge_4:.3f} pct_mfe_ge_6={pct_mfe_ge_6:.3f} "
                            f"mae_med={mae_q.get(0.5, 0.0):.3f} mae_p90={mae_q.get(0.9, 0.0):.3f} "
                            f"mae_p95={mae_q.get(0.95, 0.0):.3f} mae_p99={mae_q.get(0.99, 0.0):.3f}",
                            flush=True,
                        )
                    mfe_20 = pd.to_numeric(sub.get("MFE_20", pd.Series(dtype=float)), errors="coerce")
                    mae_20 = pd.to_numeric(sub.get("MAE_20", pd.Series(dtype=float)), errors="coerce")
                    fail_mask = (mfe_20 < 2) & mfe_20.notna()
                    fail_mae = mae_20[fail_mask].dropna()
                    if not fail_mae.empty:
                        q = fail_mae.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
                        print(
                            f"SRF fail MAE_20 ({strat}) count={int(fail_mae.size)} "
                            f"med={q.get(0.5, 0.0):.3f} p90={q.get(0.9, 0.0):.3f} "
                            f"p95={q.get(0.95, 0.0):.3f} p99={q.get(0.99, 0.0):.3f}",
                            flush=True,
                        )
                    disp_abs = pd.to_numeric(sub.get("srf_disp_ticks", pd.Series(dtype=float)), errors="coerce").abs()
                    spread_entry = pd.to_numeric(sub.get("spread_ticks_entry", pd.Series(dtype=float)), errors="coerce")
                    disp_bucket = pd.Series(pd.NA, index=sub.index, dtype=object)
                    disp_bucket[(disp_abs >= 3) & (disp_abs <= 4)] = "3-4"
                    disp_bucket[(disp_abs >= 5) & (disp_abs <= 6)] = "5-6"
                    disp_bucket[disp_abs >= 7] = "7+"
                    spread_bucket = pd.Series(pd.NA, index=sub.index, dtype=object)
                    spread_bucket[spread_entry == 1] = "1"
                    spread_bucket[spread_entry >= 2] = "2+"
                    bucket_rows = []
                    for (db, sb), g in sub.groupby([disp_bucket, spread_bucket], dropna=True):
                        r20 = pd.to_numeric(g.get("R_20", pd.Series(dtype=float)), errors="coerce").dropna()
                        r50 = pd.to_numeric(g.get("R_50", pd.Series(dtype=float)), errors="coerce").dropna()
                        if r20.empty and r50.empty:
                            continue
                        mean_spread = float(spread_entry.loc[g.index].mean()) if not spread_entry.loc[g.index].empty else 0.0
                        mean_r20 = float(r20.mean()) if not r20.empty else 0.0
                        mean_r50 = float(r50.mean()) if not r50.empty else 0.0
                        bucket_rows.append(
                            {
                                "disp_bucket": db,
                                "spread_bucket": sb,
                                "count": int(len(g)),
                                "mean_R20": mean_r20,
                                "median_R20": float(r20.median()) if not r20.empty else 0.0,
                                "mean_R50": mean_r50,
                                "median_R50": float(r50.median()) if not r50.empty else 0.0,
                                "EV_net_20": mean_r20 - mean_spread,
                            }
                        )
                    if bucket_rows:
                        print(f"SRF stratified buckets ({strat}):", flush=True)
                        print(pd.DataFrame(bucket_rows).to_string(index=False), flush=True)
        exit_by_day = _exit_breakdown_stats(all_trades)
        if not exit_by_day.empty:
            exit_by_day_path = out_dir / f"exit_breakdown_by_day_{strategy_mode}_W{w_tag}_gate_{gate_tag}.csv"
            exit_by_day.to_csv(exit_by_day_path, index=False)
            for strat in ["baseline", "gated"]:
                sub = exit_by_day[exit_by_day["strategy"] == strat].copy()
                if sub.empty:
                    continue
                counts = (
                    sub.pivot_table(
                        index="date",
                        columns="exit_reason_std",
                        values="count_trades",
                        aggfunc="sum",
                        fill_value=0,
                    )
                    .reset_index()
                )
                totals = sub.groupby("date", as_index=False)["sum_pnl_ticks"].sum().rename(columns={"sum_pnl_ticks": "total_pnl_ticks"})
                counts = counts.merge(totals, on="date", how="left")
                counts["total_trades"] = counts.drop(columns=["date", "total_pnl_ticks"]).sum(axis=1)
                print(f"Exit breakdown by day ({strat}):", flush=True)
                print(counts.to_string(index=False), flush=True)
            def _agg_reason(g: pd.DataFrame) -> pd.Series:
                pnl = pd.to_numeric(g["pnl_ticks"], errors="coerce").to_numpy()
                wins = pnl[pnl > 0]
                losses = pnl[pnl < 0]
                sum_wins = float(np.sum(wins)) if wins.size else 0.0
                sum_losses = float(np.sum(losses)) if losses.size else 0.0
                pf = (sum_wins / abs(sum_losses)) if sum_losses != 0.0 else 0.0
                return pd.Series(
                    {
                        "count_trades": int(len(pnl)),
                        "sum_pnl_ticks": float(np.sum(pnl)),
                        "mean_pnl_ticks": float(np.mean(pnl)) if pnl.size else 0.0,
                        "median_pnl_ticks": float(np.median(pnl)) if pnl.size else 0.0,
                        "win_rate": float(np.mean(pnl > 0)) if pnl.size else 0.0,
                        "profit_factor": float(pf),
                        "median_mfe_ticks": float(np.nanmedian(g["mfe_ticks"])) if "mfe_ticks" in g.columns else 0.0,
                        "median_mae_ticks": float(np.nanmedian(g["mae_ticks"])) if "mae_ticks" in g.columns else 0.0,
                        "median_hold_bars_realized": float(np.nanmedian(g["hold_bars_realized"]))
                        if "hold_bars_realized" in g.columns
                        else 0.0,
                        "p5_pnl_ticks": float(np.quantile(pnl, 0.05)) if pnl.size else 0.0,
                        "p1_pnl_ticks": float(np.quantile(pnl, 0.01)) if pnl.size else 0.0,
                    }
                )
            exit_agg = (
                all_trades.groupby(["strategy", "exit_reason_std"], sort=False)
                .apply(_agg_reason)
                .reset_index()
            )
            exit_agg_path = out_dir / f"exit_breakdown_aggregate_{strategy_mode}_W{w_tag}_gate_{gate_tag}.csv"
            exit_agg.to_csv(exit_agg_path, index=False)
            for strat in ["baseline", "gated"]:
                sub = exit_agg[exit_agg["strategy"] == strat].copy()
                if sub.empty:
                    continue
                print(f"Exit breakdown aggregate ({strat}):", flush=True)
                print(sub.to_string(index=False), flush=True)
    agg_rows = []
    if not sweep_df.empty:
        for (w, mode), g in sweep_df.groupby(["W", "gate_mode"], sort=False):
            days_count = int(g.shape[0])
            pnl_imp = pd.to_numeric(g["pnl_improvement_ticks"], errors="coerce")
            imp_per_block = pd.to_numeric(g["improvement_per_blocked"], errors="coerce")
            dd_imp = pd.to_numeric(g["dd_improvement"], errors="coerce")
            base_mean = pd.to_numeric(g["baseline_mean_trade_ticks"], errors="coerce")
            gate_mean = pd.to_numeric(g["gated_mean_trade_ticks"], errors="coerce")
            base_median = pd.to_numeric(g["baseline_median_trade_ticks"], errors="coerce")
            gate_median = pd.to_numeric(g["gated_median_trade_ticks"], errors="coerce")
            base_win = pd.to_numeric(g["baseline_win_rate"], errors="coerce")
            gate_win = pd.to_numeric(g["gated_win_rate"], errors="coerce")
            base_pf = pd.to_numeric(g["baseline_profit_factor"], errors="coerce")
            gate_pf = pd.to_numeric(g["gated_profit_factor"], errors="coerce")
            base_mean_win = pd.to_numeric(g["baseline_mean_win_ticks"], errors="coerce")
            gate_mean_win = pd.to_numeric(g["gated_mean_win_ticks"], errors="coerce")
            base_mean_loss = pd.to_numeric(g["baseline_mean_loss_ticks"], errors="coerce")
            gate_mean_loss = pd.to_numeric(g["gated_mean_loss_ticks"], errors="coerce")
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
                    "median_baseline_mean_trade_ticks": float(np.nanmedian(base_mean)) if days_count else np.nan,
                    "median_gated_mean_trade_ticks": float(np.nanmedian(gate_mean)) if days_count else np.nan,
                    "median_baseline_median_trade_ticks": float(np.nanmedian(base_median)) if days_count else np.nan,
                    "median_gated_median_trade_ticks": float(np.nanmedian(gate_median)) if days_count else np.nan,
                    "median_baseline_win_rate": float(np.nanmedian(base_win)) if days_count else np.nan,
                    "median_gated_win_rate": float(np.nanmedian(gate_win)) if days_count else np.nan,
                    "median_baseline_profit_factor": float(np.nanmedian(base_pf)) if days_count else np.nan,
                    "median_gated_profit_factor": float(np.nanmedian(gate_pf)) if days_count else np.nan,
                    "median_baseline_mean_win_ticks": float(np.nanmedian(base_mean_win)) if days_count else np.nan,
                    "median_gated_mean_win_ticks": float(np.nanmedian(gate_mean_win)) if days_count else np.nan,
                    "median_baseline_mean_loss_ticks": float(np.nanmedian(base_mean_loss)) if days_count else np.nan,
                    "median_gated_mean_loss_ticks": float(np.nanmedian(gate_mean_loss)) if days_count else np.nan,
                }
            )
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "summary_aggregate.csv", index=False)
    if not agg_df.empty:
        print("Aggregate summary by W, gate_mode:", flush=True)
        print(agg_df.to_string(index=False), flush=True)

    if run_afr2_sweep and afr2_sweep_rows:
        afr2_sweep_df = pd.DataFrame(afr2_sweep_rows)
        afr2_sweep_path = Path("artifacts") / "afr2_sweep_summary.csv"
        afr2_sweep_df.to_csv(afr2_sweep_path, index=False)
        print(f"Wrote {afr2_sweep_path}", flush=True)
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
