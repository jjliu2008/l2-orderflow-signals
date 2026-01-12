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
  STRATEGY_MODE           baseline_flat|micro_momo_v1 (default: micro_momo_v1)
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
    for day_dir in root.glob("date=*"):
        day = day_dir.name.replace("date=", "")
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
    gate_mode: str,
    debug_entry_print: bool,
    debug_entry_limit: int,
    debug_entry_tag: str,
) -> Tuple[
    pd.DataFrame,
    float,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    float,
    float,
    int,
    int,
    int,
    int,
    Dict[str, int],
]:
    bid = pd.to_numeric(df_day["bid_price_1"], errors="coerce").to_numpy()
    ask = pd.to_numeric(df_day["ask_price_1"], errors="coerce").to_numpy()
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
    entry_flow_sum = 0.0
    entry_count = 0
    mmas_signals_checked = 0
    mmas_passed_filters = 0
    mmas_signaled = 0
    mmas_entered = 0
    skip_reasons = {
        "gated_blocked": 0,
        "no_recent_event": 0,
        "no_threshold_yet": 0,
        "no_event_in_window": 0,
    }

    signed_vol = pd.to_numeric(df_day["signed_volume"], errors="coerce").fillna(0.0).to_numpy()
    spread_ticks = np.rint((ask - bid) / tick_size).astype(np.int64)
    cs = np.concatenate([[0.0], np.cumsum(signed_vol)])
    if tick_size <= 0:
        raise ValueError("tick_size must be positive for micro_momo_v1.")
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
            desired_side = _signal(i)
            if desired_side is not None:
                total_signals += 1
                signals_in_position += 1
            if i >= entry_bar + 1:
                mark_px = bid[i] if side == "long" else ask[i]
                if np.isfinite(mark_px):
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
            if i >= int(pos["exit_bar"]):
                exit_bar = int(pos["exit_bar"])
                if not np.isfinite(bid[exit_bar]) or not np.isfinite(ask[exit_bar]):
                    raise RuntimeError("Non-finite exit price; check data integrity.")
                exit_px = bid[exit_bar] if side == "long" else ask[exit_bar]
                pnl_ticks = (exit_px - entry_px) / tick_size if side == "long" else (entry_px - exit_px) / tick_size
                exit_reason = str(pos.get("exit_reason", "TIME"))
                eps = 1e-9
                entry_spread = float(pos.get("entry_spread_ticks", 0.0))
                floor_ticks = -(sl_ticks + entry_spread + eps)
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
        mmas_passed = False
        if strategy_mode == "micro_momo_v1":
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
        total_signals += 1
        entry_bar = i + 1
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

        if gated:
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
                    i += 1
                    continue
                elif event_pos[idx_pos] < entry_bar - gate_lookback_bars:
                    skip_reasons["no_event_in_window"] += 1
                    skipped += 1
                    blocked_signals += 1
                    gate_allowed = False
                    gate_reason = "no_event_in_window"
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

        entry_px = ask[entry_bar] if desired_side == "long" else bid[entry_bar]
        entry_spread_ticks = float(spread_ticks[entry_bar])
        tp_level = entry_px + (tp_ticks * tick_size if desired_side == "long" else -tp_ticks * tick_size)
        sl_level = entry_px - (sl_ticks * tick_size if desired_side == "long" else -sl_ticks * tick_size)
        pos = {
            "entry_time": df_day["Time"].iloc[entry_bar],
            "entry_bar": entry_bar,
            "entry_px": float(entry_px),
            "exit_bar": int(min(entry_bar + hold_bars, n - 1)),
            "exit_reason": "TIME",
            "tp_level": float(tp_level),
            "sl_level": float(sl_level),
            "entry_spread_ticks": entry_spread_ticks,
            "side": desired_side,
        }
        in_position = True
        entries_taken += 1
        impulse_for_entry = impulse_ticks if np.isfinite(impulse_ticks) else dmid_ticks
        entry_impulse_sum += float(impulse_for_entry) if np.isfinite(impulse_for_entry) else 0.0
        entry_flow_sum += float(flow) if np.isfinite(flow) else 0.0
        entry_count += 1
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
        float(entry_flow_sum),
        int(entry_count),
        mmas_signals_checked,
        mmas_passed_filters,
        mmas_signaled,
        mmas_entered,
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
    tick_size = float(os.environ.get("TICK_SIZE", "0.25"))
    lookback_bars = int(os.environ.get("LOOKBACK_BARS", "10"))
    entry_threshold_ticks = int(os.environ.get("ENTRY_THRESHOLD_TICKS", "1"))
    hold_bars = int(os.environ.get("HOLD_BARS", "20"))
    strategy_mode = os.environ.get("STRATEGY_MODE", "").strip().lower()
    baseline_mode = os.environ.get("BASELINE_MODE", "flat").strip().lower()
    if not strategy_mode:
        strategy_mode = "micro_momo_v1"
    baseline_k_bars = int(os.environ.get("BASELINE_K_BARS", "5"))
    micro_k_bars = int(os.environ.get("MICRO_K_BARS", "5"))
    micro_impulse_ticks = int(os.environ.get("MICRO_IMPULSE_TICKS", "1"))
    micro_flow_min = float(os.environ.get("MICRO_FLOW_MIN", "0"))
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
            "gate_mode_list": gate_modes,
            "sweep_ws": sweep_ws,
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

    health_dir = out_dir / "health"
    health_dir.mkdir(parents=True, exist_ok=True)

    for gate_mode in gate_modes:
        for gate_lookback_bars in sweep_ws:
            out_dir_w = out_dir / f"W{gate_lookback_bars}" / f"mode={gate_mode}"
            out_dir_w.mkdir(parents=True, exist_ok=True)

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
                        entry_flow_sum_base,
                        entry_count_base,
                        mmas_checked_base,
                        mmas_passed_base,
                        mmas_signaled_base,
                        mmas_entered_base,
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
                        gate_mode=gate_mode,
                        debug_entry_print=debug_entry,
                        debug_entry_limit=5,
                        debug_entry_tag="baseline",
                    )
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
                        entry_flow_sum_gate,
                        entry_count_gate,
                        mmas_checked_gate,
                        mmas_passed_gate,
                        mmas_signaled_gate,
                        mmas_entered_gate,
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
                        gate_mode=gate_mode,
                        debug_entry_print=debug_entry,
                        debug_entry_limit=5,
                        debug_entry_tag="gated",
                    )

                    if not sample_printed and day == selected_days[0]:
                        base_eq = _equity_stats(trades_base["pnl_ticks"].to_numpy())
                        gate_eq = _equity_stats(trades_gate["pnl_ticks"].to_numpy())
                        base_stats_sample = _metrics(trades_base)
                        gate_stats_sample = _metrics(trades_gate)
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
                        assert gate_eq["peak"] >= gate_eq["final"] or trades_gate.empty
                        sample_printed = True

                    if len(trades_base) == 0:
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
                            "skip_rate": float(blocked_gate / signals_flat_gate) if signals_flat_gate else 0.0,
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
                            "coverage": float(eligible_gate / signals_flat_gate) if signals_flat_gate else 0.0,
                            "block_rate": float(blocked_gate / signals_flat_gate) if signals_flat_gate else 0.0,
                            "mmas_signals_checked": int(mmas_checked_gate),
                            "mmas_passed_filters": int(mmas_passed_gate),
                            "mmas_signaled": int(mmas_signaled_gate),
                            "mmas_entered": int(mmas_entered_gate),
                        }
                    )
                    drop_reasons = []
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
                    strategy_rows.append(
                        {
                            "Symbol": instrument,
                            "date": day,
                            "W": gate_lookback_bars,
                            "gate_mode": gate_mode,
                            "strategy_mode": strategy_mode,
                            "strategy_signals_when_flat": int(signals_flat_base),
                            "strategy_long_signals": int(strategy_long_base),
                            "strategy_short_signals": int(strategy_short_base),
                            "avg_impulse_ticks_entry_base": float(entry_impulse_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_flow_entry_base": float(entry_flow_sum_base / entry_count_base)
                            if entry_count_base
                            else 0.0,
                            "avg_impulse_ticks_entry_gate": float(entry_impulse_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                            "avg_flow_entry_gate": float(entry_flow_sum_gate / entry_count_gate)
                            if entry_count_gate
                            else 0.0,
                        }
                    )
                    if sum(skip_gate.values()) != skipped_gate:
                        raise RuntimeError(
                            f"Skip reasons mismatch: sum={sum(skip_gate.values())} skipped={skipped_gate} "
                            f"day={day} W={gate_lookback_bars} mode={gate_mode}"
                        )
                    if entries_taken_base != signals_flat_base:
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
