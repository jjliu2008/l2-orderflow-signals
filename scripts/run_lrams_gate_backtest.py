"""
LRAMS gate backtest: compare baseline vs baseline + LRAMS gate.

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

Gate:
  GATE_LOOKBACK_BARS      event lookback W (default: 10)
  GATE_MODE               side_matched or any_side (default: side_matched)
  GATE_MODE_LIST          comma list; overrides GATE_MODE if set
  SWEEP_W                 comma list of W values (optional)
  WORST_Q                 worst-quantile threshold (default: 0.90)

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
    gate_lookback_bars: int,
    gated: bool,
    gate_mode: str,
) -> Tuple[pd.DataFrame, int, int, int, int, Dict[str, int]]:
    bid = pd.to_numeric(df_day["bid_price_1"], errors="coerce").to_numpy()
    ask = pd.to_numeric(df_day["ask_price_1"], errors="coerce").to_numpy()
    mid = 0.5 * (bid + ask)
    n = len(df_day)
    trades: List[Dict[str, float]] = []
    skipped = 0
    total_signals = 0
    eligible_signals = 0
    blocked_signals = 0
    skip_reasons = {
        "gated_blocked": 0,
        "no_recent_event": 0,
        "no_threshold_yet": 0,
        "no_event_in_window": 0,
    }

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

    i = lookback_bars
    max_i = n - hold_bars - 2
    while i <= max_i:
        if not np.isfinite(mid[i]) or not np.isfinite(mid[i - lookback_bars]):
            i += 1
            continue
        ret_ticks = (mid[i] - mid[i - lookback_bars]) / tick_size
        desired_side = None
        if ret_ticks >= entry_threshold_ticks:
            desired_side = "long"
        elif ret_ticks <= -entry_threshold_ticks:
            desired_side = "short"
        if desired_side is None:
            i += 1
            continue
        entry_bar = i + 1
        exit_bar = i + hold_bars + 1
        if entry_bar >= n or exit_bar >= n:
            break
        if not np.isfinite(bid[entry_bar]) or not np.isfinite(ask[entry_bar]):
            i += 1
            continue
        if not np.isfinite(bid[exit_bar]) or not np.isfinite(ask[exit_bar]):
            i += 1
            continue
        total_signals += 1

        if gated:
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
            else:
                pos = np.searchsorted(event_pos, entry_bar - 1, side="right") - 1
                if pos < 0:
                    skip_reasons["no_recent_event"] += 1
                elif event_pos[pos] < entry_bar - gate_lookback_bars:
                    skip_reasons["no_event_in_window"] += 1
                else:
                    thr = event_thr[pos]
                    if not np.isfinite(thr):
                        skip_reasons["no_threshold_yet"] += 1
                    else:
                        eligible_signals += 1
                        if event_asym[pos] >= thr:
                            skip_reasons["gated_blocked"] += 1
                            skipped += 1
                            blocked_signals += 1
                            i = exit_bar
                            continue

        if desired_side == "long":
            entry_px = ask[entry_bar]
            exit_px = bid[exit_bar]
            pnl_ticks = (exit_px - entry_px) / tick_size
        else:
            entry_px = bid[entry_bar]
            exit_px = ask[exit_bar]
            pnl_ticks = (entry_px - exit_px) / tick_size
        trades.append(
            {
                "entry_time": df_day["Time"].iloc[entry_bar],
                "exit_time": df_day["Time"].iloc[exit_bar],
                "side": desired_side,
                "entry_px": float(entry_px),
                "exit_px": float(exit_px),
                "pnl_ticks": float(pnl_ticks),
                "entry_bar": int(entry_bar),
                "exit_bar": int(exit_bar),
            }
        )
        i = exit_bar

    return pd.DataFrame(trades), skipped, total_signals, eligible_signals, blocked_signals, skip_reasons


def _metrics(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0,
            "total_pnl_ticks": 0.0,
            "mean_pnl_ticks": 0.0,
            "win_rate": 0.0,
            "max_drawdown_ticks": 0.0,
            "p1": 0.0,
            "p5": 0.0,
            "p10": 0.0,
            "worst_trade_ticks": 0.0,
        }
    pnl = trades["pnl_ticks"].to_numpy()
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return {
        "trade_count": int(len(trades)),
        "total_pnl_ticks": float(np.sum(pnl)),
        "mean_pnl_ticks": float(np.mean(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "max_drawdown_ticks": float(np.max(drawdown)) if len(drawdown) else 0.0,
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
    gate_lookback_bars = int(os.environ.get("GATE_LOOKBACK_BARS", "10"))
    gate_mode = os.environ.get("GATE_MODE", "side_matched").strip().lower()
    gate_mode_list_env = os.environ.get("GATE_MODE_LIST", "").strip()
    if gate_mode_list_env:
        gate_modes = [m.strip().lower() for m in gate_mode_list_env.split(",") if m.strip()]
    else:
        gate_modes = [gate_mode]
    worst_q = float(os.environ.get("WORST_Q", "0.90"))

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
    if len(selected_days) < 3:
        print("Warning: fewer than 3 days selected; continuing.", flush=True)
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

            day_index = 0
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

                    trades_base, skipped_base, total_signals_base, eligible_base, blocked_base, skip_base = _simulate_day(
                        df_day,
                        events_day,
                        tick_size=tick_size,
                        lookback_bars=lookback_bars,
                        entry_threshold_ticks=entry_threshold_ticks,
                        hold_bars=hold_bars,
                        gate_lookback_bars=gate_lookback_bars,
                        gated=False,
                        gate_mode=gate_mode,
                    )
                    trades_gate, skipped_gate, total_signals_gate, eligible_gate, blocked_gate, skip_gate = _simulate_day(
                        df_day,
                        events_day,
                        tick_size=tick_size,
                        lookback_bars=lookback_bars,
                        entry_threshold_ticks=entry_threshold_ticks,
                        hold_bars=hold_bars,
                        gate_lookback_bars=gate_lookback_bars,
                        gated=True,
                        gate_mode=gate_mode,
                    )

                    if len(trades_base) == 0:
                        reasons = health.get("likely_reasons", [])
                        if reasons:
                            print(
                                f"{instrument} {day} baseline_trades=0 likely_reasons={reasons}",
                                flush=True,
                            )
                        else:
                            print(f"{instrument} {day} baseline_trades=0 reason=unknown", flush=True)

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
                            "skipped_trades": int(skipped_base),
                            "skip_rate": float(skipped_base / total_signals_base) if total_signals_base else 0.0,
                            "total_signals": int(total_signals_base),
                            "eligible_signals": int(eligible_base),
                            "blocked_signals": int(blocked_base),
                            "coverage": float(eligible_base / total_signals_base) if total_signals_base else 0.0,
                            "block_rate": float(blocked_base / total_signals_base) if total_signals_base else 0.0,
                        }
                    )
                    gate_stats.update(
                        {
                            "strategy": "gated",
                            "date": day,
                            "Symbol": instrument,
                            "W": gate_lookback_bars,
                            "gate_mode": gate_mode,
                            "skipped_trades": int(skipped_gate),
                            "skip_rate": float(skipped_gate / total_signals_gate) if total_signals_gate else 0.0,
                            "total_signals": int(total_signals_gate),
                            "eligible_signals": int(eligible_gate),
                            "blocked_signals": int(blocked_gate),
                            "coverage": float(eligible_gate / total_signals_gate) if total_signals_gate else 0.0,
                            "block_rate": float(blocked_gate / total_signals_gate) if total_signals_gate else 0.0,
                        }
                    )
                    summaries.extend([base_stats, gate_stats])
                    sweep_rows.append(
                        {
                            "Symbol": instrument,
                            "date": day,
                            "W": gate_lookback_bars,
                            "gate_mode": gate_mode,
                            "baseline_pnl_ticks": base_stats["total_pnl_ticks"],
                            "gated_pnl_ticks": gate_stats["total_pnl_ticks"],
                            "pnl_improvement_ticks": gate_stats["total_pnl_ticks"] - base_stats["total_pnl_ticks"],
                            "blocked_signals": int(blocked_gate),
                            "improvement_per_blocked": (gate_stats["total_pnl_ticks"] - base_stats["total_pnl_ticks"])
                            / max(int(blocked_gate), 1),
                            "baseline_max_dd_ticks": base_stats["max_drawdown_ticks"],
                            "gated_max_dd_ticks": gate_stats["max_drawdown_ticks"],
                            "dd_improvement": base_stats["max_drawdown_ticks"] - gate_stats["max_drawdown_ticks"],
                            "baseline_worst_trade_ticks": base_stats["worst_trade_ticks"],
                            "gated_worst_trade_ticks": gate_stats["worst_trade_ticks"],
                            "baseline_p1_trade_ticks": base_stats["p1"],
                            "gated_p1_trade_ticks": gate_stats["p1"],
                            "baseline_p5_trade_ticks": base_stats["p5"],
                            "gated_p5_trade_ticks": gate_stats["p5"],
                            "baseline_mean_trade_ticks": base_stats["mean_pnl_ticks"],
                            "gated_mean_trade_ticks": gate_stats["mean_pnl_ticks"],
                        }
                    )

                    day_dir = out_dir_w / f"{instrument}_{day}"
                    day_dir.mkdir(parents=True, exist_ok=True)
                    trades_base.to_csv(day_dir / "trades_baseline.csv", index=False)
                    trades_gate.to_csv(day_dir / "trades_gated.csv", index=False)

                    print(
                        f"[{day_index}/{day_count}] {instrument} {day} W={gate_lookback_bars} mode={gate_mode} "
                        f"baseline trades={len(trades_base)} pnl={base_stats['total_pnl_ticks']:.2f} "
                        f"gated trades={len(trades_gate)} pnl={gate_stats['total_pnl_ticks']:.2f} "
                        f"skipped={skipped_gate} block_rate={gate_stats['block_rate']:.2%} coverage={gate_stats['coverage']:.2%}",
                        flush=True,
                    )
                    print(
                        f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode} skip_reasons={skip_gate}",
                        flush=True,
                    )
                    expected_skipped = len(trades_base) - len(trades_gate)
                    if skipped_gate != expected_skipped:
                        print(
                            f"{instrument} {day} W={gate_lookback_bars} mode={gate_mode} skip_mismatch skipped={skipped_gate} "
                            f"expected={expected_skipped} signals={total_signals_gate} reasons={skip_gate}",
                            flush=True,
                        )
                    assert skipped_gate == expected_skipped, (
                        f"Skipped mismatch: skipped={skipped_gate} expected={expected_skipped} "
                        f"symbol={instrument} day={day} W={gate_lookback_bars} mode={gate_mode} reasons={skip_gate}"
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

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(out_dir / "summary_sweep.csv", index=False)
    agg_rows = []
    if not sweep_df.empty:
        for (w, mode), g in sweep_df.groupby(["W", "gate_mode"], sort=False):
            days_count = int(g.shape[0])
            pnl_imp = g["pnl_improvement_ticks"]
            imp_per_block = g["improvement_per_blocked"]
            dd_imp = g["dd_improvement"]
            worst_imp = g["baseline_worst_trade_ticks"] - g["gated_worst_trade_ticks"]
            p1_imp = g["baseline_p1_trade_ticks"] - g["gated_p1_trade_ticks"]
            p5_imp = g["baseline_p5_trade_ticks"] - g["gated_p5_trade_ticks"]
            agg_rows.append(
                {
                    "W": w,
                    "gate_mode": mode,
                    "days_count": days_count,
                    "pct_days_improved": float(np.mean(pnl_imp > 0)) if days_count else 0.0,
                    "median_pnl_improvement_ticks": float(np.median(pnl_imp)) if days_count else np.nan,
                    "mean_pnl_improvement_ticks": float(np.mean(pnl_imp)) if days_count else np.nan,
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

    if skipped_days:
        skipped_path = out_dir / "skipped_days.txt"
        with open(skipped_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_days))


if __name__ == "__main__":
    main()
