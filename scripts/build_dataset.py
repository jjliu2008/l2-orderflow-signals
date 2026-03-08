"""
Build fixed-interval feature rows + labels from Databento DBN (MBP-10 + TRADES).

Output:
  data/processed/instrument=ES/date=YYYY-MM-DD/features_labels.parquet

Env vars:
  INPUT_DBN            path to .dbn file or directory of .dbn files (optional if INPUT_MBP_DBN/TRADES set)
  INPUT_MBP_DBN        path to MBP-10 .dbn file (optional)
  INPUT_TRADES_DBN     path to TRADES .dbn file (optional)
  OUTPUT_DIR           output root (default: data/processed)
  INSTRUMENT           instrument symbol (default: ES)
  FRONT_MONTH          optional exact symbol to keep (e.g., ESU5); others skipped (requires symbol field)
  FRONT_MONTH_RAW      optional raw symbol in DBN metadata (e.g., ESZ5) to keep
  FRONT_MONTH_ID       optional instrument_id to keep (fastest filter)
  STEP_MS              emit interval in ms (default: 100)
  HORIZON_MS           label horizon in ms (default: 1000)
  NEUTRAL_BAND_BPS     neutral band in bps for direction label (default: 0.0)
  REPRICE_BPS          repricing event threshold in bps (default: 4.0)
  FLOW_WINDOW          rolling window (rows) for flow/lag features (default: 20)
  LFP_HORIZONS         comma-separated horizons for LFP AE labels (default: 10,20,50)
  LFP_AE_X_PCT         percentile for AE threshold X (default: 0.90)
  LFP_AE_X_MODE        global|day (default: day)
  LFP_LAMBDA_FAST      rolling window for lambda fast (default: 20)
  LFP_LAMBDA_SLOW      rolling window for lambda slow (default: 100)
  LFP_STRESS_WINDOW    rolling window for stress/imbalance/spread (default: 50)
  LFP_TICK_SIZE        tick size override (default: 0.25 for ES)
  SESSION_START        optional session start (HH:MM:SS) in exchange time (treated as UTC if SESSION_TZ unset)
  SESSION_END          optional session end (HH:MM:SS) in exchange time (treated as UTC if SESSION_TZ unset)
  SESSION_TZ           optional IANA timezone for session filtering (e.g., America/Chicago)
  EMIT_EMPTY           if "1", emit empty intervals even with no events (default: 0)
  DATE_FILTER         optional YYYY-MM-DD to write a single day only
  START_DATE          optional YYYY-MM-DD inclusive lower bound for output days
  END_DATE            optional YYYY-MM-DD inclusive upper bound for output days
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import datetime

import numpy as np
import pandas as pd
import heapq


MONTH_CODE = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}


@dataclass
class TradeAgg:
    volume: float = 0.0
    signed_volume: float = 0.0
    count: int = 0

    def reset(self):
        self.volume = 0.0
        self.signed_volume = 0.0
        self.count = 0


def _iter_dbn_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("*.dbn"))
    if path.is_file() and path.suffix.lower() == ".dbn":
        return [path]
    raise FileNotFoundError(f"INPUT_DBN not found or not a .dbn file/dir: {path}")


def _split_mbp_trades(paths: List[Path]) -> tuple[List[Path], List[Path]]:
    mbp = [p for p in paths if "mbp" in p.name.lower()]
    trades = [p for p in paths if "trade" in p.name.lower()]
    return mbp, trades


def _trade_side_sign(side_val) -> float:
    # Databento side is often 1=buy, 2=sell; some schemas use 'B'/'S'.
    if side_val in (1, "B", "b", "BUY", "buy", "BID", "bid"):
        return 1.0
    if side_val in (2, "S", "s", "SELL", "sell", "ASK", "ask", "A"):
        return -1.0
    return 0.0


def _vwap(prices: List[float], sizes: List[float]) -> Optional[float]:
    tot = sum(sizes)
    if tot == 0:
        return None
    return sum(p * s for p, s in zip(prices, sizes)) / tot


def _sweep_cost(prices: List[float], sizes: List[float], mid: float, qty: float = 1.0) -> Optional[float]:
    remaining = qty
    cost = 0.0
    filled = 0.0
    for p, s in zip(prices, sizes):
        take = min(s, remaining)
        cost += take * p
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled == 0 or mid == 0:
        return None
    avg_price = cost / filled
    return (avg_price - mid) / mid


def _sweep_cost_ticks(
    prices: List[float],
    sizes: List[float],
    best_price: float,
    tick_size: float,
    qty: float = 5.0,
) -> Optional[float]:
    """Slippage in ticks from best_price to fill qty lots through the book.

    For a buy sweep: prices are ask levels ascending, best_price = ask[0].
    For a sell sweep: prices are bid levels descending, best_price = bid[0].
    Returns a non-negative number of ticks of slippage (0.0 = filled at best).
    """
    if tick_size <= 0 or not np.isfinite(best_price):
        return None
    remaining = qty
    cost = 0.0
    filled = 0.0
    for p, s in zip(prices, sizes):
        if not np.isfinite(p) or s <= 0:
            continue
        take = min(s, remaining)
        cost += take * p
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled == 0:
        return None
    avg_price = cost / filled
    return abs(avg_price - best_price) / tick_size


def _price_to_float(raw: int) -> float:
    import databento as db
    if raw == db.UNDEF_PRICE:
        return np.nan
    return raw / db.FIXED_PRICE_SCALE


def _parse_mbp10(rec, tick_size: float = 0.25) -> Dict[str, float]:
    # Extract top-10 book arrays from MBP-10 record levels.
    levels = getattr(rec, "levels", [])
    if not levels:
        return {}
    bid_px = [_price_to_float(lvl.bid_px) for lvl in levels[:10]]
    ask_px = [_price_to_float(lvl.ask_px) for lvl in levels[:10]]
    bid_sz = [float(lvl.bid_sz) for lvl in levels[:10]]
    ask_sz = [float(lvl.ask_sz) for lvl in levels[:10]]

    bid0 = bid_px[0]
    ask0 = ask_px[0]
    if not np.isfinite(bid0) or not np.isfinite(ask0):
        return {}
    if ask0 < bid0:
        # Ignore crossed book updates to avoid invalid top-of-book snapshots.
        return {}

    mid = (bid0 + ask0) / 2
    spread = ask0 - bid0
    top_bid_depth = bid_sz[0]
    top_ask_depth = ask_sz[0]
    denom = top_bid_depth + top_ask_depth
    order_book_imbalance = (top_bid_depth / denom) if denom else np.nan

    depth_bid_top5 = sum(bid_sz[:5])
    depth_ask_top5 = sum(ask_sz[:5])
    depth_denom = depth_bid_top5 + depth_ask_top5
    depth_imbalance_top5 = (depth_bid_top5 - depth_ask_top5) / depth_denom if depth_denom else np.nan

    vwap_bid = _vwap(bid_px[:5], bid_sz[:5])
    vwap_ask = _vwap(ask_px[:5], ask_sz[:5])
    book_slope = (vwap_ask - vwap_bid) / mid if (vwap_bid and vwap_ask and mid) else np.nan

    sweep_cost_buy1 = _sweep_cost(ask_px, ask_sz, mid=mid, qty=1.0)
    sweep_cost_sell1 = _sweep_cost(bid_px, bid_sz, mid=mid, qty=1.0)

    # ── New level-2-10 features ──────────────────────────────────────────────

    # Per-level bid/ask sizes for levels 2-10 (level 1 already stored above).
    n_levels = len(bid_sz)
    per_level: Dict[str, float] = {}
    for i in range(1, 10):  # 0-indexed: level i+1
        per_level[f"bid_size_{i + 1}"] = bid_sz[i] if i < n_levels else np.nan
        per_level[f"ask_size_{i + 1}"] = ask_sz[i] if i < n_levels else np.nan

    # Depth shape ratio: fraction of top-5 cumulative depth sitting at level 1.
    depth_shape_ratio_bid = top_bid_depth / depth_bid_top5 if depth_bid_top5 > 0 else np.nan
    depth_shape_ratio_ask = top_ask_depth / depth_ask_top5 if depth_ask_top5 > 0 else np.nan

    # Book depth slope: OLS slope of size vs level index over levels 0-9.
    # For fixed x=[0..9]: denom = 10*285 - 45^2 = 825.
    _N = min(10, n_levels)
    _sum_y_bid = sum(bid_sz[:_N])
    _sum_y_ask = sum(ask_sz[:_N])
    _sum_xy_bid = sum(i * bid_sz[i] for i in range(_N))
    _sum_xy_ask = sum(i * ask_sz[i] for i in range(_N))
    _sx = sum(range(_N))
    _sx2 = sum(i * i for i in range(_N))
    _slope_denom = _N * _sx2 - _sx * _sx
    book_depth_slope_bid = (_N * _sum_xy_bid - _sx * _sum_y_bid) / _slope_denom if _slope_denom else np.nan
    book_depth_slope_ask = (_N * _sum_xy_ask - _sx * _sum_y_ask) / _slope_denom if _slope_denom else np.nan

    # Sweep cost in ticks: how many ticks above/below best to fill N lots.
    sweep_cost_buy1_ticks = _sweep_cost_ticks(ask_px[:5], ask_sz[:5], best_price=ask0, tick_size=tick_size, qty=1.0)
    sweep_cost_sell1_ticks = _sweep_cost_ticks(bid_px[:5], bid_sz[:5], best_price=bid0, tick_size=tick_size, qty=1.0)
    sweep_cost_buy5_ticks = _sweep_cost_ticks(ask_px[:5], ask_sz[:5], best_price=ask0, tick_size=tick_size, qty=5.0)
    sweep_cost_sell5_ticks = _sweep_cost_ticks(bid_px[:5], bid_sz[:5], best_price=bid0, tick_size=tick_size, qty=5.0)

    return {
        "bid_price_1": bid0,
        "ask_price_1": ask0,
        "bid_size_1": top_bid_depth,
        "ask_size_1": top_ask_depth,
        "mid": mid,
        "spread": spread,
        "top_bid_depth": top_bid_depth,
        "top_ask_depth": top_ask_depth,
        "order_book_imbalance": order_book_imbalance,
        "depth_bid_top5": depth_bid_top5,
        "depth_ask_top5": depth_ask_top5,
        "depth_imbalance_top5": depth_imbalance_top5,
        "book_slope_top5": book_slope,
        "sweep_cost_buy1": sweep_cost_buy1,
        "sweep_cost_sell1": sweep_cost_sell1,
        # Level 2-10
        **per_level,
        "depth_shape_ratio_bid": depth_shape_ratio_bid,
        "depth_shape_ratio_ask": depth_shape_ratio_ask,
        "book_depth_slope_bid": book_depth_slope_bid,
        "book_depth_slope_ask": book_depth_slope_ask,
        "sweep_cost_buy1_ticks": sweep_cost_buy1_ticks,
        "sweep_cost_sell1_ticks": sweep_cost_sell1_ticks,
        "sweep_cost_buy5_ticks": sweep_cost_buy5_ticks,
        "sweep_cost_sell5_ticks": sweep_cost_sell5_ticks,
    }


def _emit_row(ts: pd.Timestamp, symbol: str, book: Dict[str, float], trade_agg: TradeAgg) -> Dict[str, float]:
    row = {
        "Time": ts,
        "Symbol": symbol,
        **book,
        "trade_volume": trade_agg.volume,
        "signed_volume": trade_agg.signed_volume,
        "trade_count": trade_agg.count,
    }
    return row


def _label_rows(
    df: pd.DataFrame,
    horizon_steps: int,
    neutral_band_bps: float,
    reprice_bps: float,
) -> Tuple[pd.DataFrame, float]:
    df = df.copy()
    if horizon_steps < 1:
        horizon_steps = 1
    mid = df["mid"].astype(float)
    future = mid.shift(-horizon_steps)
    fwd_ret = (future - mid) / mid
    missing_pct = float(fwd_ret.isna().mean())
    band = neutral_band_bps / 1e4
    label = (fwd_ret > band).astype(int) - (fwd_ret < -band).astype(int)
    df["fwd_ret"] = fwd_ret
    df["label"] = label
    # Cost-aware label (relative)
    if {"spread", "sweep_cost_buy1", "sweep_cost_sell1", "mid"}.issubset(df.columns):
        spread_rel = df["spread"] / df["mid"]
        sweep_mag = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
        cost_proxy = spread_rel / 2 + sweep_mag
        df["fwd_ret_net"] = df["fwd_ret"] - cost_proxy
    # Favorable excursion label (directional)
    if "label" in df.columns:
        dir_sign = df["label"].astype(float).clip(-1, 1)
        fwd = mid.shift(-1)
        rel = (fwd - mid) / mid
        rel_signed = rel * dir_sign
        fav = rel_signed.rolling(horizon_steps, min_periods=horizon_steps).max()
        df["fav_excursion"] = fav
    # Repricing event: directional jump relative to flow sign
    if "signed_volume" in df.columns:
        flow_sign = np.sign(df["signed_volume"].astype(float))
        fwd = mid.shift(-1)
        rel = (fwd - mid) / mid
        rel_signed = rel * flow_sign
        thresh = reprice_bps / 1e4
        max_rel_signed = rel_signed.rolling(horizon_steps, min_periods=horizon_steps).max()
        df["repriced_within_H"] = (max_rel_signed >= thresh).astype(int)
    return df.dropna(subset=["fwd_ret", "label"]), missing_pct


def _add_derived_features(df: pd.DataFrame, step_ms: int, window: int) -> pd.DataFrame:
    df = df.copy()
    if "signed_volume" in df.columns:
        df["abs_signed_volume"] = df["signed_volume"].abs()
    if {"signed_volume", "trade_volume"}.issubset(df.columns):
        denom = df["trade_volume"].astype(float).replace(0.0, np.nan)
        df["trade_imbalance_ratio"] = df["signed_volume"] / denom
    if "trade_volume" in df.columns:
        per_sec = 1000.0 / max(step_ms, 1)
        df["flow_intensity"] = df["trade_volume"] * per_sec
        df["flow_intensity_roll"] = df["flow_intensity"].rolling(window, min_periods=1).mean()
        df["flow_accel"] = df["flow_intensity_roll"].diff()
    if "mid" in df.columns:
        prev = df["mid"].shift(1)
        df["dmid_bps"] = ((df["mid"] - prev) / prev) * 1e4
        prev_w = df["mid"].shift(window)
        df["dmid_bps_roll"] = ((df["mid"] - prev_w) / prev_w) * 1e4
    if "abs_signed_volume" in df.columns and "dmid_bps" in df.columns:
        eps = 1e-9
        df["impact_per_flow"] = df["dmid_bps"] / (df["abs_signed_volume"] + eps)
    if "flow_intensity" in df.columns and "dmid_bps" in df.columns:
        eps = 1e-9
        df["lag_score"] = df["flow_intensity"] / (df["dmid_bps"].abs() + eps)
    if {"depth_bid_top5", "depth_ask_top5"}.issubset(df.columns):
        df["depth_total_top5"] = df["depth_bid_top5"] + df["depth_ask_top5"]
    if {"spread", "mid"}.issubset(df.columns):
        spread_bps = (df["spread"] / df["mid"]) * 1e4
        df["spread_stability"] = spread_bps.rolling(window, min_periods=1).std()
    return df


def _day_health_stats(df_day: pd.DataFrame) -> Dict[str, float]:
    bid = pd.to_numeric(df_day["bid_price_1"], errors="coerce").to_numpy()
    ask = pd.to_numeric(df_day["ask_price_1"], errors="coerce").to_numpy()
    mid = 0.5 * (bid + ask)
    mid_diff = np.diff(mid)
    mid_change_pct = float(np.mean(np.isfinite(mid_diff) & (mid_diff != 0))) if mid_diff.size else 0.0
    return {
        "n_rows": int(len(df_day)),
        "nunique_bid": int(pd.Series(bid).nunique(dropna=True)),
        "nunique_ask": int(pd.Series(ask).nunique(dropna=True)),
        "nunique_mid": int(pd.Series(mid).nunique(dropna=True)),
        "mid_change_pct": mid_change_pct,
    }


def _tick_size_for_symbol(symbol: str, env_tick_size: float) -> float:
    if env_tick_size > 0:
        return env_tick_size
    if symbol.upper().startswith("ES"):
        return 0.25
    return float("nan")


def _auto_front_month_raw(
    instrument: str,
    ts: pd.Timestamp,
    mappings: dict,
) -> str | None:
    if not mappings:
        return None
    month_code = MONTH_CODE.get(ts.month)
    if not month_code:
        return None
    year_digit = str(ts.year % 10)
    candidate = f"{instrument}{month_code}{year_digit}"
    if candidate in mappings:
        return candidate
    # Fallback: look for outrights with same month code and year digit.
    matches = [
        k for k in mappings.keys()
        if k.startswith(instrument) and "-" not in k and k.endswith(year_digit) and k[-2] == month_code
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _validate_book_sanity(
    df: pd.DataFrame,
    step_ms: int,
    tick_size: float,
    jump_bound_points: float = 20.0,
) -> None:
    bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
    spread = ask - bid
    bad_spread = spread < -1e-9
    if bad_spread.any():
        sample = df.loc[bad_spread, ["Time", "bid_price_1", "ask_price_1"]].head(5)
        raise ValueError(f"Negative spread detected; sample:\n{sample}")
    if np.isfinite(tick_size) and tick_size > 0:
        ratio = spread / tick_size
        off = (ratio - np.round(ratio)).abs()
        eps = 1e-6 * max(1.0, tick_size)
        bad_tick = off > eps
        if bad_tick.any():
            sample = df.loc[bad_tick, ["Time", "bid_price_1", "ask_price_1"]].head(5)
            raise ValueError(f"Spread not multiple of tick size; sample:\n{sample}")
    bound = jump_bound_points * (step_ms / 100.0)
    bid_jump = bid.diff().abs()
    ask_jump = ask.diff().abs()
    if bid_jump.max() > bound or ask_jump.max() > bound:
        bad_rows = (bid_jump > bound) | (ask_jump > bound)
        sample = df.loc[bad_rows, ["Time", "bid_price_1", "ask_price_1"]].head(5)
        print(
            f"Warning: large bid/ask jump detected (>{bound:.2f} points per bar). "
            f"Sample:\n{sample}"
        )


def _lambda_feature(q: pd.Series, dp: pd.Series, group_keys, window: int, eps: float) -> pd.Series:
    qdp = q * dp
    numer = qdp.groupby(group_keys).transform(lambda s: s.rolling(window, min_periods=window).sum())
    denom = (q * q).groupby(group_keys).transform(lambda s: s.rolling(window, min_periods=window).sum())
    return numer / (denom + eps)


def _add_lfp_features_labels(
    df: pd.DataFrame,
    horizons: list[int],
    x_pct: float,
    x_mode: str,
    win_fast: int,
    win_slow: int,
    stress_window: int,
) -> pd.DataFrame:
    df = df.copy()
    df["dmid_bps"] = df.get("dmid_bps", ((df["mid"] - df["mid"].shift(1)) / df["mid"].shift(1)) * 1e4)
    df["signed_volume"] = pd.to_numeric(df.get("signed_volume", 0.0), errors="coerce").fillna(0.0)
    group_keys = [df["Symbol"], df["Time"].dt.date]

    lam_fast = _lambda_feature(df["signed_volume"], df["dmid_bps"], group_keys, win_fast, eps=1e-9)
    lam_slow = _lambda_feature(df["signed_volume"], df["dmid_bps"], group_keys, win_slow, eps=1e-9)
    df[f"kyle_lambda_{win_fast}"] = lam_fast
    df[f"kyle_lambda_{win_slow}"] = lam_slow

    if "flow_intensity_roll" in df.columns and "depth_total_top5" in df.columns:
        df["stress_ratio"] = df["flow_intensity_roll"] / (df["depth_total_top5"].replace(0, np.nan))
    if "order_book_imbalance" in df.columns:
        df["imbalance_delta"] = df["order_book_imbalance"].diff()
        df["imbalance_vol"] = df["order_book_imbalance"].rolling(stress_window, min_periods=stress_window).std()
    if "spread" in df.columns and "mid" in df.columns:
        spread_bps = (df["spread"] / df["mid"]) * 1e4
        df["spread_bps"] = spread_bps
        df["spread_bps_vol"] = spread_bps.rolling(stress_window, min_periods=stress_window).std()

    mid = pd.to_numeric(df["mid"], errors="coerce")
    for h in horizons:
        future_min = mid.groupby(group_keys).transform(lambda s: s.shift(-1).rolling(h, min_periods=h).min())
        future_max = mid.groupby(group_keys).transform(lambda s: s.shift(-1).rolling(h, min_periods=h).max())
        ae_buy = ((future_min - mid) / mid) * 1e4
        ae_sell = ((future_max - mid) / mid) * 1e4
        df[f"ae_buy_bps_{h}"] = ae_buy
        df[f"ae_sell_bps_{h}"] = ae_sell

        if x_mode == "day":
            def _day_x(g: pd.DataFrame) -> float:
                vals = pd.concat([g[f"ae_buy_bps_{h}"].abs(), g[f"ae_sell_bps_{h}"].abs()])
                return float(vals.quantile(x_pct))
            x_by_day = df.groupby(df["Time"].dt.date).apply(_day_x)
            x_val = df["Time"].dt.date.map(x_by_day)
        else:
            vals = pd.concat([ae_buy.abs(), ae_sell.abs()])
            x_val = float(vals.quantile(x_pct))

        df[f"lfp_event_buy_{h}"] = ae_buy <= -x_val
        df[f"lfp_event_sell_{h}"] = ae_sell >= x_val

    return df


def _parse_time(value: str) -> Optional[datetime.time]:
    if not value:
        return None
    return pd.to_datetime(value).time()


def _in_session(
    ts: pd.Timestamp,
    start: Optional[datetime.time],
    end: Optional[datetime.time],
    tz: Optional[str],
) -> bool:
    if start is None and end is None:
        return True
    if tz:
        ts = ts.tz_convert(tz)
    t = ts.time()
    if start and end:
        if start <= end:
            return start <= t <= end
        return t >= start or t <= end
    if start:
        return t >= start
    return t <= end


def main():
    try:
        import databento as db  # type: ignore
    except ImportError as exc:
        raise ImportError("Databento Python package is required. Install with `pip install databento`.") from exc

    input_path = Path(os.environ.get("INPUT_DBN", "")).expanduser().resolve() if os.environ.get("INPUT_DBN") else None
    input_mbp = Path(os.environ.get("INPUT_MBP_DBN", "")).expanduser().resolve() if os.environ.get("INPUT_MBP_DBN") else None
    input_trades = Path(os.environ.get("INPUT_TRADES_DBN", "")).expanduser().resolve() if os.environ.get("INPUT_TRADES_DBN") else None
    if input_path is None and input_mbp is None and input_trades is None:
        raise FileNotFoundError("Set INPUT_DBN or INPUT_MBP_DBN/INPUT_TRADES_DBN.")

    output_root = Path(os.environ.get("OUTPUT_DIR", "data/processed")).expanduser().resolve()
    instrument = os.environ.get("INSTRUMENT", "ES")
    front_month = os.environ.get("FRONT_MONTH", "").strip()
    front_month_raw = os.environ.get("FRONT_MONTH_RAW", "").strip()
    front_month_id = os.environ.get("FRONT_MONTH_ID", "").strip()
    step_ms = int(os.environ.get("STEP_MS", "100"))
    horizon_ms = int(os.environ.get("HORIZON_MS", "1000"))
    neutral_band_bps = float(os.environ.get("NEUTRAL_BAND_BPS", "0.0"))
    reprice_bps = float(os.environ.get("REPRICE_BPS", "4.0"))
    flow_window = int(os.environ.get("FLOW_WINDOW", "20"))
    lfp_horizons = [int(x.strip()) for x in os.environ.get("LFP_HORIZONS", "10,20,50").split(",") if x.strip()]
    lfp_x_pct = float(os.environ.get("LFP_AE_X_PCT", "0.90"))
    lfp_x_mode = os.environ.get("LFP_AE_X_MODE", "day").strip().lower()
    lfp_lambda_fast = int(os.environ.get("LFP_LAMBDA_FAST", "20"))
    lfp_lambda_slow = int(os.environ.get("LFP_LAMBDA_SLOW", "100"))
    lfp_stress_window = int(os.environ.get("LFP_STRESS_WINDOW", "50"))
    lfp_tick_size = float(os.environ.get("LFP_TICK_SIZE", "0"))
    # Resolve tick_size early so _parse_mbp10 can compute sweep-cost-in-ticks.
    tick_size = _tick_size_for_symbol(instrument, lfp_tick_size)
    emit_empty = os.environ.get("EMIT_EMPTY", "0").strip() in {"1", "true", "yes", "y"}
    session_start = _parse_time(os.environ.get("SESSION_START", "").strip())
    session_end = _parse_time(os.environ.get("SESSION_END", "").strip())
    session_tz = os.environ.get("SESSION_TZ", "").strip() or None
    date_filter = os.environ.get("DATE_FILTER", "").strip()
    start_date = os.environ.get("START_DATE", "").strip()
    end_date = os.environ.get("END_DATE", "").strip()

    files = _iter_dbn_files(input_path) if input_path is not None else []
    if input_mbp is not None:
        files.append(input_mbp)
    if input_trades is not None:
        files.append(input_trades)
    files = list({p for p in files})
    mbp_files, trade_files = _split_mbp_trades(files)
    if not mbp_files and not trade_files:
        raise FileNotFoundError("No .dbn files found for MBP/TRADES.")

    # Build iterators per file type and merge by ts_event
    def _iter_store(path: Path):
        store = db.DBNStore.from_file(path)
        for rec in store:
            yield rec

    iterators = []
    for p in mbp_files:
        iterators.append(_iter_store(p))
    for p in trade_files:
        iterators.append(_iter_store(p))

    # Merge iterators by timestamp
    heap = []
    for idx, it in enumerate(iterators):
        try:
            rec = next(it)
            heapq.heappush(heap, (rec.ts_event, idx, rec, it))
        except StopIteration:
            continue

    if not heap:
        raise ValueError("No records found in DBN input.")

    rows: List[Dict[str, float]] = []
    book: Dict[str, float] = {}
    agg = TradeAgg()
    next_emit_ns: Optional[int] = None
    allowed_ids: Optional[set[int]] = None

    meta = None
    if mbp_files:
        try:
            meta = db.DBNStore.from_file(mbp_files[0]).metadata
        except Exception:
            meta = None
    elif trade_files:
        try:
            meta = db.DBNStore.from_file(trade_files[0]).metadata
        except Exception:
            meta = None

    if front_month_id:
        try:
            allowed_ids = {int(front_month_id)}
        except ValueError:
            raise ValueError("FRONT_MONTH_ID must be an integer instrument_id.")
    elif front_month_raw:
        # Build allowed instrument_id set from DBN metadata mappings.
        if meta and hasattr(meta, "mappings"):
            ids = set()
            for raw, intervals in meta.mappings.items():
                if raw != front_month_raw:
                    continue
                for interval in intervals:
                    try:
                        ids.add(int(interval.get("symbol")))
                    except Exception:
                        continue
            allowed_ids = ids if ids else None
        if not allowed_ids:
            raise ValueError(f"FRONT_MONTH_RAW={front_month_raw} not found in DBN metadata mappings.")
    elif meta and hasattr(meta, "mappings"):
        ts0 = pd.to_datetime(heap[0][0], unit="ns", utc=True)
        auto_raw = _auto_front_month_raw(instrument, ts0, meta.mappings)
        if auto_raw:
            ids = set()
            for raw, intervals in meta.mappings.items():
                if raw != auto_raw:
                    continue
                for interval in intervals:
                    try:
                        ids.add(int(interval.get("symbol")))
                    except Exception:
                        continue
            allowed_ids = ids if ids else None
            if allowed_ids:
                print(f"Auto-selected FRONT_MONTH_RAW={auto_raw} (instrument_id={sorted(allowed_ids)})")
        if not allowed_ids:
            raise ValueError(
                "No FRONT_MONTH/FRONT_MONTH_RAW/FRONT_MONTH_ID provided and auto-detect failed. "
                "Set FRONT_MONTH_RAW (e.g., ESZ5) or FRONT_MONTH_ID."
            )

    while heap:
        ts_event, idx, rec, it = heapq.heappop(heap)
        ts = pd.to_datetime(ts_event, unit="ns", utc=True)

        if not _in_session(ts, session_start, session_end, session_tz):
            book = {}
            agg.reset()
            next_emit_ns = None
            try:
                rec_next = next(it)
                heapq.heappush(heap, (rec_next.ts_event, idx, rec_next, it))
            except StopIteration:
                pass
            continue

        if next_emit_ns is None:
            next_emit_ns = int(ts_event // (step_ms * 1_000_000) * (step_ms * 1_000_000) + step_ms * 1_000_000)

        if allowed_ids is not None:
            if getattr(rec, "instrument_id", None) not in allowed_ids:
                try:
                    rec = next(it)
                    heapq.heappush(heap, (rec.ts_event, idx, rec, it))
                except StopIteration:
                    pass
                continue
        elif front_month:
            rec_symbol = getattr(rec, "symbol", None) or getattr(rec, "instrument", None)
            if rec_symbol and rec_symbol != front_month:
                try:
                    rec = next(it)
                    heapq.heappush(heap, (rec.ts_event, idx, rec, it))
                except StopIteration:
                    pass
                continue

        # Update book or trades
        if rec.__class__.__name__.lower().startswith("mbp"):
            parsed = _parse_mbp10(rec, tick_size=tick_size)
            if parsed:
                book = parsed
            # MBP-10 embeds trade records inline (action='T'); accumulate them.
            if str(getattr(rec, "action", "")) == "T":
                size = getattr(rec, "size", None)
                side = getattr(rec, "side", None)
                if size is not None:
                    agg.volume += float(size)
                    agg.signed_volume += float(size) * _trade_side_sign(side)
                    agg.count += 1
        elif rec.__class__.__name__.lower().startswith("trade"):
            price = getattr(rec, "price", None)
            size = getattr(rec, "size", None)
            side = getattr(rec, "side", None)
            if price is not None and size is not None:
                if isinstance(price, int):
                    price = _price_to_float(price)
                agg.volume += float(size)
                agg.signed_volume += float(size) * _trade_side_sign(side)
                agg.count += 1

        # Emit rows on clock
        while next_emit_ns is not None and ts_event >= next_emit_ns:
            if book or emit_empty:
                emit_ts = pd.to_datetime(next_emit_ns, unit="ns", utc=True)
                rows.append(_emit_row(emit_ts, instrument, book, agg))
            agg.reset()
            next_emit_ns += step_ms * 1_000_000

        # advance iterator
        try:
            rec_next = next(it)
            heapq.heappush(heap, (rec_next.ts_event, idx, rec_next, it))
        except StopIteration:
            pass

    if not rows:
        print("No rows emitted; check inputs/session filter.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("Time")
    if not {"bid_price_1", "ask_price_1"}.issubset(df.columns):
        raise ValueError("Missing bid_price_1/ask_price_1 in emitted rows; check MBP-10 parsing.")
    bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
    df["mid"] = 0.5 * (bid + ask)
    df["spread"] = ask - bid
    _validate_book_sanity(df, step_ms=step_ms, tick_size=tick_size, jump_bound_points=20.0)
    df = _add_derived_features(df, step_ms=step_ms, window=flow_window)
    df = _add_lfp_features_labels(
        df,
        horizons=lfp_horizons,
        x_pct=lfp_x_pct,
        x_mode=lfp_x_mode,
        win_fast=lfp_lambda_fast,
        win_slow=lfp_lambda_slow,
        stress_window=lfp_stress_window,
    )
    horizon_steps = max(1, horizon_ms // step_ms)
    df, missing_pct = _label_rows(
        df,
        horizon_steps=horizon_steps,
        neutral_band_bps=neutral_band_bps,
        reprice_bps=reprice_bps,
    )

    # Basic summary diagnostics
    if {"spread", "mid"}.issubset(df.columns):
        spread_bps = (df["spread"] / df["mid"]) * 1e4
        q = spread_bps.quantile([0.05, 0.5, 0.95]).to_dict()
        print(f"Spread bps quantiles (5/50/95): {q}")
    if {"spread", "mid", "sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df.columns):
        sweep_mag = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
        cost_bps = ((df["spread"] / df["mid"]) / 2 + sweep_mag) * 1e4
        q = cost_bps.quantile([0.05, 0.5, 0.95]).to_dict()
        print(f"Cost proxy bps quantiles (5/50/95): {q}")
    print(f"Label missing rate: {missing_pct:.2%}")

    # Dtype tightening for Parquet size/speed
    float32_cols = [
        "bid_price_1",
        "ask_price_1",
        "mid",
        "spread",
        "order_book_imbalance",
        "depth_imbalance_top5",
        "book_slope_top5",
        "sweep_cost_buy1",
        "sweep_cost_sell1",
        "fwd_ret",
        "fwd_ret_net",
        "fav_excursion",
        "trade_volume",
        "signed_volume",
        # Level 2-10 derived features
        "depth_shape_ratio_bid",
        "depth_shape_ratio_ask",
        "book_depth_slope_bid",
        "book_depth_slope_ask",
        "sweep_cost_buy1_ticks",
        "sweep_cost_sell1_ticks",
        "sweep_cost_buy5_ticks",
        "sweep_cost_sell5_ticks",
    ]
    int32_cols = [
        "trade_count",
        "bid_size_1",
        "ask_size_1",
        "top_bid_depth",
        "top_ask_depth",
        "depth_bid_top5",
        "depth_ask_top5",
        # Per-level bid/ask sizes for levels 2-10
        *[f"bid_size_{i}" for i in range(2, 11)],
        *[f"ask_size_{i}" for i in range(2, 11)],
    ]
    int8_cols = ["label"]
    for col in float32_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in int32_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
    for col in int8_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    # Partition by date (optionally filter)
    df["date"] = df["Time"].dt.date.astype(str)
    if date_filter:
        df = df[df["date"] == date_filter]
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    for date, df_day in df.groupby("date"):
        out_dir = output_root / f"instrument={instrument}" / f"date={date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "features_labels.parquet"
        health = _day_health_stats(df_day)
        health_path = out_dir / "health.json"
        with open(health_path, "w", encoding="utf-8") as f:
            json.dump(health, f, indent=2)
        if (
            health["nunique_bid"] < 50
            or health["nunique_ask"] < 50
            or health["mid_change_pct"] < 0.005
        ):
            raise RuntimeError(
                f"Day {date} failed health checks: {health}. "
                "Refusing to write processed output."
            )
        df_day.drop(columns=["date"]).to_parquet(out_path, index=False)
        print(f"Wrote {len(df_day)} rows to {out_path}")


if __name__ == "__main__":
    main()
