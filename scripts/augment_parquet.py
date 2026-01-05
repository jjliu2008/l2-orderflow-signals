"""
Augment existing Parquet files with LFP labels and lambda features.

Env vars:
  INPUT_DIR            root directory with existing parquet (default: data/processed)
  OUTPUT_DIR           output root (default: data/processed_lfp)
  HORIZON_LIST         comma-separated horizons in rows (default: 10,20,50)
  P_LFP_PCT            percentile for AE threshold X (default: 0.90)
  P_LFP_LAMBDA_WIN     rolling window for lambda (default: 200)
  P_LFP_FLOW_WIN       rolling window for flow/stress (default: 50)
  P_LFP_DEPTH_DROP     depth drop threshold for gap_flag (default: 0.5)
  LFP_STRUCT_H_MS      structural failure horizon in ms (default: 2000)
  LFP_STRUCT_X_TICKS   spread widening threshold in ticks (default: 2)
  LFP_STRUCT_Y_PCT     depth withdrawal threshold in percent (default: 60)
  LFP_STRUCT_R_MS      persistence window in ms (default: 300)
  LFP_STRUCT_DMIN      minimum baseline depth for B condition (default: 100)
  LFP_STRUCT_B_MODE    OR | AND_CANY (default: AND_CANY)
  LFP_STRUCT_G_TICKS   gap threshold in ticks (default: 4)
  LFP_STRUCT_STEP_MS   override bar step size in ms (default: infer per file)
  LFP_STRUCT_AUTOCAL   set to 1 to calibrate X/Y to target base rate (default: 0)
  LFP_STRUCT_TGT_MIN   target min base rate (default: 0.01)
  LFP_STRUCT_TGT_MAX   target max base rate (default: 0.05)
  LFP_STRUCT_X_GRID    comma-separated X ticks grid (default: 1,2,3)
  LFP_STRUCT_Y_GRID    comma-separated Y pct grid (default: 40,50,60,70)
  LFP_STRUCT_G_GRID    comma-separated G ticks grid (default: 2,3,4,5,6)
  LFP_STRUCT_DMIN_GRID comma-separated DMIN grid (default: 50,100,200,300)
  LFP_STRUCT_B_MODE_GRID comma-separated B_MODE grid (default: AND_CANY,OR)
  LFP_TICK_SIZE        tick size override (default: 0.25 when Symbol starts with ES)
  EPS                 small epsilon for denom (default: 1e-9)
  MID_MIN             optional min mid filter (default: 0 = disabled)
  MID_MAX             optional max mid filter (default: 0 = disabled)
  OVERWRITE            set to 1 to overwrite existing output files (default: 0)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def _iter_parquet_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    return df


def _compute_dmid_bps(df: pd.DataFrame, group_keys: Iterable) -> pd.Series:
    if "dmid_bps" in df.columns:
        return pd.to_numeric(df["dmid_bps"], errors="coerce")
    mid = pd.to_numeric(df["mid"], errors="coerce")
    prev = mid.groupby(group_keys).transform(lambda s: s.shift(1))
    return ((mid - prev) / prev) * 1e4


def _compute_signed_volume(df: pd.DataFrame) -> pd.Series:
    if "signed_volume" in df.columns:
        return pd.to_numeric(df["signed_volume"], errors="coerce").fillna(0.0)
    if "trade_volume" in df.columns and "dmid_bps" in df.columns:
        return pd.to_numeric(df["trade_volume"], errors="coerce").fillna(0.0) * np.sign(df["dmid_bps"])
    return pd.Series(0.0, index=df.index)


def _lambda_feature(q: pd.Series, dp: pd.Series, group_keys: Iterable, window: int, eps: float) -> pd.Series:
    qdp = q * dp
    numer = qdp.groupby(group_keys).transform(lambda s: s.rolling(window, min_periods=window).sum())
    denom = (q * q).groupby(group_keys).transform(lambda s: s.rolling(window, min_periods=window).sum())
    return numer / (denom + eps), denom


def _forward_rolling_min(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series.copy()
    rev = series.iloc[::-1]
    return rev.rolling(window, min_periods=window).min().iloc[::-1]


def _forward_rolling_max(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series.copy()
    rev = series.iloc[::-1]
    return rev.rolling(window, min_periods=window).max().iloc[::-1]


def _infer_tick_size(spread: pd.Series) -> float:
    vals = pd.to_numeric(spread, errors="coerce")
    vals = vals[(vals > 0) & np.isfinite(vals)]
    if vals.empty:
        return float("nan")
    return float(vals.min())


def _infer_step_ms(times: pd.Series) -> int:
    times = pd.to_datetime(times, utc=True, errors="coerce")
    diffs = times.diff().dt.total_seconds().mul(1000.0)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        return 100
    return int(round(float(diffs.median())))


def _tick_size_for_symbol(symbol: str, env_tick_size: float, spread: pd.Series) -> float:
    if env_tick_size > 0:
        return env_tick_size
    if symbol.upper().startswith("ES"):
        return 0.25
    return _infer_tick_size(spread)


def _prices_to_ticks(prices: pd.Series, tick_size: float) -> np.ndarray:
    sentinel = np.iinfo(np.int64).min
    arr = pd.to_numeric(prices, errors="coerce").to_numpy()
    ticks = np.full(arr.shape, sentinel, dtype=np.int64)
    if not np.isfinite(tick_size) or tick_size <= 0:
        return ticks
    valid = np.isfinite(arr)
    if valid.any():
        ticks[valid] = np.rint(arr[valid] / tick_size).astype(np.int64)
    return ticks


def _gap_flags_for_side(
    ticks: np.ndarray,
    gap_ticks: int,
    horizon_steps: int,
    bars_r: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    sentinel = np.iinfo(np.int64).min
    n = len(ticks)
    any_flags = np.zeros(n, dtype=bool)
    persist_flags = np.zeros(n, dtype=bool)
    if horizon_steps <= 0 or bars_r <= 0:
        return any_flags, persist_flags
    for i in range(n):
        base = ticks[i]
        if base == sentinel:
            continue
        start = i + 1
        end = min(i + 1 + horizon_steps, n)
        if start >= end:
            continue
        future = ticks[start:end]
        valid = future != sentinel
        if not valid.any():
            continue
        cond_full = np.zeros(len(future), dtype=bool)
        if direction == "buy":
            cond_full[valid] = future[valid] <= base - gap_ticks
        else:
            cond_full[valid] = future[valid] >= base + gap_ticks
        any_flags[i] = cond_full.any()
        if len(cond_full) >= bars_r:
            window = np.convolve(cond_full.astype(int), np.ones(bars_r, dtype=int), mode="valid")
            persist_flags[i] = np.any(window == bars_r)
    return any_flags, persist_flags


def _depth_persist_flags(
    depth: np.ndarray,
    y_pct: float,
    dmin: float,
    horizon_steps: int,
    bars_r: int,
) -> np.ndarray:
    n = len(depth)
    flags = np.zeros(n, dtype=bool)
    if horizon_steps <= 0 or bars_r <= 0:
        return flags
    for i in range(n):
        d0 = depth[i]
        if not np.isfinite(d0) or d0 < dmin:
            continue
        start = i + 1
        end = min(i + 1 + horizon_steps, n)
        if start >= end:
            continue
        future = depth[start:end]
        valid = np.isfinite(future)
        if not valid.any():
            continue
        thresh = d0 * (1.0 - y_pct / 100.0)
        cond = np.zeros(len(future), dtype=bool)
        cond[valid] = future[valid] <= thresh
        if len(cond) >= bars_r:
            window = np.convolve(cond.astype(int), np.ones(bars_r, dtype=int), mode="valid")
            flags[i] = np.any(window == bars_r)
    return flags


def _gap_flags_strict(
    ticks: np.ndarray,
    gap_ticks: int,
    horizon_steps: int,
    bars_r: int,
    direction: str,
) -> np.ndarray:
    sentinel = np.iinfo(np.int64).min
    n = len(ticks)
    persist_flags = np.zeros(n, dtype=bool)
    if horizon_steps <= 0 or bars_r <= 0:
        return persist_flags
    for i in range(n):
        base = ticks[i]
        if base == sentinel:
            continue
        start = i + 1
        end = min(i + 1 + horizon_steps, n)
        if start >= end:
            continue
        future = ticks[start:end]
        if len(future) < bars_r:
            continue
        for j in range(0, len(future) - bars_r + 1):
            window = future[j : j + bars_r]
            if (window == sentinel).any():
                continue
            if direction == "buy":
                if window.max() <= base - gap_ticks:
                    persist_flags[i] = True
                    break
            else:
                if window.min() >= base + gap_ticks:
                    persist_flags[i] = True
                    break
    return persist_flags


def _structural_flags_for_day(
    day: pd.DataFrame,
    spread_ticks: int,
    depth_drop_pct: float,
    gap_ticks: int,
    dmin: float,
    tick_size: float,
    horizon_steps: int,
    bars_r: int,
) -> dict:
    spread_day = pd.to_numeric(day["spread"], errors="coerce")
    bid_depth = pd.to_numeric(day["top_bid_depth"], errors="coerce")
    ask_depth = pd.to_numeric(day["top_ask_depth"], errors="coerce")
    bid_px = pd.to_numeric(day["bid_price_1"], errors="coerce")
    ask_px = pd.to_numeric(day["ask_price_1"], errors="coerce")

    window_span = horizon_steps - bars_r + 1
    fwd_spread_min = _forward_rolling_min(spread_day, bars_r)
    fwd_spread_max_of_min = _forward_rolling_max(fwd_spread_min, window_span).shift(-1)

    fwd_bid_depth_max = _forward_rolling_max(bid_depth, bars_r)
    fwd_bid_depth_min_of_max = _forward_rolling_min(fwd_bid_depth_max, window_span).shift(-1)

    fwd_ask_depth_max = _forward_rolling_max(ask_depth, bars_r)
    fwd_ask_depth_min_of_max = _forward_rolling_min(fwd_ask_depth_max, window_span).shift(-1)

    if np.isfinite(tick_size):
        spread_thresh = spread_day + spread_ticks * tick_size
        spread_fail = fwd_spread_max_of_min >= spread_thresh
    else:
        spread_fail = pd.Series(False, index=day.index)

    bid_depth_thresh = bid_depth * (1.0 - depth_drop_pct / 100.0)
    ask_depth_thresh = ask_depth * (1.0 - depth_drop_pct / 100.0)
    bid_depth_fail = fwd_bid_depth_min_of_max <= bid_depth_thresh
    ask_depth_fail = fwd_ask_depth_min_of_max <= ask_depth_thresh
    bid_depth_persist = pd.Series(
        _depth_persist_flags(bid_depth.to_numpy(), depth_drop_pct, dmin, horizon_steps, bars_r),
        index=day.index,
    )
    ask_depth_persist = pd.Series(
        _depth_persist_flags(ask_depth.to_numpy(), depth_drop_pct, dmin, horizon_steps, bars_r),
        index=day.index,
    )

    bid_gap_any = pd.Series(False, index=day.index)
    ask_gap_any = pd.Series(False, index=day.index)
    bid_gap_persist = pd.Series(False, index=day.index)
    ask_gap_persist = pd.Series(False, index=day.index)
    if np.isfinite(tick_size):
        bid_ticks = _prices_to_ticks(bid_px, tick_size)
        ask_ticks = _prices_to_ticks(ask_px, tick_size)
        bid_any, bid_persist = _gap_flags_for_side(
            bid_ticks, gap_ticks, horizon_steps, bars_r, direction="buy"
        )
        ask_any, ask_persist = _gap_flags_for_side(
            ask_ticks, gap_ticks, horizon_steps, bars_r, direction="sell"
        )
        bid_gap_any = pd.Series(bid_any, index=day.index)
        ask_gap_any = pd.Series(ask_any, index=day.index)
        bid_gap_persist = pd.Series(bid_persist, index=day.index)
        ask_gap_persist = pd.Series(ask_persist, index=day.index)

    return {
        "spread_fail": spread_fail,
        "bid_depth_fail": bid_depth_fail,
        "ask_depth_fail": ask_depth_fail,
        "bid_depth_persist": bid_depth_persist,
        "ask_depth_persist": ask_depth_persist,
        "bid_gap_any": bid_gap_any,
        "ask_gap_any": ask_gap_any,
        "bid_gap_persist": bid_gap_persist,
        "ask_gap_persist": ask_gap_persist,
    }


def _add_structural_lfp_labels(
    df: pd.DataFrame,
    horizon_ms: int,
    step_ms: int,
    spread_ticks: int,
    depth_drop_pct: float,
    persist_ms: int,
    gap_ticks: int,
    dmin: float,
    b_mode: str,
    tick_size_env: float,
    autocal: bool,
    tgt_min: float,
    tgt_max: float,
    x_grid: list[int],
    y_grid: list[float],
    g_grid: list[int],
    dmin_grid: list[float],
    b_mode_grid: list[str],
) -> pd.DataFrame:
    """
    Add structural liquidity failure labels for passive makers.

    Failure modes:
      A) Spread failure: spread widens by >= X ticks and stays wide for R ms.
      B) Depth withdrawal: top-of-book depth on fill side drops by >= Y% and stays low for R ms.
      C) Gap persistence: best price moves by >= G ticks and stays through that gap for R ms.
    """
    required = {
        "bid_price_1",
        "ask_price_1",
        "bid_size_1",
        "ask_size_1",
        "top_bid_depth",
        "top_ask_depth",
        "spread",
    }
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(
            f"Missing required columns for structural labels: {missing}. "
            "Regenerate processed data with L1 columns (bid_price_1/ask_price_1)."
        )

    horizon_steps = int(np.ceil(horizon_ms / max(step_ms, 1)))
    bars_r = int(np.ceil(persist_ms / max(step_ms, 1)))
    window_span = horizon_steps - bars_r + 1
    if window_span <= 0:
        df["lfp_structural_fail_buy_{}".format(horizon_ms)] = 0
        df["lfp_structural_fail_sell_{}".format(horizon_ms)] = 0
        return df

    out_buy = pd.Series(False, index=df.index)
    out_sell = pd.Series(False, index=df.index)
    group_keys = [df["Symbol"], df["Time"].dt.date]

    chosen_x = spread_ticks
    chosen_y = depth_drop_pct
    chosen_g = gap_ticks
    chosen_dmin = dmin
    chosen_b_mode = b_mode

    if autocal and x_grid and y_grid:
        best = None
        best_below = None
        best_above = None
        for x in x_grid:
            for y in y_grid:
                for g in g_grid:
                    for dmin_val in dmin_grid:
                        for mode in b_mode_grid:
                            flags_buy = []
                            flags_sell = []
                            for _, idx in df.groupby(group_keys).groups.items():
                                idx = pd.Index(idx)
                                if idx.empty:
                                    continue
                                day = df.loc[idx]
                                symbol = str(day["Symbol"].iloc[0])
                                tick_size = _tick_size_for_symbol(symbol, tick_size_env, day["spread"])
                                flags = _structural_flags_for_day(
                                    day, x, y, g, dmin_val, tick_size, horizon_steps, bars_r
                                )
                                if mode == "AND_CANY":
                                    day_buy = flags["spread_fail"] | (
                                        flags["bid_depth_persist"] & flags["bid_gap_any"]
                                    )
                                    day_sell = flags["spread_fail"] | (
                                        flags["ask_depth_persist"] & flags["ask_gap_any"]
                                    )
                                else:
                                    day_buy = flags["spread_fail"] | flags["bid_depth_persist"] | flags["bid_gap_persist"]
                                    day_sell = flags["spread_fail"] | flags["ask_depth_persist"] | flags["ask_gap_persist"]
                                flags_buy.append(day_buy)
                                flags_sell.append(day_sell)
                            if not flags_buy:
                                continue
                            base = pd.concat(flags_buy + flags_sell).mean()
                            if tgt_min <= base <= tgt_max:
                                best = (x, y, g, dmin_val, mode, base)
                                break
                            if base <= tgt_max:
                                if best_below is None or base > best_below[5]:
                                    best_below = (x, y, g, dmin_val, mode, base)
                            if best_above is None or base < best_above[5]:
                                best_above = (x, y, g, dmin_val, mode, base)
                        if best is not None:
                            break
                    if best is not None:
                        break
                if best is not None:
                    break
            if best is not None:
                break
        if best is None and best_below is not None:
            best = best_below
            print(
                "Warning: no autocal combo met target; using closest below TGT_MAX "
                f"base_rate={best[5]:.3f} target=[{tgt_min:.3f},{tgt_max:.3f}]"
            )
        if best is None and best_above is not None:
            best = best_above
            print(
                "Warning: no autocal combo met or fell below TGT_MAX; using minimum base_rate "
                f"base_rate={best[5]:.3f} target=[{tgt_min:.3f},{tgt_max:.3f}]"
            )
        if best is not None:
            chosen_x, chosen_y, chosen_g, chosen_dmin, chosen_b_mode, base = best
            print(
                f"Autocal LFP struct: X={chosen_x} ticks, Y={chosen_y:.1f}%, "
                f"G={chosen_g} ticks, DMIN={chosen_dmin:.0f}, B_MODE={chosen_b_mode} "
                f"base_rate={base:.3f} target=[{tgt_min:.3f},{tgt_max:.3f}]"
            )

    for _, idx in df.groupby(group_keys).groups.items():
        idx = pd.Index(idx)
        if idx.empty:
            continue
        day = df.loc[idx]
        symbol = str(day["Symbol"].iloc[0])
        tick_size = _tick_size_for_symbol(symbol, tick_size_env, day["spread"])
        flags = _structural_flags_for_day(
            day, chosen_x, chosen_y, chosen_g, chosen_dmin, tick_size, horizon_steps, bars_r
        )

        spread_fail = flags["spread_fail"]
        bid_depth_fail = flags["bid_depth_fail"]
        ask_depth_fail = flags["ask_depth_fail"]
        bid_depth_persist = flags["bid_depth_persist"]
        ask_depth_persist = flags["ask_depth_persist"]
        bid_gap_any = flags["bid_gap_any"]
        ask_gap_any = flags["ask_gap_any"]
        bid_gap_persist = flags["bid_gap_persist"]
        ask_gap_persist = flags["ask_gap_persist"]

        if chosen_g >= 12 and bid_gap_persist.mean() > 0.05:
            bid_ticks = _prices_to_ticks(day["bid_price_1"], tick_size)
            strict = _gap_flags_strict(bid_ticks, chosen_g, horizon_steps, bars_r, direction="buy")
            bid_gap_persist = pd.Series(strict, index=day.index)
        if chosen_g >= 12 and ask_gap_persist.mean() > 0.05:
            ask_ticks = _prices_to_ticks(day["ask_price_1"], tick_size)
            strict = _gap_flags_strict(ask_ticks, chosen_g, horizon_steps, bars_r, direction="sell")
            ask_gap_persist = pd.Series(strict, index=day.index)

        if chosen_b_mode == "AND_CANY":
            out_buy.loc[idx] = spread_fail | (bid_depth_persist & bid_gap_any)
            out_sell.loc[idx] = spread_fail | (ask_depth_persist & ask_gap_any)
        else:
            out_buy.loc[idx] = spread_fail | bid_depth_persist | bid_gap_persist
            out_sell.loc[idx] = spread_fail | ask_depth_persist | ask_gap_persist

        day_str = str(day["Time"].iloc[0].date())
        buy_rate = out_buy.loc[idx].mean()
        sell_rate = out_sell.loc[idx].mean()
        print(
            f"{day_str} struct_fail_rates "
            f"buy={buy_rate:.3f} sell={sell_rate:.3f} "
            f"A_spread={spread_fail.mean():.3f} "
            f"B_bid_depth={bid_depth_persist.mean():.3f} "
            f"B_ask_depth={ask_depth_persist.mean():.3f} "
            f"C_any_bid={bid_gap_any.mean():.3f} "
            f"C_any_ask={ask_gap_any.mean():.3f} "
            f"C_persist_bid={bid_gap_persist.mean():.3f} "
            f"C_persist_ask={ask_gap_persist.mean():.3f} "
            f"bars_R={bars_r} bar_ms={step_ms} H_ms={horizon_ms} H_bars={horizon_steps} "
            f"DMIN={chosen_dmin:.0f} B_MODE={chosen_b_mode}"
        )

    df["lfp_structural_fail_buy_{}".format(horizon_ms)] = out_buy.astype(int)
    df["lfp_structural_fail_sell_{}".format(horizon_ms)] = out_sell.astype(int)
    return df


def _validate_book_columns(
    df: pd.DataFrame,
    step_ms: int,
    tick_size_env: float,
    max_spread_points: float = 5.0,
    max_jump_points: float = 20.0,
) -> None:
    required = {"bid_price_1", "ask_price_1"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not {"bid_size_1", "ask_size_1"}.issubset(df.columns):
        print("Warning: missing bid_size_1/ask_size_1 columns; depth checks may be degraded.")
    bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
    spread = ask - bid
    tick_size = _tick_size_for_symbol(str(df["Symbol"].iloc[0]), tick_size_env, spread)
    if (spread < -1e-9).any():
        sample = df.loc[spread < -1e-9, ["Time", "bid_price_1", "ask_price_1"]].head(5)
        raise ValueError(f"Negative spread detected; sample:\n{sample}")
    if np.isfinite(tick_size) and tick_size > 0:
        ratio = spread / tick_size
        off = (ratio - np.round(ratio)).abs()
        eps = 1e-6 * max(1.0, tick_size)
        bad_tick = off > eps
        if bad_tick.any():
            sample = df.loc[bad_tick, ["Time", "bid_price_1", "ask_price_1"]].head(5)
            raise ValueError(f"Spread not multiple of tick size; sample:\n{sample}")
    if spread.max() > max_spread_points:
        sample = df.loc[spread > max_spread_points, ["Time", "bid_price_1", "ask_price_1"]].head(5)
        print(f"Warning: large spread > {max_spread_points} points; sample:\n{sample}")
    bound = max_jump_points * (step_ms / 100.0)
    bid_jump = bid.diff().abs()
    ask_jump = ask.diff().abs()
    if bid_jump.max() > bound or ask_jump.max() > bound:
        bad = (bid_jump > bound) | (ask_jump > bound)
        sample = df.loc[bad, ["Time", "bid_price_1", "ask_price_1"]].head(5)
        print(f"Warning: large bid/ask jump > {bound:.2f} points; sample:\n{sample}")


def _ae_labels(df: pd.DataFrame, horizons: List[int], pct: float, group_keys: Iterable) -> pd.DataFrame:
    mid = pd.to_numeric(df["mid"], errors="coerce")
    for h in horizons:
        future_min = mid.groupby(group_keys).transform(lambda s: s.shift(-1).rolling(h, min_periods=h).min())
        future_max = mid.groupby(group_keys).transform(lambda s: s.shift(-1).rolling(h, min_periods=h).max())
        ae_buy = ((future_min - mid) / mid) * 1e4
        ae_sell = ((future_max - mid) / mid) * 1e4
        df[f"ae_buy_bps_{h}"] = ae_buy
        df[f"ae_sell_bps_{h}"] = ae_sell

        # Adverse magnitude (direction-aware)
        adv_buy = -ae_buy
        adv_sell = ae_sell

        def _day_x(g: pd.DataFrame) -> float:
            vals = pd.concat([g[f"ae_buy_bps_{h}"].mul(-1), g[f"ae_sell_bps_{h}"]])
            return float(vals.quantile(pct))

        x_by_day = df.groupby(df["Time"].dt.date).apply(_day_x)
        x_val = df["Time"].dt.date.map(x_by_day)

        df[f"x_buy_bps_{h}"] = x_val
        df[f"x_sell_bps_{h}"] = x_val
        df[f"lfp_event_buy_{h}"] = adv_buy >= x_val
        df[f"lfp_event_sell_{h}"] = adv_sell >= x_val
    return df


def main() -> None:
    input_dir = Path(os.environ.get("INPUT_DIR", "data/processed")).expanduser().resolve()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "data/processed_lfp")).expanduser().resolve()
    horizons = [int(h.strip()) for h in os.environ.get("HORIZON_LIST", "10,20,50").split(",") if h.strip()]
    pct = float(os.environ.get("P_LFP_PCT", "0.90"))
    lambda_win = int(os.environ.get("P_LFP_LAMBDA_WIN", "200"))
    flow_win = int(os.environ.get("P_LFP_FLOW_WIN", "50"))
    depth_drop = float(os.environ.get("P_LFP_DEPTH_DROP", "0.5"))
    struct_h_ms = int(os.environ.get("LFP_STRUCT_H_MS", "2000"))
    struct_x_ticks = int(os.environ.get("LFP_STRUCT_X_TICKS", "2"))
    struct_y_pct = float(os.environ.get("LFP_STRUCT_Y_PCT", "60"))
    struct_r_ms = int(os.environ.get("LFP_STRUCT_R_MS", "300"))
    struct_g_ticks = int(os.environ.get("LFP_STRUCT_G_TICKS", "4"))
    struct_dmin = float(os.environ.get("LFP_STRUCT_DMIN", "100"))
    struct_b_mode = os.environ.get("LFP_STRUCT_B_MODE", "AND_CANY").strip().upper()
    struct_step_ms = int(os.environ.get("LFP_STRUCT_STEP_MS", "0"))
    struct_autocal = os.environ.get("LFP_STRUCT_AUTOCAL", "0").strip() in {"1", "true", "yes", "y"}
    struct_tgt_min = float(os.environ.get("LFP_STRUCT_TGT_MIN", "0.01"))
    struct_tgt_max = float(os.environ.get("LFP_STRUCT_TGT_MAX", "0.05"))
    x_grid = [int(x.strip()) for x in os.environ.get("LFP_STRUCT_X_GRID", "1,2,3").split(",") if x.strip()]
    y_grid = [float(y.strip()) for y in os.environ.get("LFP_STRUCT_Y_GRID", "40,50,60,70").split(",") if y.strip()]
    g_grid = [int(x.strip()) for x in os.environ.get("LFP_STRUCT_G_GRID", "2,3,4,5,6").split(",") if x.strip()]
    dmin_grid = [float(x.strip()) for x in os.environ.get("LFP_STRUCT_DMIN_GRID", "50,100,200,300").split(",") if x.strip()]
    b_mode_grid = [m.strip().upper() for m in os.environ.get("LFP_STRUCT_B_MODE_GRID", "AND_CANY,OR").split(",") if m.strip()]
    tick_size_env = float(os.environ.get("LFP_TICK_SIZE", "0"))
    eps = float(os.environ.get("EPS", "1e-9"))
    mid_min = float(os.environ.get("MID_MIN", "0"))
    mid_max = float(os.environ.get("MID_MAX", "0"))
    overwrite = os.environ.get("OVERWRITE", "0").strip() in {"1", "true", "yes", "y"}

    files = _iter_parquet_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")

    for path in files:
        rel = path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            print(f"Skipping existing {out_path}")
            continue

        df = pd.read_parquet(path)
        df = _ensure_datetime(df)
        if "Symbol" not in df.columns or "Time" not in df.columns:
            raise ValueError(f"Missing Symbol/Time in {path}")
        df = df.sort_values(["Symbol", "Time"])
        if mid_min > 0:
            df = df[df["mid"] >= mid_min]
        if mid_max > 0:
            df = df[df["mid"] <= mid_max]
        if df.empty:
            print(f"{rel}: no rows left after mid filter; skipping")
            continue

        group_keys = [df["Symbol"], df["Time"].dt.date]
        step_ms = struct_step_ms or _infer_step_ms(df["Time"])
        _validate_book_columns(df, step_ms=step_ms, tick_size_env=tick_size_env)
        df["dmid_bps"] = _compute_dmid_bps(df, group_keys)
        q = _compute_signed_volume(df)
        df["signed_volume"] = q
        df["abs_signed_volume"] = q.abs()
        dp = df["dmid_bps"]

        lam, denom = _lambda_feature(q, dp, group_keys, lambda_win, eps)
        df[f"kyle_lambda_{lambda_win}"] = lam
        df["impact_est_bps"] = lam * q
        df["impact_per_flow"] = dp / (q.replace(0, np.nan) + eps)

        if "depth_total_top5" in df.columns:
            df["flow_intensity"] = df["abs_signed_volume"] / (df["depth_total_top5"] + eps)
            df["flow_intensity_roll"] = df["flow_intensity"].rolling(flow_win, min_periods=flow_win).mean()
            df["flow_accel"] = df["flow_intensity"] - df["flow_intensity"].shift(flow_win)

            prev_depth = df["depth_total_top5"].shift(1)
            df["depth_drop"] = (df["depth_total_top5"] - prev_depth) / (prev_depth + eps)
            df["gap_flag"] = (df["depth_drop"] <= -abs(depth_drop)).astype(int)
            df["stress_ratio"] = df["flow_intensity_roll"] / (df["depth_total_top5"] + eps)

        if "order_book_imbalance" in df.columns:
            df["imbalance_delta"] = df["order_book_imbalance"].diff()
            df["imbalance_vol"] = df["order_book_imbalance"].rolling(flow_win, min_periods=flow_win).std()

        if "spread" in df.columns and "mid" in df.columns:
            spread_bps = (df["spread"] / df["mid"]) * 1e4
            df["spread_bps"] = spread_bps
            df["spread_bps_vol"] = spread_bps.rolling(flow_win, min_periods=flow_win).std()
            df["spread_stability"] = df["spread_bps_vol"]

        df = _ae_labels(df, horizons, pct=pct, group_keys=group_keys)
        df = _add_structural_lfp_labels(
            df,
            horizon_ms=struct_h_ms,
            step_ms=step_ms,
            spread_ticks=struct_x_ticks,
            depth_drop_pct=struct_y_pct,
            persist_ms=struct_r_ms,
            gap_ticks=struct_g_ticks,
            dmin=struct_dmin,
            b_mode=struct_b_mode,
            tick_size_env=tick_size_env,
            autocal=struct_autocal,
            tgt_min=struct_tgt_min,
            tgt_max=struct_tgt_max,
            x_grid=x_grid,
            y_grid=y_grid,
            g_grid=g_grid,
            dmin_grid=dmin_grid,
            b_mode_grid=b_mode_grid,
        )

        tiny = (denom.abs() < eps).mean()
        print(
            f"{rel}: rows={len(df)}, lambda_abs_mean={lam.abs().mean():.6g}, "
            f"lambda_abs_median={lam.abs().median():.6g}, tiny_lambda={tiny:.2%}"
        )

        df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
