"""
Inspect a single-day parquet for best bid/ask sanity and gap-condition rates.

Env vars:
  INPUT_PARQUET   path to parquet file
  BID_COL         bid column name (default: best_bid)
  ASK_COL         ask column name (default: best_ask)
  BAR_MS          bar size in ms (default: 100)
  H_MS            horizon in ms (default: 2000)
  R_MS            persistence window in ms (default: 300)
  G_TICKS         gap threshold in ticks (default: 12)
  LFP_TICK_SIZE   optional tick size override
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _pctiles(series: pd.Series, qs: Iterable[float]) -> dict:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.quantile(list(qs)).to_dict()


def _infer_tick_size_from_spread(bid: pd.Series, ask: pd.Series) -> float | None:
    spread = pd.to_numeric(ask, errors="coerce") - pd.to_numeric(bid, errors="coerce")
    spread = spread[(spread > 0) & np.isfinite(spread)]
    if spread.empty:
        return None
    return float(spread.min())


def _gap_any_persist(
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
        cond = np.zeros(len(future), dtype=bool)
        if direction == "buy":
            cond[valid] = future[valid] <= base - gap_ticks
        else:
            cond[valid] = future[valid] >= base + gap_ticks
        any_flags[i] = cond.any()
        if len(cond) >= bars_r:
            window = np.convolve(cond.astype(int), np.ones(bars_r, dtype=int), mode="valid")
            persist_flags[i] = np.any(window == bars_r)
    return any_flags, persist_flags


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


def _print_examples(
    df: pd.DataFrame,
    bid: pd.Series,
    ask: pd.Series,
    persist_flags: np.ndarray,
    horizon_steps: int,
    bars_r: int,
    tick_size: float,
    gap_ticks: int,
    side: str,
):
    idx = np.where(persist_flags)[0][:5]
    if len(idx) == 0:
        print(f"Examples {side}: none")
        return
    print(f"Examples {side}:")
    for i in idx:
        start = i + 1
        end = min(i + 1 + horizon_steps, len(df))
        if side == "buy":
            base = bid.iloc[i]
            window = bid.iloc[start:end]
        else:
            base = ask.iloc[i]
            window = ask.iloc[start:end]
        spread = (ask - bid).iloc[max(0, i - bars_r) : min(len(df), i + bars_r + 1)]
        window_min = window.min()
        window_max = window.max()
        delta = (window_min - base) if side == "buy" else (window_max - base)
        delta_ticks = delta / tick_size if tick_size else float("nan")
        print(
            f"  idx={i} base={base:.6g} win_min={window_min:.6g} win_max={window_max:.6g} "
            f"delta={delta:.6g} delta_ticks={delta_ticks:.3f} "
            f"spread_min={spread.min():.6g} spread_max={spread.max():.6g}"
        )


def main() -> None:
    input_path = Path(
        os.environ.get(
            "INPUT_PARQUET",
            "data/processed/instrument=ES/date=2025-12-16/features_labels.parquet",
        )
    ).expanduser().resolve()
    bid_col = os.environ.get("BID_COL", "best_bid").strip()
    ask_col = os.environ.get("ASK_COL", "best_ask").strip()

    if not input_path.exists():
        raise FileNotFoundError(f"Parquet not found: {input_path}")

    df = pd.read_parquet(input_path)
    if bid_col not in df.columns or ask_col not in df.columns:
        raise ValueError(f"Missing bid/ask columns: {bid_col}, {ask_col}")

    bid = pd.to_numeric(df[bid_col], errors="coerce")
    ask = pd.to_numeric(df[ask_col], errors="coerce")
    spread = ask - bid

    print("== Validity ==")
    bid_nf = (~np.isfinite(bid)).mean()
    ask_nf = (~np.isfinite(ask)).mean()
    print(f"% nonfinite bid={bid_nf:.3%} ask={ask_nf:.3%}")
    print(f"% bid==0={(bid == 0).mean():.3%} % ask==0={(ask == 0).mean():.3%}")
    print(f"% ask < bid={(ask < bid).mean():.3%}")
    print("spread percentiles:", _pctiles(spread, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]))
    print(f"max abs jump bid={bid.diff().abs().max():.6g} ask={ask.diff().abs().max():.6g}")

    print("\n== Increment scale ==")
    bid_diffs = bid.diff().abs()
    ask_diffs = ask.diff().abs()
    bid_nz = bid_diffs[(bid_diffs > 0) & np.isfinite(bid_diffs)]
    ask_nz = ask_diffs[(ask_diffs > 0) & np.isfinite(ask_diffs)]
    print("bid diff percentiles:", _pctiles(bid_nz, [0.01, 0.05, 0.10, 0.50]))
    print("bid diff min:", bid_nz.min())
    print("ask diff percentiles:", _pctiles(ask_nz, [0.01, 0.05, 0.10, 0.50]))
    print("ask diff min:", ask_nz.min())
    diff_p10 = float(bid_nz.quantile(0.10)) if not bid_nz.empty else float("nan")
    if np.isfinite(diff_p10):
        units = "ticks" if diff_p10 >= 0.75 else "dollars"
        print(f"LIKELY_UNITS: {units} (bid diff p10={diff_p10:.6g})")
    else:
        print("LIKELY_UNITS: unknown (no nonzero diffs)")

    print("\n== Gap condition check ==")
    bar_ms = int(os.environ.get("BAR_MS", "100"))
    h_ms = int(os.environ.get("H_MS", "2000"))
    r_ms = int(os.environ.get("R_MS", "300"))
    g_ticks = int(os.environ.get("G_TICKS", "12"))
    tick_env = float(os.environ.get("LFP_TICK_SIZE", "0")) if os.environ.get("LFP_TICK_SIZE") else 0.0
    tick_size = tick_env or _infer_tick_size_from_spread(bid, ask) or float("nan")
    h_bars = int(np.ceil(h_ms / max(bar_ms, 1)))
    bars_r = int(np.ceil(r_ms / max(bar_ms, 1)))
    print(f"bar_ms={bar_ms} H_ms={h_ms} R_ms={r_ms} H_bars={h_bars} bars_R={bars_r} tick_size={tick_size}")

    bid_ticks = _prices_to_ticks(bid, tick_size)
    ask_ticks = _prices_to_ticks(ask, tick_size)
    bid_any, bid_persist = _gap_any_persist(bid_ticks, g_ticks, h_bars, bars_r, direction="buy")
    ask_any, ask_persist = _gap_any_persist(ask_ticks, g_ticks, h_bars, bars_r, direction="sell")
    print(f"BUY any={bid_any.mean():.3%} persist={bid_persist.mean():.3%}")
    print(f"SELL any={ask_any.mean():.3%} persist={ask_persist.mean():.3%}")

    _print_examples(df, bid, ask, bid_persist, h_bars, bars_r, tick_size, g_ticks, side="buy")
    _print_examples(df, bid, ask, ask_persist, h_bars, bars_r, tick_size, g_ticks, side="sell")


if __name__ == "__main__":
    main()
