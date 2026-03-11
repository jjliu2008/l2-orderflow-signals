"""
Build 100ms last-price bars from NQH6 TRADES DBN files.

Reads each daily TRADES file, filters to NQH6, resamples to 100ms bars
using the last trade price per bar (forward-filled across empty bars).
Output is a minimal parquet with Time + mid columns, matching the schema
expected by signal_esnq_spread.py.

NQ tick = 0.25 points = $5/tick.  Prices in DBN: fixed-point, divide by 1e9.

Usage:
    python scripts/build_nqh6_price_bars.py
    python scripts/build_nqh6_price_bars.py --start 2026-01-02 --end 2026-02-28

Output:
    data/processed_nqh6/instrument=NQ/date=YYYY-MM-DD/prices.parquet
    Columns: Time (UTC datetime64), mid (float64 price)
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRADES_ROOT   = Path("c:/Users/majin/Downloads/NQH6-TRADES")
OUTPUT_ROOT   = PROJECT_ROOT / "data/processed_nqh6/instrument=NQ"

SYMBOL        = "NQH6"
BAR_MS        = 100          # 100ms bars
PRICE_SCALE   = db.FIXED_PRICE_SCALE   # 1e-9

# Full 24h day: 00:00 to 23:59:59.9
DAY_START_UTC = "00:00:00"
DAY_END_UTC   = "23:59:59"


def build_date(date_str: str) -> bool:
    compact   = date_str.replace("-", "")
    dbn_path  = TRADES_ROOT / f"glbx-mdp3-{compact}.trades.dbn"
    out_dir   = OUTPUT_ROOT / f"date={date_str}"
    out_path  = out_dir / "prices.parquet"

    if not dbn_path.exists():
        print(f"  {date_str}  DBN not found, skip")
        return False

    if out_path.exists():
        print(f"  {date_str}  already built, skip")
        return True

    # Read trades
    store = db.DBNStore.from_file(str(dbn_path))
    df    = store.to_df()

    if df.empty:
        print(f"  {date_str}  empty file, skip")
        return False

    # Keep only NQH6 trades (filter out spread/calendar trades if any)
    if "symbol" in df.columns:
        df = df[df["symbol"] == SYMBOL]
    elif "raw_symbol" in df.columns:
        df = df[df["raw_symbol"] == SYMBOL]

    if df.empty:
        print(f"  {date_str}  no NQH6 trades, skip")
        return False

    # Parse timestamp — DBN gives nanosecond ts_event as index or column
    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], unit="ns", utc=True)
    else:
        # ts_event is often the index after to_df()
        ts = pd.to_datetime(df.index, utc=True)

    # Price: fixed-point integer → float
    prices = df["price"].values.astype(float) * PRICE_SCALE

    # Build a temporary time-indexed Series of trade prices
    trade_series = pd.Series(prices, index=ts, name="price")
    trade_series = trade_series[trade_series > 0]   # drop zero/invalid prices

    if len(trade_series) == 0:
        print(f"  {date_str}  no valid prices, skip")
        return False

    # Resample to 100ms bars: last trade price per bar
    resampled = trade_series.resample(f"{BAR_MS}ms").last()

    # Build complete 100ms grid for the full day
    day_start = pd.Timestamp(f"{date_str} 00:00:00", tz="UTC")
    day_end   = pd.Timestamp(f"{date_str} 23:59:59.9", tz="UTC")
    full_grid = pd.date_range(day_start, day_end, freq=f"{BAR_MS}ms")

    bars = resampled.reindex(full_grid)
    bars = bars.ffill()   # forward-fill gaps (no trades in this bar)
    bars = bars.bfill()   # back-fill leading NaN at start of day

    out_df = pd.DataFrame({
        "Time": bars.index,
        "mid":  bars.values.astype(np.float64),
    })

    # Drop bars with no valid price (shouldn't happen after ffill/bfill)
    out_df = out_df.dropna(subset=["mid"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    n_trades = len(trade_series)
    print(f"  {date_str}  {n_trades:,} trades -> {len(out_df):,} bars  ({out_path.name})")
    return True


def trading_days(start: str, end: str) -> list[str]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    out = []
    cur = d0
    while cur <= d1:
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--end",   default="2026-02-28")
    args = parser.parse_args()

    dates = trading_days(args.start, args.end)
    print(f"Building NQH6 100ms price bars from TRADES data")
    print(f"Range: {args.start} to {args.end}  ({len(dates)} trading days)")
    print(f"Input:  {TRADES_ROOT}")
    print(f"Output: {OUTPUT_ROOT}\n")

    n_ok = n_skip = n_fail = 0
    for d in dates:
        result = build_date(d)
        if result:
            n_ok += 1
        else:
            # Check if it was a skip (output already exists) or a real fail
            out_path = OUTPUT_ROOT / f"date={d}" / "prices.parquet"
            if out_path.exists():
                n_skip += 1
            else:
                n_fail += 1

    print(f"\nDone.  Built={n_ok}  Skipped={n_skip}  Failed={n_fail}")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
