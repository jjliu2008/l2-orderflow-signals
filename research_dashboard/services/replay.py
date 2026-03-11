from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_trade_replay(replay_index_row: pd.Series) -> pd.DataFrame:
    source = Path(str(replay_index_row["replay_source"]))
    if not source.exists():
        return pd.DataFrame()

    window_start = pd.to_datetime(replay_index_row["window_start_ts"], utc=True)
    window_end = pd.to_datetime(replay_index_row["window_end_ts"], utc=True)
    cols = [
        "Time",
        "mid",
        "trade_volume",
        "signed_volume",
        "order_book_imbalance",
        "stress_ratio",
        "bid_price_1",
        "ask_price_1",
    ]
    raw = pd.read_parquet(source)
    keep = [c for c in cols if c in raw.columns]
    if "Time" not in keep:
        return pd.DataFrame()

    df = raw[keep].copy()
    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df = df[(df["Time"] >= window_start) & (df["Time"] <= window_end)].copy()
    if df.empty:
        return df
    df = df.sort_values("Time")
    df["Time_et"] = df["Time"].dt.tz_convert("America/New_York")
    return df


def build_replay_chart_data(trade_row: pd.Series, replay_df: pd.DataFrame) -> pd.DataFrame:
    if replay_df.empty:
        return replay_df

    out = replay_df.copy()
    if {"bid_price_1", "ask_price_1"}.issubset(out.columns):
        out["spread_ticks"] = (out["ask_price_1"] - out["bid_price_1"]) / 0.25

    out["cum_vol"] = pd.to_numeric(out.get("trade_volume", 0.0), errors="coerce").fillna(0.0).cumsum()
    pv = out["mid"] * pd.to_numeric(out.get("trade_volume", 0.0), errors="coerce").fillna(0.0)
    out["cum_pv"] = pv.cumsum()
    out["vwap"] = out["cum_pv"] / out["cum_vol"].replace(0.0, pd.NA)

    out["entry_ts"] = pd.to_datetime(trade_row["entry_ts"], utc=True).tz_convert("America/New_York")
    out["exit_ts"] = pd.to_datetime(trade_row["exit_ts"], utc=True).tz_convert("America/New_York")
    return out
