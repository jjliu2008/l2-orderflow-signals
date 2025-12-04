from typing import Union
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple features to market data.

    - If level-1 order book columns exist (BidPrice1/AskPrice1), use them for mid/spread.
    - Otherwise fall back to High/Low or Open/Close when available.
    - Depth/imbalance rely on BidVolume1/AskVolume1; if absent they are left as NA.
    """
    df = df.copy()

    # Mid/spread from best bid/ask when available; otherwise use high/low or open/close as fallbacks.
    if {"BidPrice1", "AskPrice1"}.issubset(df.columns):
        mid = (df["BidPrice1"] + df["AskPrice1"]) / 2
        spread = df["AskPrice1"] - df["BidPrice1"]
    elif {"High", "Low"}.issubset(df.columns):
        mid = (df["High"] + df["Low"]) / 2
        spread = df["High"] - df["Low"]
    elif {"Open", "Close"}.issubset(df.columns):
        mid = (df[["Open", "Close"]].max(axis=1) + df[["Open", "Close"]].min(axis=1)) / 2
        spread = df[["Open", "Close"]].max(axis=1) - df[["Open", "Close"]].min(axis=1)
    else:
        raise ValueError(
            "Cannot compute mid/spread; expected bid/ask or OHLC columns."
        )

    df["mid"] = mid
    df["spread"] = spread

    # Depths: use L1 volumes if present; otherwise fall back to total Volume split evenly.
    if {"BidVolume1", "AskVolume1"}.issubset(df.columns):
        bid_depth = pd.to_numeric(df["BidVolume1"], errors="coerce")
        ask_depth = pd.to_numeric(df["AskVolume1"], errors="coerce")
    elif "Volume" in df.columns:
        # No L1 book data; approximate with total traded volume split evenly.
        vol = pd.to_numeric(df["Volume"], errors="coerce")
        bid_depth = vol / 2
        ask_depth = vol / 2
    else:
        bid_depth = pd.Series(pd.NA, index=df.index, dtype="Float64")
        ask_depth = pd.Series(pd.NA, index=df.index, dtype="Float64")

    df["top_bid_depth"] = bid_depth
    df["top_ask_depth"] = ask_depth

    denom = bid_depth.fillna(0).astype("Float64") + ask_depth.fillna(0).astype("Float64")
    df["order_book_imbalance"] = bid_depth / denom.replace(0, pd.NA)

    return df
