from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# Default set of feature columns expected after add_basic_features/add_orderflow_features
DEFAULT_FEATURE_COLUMNS = [
    "mid",
    "spread",
    "top_bid_depth",
    "top_ask_depth",
    "order_book_imbalance",
    "ret_1",
    "ret_5",
    "volatility_20",
    "signed_volume",
    "cvd_20",
]


def ensure_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee the data is indexed by (Symbol, Time) so downstream features/labels
    stay aligned. If the MultiIndex is already present it is returned as-is.
    """
    if isinstance(df.index, pd.MultiIndex):
        return df

    if {"Symbol", "Time"}.issubset(df.columns):
        return df.set_index(["Symbol", "Time"]).sort_index()

    raise ValueError("Expected MultiIndex on ['Symbol', 'Time'] or matching columns to build it.")


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


def add_orderflow_features(
    df: pd.DataFrame,
    return_windows: Tuple[int, int] = (1, 5),
    volatility_window: int = 20,
    cvd_window: int = 20,
) -> pd.DataFrame:
    """
    Add simple orderflow-style features using price/volume because depth levels are unavailable.

    - ret_1 / ret_5: percent returns over short windows.
    - volatility_20: rolling std of log returns.
    - signed_volume: volume signed by 1-bar return direction.
    - cvd_20: rolling sum of signed_volume (a short-horizon CVD proxy).
    """
    df = ensure_multiindex(df).copy()

    short_win, long_win = return_windows
    price = df["mid"]

    ret_1 = price.groupby(level=0).pct_change(periods=short_win)
    ret_5 = price.groupby(level=0).pct_change(periods=long_win)

    log_ret = price.groupby(level=0).apply(lambda s: np.log(s.astype(float)).diff()).droplevel(0)
    volatility = log_ret.groupby(level=0).transform(lambda s: s.rolling(volatility_window).std())

    volume = pd.to_numeric(df.get("Volume"), errors="coerce") if "Volume" in df.columns else pd.Series(pd.NA, index=df.index)
    signed_volume = volume * np.sign(ret_1)
    cvd = signed_volume.groupby(level=0).transform(lambda s: s.rolling(cvd_window).sum())

    df["ret_1"] = ret_1
    df["ret_5"] = ret_5
    df["volatility_20"] = volatility
    df["signed_volume"] = signed_volume
    df["cvd_20"] = cvd

    return df


def make_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Iterable[str] | None = None,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, pd.Index]:
    """
    Stack selected feature columns into a numeric matrix `X`.

    Returns (X, index) where X is a DataFrame with only the chosen features and
    index matches the original rows that survive optional NA dropping.
    """
    df = ensure_multiindex(df)

    cols: List[str] = list(feature_cols) if feature_cols is not None else DEFAULT_FEATURE_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[cols]
    if drop_na:
        X = X.dropna()

    return X, X.index
