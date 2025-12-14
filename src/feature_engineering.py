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
    "depth_bid_top5",
    "depth_ask_top5",
    "depth_imbalance_top5",
    "book_slope_top5",
    "sweep_cost_buy1",
    "sweep_cost_sell1",
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "volatility_20",
    "volatility_50",
    "signed_volume",
    "cvd_20",
    "volume_trend_20",
    "volume_zscore_20",
    "vol_regime_score",
    "risk_adj_momentum_10",
    "risk_adj_momentum_20",
    "trend_flip_flag",
    "price_drawdown_50",
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

    # If richer depth metrics are present (from Kraken book), keep them; otherwise set to NA for compatibility.
    for col in ["depth_bid_top5", "depth_ask_top5", "depth_imbalance_top5", "book_slope_top5", "sweep_cost_buy1", "sweep_cost_sell1"]:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="Float64")

    return df


def add_orderflow_features(
    df: pd.DataFrame,
    return_windows: Tuple[int, ...] = (1, 5, 10, 20),
    volatility_windows: Tuple[int, ...] = (20, 50),
    cvd_window: int = 20,
    volume_trend_windows: Tuple[int, ...] = (20,),
) -> pd.DataFrame:
    """
    Add simple orderflow-style features using price/volume because depth levels are unavailable.

    - ret_{w}: percent returns over multiple short windows.
    - volatility_{w}: rolling std of log returns.
    - signed_volume: volume signed by 1-bar return direction.
    - cvd_{cvd_window}: rolling sum of signed_volume (a short-horizon CVD proxy).
    - volume_trend_{w}: volume deviation vs rolling mean.
    - volume_zscore_{w}: volume z-score vs rolling stats.
    """
    df = ensure_multiindex(df).copy()

    price = df["mid"]

    grouped_price = price.groupby(level=0)
    for w in return_windows:
        df[f"ret_{w}"] = grouped_price.pct_change(periods=w)

    log_ret = grouped_price.apply(lambda s: np.log(s.astype(float)).diff()).droplevel(0)
    for w in volatility_windows:
        df[f"volatility_{w}"] = log_ret.groupby(level=0).transform(lambda s: s.rolling(w).std())

    volume = pd.to_numeric(df.get("Volume"), errors="coerce") if "Volume" in df.columns else pd.Series(pd.NA, index=df.index)
    df["signed_volume"] = volume * np.sign(df["ret_1"])
    df[f"cvd_{cvd_window}"] = df["signed_volume"].groupby(level=0).transform(lambda s: s.rolling(cvd_window).sum())

    for w in volume_trend_windows:
        roll_mean = volume.groupby(level=0).transform(lambda s: s.rolling(w).mean())
        roll_std = volume.groupby(level=0).transform(lambda s: s.rolling(w).std())
        df[f"volume_trend_{w}"] = (volume - roll_mean) / roll_mean.replace(0, np.nan)
        df[f"volume_zscore_{w}"] = (volume - roll_mean) / roll_std.replace(0, np.nan)

    # Volatility regime and risk-adjusted momentum helpers
    if "volatility_20" in df and "volatility_50" in df:
        df["vol_regime_score"] = df["volatility_20"] / df["volatility_50"].replace(0, np.nan)
    else:
        df["vol_regime_score"] = pd.NA

    if "volatility_20" in df:
        vol_denom = df["volatility_20"].abs().replace(0, np.nan)
        df["risk_adj_momentum_10"] = df.get("ret_10") / vol_denom
        df["risk_adj_momentum_20"] = df.get("ret_20") / vol_denom
    else:
        df["risk_adj_momentum_10"] = pd.NA
        df["risk_adj_momentum_20"] = pd.NA

    # Trend flip indicator: short vs medium horizon return sign disagreement
    df["trend_flip_flag"] = ((df.get("ret_5") * df.get("ret_20")) < 0).astype("Int64")

    # Rolling price drawdown over a medium window as a stress/regime proxy
    grouped_price = price.groupby(level=0)
    roll_max_50 = grouped_price.transform(lambda s: s.rolling(50).max())
    df["price_drawdown_50"] = (roll_max_50 - price) / roll_max_50.replace(0, np.nan)

    # Simple order flow imbalance proxy: change in top-level depth imbalance
    if "order_book_imbalance" in df:
        df["ofi_1"] = df["order_book_imbalance"].groupby(level=0).diff()
    else:
        df["ofi_1"] = pd.NA

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
