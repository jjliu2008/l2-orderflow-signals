from typing import Iterable, Tuple

import pandas as pd


def make_labels(
    df: pd.DataFrame,
    price_col: str = "mid",
    horizon: int = 1,
    neutral_threshold: float = 0.0,
) -> Tuple[pd.Series, pd.Index]:
    """
    Create classification labels (y) from future price moves.

    - Looks `horizon` steps ahead on `price_col` to compute forward returns.
    - Labels: 1 for up move, -1 for down move, 0 for neutral (|ret| < threshold).
    - Returns (y, index) aligned to the surviving rows (tail rows without future price are removed).
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    price = pd.to_numeric(df[price_col], errors="coerce")
    future_price = price.shift(-horizon)
    forward_ret = (future_price - price) / price

    y = (
        (forward_ret > neutral_threshold).astype(int)
        - (forward_ret < -neutral_threshold).astype(int)
    ).astype("Int8")

    # Keep only rows with valid future data
    valid = future_price.notna() & price.notna()
    y = y[valid]
    return y, y.index
