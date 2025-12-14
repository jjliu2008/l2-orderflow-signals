from typing import Iterable, Tuple, Optional
import pandas as pd

def _future_price_time_based(price: pd.Series, delta: pd.Timedelta) -> pd.Series:
    """
    Compute future price using a time-based horizon per symbol.
    Uses forward merge_asof to find the first quote at/after t+delta.
    """
    base = price.reset_index()
    base = base.rename(columns={price.name if price.name else 0: "price"})
    base = base.sort_values(["Symbol", "Time"])

    target = base[["Symbol", "Time", "price"]].copy()
    target["target_time"] = target["Time"] + delta

    future_lookup = base[["Symbol", "Time", "price"]].rename(columns={"Time": "future_time", "price": "future_price"})
    merged = pd.merge_asof(
        target.sort_values(["Symbol", "target_time"]),
        future_lookup.sort_values(["Symbol", "future_time"]),
        left_on="target_time",
        right_on="future_time",
        by="Symbol",
        direction="forward",
        allow_exact_matches=True,
    )
    future_price = merged["future_price"]
    # Restore original index alignment
    future_price.index = price.reset_index().index
    future_price.index = price.index
    return future_price



def make_labels(
    df: pd.DataFrame,
    price_col: str = "mid",
    horizon: int = 1,
    time_horizon: Optional[pd.Timedelta] = None,
    neutral_threshold: float = 0.0,
) -> Tuple[pd.Series, pd.Index]:
    """
    Create classification labels (y) from future price moves.

    - If `time_horizon` is provided, looks that far ahead in time (per symbol) using forward merge.
      Otherwise, looks `horizon` steps ahead on `price_col` to compute forward returns.
    - Labels: 1 for up move, -1 for down move, 0 for neutral (|ret| < threshold).
    - Returns (y, index) aligned to the surviving rows (tail rows without future price are removed).
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")
    if time_horizon is None and horizon < 1:
        raise ValueError("horizon must be >= 1")
    if time_horizon is not None and not isinstance(time_horizon, pd.Timedelta):
        time_horizon = pd.to_timedelta(time_horizon)

    price = pd.to_numeric(df[price_col], errors="coerce")
    # Compute horizon-ahead price per symbol to avoid cross-symbol leakage
    if time_horizon is not None:
        future_price = _future_price_time_based(price, delta=time_horizon)
    else:
        future_price = price.groupby(level=0).shift(-horizon)
    forward_ret = (future_price - price) / price

    y = (
        (forward_ret > neutral_threshold).astype(int)
        - (forward_ret < -neutral_threshold).astype(int)
    ).astype("Int8")

    # Keep only rows with valid future data
    valid = future_price.notna() & price.notna()
    y = y[valid]
    return y, y.index
