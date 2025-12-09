import os
from pathlib import Path

import numpy as np
import pandas as pd

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, ensure_multiindex


def bootstrap_paths(log_returns: np.ndarray, start_price: float, n_paths: int, path_len: int) -> np.ndarray:
    """
    Generate price paths by bootstrapping historical log returns.
    Returns array shape (n_paths, path_len+1) including the start price.
    """
    # Sample with replacement
    samples = np.random.choice(log_returns, size=(n_paths, path_len), replace=True)
    log_paths = np.cumsum(samples, axis=1)
    prices = start_price * np.exp(log_paths)
    # prepend start price
    start_col = np.full((n_paths, 1), start_price)
    return np.concatenate([start_col, prices], axis=1)


def regime_bootstrap_paths(log_returns: pd.Series, start_price: float, n_paths: int, path_len: int, vol_quantile: float = 0.5) -> np.ndarray:
    """
    Regime-aware bootstrap: split returns into low/high vol regimes using rolling std quantile,
    then for each simulated step, pick the regime of a randomly chosen historical day and sample
    a return from that regime's pool. This preserves some vol clustering vs plain iid sampling.
    """
    roll_vol = log_returns.rolling(50, min_periods=10).std()
    thresh = roll_vol.quantile(vol_quantile)
    low_mask = roll_vol <= thresh
    high_mask = roll_vol > thresh

    low_pool = log_returns[low_mask].dropna().values
    high_pool = log_returns[high_mask].dropna().values
    if len(low_pool) == 0 or len(high_pool) == 0:
        # Fallback to plain bootstrap if we can't form pools
        return bootstrap_paths(log_returns.dropna().values, start_price, n_paths, path_len)

    # Build a sequence of regimes by sampling historical rolling vol states
    regimes = np.random.choice(["low", "high"], size=(n_paths, path_len))
    # Weight regime sampling by historical frequency
    p_high = len(high_pool) / (len(high_pool) + len(low_pool))
    regimes = np.random.choice(["low", "high"], size=(n_paths, path_len), p=[1 - p_high, p_high])

    samples = np.empty((n_paths, path_len))
    for i in range(n_paths):
        low_choices = np.random.choice(low_pool, size=(regimes[i] == "low").sum(), replace=True) if (regimes[i] == "low").any() else np.array([])
        high_choices = np.random.choice(high_pool, size=(regimes[i] == "high").sum(), replace=True) if (regimes[i] == "high").any() else np.array([])
        # fill preserving order
        low_idx = 0
        high_idx = 0
        for t, r in enumerate(regimes[i]):
            if r == "low":
                samples[i, t] = low_choices[low_idx]
                low_idx += 1
            else:
                samples[i, t] = high_choices[high_idx]
                high_idx += 1

    log_paths = np.cumsum(samples, axis=1)
    prices = start_price * np.exp(log_paths)
    start_col = np.full((n_paths, 1), start_price)
    return np.concatenate([start_col, prices], axis=1)


def max_drawdown(path: np.ndarray) -> float:
    """
    Compute max drawdown as min((price/peak) - 1).
    """
    peak = np.maximum.accumulate(path)
    dd = path / peak - 1.0
    return dd.min()


def main():
    raw_dir_env = os.environ.get("MONTE_CARLO_DATA_DIR")
    raw_dir = Path(raw_dir_env).expanduser().resolve() if raw_dir_env else PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    print(f"Loading data from {raw_dir} ...")
    df = ensure_multiindex(load_all_raw_data(raw_dir))
    df = add_basic_features(df)

    # Use all symbols' returns together for sampling
    mid = df["mid"].dropna()
    log_ret = mid.groupby(level=0).apply(lambda s: np.log(s.astype(float)).diff()).droplevel(0)
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna()

    if log_ret.empty:
        raise ValueError("No returns available for simulation. Ensure raw data has price information.")

    start_price = float(mid.iloc[-1])

    n_paths = int(os.environ.get("MONTE_CARLO_PATHS", "100"))
    path_len = int(os.environ.get("MONTE_CARLO_STEPS", "200"))
    mode = os.environ.get("MONTE_CARLO_MODE", "plain").lower()
    compare = mode == "both"
    modes_to_run = ["plain", "regime"] if compare else [mode]

    print(f"Simulating {n_paths} paths of length {path_len} from start price {start_price:.2f} ...")

    def pct(x, q):
        return np.percentile(x, q)

    results = {}

    for current_mode in modes_to_run:
        if current_mode == "regime":
            print("\nUsing regime-aware bootstrap (low/high vol splits).")
            paths = regime_bootstrap_paths(log_ret, start_price, n_paths, path_len)
        else:
            print("\nUsing plain bootstrap of log returns.")
            paths = bootstrap_paths(log_ret.values, start_price, n_paths, path_len)

        finals = paths[:, -1]
        returns = finals / start_price - 1.0
        drawdowns = np.array([max_drawdown(p) for p in paths])

        print("Price stats:")
        print(f"  Final price mean: {finals.mean():.2f}")
        print(f"  Final price 5th/50th/95th pct: {pct(finals,5):.2f} / {pct(finals,50):.2f} / {pct(finals,95):.2f}")

        print("Return stats:")
        print(f"  Mean return: {returns.mean()*100:.2f}%")
        print(f"  5th/50th/95th pct: {pct(returns,5)*100:.2f}% / {pct(returns,50)*100:.2f}% / {pct(returns,95)*100:.2f}%")

        print("Max drawdown stats:")
        print(f"  Mean max DD: {drawdowns.mean()*100:.2f}%")
        print(f"  5th/50th/95th pct: {pct(drawdowns,5)*100:.2f}% / {pct(drawdowns,50)*100:.2f}% / {pct(drawdowns,95)*100:.2f}%")

        # Preview first few paths
        preview = min(3, n_paths)
        print(f"Preview of {preview} simulated paths (first 5 steps):")
        for i in range(preview):
            print(f"  Path {i+1}: {np.round(paths[i, :6], 2)} ...")

        results[current_mode] = paths

    # No plotting or artifact saves here; visualization handled by the live viewer.


if __name__ == "__main__":
    main()
