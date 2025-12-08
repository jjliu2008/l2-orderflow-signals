import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, add_orderflow_features, ensure_multiindex, make_feature_matrix


def forward_returns(mid: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Compute forward returns over `horizon` steps per symbol.
    """
    future = mid.groupby(level=0).shift(-horizon)
    ret = (future - mid) / mid
    return ret


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Max drawdown of an equity curve (as a negative number).
    """
    peak = equity_curve.cummax()
    dd = equity_curve / peak - 1.0
    return dd.min()


def _choose_data_dir() -> Path:
    default_raw = PROJECT_ROOT / "data" / "raw"
    default_live = PROJECT_ROOT / "data" / "live"
    env_dir = os.environ.get("BT_DATA_DIR")
    print("Select data directory:")
    print(f"  1) raw ({default_raw})")
    print(f"  2) live ({default_live})")
    choice = input(f"Enter 1/2 or a custom path [{env_dir or default_raw}]: ").strip()
    if choice == "1" or choice == "":
        target = env_dir or default_raw
    elif choice == "2":
        target = default_live
    else:
        target = choice

    raw_dir = Path(target).expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    return raw_dir


def main():
    raw_dir = _choose_data_dir()

    model_path = Path(os.environ.get("BT_MODEL_PATH", PROJECT_ROOT / "artifacts" / "hgb_model.joblib"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}; set BT_MODEL_PATH to override.")

    long_thr = float(os.environ.get("BT_LONG_THRESHOLD", "0.1"))
    short_thr = float(os.environ.get("BT_SHORT_THRESHOLD", "-0.1"))
    fee_bps = float(os.environ.get("BT_FEE_BPS", "0.0"))  # fee per trade (entry/flip), in bps
    slippage_bps = float(os.environ.get("BT_SLIPPAGE_BPS", "0.0"))  # slippage per trade, in bps
    latency_ticks = int(os.environ.get("BT_LATENCY_TICKS", "0"))  # delay execution by N ticks
    horizon = int(os.environ.get("BT_HORIZON", "1"))
    save_artifacts = os.environ.get("BACKTEST_SAVE", "1").strip().lower() in {"1", "true", "yes", "y"}

    print(f"Loading data from {raw_dir} ...")
    df = ensure_multiindex(load_all_raw_data(raw_dir))
    df_feat = add_basic_features(df)
    df_feat = add_orderflow_features(df_feat)

    X_df, _ = make_feature_matrix(df_feat, drop_na=False)

    mid = df_feat["mid"]
    fwd_ret = forward_returns(mid, horizon=horizon).rename("fwd_ret")

    # Merge to align features with forward returns on Symbol/Time
    merged = X_df.reset_index().merge(fwd_ret.reset_index(), on=["Symbol", "Time"], how="inner")
    merged = merged.sort_values(["Time", "Symbol"])
    merged = merged.dropna(subset=["fwd_ret"])
    if merged.empty:
        raise ValueError("No overlapping rows between features and forward returns for backtest.")

    feature_cols = [c for c in merged.columns if c not in {"Symbol", "Time", "fwd_ret"}]
    X_bt = merged[feature_cols]
    fwd = merged["fwd_ret"]

    print(f"Loaded {len(X_bt):,} rows for backtest.")
    model = joblib.load(model_path)
    classes = model.classes_.astype(float)
    proba = model.predict_proba(X_bt)
    expected = (proba * classes.reshape(1, -1)).sum(axis=1)

    positions_signal = np.where(expected > long_thr, 1, np.where(expected < short_thr, -1, 0))
    if latency_ticks > 0:
        positions = np.concatenate([np.zeros(latency_ticks, dtype=int), positions_signal[:-latency_ticks]])
    else:
        positions = positions_signal
    fwd_values = fwd.values

    # Simple PnL: position * forward return. Apply fee when position changes.
    returns = positions * fwd_values
    pos_change = np.concatenate([[0], np.abs(np.diff(positions))])
    trade_cost = (fee_bps + slippage_bps) / 10000.0
    returns -= pos_change * trade_cost

    equity = pd.Series(returns).cumsum() + 1.0  # start at 1.0

    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0  # per-sample, scaled
    mdd = max_drawdown(equity)

    print("\nBacktest summary (per-sample returns):")
    print(f"  Samples: {len(returns):,}")
    print(f"  Mean return: {mean_ret:.6f}")
    print(f"  Std return:  {std_ret:.6f}")
    print(f"  Sharpe (scaled): {sharpe:.3f}")
    print(f"  Max drawdown: {mdd:.3%}")
    print(f"  Final equity: {equity.iloc[-1]:.3f}")

    ev_series = pd.Series(expected)
    print("\nExpected value quantiles (5/50/95):", ev_series.quantile([0.05, 0.5, 0.95]).to_dict())
    print(f"Thresholds: long>{long_thr}, short<{short_thr}, fee_bps={fee_bps}, slippage_bps={slippage_bps}, latency_ticks={latency_ticks}")

    if save_artifacts:
        out_dir = PROJECT_ROOT / "artifacts"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "backtest_results.csv"
        out_df = pd.DataFrame(
            {
                "Symbol": merged["Symbol"],
                "Time": merged["Time"],
                "expected": expected,
                "position_signal": positions_signal,
                "position_exec": positions,
                "fwd_ret": fwd_values,
                "return": returns,
                "equity": equity.values,
            }
        )
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved backtest results to {out_path}")

        # Equity curve plot
        plt.figure(figsize=(10, 4))
        plt.plot(merged["Time"], equity.values, label="Equity")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.title("Backtest Equity Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        eq_path = out_dir / "backtest_equity.png"
        plt.savefig(eq_path, dpi=150)
        plt.close()
        print(f"Saved equity curve plot to {eq_path}")
    else:
        print("\nSkipping save (BACKTEST_SAVE is falsy).")


if __name__ == "__main__":
    main()
