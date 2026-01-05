"""
LFP diagnostics: model-free baseline score, event capture, and AE tail checks.

Env vars:
  DATA_DIR         root data dir (default: data/processed_lfp)
  LABEL            lfp_event_buy_10 | lfp_event_sell_10 (default: lfp_event_buy_10)
  HORIZON          horizon in bars (default: 20)
  TEST_DAYS        comma-separated dates (YYYY-MM-DD) for test split (default: last day)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data


def _pick_test_days(df: pd.DataFrame, requested: str | None) -> List[str]:
    dates = pd.to_datetime(df["Time"], utc=True, errors="coerce").dt.date
    unique = sorted({d.isoformat() for d in dates if pd.notna(d)})
    if requested:
        return [d.strip() for d in requested.split(",") if d.strip()]
    return unique[-1:]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "spread_bps" not in df.columns and {"spread", "mid"}.issubset(df.columns):
        df["spread_bps"] = (df["spread"] / df["mid"]) * 1e4
    if "impact_est_bps" not in df.columns and {"kyle_lambda_200", "signed_volume"}.issubset(df.columns):
        df["impact_est_bps"] = df["kyle_lambda_200"] * df["signed_volume"]
    if "depth_total_top5" not in df.columns:
        df["depth_total_top5"] = df.get("top_bid_depth", 0) + df.get("top_ask_depth", 0)
    if "flow_intensity_roll" not in df.columns and "flow_intensity" in df.columns:
        df["flow_intensity_roll"] = df["flow_intensity"].rolling(50, min_periods=50).mean()
    return df


def _compute_mid_from_l1(df: pd.DataFrame) -> pd.Series:
    if not {"bid_price_1", "ask_price_1"}.issubset(df.columns):
        raise ValueError("Missing bid_price_1/ask_price_1; regenerate data with L1 columns.")
    bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
    return 0.5 * (bid + ask)


def _forward_window_min(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if window <= 0 or n == 0:
        return out
    from collections import deque

    dq: deque[int] = deque()
    for i, val in enumerate(arr):
        while dq and dq[0] <= i - window:
            dq.popleft()
        while dq and arr[dq[-1]] >= val:
            dq.pop()
        dq.append(i)
        if i >= window - 1:
            out[i - window + 1] = arr[dq[0]]
    return out


def _forward_window_max(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if window <= 0 or n == 0:
        return out
    from collections import deque

    dq: deque[int] = deque()
    for i, val in enumerate(arr):
        while dq and dq[0] <= i - window:
            dq.popleft()
        while dq and arr[dq[-1]] <= val:
            dq.pop()
        dq.append(i)
        if i >= window - 1:
            out[i - window + 1] = arr[dq[0]]
    return out


def _compute_ae_from_l1(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    if horizon < 1:
        horizon = 1
    bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
    mid = 0.5 * (bid + ask)
    spread = ask - bid
    valid_row = pd.Series(False, index=df.index)

    ae_buy = pd.Series(np.nan, index=mid.index)
    ae_sell = pd.Series(np.nan, index=mid.index)
    fwd_min_series = pd.Series(np.nan, index=mid.index)
    fwd_max_series = pd.Series(np.nan, index=mid.index)
    valid_t = pd.Series(False, index=mid.index)

    day = pd.to_datetime(df["Time"], utc=True, errors="coerce").dt.date
    for (symbol, d), group in df.groupby([df["Symbol"], day], sort=False):
        idx = group.index
        if idx.empty:
            continue
        m = mid.loc[idx].to_numpy()
        sp = spread.loc[idx].to_numpy()
        b = bid.loc[idx].to_numpy()
        a = ask.loc[idx].to_numpy()
        vr = (
            np.isfinite(m)
            & np.isfinite(sp)
            & (sp >= 0.25)
            & (sp <= 2.0)
        )
        if np.any(vr):
            lo = np.nanpercentile(m, 0.1)
            hi = np.nanpercentile(m, 99.9)
            vr = vr & (m >= lo) & (m <= hi)
        valid_row.loc[idx] = vr
        if np.any(vr):
            if np.any(~np.isfinite(b[vr])) or np.any(~np.isfinite(a[vr])) or np.any(b[vr] <= 0) or np.any(a[vr] <= 0):
                raise ValueError("Invalid bid/ask values in valid rows (nonfinite or <= 0).")
        vr = vr.astype(np.int32)
        n = len(m)
        window_len = horizon + 1
        cs = np.concatenate([[0], np.cumsum(vr)])
        valid_count = cs[window_len:] - cs[:-window_len]
        vt = valid_count == window_len
        vt_full = np.zeros(n, dtype=bool)
        vt_full[: len(vt)] = vt
        valid_t.loc[idx] = vt_full
        if n <= horizon:
            continue
        fwd_min = _forward_window_min(m[1:], horizon)
        fwd_max = _forward_window_max(m[1:], horizon)
        fwd_min_arr = np.full(n, np.nan, dtype=float)
        fwd_max_arr = np.full(n, np.nan, dtype=float)
        fwd_min_arr[: len(fwd_min)] = fwd_min
        fwd_max_arr[: len(fwd_max)] = fwd_max
        fwd_min_series.loc[idx] = fwd_min_arr
        fwd_max_series.loc[idx] = fwd_max_arr
        ae_buy.loc[idx] = fwd_min_arr - m
        ae_sell.loc[idx] = fwd_max_arr - m

    ae_buy = ae_buy.where(valid_t)
    ae_sell = ae_sell.where(valid_t)
    if (valid_t & (~np.isfinite(fwd_min_series) | ~np.isfinite(fwd_max_series))).any():
        raise ValueError("Nonfinite forward min/max on evaluated t; integrity violation.")
    print(f"invalid_row_rate={(~valid_row).mean():.3%}")
    print("spread percentiles:", spread[valid_row].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).to_dict())
    return ae_buy, ae_sell, valid_t, fwd_min_series, fwd_max_series, valid_row


def _adverse_series(df: pd.DataFrame, label_col: str, horizon: int) -> pd.Series:
    ae_buy, ae_sell, valid_t, _, _, _ = _compute_ae_from_l1(df, horizon)
    if "buy" in label_col:
        return -ae_buy
    return ae_sell


def main() -> None:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    data_dir = Path(os.environ.get("DATA_DIR", "data/processed_lfp")).expanduser().resolve()
    label_col = os.environ.get("LABEL", "lfp_event_buy_10").strip()
    horizon = int(os.environ.get("HORIZON", "20").strip())
    test_days_env = os.environ.get("TEST_DAYS", "").strip()

    df = load_all_raw_data(data_dir)
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    if not {"bid_price_1", "ask_price_1"}.issubset(df.columns):
        raise ValueError("Missing bid_price_1/ask_price_1; regenerate data with L1 columns.")
    bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
    spread = ask - bid
    if (spread < 0).any():
        sample = df.loc[spread < 0, ["bid_price_1", "ask_price_1"]].head(5).reset_index()
        raise ValueError(f"Negative spread detected; sample:\n{sample}")
    if spread.max() > 10.0:
        sample = df.loc[spread > 10.0, ["bid_price_1", "ask_price_1"]].head(5).reset_index()
        print("Warning: spread > 10 points; sample:\n", sample)

    df = df.reset_index().sort_values(["Symbol", "Time"]).reset_index(drop=True)
    time_ok = df.groupby("Symbol")["Time"].apply(lambda s: s.is_monotonic_increasing)
    if not bool(time_ok.all()):
        bad = time_ok[~time_ok].index.tolist()
        raise ValueError(f"Time is not monotonic within symbols: {bad}")
    df = _build_features(df)

    feature_cols = [
        "impact_est_bps",
        "flow_intensity_roll",
        "spread_bps",
        "depth_total_top5",
    ]
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing feature column: {c}")

    df = df.dropna(subset=feature_cols + [label_col, "bid_price_1", "ask_price_1"])
    dates = pd.to_datetime(df["Time"], utc=True, errors="coerce").dt.date
    test_days = set(_pick_test_days(df, test_days_env))
    is_test = pd.Series([d.isoformat() in test_days for d in dates], index=df.index)

    X = df[feature_cols].astype(float)
    y = df[label_col].astype(int)

    X_test = X[is_test]
    y_test = y[is_test]
    if X_test.empty:
        raise ValueError("Test split is empty; check TEST_DAYS or data range.")

    eps = 1e-9
    raw_score = (
        X["impact_est_bps"].abs().fillna(0)
        + X["flow_intensity_roll"].fillna(0)
        + X["spread_bps"].fillna(0)
        + (1.0 / (X["depth_total_top5"].replace(0, np.nan) + eps))
    )
    days = pd.to_datetime(df["Time"], utc=True, errors="coerce").dt.date
    score = raw_score.groupby(days).transform(lambda s: (s - s.mean()) / (s.std() + eps))
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proba = score[is_test]
    auc = roc_auc_score(y_test, proba) if roc_auc_score and len(np.unique(y_test)) > 1 else float("nan")
    print(f"Test days: {sorted(test_days)}")
    print(f"ROC AUC: {auc:.4f}")

    test_df = df.loc[is_test].copy()
    test_df["p_lfp"] = proba
    test_df["decile"] = pd.qcut(test_df["p_lfp"], 10, labels=False, duplicates="drop")
    test_df["adverse_points"] = _adverse_series(test_df, label_col, horizon)

    # AE decile monotonicity
    bar_ms = 100
    seconds = horizon * bar_ms / 1000.0
    print(f"AE horizon: bar_ms={bar_ms}, H_bars={horizon}, seconds={seconds:.2f}")

    ae_buy, ae_sell, valid_t, fwd_min, fwd_max, valid_row = _compute_ae_from_l1(test_df, horizon)
    ae_col = "ae_buy_points" if "buy" in label_col else "ae_sell_points"
    test_df[ae_col] = ae_buy if "buy" in label_col else ae_sell
    test_df["mid_l1"] = 0.5 * (pd.to_numeric(test_df["bid_price_1"], errors="coerce") + pd.to_numeric(test_df["ask_price_1"], errors="coerce"))
    test_df["spread_l1"] = pd.to_numeric(test_df["ask_price_1"], errors="coerce") - pd.to_numeric(test_df["bid_price_1"], errors="coerce")
    test_df["fwd_min_mid"] = fwd_min
    test_df["fwd_max_mid"] = fwd_max
    ae_abs = test_df[ae_col].abs()
    valid_ae = test_df.loc[valid_t]
    evaluated = int(valid_t.sum())
    print(f"evaluated_t_count={evaluated}")
    if not valid_ae.empty:
        abs_ae = valid_ae[ae_col].abs()
        p50 = float(abs_ae.quantile(0.50))
        p90 = float(abs_ae.quantile(0.90))
        p95 = float(abs_ae.quantile(0.95))
        p99 = float(abs_ae.quantile(0.99))
        mx = float(abs_ae.max())
        print(
            f"AE abs tails: p50={p50:.2f} p90={p90:.2f} p95={p95:.2f} p99={p99:.2f} max={mx:.2f}",
            flush=True,
        )
        if "buy" in label_col:
            p1 = float(valid_ae[ae_col].quantile(0.01))
            p5 = float(valid_ae[ae_col].quantile(0.05))
            p95_s = float(valid_ae[ae_col].quantile(0.95))
            p99_s = float(valid_ae[ae_col].quantile(0.99))
            print(
                f"AE signed (buy): p1={p1:.2f} p5={p5:.2f} p95={p95_s:.2f} p99={p99_s:.2f}",
                flush=True,
            )
        else:
            p95_s = float(valid_ae[ae_col].quantile(0.95))
            p99_s = float(valid_ae[ae_col].quantile(0.99))
            print(
                f"AE signed (sell): p95={p95_s:.2f} p99={p99_s:.2f}",
                flush=True,
            )
        if p99 > 20.0:
            print(
                f"Warning: high AE tail (p99={p99:.2f}, max={mx:.2f}) likely due to volatility; continuing",
                flush=True,
            )

    agg = test_df.groupby("decile")[ae_col].quantile([0.1, 0.5, 0.9]).unstack()
    print("AE deciles (p10/p50/p90):")
    print(agg.to_string())

    # event capture at top-k%
    for k in [0.05, 0.1]:
        cutoff = np.quantile(test_df["p_lfp"], 1 - k)
        sel = test_df["p_lfp"] >= cutoff
        capture = test_df.loc[sel, label_col].mean()
        base = test_df[label_col].mean()
        print(f"Top {int(k*100)}% capture={capture:.3f}, base_rate={base:.3f}, lift={(capture/base if base>0 else float('nan')):.2f}")

    # Event base rate per day and tail AE checks
    test_df["date"] = pd.to_datetime(test_df["Time"], utc=True, errors="coerce").dt.date.astype(str)
    for date, day_df in test_df.groupby("date"):
        base_rate = day_df[label_col].mean()
        top5_cut = np.quantile(day_df["p_lfp"], 0.95)
        top10_cut = np.quantile(day_df["p_lfp"], 0.90)
        top5_rate = day_df.loc[day_df["p_lfp"] >= top5_cut, label_col].mean()
        top10_rate = day_df.loc[day_df["p_lfp"] >= top10_cut, label_col].mean()
        tail_all = day_df["adverse_points"].quantile(0.99)
        top10_tail = day_df.loc[day_df["p_lfp"] >= top10_cut, "adverse_points"].quantile(0.99)
        top20_cut = np.quantile(day_df["p_lfp"], 0.80)
        top20_tail = day_df.loc[day_df["p_lfp"] >= top20_cut, "adverse_points"].quantile(0.99)
        print(
            f"{date} base_rate={base_rate:.3f} "
            f"top5_rate={top5_rate:.3f} top10_rate={top10_rate:.3f} "
            f"tail99={tail_all:.2f} top10_tail99={top10_tail:.2f} top20_tail99={top20_tail:.2f}"
        )

        eval_mask = valid_t.loc[day_df.index]
        eval_df = day_df.loc[eval_mask]
        if eval_df.empty:
            print(f"{date} filter_benefit: no evaluated_t rows")
            continue
        score = eval_df["p_lfp"]
        ae_abs = eval_df[ae_col].abs()
        top10_cut_eval = np.quantile(score, 0.90)
        top5_cut_eval = np.quantile(score, 0.95)
        top10 = eval_df[score >= top10_cut_eval]
        top5 = eval_df[score >= top5_cut_eval]
        bottom90 = eval_df[score < top10_cut_eval]
        def _stats(name: str, subset: pd.DataFrame):
            if subset.empty:
                return f"{name}: n=0"
            return (
                f"{name}: n={len(subset)} "
                f"mean_abs={subset[ae_col].abs().mean():.2f} "
                f"p99_abs={subset[ae_col].abs().quantile(0.99):.2f}"
            )
        print(
            f"{date} filter_benefit "
            f"all: n={len(eval_df)} tail99_abs={ae_abs.quantile(0.99):.2f} "
            f"{_stats('top10', top10)} "
            f"{_stats('bottom90', bottom90)} "
            f"{_stats('top5', top5)}"
        )
    print(f"score std: {test_df['p_lfp'].std():.6g}")


if __name__ == "__main__":
    main()
