"""
LRAMS falsification experiment: Liquidity Refill Asymmetry After Micro-Stress.

Env vars:
  DATA_DIR         root data dir (default: data/processed)
  OUTPUT_DIR       artifacts output dir (default: artifacts/lrams)
  TEST_DAYS        comma-separated YYYY-MM-DD (optional)
  BINS             bucket count for asym (default: 10)
  BAR_MS           bar size in ms (default: 100)
  HORIZON_BARS     label horizon in bars (default: 20)
  STABLE_BARS      price stability window N (default: 3)
  REFILL_BARS      response window R (default: 10)
  V_MIN            min abs signed_volume (default: 1)
  V_MAX            max abs signed_volume (default: 10)
  SPREAD_MIN_T     min spread in ticks (default: 1)
  SPREAD_MAX_T     max spread in ticks (default: 4)
  TICK_SIZE        tick size (default: 0.25)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data


def _forward_roll_min(arr: np.ndarray, window: int) -> np.ndarray:
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


def _forward_roll_max(arr: np.ndarray, window: int) -> np.ndarray:
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


def _auc_deficit(pre_depth: float, window: np.ndarray) -> float:
    if not np.isfinite(pre_depth) or pre_depth <= 0:
        return np.nan
    if window.size == 0:
        return np.nan
    deficit = np.maximum(0.0, (pre_depth - window) / pre_depth)
    return float(np.nanmean(deficit))


def _time_to_recovery(pre_depth: float, window: np.ndarray, thresh: float, max_k: int) -> int:
    if not np.isfinite(pre_depth) or pre_depth <= 0 or window.size == 0:
        return max_k + 1
    target = pre_depth * thresh
    for k, val in enumerate(window, start=1):
        if np.isfinite(val) and val >= target:
            return k
    return max_k + 1


def _stable_mask(arr: np.ndarray, n_bars: int) -> np.ndarray:
    n = len(arr)
    if n_bars <= 0:
        return np.zeros(n, dtype=bool)
    fwd_min = _forward_roll_min(arr[1:], n_bars)
    fwd_max = _forward_roll_max(arr[1:], n_bars)
    out = np.zeros(n, dtype=bool)
    if len(fwd_min) > 0:
        pad_min = np.full(n, np.nan, dtype=float)
        pad_max = np.full(n, np.nan, dtype=float)
        pad_min[: len(fwd_min)] = fwd_min
        pad_max[: len(fwd_max)] = fwd_max
        out = (pad_min == arr) & (pad_max == arr)
    return out


def _label_first_move(
    bid_ticks: np.ndarray,
    ask_ticks: np.ndarray,
    i: int,
    horizon: int,
) -> int:
    bid0 = bid_ticks[i]
    ask0 = ask_ticks[i]
    first_bid = None
    first_ask = None
    end = min(i + horizon, len(bid_ticks) - 1)
    for j in range(i + 1, end + 1):
        if first_bid is None and bid_ticks[j] <= bid0 - 1:
            first_bid = j
        if first_ask is None and ask_ticks[j] >= ask0 + 1:
            first_ask = j
        if first_bid is not None and first_ask is not None:
            break
    if first_bid is None and first_ask is None:
        return 0
    if first_bid is None:
        return 1
    if first_ask is None:
        return -1
    if first_bid < first_ask:
        return -1
    if first_ask < first_bid:
        return 1
    return 0


def _adverse_move(mid: np.ndarray, i: int, horizon: int, side: int) -> float:
    end = min(i + horizon, len(mid) - 1)
    window = mid[i + 1 : end + 1]
    if window.size == 0:
        return np.nan
    if side > 0:
        return float(window.min() - mid[i])
    return float(window.max() - mid[i])


def _load_data() -> pd.DataFrame:
    data_dir = Path(os.environ.get("DATA_DIR", "data/processed")).expanduser().resolve()
    df = load_all_raw_data(data_dir).reset_index()
    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df = df.sort_values(["Symbol", "Time"]).reset_index(drop=True)
    days_env = os.environ.get("TEST_DAYS", "").strip()
    if days_env:
        keep = {d.strip() for d in days_env.split(",") if d.strip()}
        df = df[df["Time"].dt.date.astype(str).isin(keep)]
        df = df.sort_values(["Symbol", "Time"]).reset_index(drop=True)
    return df


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _build_events(
    df: pd.DataFrame,
    v_min: float,
    v_max: float,
    spread_min_t: int,
    spread_max_t: int,
    stable_bars: int,
    refill_bars: int,
    horizon_bars: int,
    tick_size: float,
) -> pd.DataFrame:
    required = [
        "Symbol",
        "Time",
        "bid_price_1",
        "ask_price_1",
        "bid_size_1",
        "ask_size_1",
        "top_bid_depth",
        "top_ask_depth",
        "signed_volume",
        "trade_count",
    ]
    _ensure_columns(df, required)
    has_top5 = {"depth_bid_top5", "depth_ask_top5"}.issubset(df.columns)
    has_top3 = {"depth_bid_top3", "depth_ask_top3"}.issubset(df.columns)

    bid = pd.to_numeric(df["bid_price_1"], errors="coerce").to_numpy()
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce").to_numpy()
    mid = 0.5 * (bid + ask)
    bid_ticks = np.rint(bid / tick_size).astype(np.int64)
    ask_ticks = np.rint(ask / tick_size).astype(np.int64)
    mid_ticks = np.rint(mid / tick_size).astype(np.int64)
    spread_ticks = np.rint((ask - bid) / tick_size).astype(np.int64)

    agg_trade = (df["trade_count"].to_numpy() > 0) & (df["signed_volume"].to_numpy() != 0)
    vol = np.abs(df["signed_volume"].to_numpy())
    vol_band = (vol >= v_min) & (vol <= v_max)
    spread_band = (spread_ticks >= spread_min_t) & (spread_ticks <= spread_max_t)

    stable_bid = _stable_mask(bid_ticks.astype(float), stable_bars)
    stable_ask = _stable_mask(ask_ticks.astype(float), stable_bars)
    stable_mid = _stable_mask(mid_ticks.astype(float), stable_bars)
    stable = stable_bid & stable_ask & stable_mid

    event_mask = pd.Series(agg_trade & vol_band & spread_band & stable, index=df.index)

    records = []
    max_lookahead = max(stable_bars, refill_bars, horizon_bars)
    for symbol, group in df.groupby("Symbol", sort=False):
        idx = group.index.to_numpy()
        if len(idx) <= max_lookahead:
            continue
        for i_pos, i in enumerate(idx):
            if i_pos + max_lookahead >= len(idx):
                break
            if not bool(event_mask.loc[i]):
                continue
            side = int(np.sign(df.loc[i, "signed_volume"]))
            if side == 0:
                continue

            bid0 = float(df.loc[i, "bid_size_1"])
            ask0 = float(df.loc[i, "ask_size_1"])
            if not np.isfinite(bid0) or not np.isfinite(ask0) or bid0 <= 0 or ask0 <= 0:
                continue

            depth_bid = df.loc[idx[i_pos + 1 : i_pos + refill_bars], "bid_size_1"].to_numpy(dtype=float)
            depth_ask = df.loc[idx[i_pos + 1 : i_pos + refill_bars], "ask_size_1"].to_numpy(dtype=float)
            bid_def = np.maximum(0.0, (bid0 - depth_bid) / bid0)
            ask_def = np.maximum(0.0, (ask0 - depth_ask) / ask0)
            auc_bid = float(np.nanmean(bid_def)) if depth_bid.size else np.nan
            auc_ask = float(np.nanmean(ask_def)) if depth_ask.size else np.nan

            if side > 0:
                asym = auc_ask - auc_bid
            else:
                asym = auc_bid - auc_ask

            t80_bid = _time_to_recovery(bid0, depth_bid, 0.80, refill_bars)
            t80_ask = _time_to_recovery(ask0, depth_ask, 0.80, refill_bars)
            t95_bid = _time_to_recovery(bid0, depth_bid, 0.95, refill_bars)
            t95_ask = _time_to_recovery(ask0, depth_ask, 0.95, refill_bars)

            auc_bid_top5 = np.nan
            auc_ask_top5 = np.nan
            asym_top5 = np.nan
            t80_top5_bid = np.nan
            t80_top5_ask = np.nan
            t95_top5_bid = np.nan
            t95_top5_ask = np.nan

            auc_bid_top3 = np.nan
            auc_ask_top3 = np.nan
            asym_top3 = np.nan
            t80_top3_bid = np.nan
            t80_top3_ask = np.nan
            t95_top3_bid = np.nan
            t95_top3_ask = np.nan
            if has_top5:
                bid5_0 = float(df.loc[i, "depth_bid_top5"])
                ask5_0 = float(df.loc[i, "depth_ask_top5"])
                depth_bid5 = df.loc[idx[i_pos + 1 : i_pos + refill_bars], "depth_bid_top5"].to_numpy(dtype=float)
                depth_ask5 = df.loc[idx[i_pos + 1 : i_pos + refill_bars], "depth_ask_top5"].to_numpy(dtype=float)
                auc_bid_top5 = _auc_deficit(bid5_0, depth_bid5)
                auc_ask_top5 = _auc_deficit(ask5_0, depth_ask5)
                if side > 0:
                    asym_top5 = auc_ask_top5 - auc_bid_top5
                else:
                    asym_top5 = auc_bid_top5 - auc_ask_top5
                t80_top5_bid = _time_to_recovery(bid5_0, depth_bid5, 0.80, refill_bars)
                t80_top5_ask = _time_to_recovery(ask5_0, depth_ask5, 0.80, refill_bars)
                t95_top5_bid = _time_to_recovery(bid5_0, depth_bid5, 0.95, refill_bars)
                t95_top5_ask = _time_to_recovery(ask5_0, depth_ask5, 0.95, refill_bars)
            if has_top3:
                bid3_0 = float(df.loc[i, "depth_bid_top3"])
                ask3_0 = float(df.loc[i, "depth_ask_top3"])
                depth_bid3 = df.loc[idx[i_pos + 1 : i_pos + refill_bars], "depth_bid_top3"].to_numpy(dtype=float)
                depth_ask3 = df.loc[idx[i_pos + 1 : i_pos + refill_bars], "depth_ask_top3"].to_numpy(dtype=float)
                auc_bid_top3 = _auc_deficit(bid3_0, depth_bid3)
                auc_ask_top3 = _auc_deficit(ask3_0, depth_ask3)
                if side > 0:
                    asym_top3 = auc_ask_top3 - auc_bid_top3
                else:
                    asym_top3 = auc_bid_top3 - auc_ask_top3
                t80_top3_bid = _time_to_recovery(bid3_0, depth_bid3, 0.80, refill_bars)
                t80_top3_ask = _time_to_recovery(ask3_0, depth_ask3, 0.80, refill_bars)
                t95_top3_bid = _time_to_recovery(bid3_0, depth_bid3, 0.95, refill_bars)
                t95_top3_ask = _time_to_recovery(ask3_0, depth_ask3, 0.95, refill_bars)

            label = _label_first_move(bid_ticks, ask_ticks, i, horizon_bars)
            adverse = _adverse_move(mid, i, horizon_bars, side)

            records.append(
                {
                    "Symbol": df.loc[i, "Symbol"],
                    "Time": df.loc[i, "Time"],
                    "side": side,
                    "signed_volume": float(df.loc[i, "signed_volume"]),
                    "spread_ticks": int(spread_ticks[i]),
                    "auc_bid": auc_bid,
                    "auc_ask": auc_ask,
                    "asym": asym,
                    "t80_bid": t80_bid,
                    "t80_ask": t80_ask,
                    "t95_bid": t95_bid,
                    "t95_ask": t95_ask,
                    "auc_bid_top5": auc_bid_top5,
                    "auc_ask_top5": auc_ask_top5,
                    "asym_top5": asym_top5,
                    "auc_bid_top3": auc_bid_top3,
                    "auc_ask_top3": auc_ask_top3,
                    "asym_top3": asym_top3,
                    "t80_top5_bid": t80_top5_bid,
                    "t80_top5_ask": t80_top5_ask,
                    "t95_top5_bid": t95_top5_bid,
                    "t95_top5_ask": t95_top5_ask,
                    "t80_top3_bid": t80_top3_bid,
                    "t80_top3_ask": t80_top3_ask,
                    "t95_top3_bid": t95_top3_bid,
                    "t95_top3_ask": t95_top3_ask,
                    "label": label,
                    "adverse_move": adverse,
                }
            )

    return pd.DataFrame(records)


def _bucket_stats(events: pd.DataFrame, bins: int, side: int, score_col: str) -> pd.DataFrame:
    df = events.copy()
    df = df[df["side"] == side]
    df = df.dropna(subset=[score_col, "label"])
    df["bucket"] = pd.qcut(df[score_col], bins, labels=False, duplicates="drop")
    if side > 0:
        df["y"] = (df["label"] == 1).astype(int)
    else:
        df["y"] = (df["label"] == -1).astype(int)
    out = (
        df.groupby("bucket")
        .agg(
            count=("label", "size"),
            rate=("y", "mean"),
            adverse_p90=("adverse_move", lambda s: s.quantile(0.9)),
            adverse_p95=("adverse_move", lambda s: s.quantile(0.95)),
            adverse_p99=("adverse_move", lambda s: s.quantile(0.99)),
        )
        .reset_index()
    )
    return out


def _auc_score(events: pd.DataFrame, side: int, score_col: str) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")
    df = events[events["side"] == side].copy()
    df = df[df["label"] != 0].dropna(subset=[score_col])
    if df.empty:
        return float("nan")
    y = (df["label"] == (1 if side > 0 else -1)).astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, df[score_col]))


def _permute_labels(events: pd.DataFrame) -> pd.DataFrame:
    df = events.copy()
    df["date"] = df["Time"].dt.date.astype(str)
    def _shuffle(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["label"] = np.random.permutation(g["label"].to_numpy())
        return g
    return df.groupby(["date", "side"], group_keys=False).apply(_shuffle)


def _top_bucket_info(df: pd.DataFrame, score_col: str, bins: int) -> Dict[str, float]:
    df = df.dropna(subset=[score_col])
    df["bucket"] = pd.qcut(df[score_col], bins, labels=False, duplicates="drop")
    top = df[df["bucket"] == df["bucket"].max()]
    return {
        "top_bucket": float(df["bucket"].max()) if not df.empty else np.nan,
        "top_mean_asym": float(top[score_col].mean()) if not top.empty else np.nan,
    }


def _corrs(df: pd.DataFrame, score_col: str) -> Dict[str, float]:
    out = {}
    if score_col not in df.columns:
        return out
    for other in ["asym_top5", "asym", "spread_ticks", "bid_size_1", "ask_size_1"]:
        if other in df.columns and df[score_col].notna().any() and df[other].notna().any():
            out[f"{score_col}_vs_{other}"] = float(df[score_col].corr(df[other], method="spearman"))
    return out


def _depth_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["top_bid_depth", "top_ask_depth", "depth_bid_top5", "depth_ask_top5"]
    out = []
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        out.append(
            {
                "metric": col,
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p10": float(series.quantile(0.10)),
                "p90": float(series.quantile(0.90)),
            }
        )
    return pd.DataFrame(out)


def _conditional_means(df: pd.DataFrame, score_col: str, side: int) -> Dict[str, float]:
    df = df[df["side"] == side].copy()
    df = df[df["label"] != 0].dropna(subset=[score_col])
    if df.empty:
        return {}
    y = (df["label"] == (1 if side > 0 else -1)).astype(int)
    return {
        "mean_y1": float(df.loc[y == 1, score_col].mean()),
        "mean_y0": float(df.loc[y == 0, score_col].mean()),
    }


def main() -> None:
    bar_ms = int(os.environ.get("BAR_MS", "100"))
    horizon_bars = int(os.environ.get("HORIZON_BARS", "20"))
    stable_bars = int(os.environ.get("STABLE_BARS", "3"))
    refill_bars = int(os.environ.get("REFILL_BARS", "10"))
    v_min = float(os.environ.get("V_MIN", "1"))
    v_max = float(os.environ.get("V_MAX", "10"))
    spread_min_t = int(os.environ.get("SPREAD_MIN_T", "1"))
    spread_max_t = int(os.environ.get("SPREAD_MAX_T", "4"))
    tick_size = float(os.environ.get("TICK_SIZE", "0.25"))
    bins = int(os.environ.get("BINS", "10"))
    permute = os.environ.get("PERMUTE_LABELS", "0").strip() in {"1", "true", "yes", "y"}

    df = _load_data()
    if df.empty:
        raise ValueError("No data loaded.")

    events = _build_events(
        df,
        v_min=v_min,
        v_max=v_max,
        spread_min_t=spread_min_t,
        spread_max_t=spread_max_t,
        stable_bars=stable_bars,
        refill_bars=refill_bars,
        horizon_bars=horizon_bars,
        tick_size=tick_size,
    )
    if events.empty:
        raise ValueError("No events found with current filters.")

    out_dir = Path(os.environ.get("OUTPUT_DIR", "artifacts/lrams")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.parquet"
    events_csv = out_dir / "events.csv"
    events.to_parquet(events_path, index=False)
    events.to_csv(events_csv, index=False)

    def _eval_score(score_col: str, side: int, name: str) -> Tuple[float, float]:
        auc = _auc_score(events, side=side, score_col=score_col)
        stats = _bucket_stats(events, bins=bins, side=side, score_col=score_col)
        stats_path = out_dir / f"bucket_stats_{name}_{score_col}.csv"
        stats.to_csv(stats_path, index=False)
        side_df = events[events["side"] == side]
        base = (side_df["label"] == (1 if side > 0 else -1)).mean()
        bucket0 = stats["rate"].iloc[0] if not stats.empty else float("nan")
        bucket9 = stats["rate"].iloc[-1] if not stats.empty else float("nan")
        lift_base = (bucket9 / base) if base and base > 0 else float("nan")
        lift_bottom = (bucket9 / bucket0) if bucket0 and bucket0 > 0 else float("nan")
        info = _top_bucket_info(side_df, score_col=score_col, bins=bins)
        print(
            f"{name} {score_col} AUC={auc:.4f} base_rate={base:.3f} "
            f"bucket0_rate={bucket0:.3f} bucket9_rate={bucket9:.3f} "
            f"lift_vs_base={lift_base:.2f} lift_vs_bottom={lift_bottom:.2f} "
            f"top_bucket={info['top_bucket']:.0f} top_mean_asym={info['top_mean_asym']:.6g}"
        )
        return auc, lift_base

    # Optional combined score
    events = events.copy()
    if "asym_top5" in events.columns:
        events["date"] = events["Time"].dt.date.astype(str)
        for side in [1, -1]:
            mask = events["side"] == side
            for date, g in events.loc[mask].groupby("date"):
                idx = g.index
                z1 = (g["asym"] - g["asym"].mean()) / (g["asym"].std() + 1e-9)
                z2 = (g["asym_top5"] - g["asym_top5"].mean()) / (g["asym_top5"].std() + 1e-9)
                events.loc[idx, "asym_combo"] = z1 + z2

    for side, name in [(1, "buy"), (-1, "sell")]:
        side_df = events[events["side"] == side]
        if side_df.empty:
            continue
        corrs = {}
        corrs.update(_corrs(side_df, "asym"))
        if "asym_top5" in side_df.columns:
            corrs.update(_corrs(side_df, "asym_top5"))
        if "asym_top3" in side_df.columns:
            corrs.update(_corrs(side_df, "asym_top3"))
        if corrs:
            print(f"{name} correlations: {corrs}")

    debug_rows = []
    for side, name in [(1, "buy"), (-1, "sell")]:
        side_df = events[events["side"] == side]
        print(f"{name} events: {len(side_df)}")
        _eval_score("asym", side=side, name=name)
        if "asym_top5" in events.columns and events["asym_top5"].notna().any():
            _eval_score("asym_top5", side=side, name=name)
        else:
            print(f"{name} asym_top5 unavailable")
        if "asym_top3" in events.columns and events["asym_top3"].notna().any():
            _eval_score("asym_top3", side=side, name=name)
        else:
            print(f"{name} asym_top3 unavailable")
        if "asym_combo" in events.columns and events["asym_combo"].notna().any():
            _eval_score("asym_combo", side=side, name=name)

        if {"depth_bid_top5", "depth_ask_top5", "top_bid_depth", "top_ask_depth"}.issubset(events.columns):
            corr_bid = float(side_df["depth_bid_top5"].corr(side_df["top_bid_depth"])) if not side_df.empty else float("nan")
            corr_ask = float(side_df["depth_ask_top5"].corr(side_df["top_ask_depth"])) if not side_df.empty else float("nan")
            print(f"{name} depth_top5 corr: bid={corr_bid:.3f} ask={corr_ask:.3f}")
            if (side_df["depth_bid_top5"] < side_df["top_bid_depth"]).mean() > 0.01 or (side_df["depth_ask_top5"] < side_df["top_ask_depth"]).mean() > 0.01:
                print("Warning: depth_top5 < top_depth for nontrivial fraction; top5 may not be raw depth sum.")

        for score in ["asym", "asym_top5"]:
            if score in events.columns and events[score].notna().any():
                cm = _conditional_means(events, score, side)
                y_label = "ask_up_first" if side > 0 else "bid_down_first"
                print(f"{name} {score} conditional means (y=1:{y_label}): {cm}")
                debug_rows.append(
                    {
                        "side": name,
                        "score": score,
                        "y1_label": y_label,
                        "mean_y1": cm.get("mean_y1"),
                        "mean_y0": cm.get("mean_y0"),
                    }
                )

        if "asym_top5" in events.columns and events["asym_top5"].notna().any():
            events["asym_top5_alt"] = -events["asym_top5"]
            auc_top5 = _auc_score(events, side=side, score_col="asym_top5")
            auc_top5_alt = _auc_score(events, side=side, score_col="asym_top5_alt")
            print(f"{name} asym_top5 AUC={auc_top5:.4f} asym_top5_alt AUC={auc_top5_alt:.4f}")
            debug_rows.append(
                {
                    "side": name,
                    "score": "asym_top5_alt",
                    "y1_label": "ask_up_first" if side > 0 else "bid_down_first",
                    "mean_y1": np.nan,
                    "mean_y0": np.nan,
                    "auc": auc_top5_alt,
                }
            )

    if permute:
        perm_events = _permute_labels(events)
        for side, name in [(1, "buy"), (-1, "sell")]:
            for score_col in ["asym", "asym_top5"]:
                if score_col not in perm_events.columns or perm_events[score_col].notna().sum() == 0:
                    continue
                auc = _auc_score(perm_events, side=side, score_col=score_col)
                stats = _bucket_stats(perm_events, bins=bins, side=side, score_col=score_col)
                side_df = perm_events[perm_events["side"] == side]
                base = (side_df["label"] == (1 if side > 0 else -1)).mean()
                bottom = stats["rate"].iloc[0] if not stats.empty else float("nan")
                top = stats["rate"].iloc[-1] if not stats.empty else float("nan")
                lift_base = (top / base) if base and base > 0 else float("nan")
                lift_bottom = (top / bottom) if bottom and bottom > 0 else float("nan")
                note = ""
                if not np.isnan(auc) and (auc < 0.45 or auc > 0.55):
                    note = " (perm AUC off 0.50)"
                if not np.isnan(lift_base) and (lift_base < 0.8 or lift_base > 1.2):
                    note = f"{note} (perm lift off 1.0)"
                print(f"{name} permuted {score_col} AUC={auc:.4f} lift_base={lift_base:.2f} lift_bottom={lift_bottom:.2f}{note}")

    try:
        import matplotlib.pyplot as plt

        stats_buy = _bucket_stats(events, bins=bins, side=1, score_col="asym")
        stats_sell = _bucket_stats(events, bins=bins, side=-1, score_col="asym")
        stats_buy_top5 = _bucket_stats(events, bins=bins, side=1, score_col="asym_top5")
        stats_sell_top5 = _bucket_stats(events, bins=bins, side=-1, score_col="asym_top5")
        stats_buy_top3 = _bucket_stats(events, bins=bins, side=1, score_col="asym_top3")
        stats_sell_top3 = _bucket_stats(events, bins=bins, side=-1, score_col="asym_top3")
        fig, ax = plt.subplots(figsize=(8, 4))
        if not stats_buy.empty:
            ax.plot(stats_buy["bucket"], stats_buy["rate"], label="buy")
        if not stats_sell.empty:
            ax.plot(stats_sell["bucket"], stats_sell["rate"], label="sell")
        if not stats_buy_top5.empty:
            ax.plot(stats_buy_top5["bucket"], stats_buy_top5["rate"], label="buy_top5", linestyle="--")
        if not stats_sell_top5.empty:
            ax.plot(stats_sell_top5["bucket"], stats_sell_top5["rate"], label="sell_top5", linestyle="--")
        if not stats_buy_top3.empty:
            ax.plot(stats_buy_top3["bucket"], stats_buy_top3["rate"], label="buy_top3", linestyle=":")
        if not stats_sell_top3.empty:
            ax.plot(stats_sell_top3["bucket"], stats_sell_top3["rate"], label="sell_top3", linestyle=":")
        ax.set_xlabel("asym bucket")
        ax.set_ylabel("event rate")
        ax.legend()
        fig.tight_layout()
        fig_path = out_dir / "bucket_curve.png"
        fig.savefig(fig_path)
    except Exception:
        pass

    # Debug artifact
    if {"depth_bid_top5", "depth_ask_top5", "top_bid_depth", "top_ask_depth"}.issubset(events.columns):
        summary = _depth_summary(events)
        summary["corr_bid_top5_vs_top1"] = events["depth_bid_top5"].corr(events["top_bid_depth"])
        summary["corr_ask_top5_vs_top1"] = events["depth_ask_top5"].corr(events["top_ask_depth"])
        summary_path = out_dir / "debug_depth_summary.csv"
        summary.to_csv(summary_path, index=False)
    if debug_rows:
        debug_df = pd.DataFrame(debug_rows)
        debug_path = out_dir / "debug_conditional_means.csv"
        debug_df.to_csv(debug_path, index=False)

    print(f"Artifacts written to {out_dir}")


if __name__ == "__main__":
    main()
