"""
Resample processed features_labels.parquet to coarser bar sizes.

Input:
  data/processed/instrument=ES/date=YYYY-MM-DD/features_labels.parquet

Output:
  {OUTPUT_DIR}/step_ms=XYZ/instrument=ES/date=YYYY-MM-DD/features_labels.parquet

Env vars:
  INPUT_DIR             input root (default: data/processed)
  OUTPUT_DIR            output root (default: data/processed_resampled)
  INSTRUMENT            instrument symbol (default: ES)
  TARGET_STEP_MS        target bar size in ms (default: 250)
  TARGET_STEP_MS_LIST   comma list of target bar sizes (optional)
  DATE_FILTER           optional YYYY-MM-DD to write a single day only
  START_DATE            optional YYYY-MM-DD inclusive lower bound
  END_DATE              optional YYYY-MM-DD inclusive upper bound
  HORIZON_MS            label horizon in ms (default: 1000)
  NEUTRAL_BAND_BPS      neutral band for label (default: 0.0)
  REPRICE_BPS           repricing event threshold (default: 4.0)
  FLOW_WINDOW           rolling window for derived features (default: 20)
  LFP_HORIZONS          comma-separated horizons (default: 10,20,50)
  LFP_AE_X_PCT          percentile for AE threshold X (default: 0.90)
  LFP_AE_X_MODE         global|day (default: day)
  LFP_LAMBDA_FAST       rolling window for lambda fast (default: 20)
  LFP_LAMBDA_SLOW       rolling window for lambda slow (default: 100)
  LFP_STRESS_WINDOW     rolling window for stress/imbalance/spread (default: 50)
  LFP_TICK_SIZE         tick size override (default: 0)
  STRICT_HEALTH         if "1", fail on health check (default: 0)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Reuse feature/label helpers from build_dataset.
from build_dataset import (
    _add_derived_features,
    _add_lfp_features_labels,
    _label_rows,
    _day_health_stats,
    _tick_size_for_symbol,
    _validate_book_sanity,
)


def _list_days(root: Path, instrument: str, date_filter: str, start_date: str, end_date: str) -> List[str]:
    inst_dir = root / f"instrument={instrument}"
    if not inst_dir.exists():
        return []
    days = []
    for d in inst_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith("date="):
            name = name.split("=", 1)[1]
        days.append(name)
    if date_filter:
        days = [d for d in days if d == date_filter]
    if start_date:
        days = [d for d in days if d >= start_date]
    if end_date:
        days = [d for d in days if d <= end_date]
    return sorted(set(days))


def _resolve_day_dir(root: Path, instrument: str, day: str) -> Path:
    inst_dir = root / f"instrument={instrument}"
    dated = inst_dir / f"date={day}"
    if dated.exists():
        return dated
    return inst_dir / day


def _resample_day(
    df_day: pd.DataFrame,
    step_ms: int,
    flow_window: int,
    lfp_horizons: List[int],
    lfp_x_pct: float,
    lfp_x_mode: str,
    lfp_lambda_fast: int,
    lfp_lambda_slow: int,
    lfp_stress_window: int,
    horizon_ms: int,
    neutral_band_bps: float,
    reprice_bps: float,
    tick_size: float,
) -> pd.DataFrame:
    df_day = df_day.copy()
    df_day["Time"] = pd.to_datetime(df_day["Time"], utc=True, errors="coerce")
    df_day = df_day.sort_values("Time")
    symbol = str(df_day["Symbol"].iloc[0]) if "Symbol" in df_day.columns else "NA"

    sum_cols = ["trade_volume", "signed_volume", "trade_count"]
    last_cols = [
        "bid_price_1",
        "ask_price_1",
        "bid_size_1",
        "ask_size_1",
        "top_bid_depth",
        "top_ask_depth",
        "depth_bid_top5",
        "depth_ask_top5",
        "depth_imbalance_top5",
        "book_slope_top5",
        "sweep_cost_buy1",
        "sweep_cost_sell1",
    ]

    agg: Dict[str, str] = {}
    for col in sum_cols:
        if col in df_day.columns:
            agg[col] = "sum"
    for col in last_cols:
        if col in df_day.columns:
            agg[col] = "last"
    if not agg:
        raise ValueError("No columns available for resampling.")

    resampled = (
        df_day.set_index("Time")
        .resample(f"{step_ms}ms", label="left", closed="left")
        .agg(agg)
        .reset_index()
    )
    resampled = resampled.dropna(subset=["bid_price_1", "ask_price_1"])
    resampled["Symbol"] = symbol

    # Fill missing depth columns from L1 when available.
    if "top_bid_depth" not in resampled.columns and "bid_size_1" in resampled.columns:
        resampled["top_bid_depth"] = resampled["bid_size_1"]
    if "top_ask_depth" not in resampled.columns and "ask_size_1" in resampled.columns:
        resampled["top_ask_depth"] = resampled["ask_size_1"]
    if "depth_bid_top5" not in resampled.columns and "top_bid_depth" in resampled.columns:
        resampled["depth_bid_top5"] = resampled["top_bid_depth"]
    if "depth_ask_top5" not in resampled.columns and "top_ask_depth" in resampled.columns:
        resampled["depth_ask_top5"] = resampled["top_ask_depth"]
    for col in ["book_slope_top5", "sweep_cost_buy1", "sweep_cost_sell1"]:
        if col not in resampled.columns:
            resampled[col] = np.nan

    bid = pd.to_numeric(resampled["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(resampled["ask_price_1"], errors="coerce")
    resampled["mid"] = 0.5 * (bid + ask)
    resampled["spread"] = ask - bid

    if "top_bid_depth" in resampled.columns and "top_ask_depth" in resampled.columns:
        denom = (
            pd.to_numeric(resampled["top_bid_depth"], errors="coerce")
            + pd.to_numeric(resampled["top_ask_depth"], errors="coerce")
        )
        resampled["order_book_imbalance"] = (
            pd.to_numeric(resampled["top_bid_depth"], errors="coerce")
            / denom.replace(0, np.nan)
        )
    if "depth_bid_top5" in resampled.columns and "depth_ask_top5" in resampled.columns:
        denom = (
            pd.to_numeric(resampled["depth_bid_top5"], errors="coerce")
            + pd.to_numeric(resampled["depth_ask_top5"], errors="coerce")
        )
        resampled["depth_imbalance_top5"] = (
            pd.to_numeric(resampled["depth_bid_top5"], errors="coerce")
            - pd.to_numeric(resampled["depth_ask_top5"], errors="coerce")
        ) / denom.replace(0, np.nan)

    # Derived features + labels.
    resampled = _add_derived_features(resampled, step_ms=step_ms, window=flow_window)
    resampled = _add_lfp_features_labels(
        resampled,
        horizons=lfp_horizons,
        x_pct=lfp_x_pct,
        x_mode=lfp_x_mode,
        win_fast=lfp_lambda_fast,
        win_slow=lfp_lambda_slow,
        stress_window=lfp_stress_window,
    )
    horizon_steps = max(1, int(horizon_ms // max(step_ms, 1)))
    resampled, _ = _label_rows(
        resampled,
        horizon_steps=horizon_steps,
        neutral_band_bps=neutral_band_bps,
        reprice_bps=reprice_bps,
    )

    # Basic sanity checks (non-fatal by default).
    _validate_book_sanity(resampled, step_ms=step_ms, tick_size=tick_size, jump_bound_points=20.0)

    # Dtype tightening to match build_dataset output.
    float32_cols = [
        "bid_price_1",
        "ask_price_1",
        "mid",
        "spread",
        "order_book_imbalance",
        "depth_imbalance_top5",
        "book_slope_top5",
        "sweep_cost_buy1",
        "sweep_cost_sell1",
        "fwd_ret",
        "fwd_ret_net",
        "fav_excursion",
        "trade_volume",
        "signed_volume",
    ]
    int32_cols = [
        "trade_count",
        "bid_size_1",
        "ask_size_1",
        "top_bid_depth",
        "top_ask_depth",
        "depth_bid_top5",
        "depth_ask_top5",
    ]
    int8_cols = ["label"]
    for col in float32_cols:
        if col in resampled.columns:
            resampled[col] = resampled[col].astype("float32")
    for col in int32_cols:
        if col in resampled.columns:
            resampled[col] = pd.to_numeric(resampled[col], errors="coerce").fillna(0).astype("int32")
    for col in int8_cols:
        if col in resampled.columns:
            resampled[col] = pd.to_numeric(resampled[col], errors="coerce").fillna(0).astype("int8")

    return resampled


def main() -> None:
    input_dir = Path(os.environ.get("INPUT_DIR", "data/processed")).expanduser().resolve()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "data/processed_resampled")).expanduser().resolve()
    instrument = os.environ.get("INSTRUMENT", "ES").strip()
    target_step_env = os.environ.get("TARGET_STEP_MS", "250").strip()
    target_step_list_env = os.environ.get("TARGET_STEP_MS_LIST", "").strip()
    date_filter = os.environ.get("DATE_FILTER", "").strip()
    start_date = os.environ.get("START_DATE", "").strip()
    end_date = os.environ.get("END_DATE", "").strip()

    horizon_ms = int(os.environ.get("HORIZON_MS", "1000"))
    neutral_band_bps = float(os.environ.get("NEUTRAL_BAND_BPS", "0.0"))
    reprice_bps = float(os.environ.get("REPRICE_BPS", "4.0"))
    flow_window = int(os.environ.get("FLOW_WINDOW", "20"))
    lfp_horizons = [int(x.strip()) for x in os.environ.get("LFP_HORIZONS", "10,20,50").split(",") if x.strip()]
    lfp_x_pct = float(os.environ.get("LFP_AE_X_PCT", "0.90"))
    lfp_x_mode = os.environ.get("LFP_AE_X_MODE", "day").strip().lower()
    lfp_lambda_fast = int(os.environ.get("LFP_LAMBDA_FAST", "20"))
    lfp_lambda_slow = int(os.environ.get("LFP_LAMBDA_SLOW", "100"))
    lfp_stress_window = int(os.environ.get("LFP_STRESS_WINDOW", "50"))
    lfp_tick_size = float(os.environ.get("LFP_TICK_SIZE", "0"))
    strict_health = os.environ.get("STRICT_HEALTH", "0").strip() == "1"

    if target_step_list_env:
        target_steps = [int(x.strip()) for x in target_step_list_env.split(",") if x.strip()]
    else:
        target_steps = [int(target_step_env)] if target_step_env else [250]

    days = _list_days(input_dir, instrument, date_filter, start_date, end_date)
    if not days:
        raise FileNotFoundError(f"No days found under {input_dir}/instrument={instrument}")

    tick_size = _tick_size_for_symbol(instrument, lfp_tick_size)

    for step_ms in target_steps:
        for day in days:
            in_day_dir = _resolve_day_dir(input_dir, instrument, day)
            in_path = in_day_dir / "features_labels.parquet"
            if not in_path.exists():
                print(f"Missing input: {in_path}", flush=True)
                continue
            df_day = pd.read_parquet(in_path)
            if df_day.empty:
                print(f"Empty day: {in_path}", flush=True)
                continue
            out_day_dir = _resolve_day_dir(output_dir / f"step_ms={step_ms}", instrument, day)
            out_day_dir.mkdir(parents=True, exist_ok=True)

            df_out = _resample_day(
                df_day,
                step_ms=step_ms,
                flow_window=flow_window,
                lfp_horizons=lfp_horizons,
                lfp_x_pct=lfp_x_pct,
                lfp_x_mode=lfp_x_mode,
                lfp_lambda_fast=lfp_lambda_fast,
                lfp_lambda_slow=lfp_lambda_slow,
                lfp_stress_window=lfp_stress_window,
                horizon_ms=horizon_ms,
                neutral_band_bps=neutral_band_bps,
                reprice_bps=reprice_bps,
                tick_size=tick_size,
            )

            health = _day_health_stats(df_out)
            health_path = out_day_dir / "health.json"
            with open(health_path, "w", encoding="utf-8") as f:
                json.dump(health, f, indent=2)
            if strict_health and (
                health["nunique_bid"] < 50
                or health["nunique_ask"] < 50
                or health["mid_change_pct"] < 0.005
            ):
                raise RuntimeError(f"Day {day} failed health checks: {health}")

            out_path = out_day_dir / "features_labels.parquet"
            df_out.drop(columns=["date"], errors="ignore").to_parquet(out_path, index=False)
            print(f"Wrote {len(df_out)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
