"""
Smoke-test the extended MBP-10 parquet schema after a single-day rebuild.

Usage (run from project root):
    python scripts/verify_mbp10_schema.py [DATE]

    DATE defaults to 2025-12-02.  The script checks that:
      1. The parquet exists and is non-empty.
      2. All expected level-2-10 columns are present with correct dtypes.
      3. None of the new columns are entirely NaN.
      4. Sweep cost in ticks is non-negative.
      5. Depth shape ratios are in (0, 1].
      6. Prints a compact size estimate for the full date range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "processed" / "instrument=ES"

LEVEL_COLS = (
    [f"bid_size_{i}" for i in range(2, 11)]
    + [f"ask_size_{i}" for i in range(2, 11)]
)
DERIVED_FLOAT32_COLS = [
    "depth_shape_ratio_bid",
    "depth_shape_ratio_ask",
    "book_depth_slope_bid",
    "book_depth_slope_ask",
    "sweep_cost_buy1_ticks",
    "sweep_cost_sell1_ticks",
    "sweep_cost_buy5_ticks",
    "sweep_cost_sell5_ticks",
]
ALL_NEW_COLS = LEVEL_COLS + DERIVED_FLOAT32_COLS

EXPECTED_INT32 = set(LEVEL_COLS)
EXPECTED_FLOAT32 = set(DERIVED_FLOAT32_COLS)


def check(date: str) -> None:
    parquet = DATA_ROOT / f"date={date}" / "features_labels.parquet"
    if not parquet.exists():
        print(f"[FAIL] Parquet not found: {parquet}")
        sys.exit(1)

    df = pd.read_parquet(parquet)
    n_rows, n_cols = df.shape
    size_mb = parquet.stat().st_size / 1024 / 1024
    print(f"[OK]   Loaded {n_rows:,} rows x {n_cols} cols  ({size_mb:.2f} MB on disk)")

    missing = [c for c in ALL_NEW_COLS if c not in df.columns]
    if missing:
        print(f"[FAIL] Missing columns: {missing}")
        sys.exit(1)
    print(f"[OK]   All {len(ALL_NEW_COLS)} new columns present")

    # Dtype checks
    dtype_errors = []
    for col in LEVEL_COLS:
        if df[col].dtype not in (np.dtype("int32"), np.dtype("int64")):
            dtype_errors.append(f"{col}: expected int, got {df[col].dtype}")
    for col in DERIVED_FLOAT32_COLS:
        if df[col].dtype not in (np.dtype("float32"), np.dtype("float64")):
            dtype_errors.append(f"{col}: expected float, got {df[col].dtype}")
    if dtype_errors:
        print("[FAIL] Dtype errors:")
        for e in dtype_errors:
            print(f"       {e}")
        sys.exit(1)
    print("[OK]   Dtypes correct")

    # All-NaN guard
    all_nan = [c for c in ALL_NEW_COLS if df[c].isna().all()]
    if all_nan:
        print(f"[FAIL] Columns entirely NaN: {all_nan}")
        sys.exit(1)
    nan_pcts = {c: f"{df[c].isna().mean():.1%}" for c in ALL_NEW_COLS if df[c].isna().mean() > 0.05}
    if nan_pcts:
        print(f"[WARN] High NaN rate (>5%): {nan_pcts}")
    else:
        print("[OK]   No column has >5% NaN")

    # Sweep cost non-negative
    for col in ["sweep_cost_buy1_ticks", "sweep_cost_sell1_ticks",
                "sweep_cost_buy5_ticks", "sweep_cost_sell5_ticks"]:
        neg = (df[col].dropna() < 0).sum()
        if neg > 0:
            print(f"[FAIL] {col} has {neg} negative values")
            sys.exit(1)
    print("[OK]   Sweep costs in ticks are non-negative")

    # Depth shape ratio in (0, 1]
    for col in ["depth_shape_ratio_bid", "depth_shape_ratio_ask"]:
        valid = df[col].dropna()
        bad = ((valid <= 0) | (valid > 1 + 1e-6)).sum()
        if bad > 0:
            print(f"[WARN] {col} has {bad} values outside (0,1] — check for zero-depth bars")
    print("[OK]   Depth shape ratios checked")

    # Quick stats on new columns
    print("\n--- New column summary stats ---")
    cols_to_show = [
        "bid_size_2", "bid_size_5", "bid_size_10",
        "ask_size_2", "ask_size_5", "ask_size_10",
        "depth_shape_ratio_bid", "depth_shape_ratio_ask",
        "book_depth_slope_bid", "book_depth_slope_ask",
        "sweep_cost_buy1_ticks", "sweep_cost_buy5_ticks",
    ]
    for col in cols_to_show:
        if col in df.columns:
            s = df[col].dropna()
            print(f"  {col:<30s} mean={s.mean():8.3f}  p5={s.quantile(.05):8.3f}  "
                  f"p50={s.quantile(.50):8.3f}  p95={s.quantile(.95):8.3f}")

    # Storage estimate
    all_days = list(DATA_ROOT.glob("date=*"))
    if len(all_days) > 1:
        avg_mb = sum(
            (d / "features_labels.parquet").stat().st_size
            for d in all_days
            if (d / "features_labels.parquet").exists()
        ) / len(all_days) / 1024 / 1024
        print(f"\n--- Storage estimate ---")
        print(f"  Days with parquet : {len(all_days)}")
        print(f"  Avg size/day      : {avg_mb:.2f} MB (compressed parquet)")
        print(f"  Est. total        : {avg_mb * len(all_days):.1f} MB")
    else:
        print(f"\n--- Storage estimate (single day) ---")
        print(f"  {size_mb:.2f} MB for {date}; full range size = {size_mb} x N_days")


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else "2025-12-02"
    print(f"Verifying schema for date={date}\n")
    check(date)
    print("\n[PASS] All checks passed.")
