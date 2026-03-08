"""
Run frozen PFLFT_v8 backtest over ESH6 Jan 2026 in small batches to avoid OOM.
Creates temporary per-batch data dirs so the backtest only loads ~3 days at a time.
Consolidates per-date trade dirs into artifacts/esh6_oos_trades/ afterward.

Usage:
    python scripts/run_esh6_oos_backtest.py
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUTPUT_DIR   = PROJECT_ROOT / "artifacts/esh6_oos"
TRADES_DIR   = PROJECT_ROOT / "artifacts/esh6_oos_trades"
BACKTEST     = PROJECT_ROOT / "scripts/run_lrams_gate_backtest.py"

# Frozen PFLFT_v8 parameters (from experiment_agent.json + grid A)
FIXED_ENV = {
    "STRATEGY_MODE":              "entry_alpha_v1",
    "ENTRY_ALPHA_FAMILY_ALLOWLIST": "PFLFT_v8",
    "GATE_MODE":                  "side_matched",
    "VALIDATE_DEBUG":             "0",
    "ENTRY_CONFIRM_STYLE":        "price_only",
    "ENTRY_CONFIRM_BARS":         "2",
    "ENTRY_ALPHA_ALLOW_MISMATCH_V7": "1",
    "DISABLE_GATE":               "0",
    "PNL_TICK_VALUE":             "12.5",
    "PNL_COMMISSION_ROUND_TURN":  "1.2",
    "PNL_SLIPPAGE_TICKS":         "1",
    "ENTRY_COOLDOWN_BARS":        "0",
    "ENTRY_MIN_PROGRESS_TICKS":   "0",
    "PFLFT_V7_ENABLE":            "1",
    "PFLFT_V7_SPREAD_MAX":        "1",
    "PFLFT_V7_FLOWINT_PRE3_MIN":  "15",
    "PFLFT_V7_SV_PRE3_MIN":       "12",
    "PFLFT_V7_DMID1_MAX":         "1",
    "PFLFT_V7_PROOF_BARS":        "2",
    "PFLFT_V7_PROOF_TICKS":       "1",
    "PFLFT_V7_PROOF_MIN_FLOW":    "0",
    "PFLFT_V7_TIME_STOP_BARS":    "5",
    "PFLFT_V7_TP_TICKS":          "1",
    "PFLFT_V7_SL_TICKS":          "1",
    "PFLFT_V7_FLOWINT_PRE3_P90":  "25",
    "PFLFT_V7_RUNNER_TP_TICKS":   "4",
    "PFLFT_V7_PREV_RANGE_MAX_TICKS": "15",
    "PFLFT_V7_PREV_ABS_FLOW_MAX": "20000",
    "PFLFT_V8_ENABLE":            "1",
    "PFLFT_V8_LFP_ALIGNED_10_MIN": "0.58",
    "PFLFT_V8_STRESS_RATIO_MIN":  "0.18",
    "PFLFT_V8_DEPTH_TOTAL_TOP5_MAX": "640",
    "PFLFT_V8_ALIGNED_IMB_DELTA_MIN": "0.05",
    "PFLFT_V8_TOXICITY_MAX":      "0.156",
    "PFLFT_V8_TOX_PROXY_MAX":     "0",
}

# All ESH6 Jan 2026 trading dates
ALL_DATES = [
    "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09",
    "2026-01-12", "2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16", "2026-01-19",
    "2026-01-20", "2026-01-21", "2026-01-22", "2026-01-23",
    "2026-01-26", "2026-01-27", "2026-01-28", "2026-01-29", "2026-01-30",
]

BATCH_SIZE = 3  # days per backtest run


def make_batch_data_dir(dates: list[str]) -> Path:
    """Create a temp dir with symlinks/copies to just the given dates' parquets."""
    tag = f"batch_{dates[0].replace('-','')}_to_{dates[-1].replace('-','')}"
    batch_dir = PROJECT_ROOT / f"data/_esh6_batch_{tag}" / "instrument=ES"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for d in dates:
        src = DATA_ROOT / f"date={d}"
        dst = batch_dir / f"date={d}"
        if not dst.exists():
            shutil.copytree(src, dst)
    return batch_dir.parent


def run_batch(dates: list[str], batch_num: int, n_batches: int) -> None:
    print(f"\n=== Batch {batch_num}/{n_batches}: {dates[0]} to {dates[-1]} ({len(dates)} days) ===")
    batch_data_dir = make_batch_data_dir(dates)

    env = os.environ.copy()
    env.update(FIXED_ENV)
    env["DATA_DIR"]   = str(batch_data_dir)
    env["OUTPUT_DIR"] = str(OUTPUT_DIR)
    env["DATES"]      = ",".join(dates)

    try:
        subprocess.check_call([sys.executable, str(BACKTEST)], env=env)
    finally:
        shutil.rmtree(batch_data_dir, ignore_errors=True)


def consolidate() -> None:
    """Merge per-date trade dirs from all run_*/W10/mode=side_matched/ES_*/ into TRADES_DIR."""
    dest_base = TRADES_DIR / "W10/mode=side_matched"
    dest_base.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Consolidating into {TRADES_DIR} ===")
    for run_dir in sorted(OUTPUT_DIR.glob("run_*")):
        w10 = run_dir / "W10/mode=side_matched"
        if not w10.exists():
            continue
        for date_dir in sorted(w10.iterdir()):
            if not date_dir.is_dir() or not date_dir.name.startswith("ES_2026-"):
                continue
            dest = dest_base / date_dir.name
            if dest.exists():
                print(f"  {date_dir.name} already consolidated, skipping")
            else:
                shutil.copytree(date_dir, dest)
                print(f"  copied {date_dir.name}")


def main() -> None:
    missing = [d for d in ALL_DATES if not (DATA_ROOT / f"date={d}").exists()]
    if missing:
        print(f"WARNING: missing parquets for: {missing}")

    dates = [d for d in ALL_DATES if (DATA_ROOT / f"date={d}").exists()]
    batches = [dates[i:i+BATCH_SIZE] for i in range(0, len(dates), BATCH_SIZE)]
    n = len(batches)

    for i, batch in enumerate(batches, 1):
        run_batch(batch, i, n)

    consolidate()

    print(f"\nDone. Gated trades in: {TRADES_DIR}")
    print("\nNext step — run absorption detector per date, e.g.:")
    print(f"  python scripts/run_absorption_detector.py --date 2026-01-02 \\")
    print(f"    --trades-root {TRADES_DIR} \\")
    print(f"    --dbn-root c:/Users/majin/Downloads/ESH6-MBP10 \\")
    print(f"    --symbol ESH6 \\")
    print(f"    --output-dir artifacts/absorption_detector_esh6")


if __name__ == "__main__":
    main()
