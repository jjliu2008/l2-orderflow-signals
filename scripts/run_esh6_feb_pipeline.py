"""
Full Feb 2026 ESH6 pipeline — runs after parquets are already built.

Steps:
  1. Run PFLFT_v8 backtest for all Feb dates (run_esh6_feb_backtest.py)
  2. Run absorption detector for each Feb date
  3. Run combined Jan+Feb evaluation (evaluate_esh6_combined.py)

Usage:
    python scripts/run_esh6_feb_pipeline.py

Prerequisite: parquets must exist in data/processed_esh6/instrument=ES/date=2026-02-*/
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEB_DATES = [
    "2026-02-02", "2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
    "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-02-23", "2026-02-24",
]

DBN_ROOT    = "c:/Users/majin/Downloads/ESH6-MBP10"
TRADES_ROOT = str(PROJECT_ROOT / "artifacts/esh6_oos_trades")
DETECTOR_DIR = str(PROJECT_ROOT / "artifacts/absorption_detector_esh6")


def run(cmd: list[str], **kwargs) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, **kwargs)


def main() -> None:
    # ── Step 1: backtest ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 1: Run PFLFT_v8 backtest for Feb 2026 dates")
    print("="*60)
    run([sys.executable, str(PROJECT_ROOT / "scripts/run_esh6_feb_backtest.py")])

    # ── Step 2: absorption detector per date ─────────────────────────────────
    print("\n" + "="*60)
    print("Step 2: Run absorption detector for Feb dates")
    print("="*60)
    detector = str(PROJECT_ROOT / "scripts/run_absorption_detector.py")
    for date in FEB_DATES:
        out_file = Path(DETECTOR_DIR) / f"absorption_{date}.jsonl"
        if out_file.exists():
            print(f"  {date}: already exists, skipping")
            continue
        trades_dir = Path(TRADES_ROOT) / "W10/mode=side_matched" / f"ES_{date}"
        if not trades_dir.exists():
            print(f"  {date}: no trades dir {trades_dir}, skipping")
            continue
        run([
            sys.executable, detector,
            "--date", date,
            "--trades-root", TRADES_ROOT,
            "--dbn-root", DBN_ROOT,
            "--symbol", "ESH6",
            "--output-dir", DETECTOR_DIR,
        ])

    # ── Step 3: combined evaluation ───────────────────────────────────────────
    print("\n" + "="*60)
    print("Step 3: Combined Jan+Feb evaluation")
    print("="*60)
    run([sys.executable, str(PROJECT_ROOT / "scripts/evaluate_esh6_combined.py")])


if __name__ == "__main__":
    main()
