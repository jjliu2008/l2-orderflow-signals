"""
Download NQH6 TRADES DBN data from Databento.

TRADES schema is much smaller than MBP-10 (only executed transactions, no book depth).
Sufficient to build 100ms last-price bars for correlation and spread analysis.

Cost: ~$19 for Jan-Feb 2026 (vs $124 for MBP-10).

Usage:
    python scripts/download_nqh6_trades.py --dry-run
    python scripts/download_nqh6_trades.py --api-key db-xxx

Files saved to:
    c:/Users/majin/Downloads/NQH6-TRADES/glbx-mdp3-YYYYMMDD.trades.dbn
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import databento as db

DOWNLOAD_ROOT = Path("c:/Users/majin/Downloads/NQH6-TRADES")
DATASET       = "GLBX.MDP3"
SCHEMA        = "trades"
SYMBOLS       = ["NQH6"]

DEFAULT_START = "2026-01-02"
DEFAULT_END   = "2026-02-28"


def trading_days(start: str, end: str) -> list[str]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    out = []
    cur = d0
    while cur <= d1:
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def estimate_cost(client: db.Historical, dates: list[str]) -> None:
    print(f"Estimating cost for {len(dates)} dates ({dates[0]} - {dates[-1]})...")
    try:
        cost = client.metadata.get_cost(
            dataset=DATASET,
            symbols=SYMBOLS,
            schema=SCHEMA,
            stype_in="raw_symbol",
            start=dates[0],
            end=(date.fromisoformat(dates[-1]) + timedelta(days=1)).isoformat(),
        )
        print(f"  Estimated cost: ${cost:.2f}")
    except Exception as e:
        print(f"  Cost estimate failed: {e}")


def download_date(client: db.Historical, date_str: str, out_dir: Path) -> bool:
    compact  = date_str.replace("-", "")
    out_path = out_dir / f"glbx-mdp3-{compact}.trades.dbn"
    if out_path.exists():
        print(f"  {date_str}  already exists, skipping")
        return True

    next_day = (date.fromisoformat(date_str) + timedelta(days=1)).isoformat()
    try:
        client.timeseries.get_range(
            dataset=DATASET,
            symbols=SYMBOLS,
            schema=SCHEMA,
            stype_in="raw_symbol",
            start=date_str,
            end=next_day,
            path=str(out_path),
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f"  {date_str}  saved  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  {date_str}  FAILED: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   default=DEFAULT_START)
    parser.add_argument("--end",     default=DEFAULT_END)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: pass --api-key or set DATABENTO_API_KEY")
        sys.exit(1)

    client = db.Historical(api_key)
    dates  = trading_days(args.start, args.end)
    print(f"Target: {len(dates)} trading days  ({dates[0]} - {dates[-1]})")

    if args.dry_run:
        estimate_cost(client, dates)
        return

    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output: {DOWNLOAD_ROOT}\n")

    n_ok = n_fail = 0
    for d in dates:
        if download_date(client, d, DOWNLOAD_ROOT):
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone. OK={n_ok}  Failed={n_fail}")
    print(f"Files: {DOWNLOAD_ROOT}")


if __name__ == "__main__":
    main()
