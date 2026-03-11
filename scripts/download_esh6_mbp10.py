"""
Download ESH6 MBP-10 DBN data from Databento for OOS absorption detector validation.

Usage:
    # Dry run — prints cost estimate only, downloads nothing:
    python scripts/download_esh6_mbp10.py --dry-run

    # Download:
    DATABENTO_API_KEY=db-xxx python scripts/download_esh6_mbp10.py

    # Specific date range:
    python scripts/download_esh6_mbp10.py --start 2026-01-02 --end 2026-01-31

Files are saved to:
    c:/Users/majin/Downloads/ESH6-MBP10/glbx-mdp3-YYYYMMDD.mbp-10.dbn/

The reprocess_range.py script expects the DBN_ROOT env var or the default
c:/Users/majin/Downloads/GLBX-20260214-5X9XQHYJXV structure. We save to a
separate ESH6 directory and point DBN_ROOT at it when building parquets.
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import databento as db

DOWNLOAD_ROOT = Path("c:/Users/majin/Downloads/ESH6-MBP10")
DATASET       = "GLBX.MDP3"
SCHEMA        = "mbp-10"
SYMBOLS       = ["ESH6"]
SYM_MAP_SYM   = "ESH6"

# ESH6 trading days Jan–Feb 2026 (Mon–Fri, excl US market holidays)
# CME Globex holidays: New Year's Day (Jan 1), MLK Day (Jan 19), Presidents Day (Feb 16)
DEFAULT_START = "2026-01-02"
DEFAULT_END   = "2026-02-27"


def trading_days(start: str, end: str) -> list[str]:
    """Return Mon–Fri dates in [start, end], as YYYY-MM-DD strings."""
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    out = []
    cur = d0
    while cur <= d1:
        if cur.weekday() < 5:   # Mon–Fri
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def estimate_cost(client: db.Historical, dates: list[str]) -> None:
    """Print cost estimate for the full date list without downloading."""
    print(f"Estimating cost for {len(dates)} dates ({dates[0]} – {dates[-1]})...")
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


def download_date(client: db.Historical, date_str: str, out_dir: Path) -> Path | None:
    """
    Download one day of ESH6 MBP-10 data. Returns path to the saved file or None.
    Skips if already downloaded.
    """
    compact = date_str.replace("-", "")
    # Databento writes files as glbx-mdp3-YYYYMMDD.mbp-10.dbn
    out_path = out_dir / f"glbx-mdp3-{compact}.mbp-10.dbn"
    if out_path.exists():
        print(f"  {date_str}  already exists, skipping")
        return out_path

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
        return out_path
    except Exception as e:
        print(f"  {date_str}  FAILED: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ESH6 MBP-10 from Databento")
    parser.add_argument("--start",     default=DEFAULT_START)
    parser.add_argument("--end",       default=DEFAULT_END)
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print cost estimate only, do not download")
    parser.add_argument("--api-key",   default=None,
                        help="Databento API key (default: DATABENTO_API_KEY env var)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        print("ERROR: No API key. Set DATABENTO_API_KEY or pass --api-key")
        sys.exit(1)

    client = db.Historical(api_key)
    dates  = trading_days(args.start, args.end)
    print(f"Target dates: {len(dates)} ({dates[0]} – {dates[-1]})")

    if args.dry_run:
        estimate_cost(client, dates)
        print("\nDry run complete. Run without --dry-run to download.")
        return

    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {DOWNLOAD_ROOT}")
    print()

    n_ok = 0
    n_fail = 0
    for d in dates:
        result = download_date(client, d, DOWNLOAD_ROOT)
        if result:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone. Downloaded: {n_ok}, Failed/skipped: {n_fail}")
    print(f"Files in: {DOWNLOAD_ROOT}")
    print()
    print("Next step — build parquets:")
    print(f"  FRONT_MONTH_RAW=ESH6 DBN_ROOT={DOWNLOAD_ROOT} \\")
    print(f"  python scripts/reprocess_range.py --start {args.start} --end {args.end}")


if __name__ == "__main__":
    main()
