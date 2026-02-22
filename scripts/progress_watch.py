import argparse
import os
import re
import time
from pathlib import Path


HEALTH_RE = re.compile(r"^ES (\d{4}-\d{2}-\d{2}) health ")
PNL_RE = re.compile(r"^PNL_DAY ES (\d{4}-\d{2}-\d{2}) ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch backtest log progress.")
    parser.add_argument("--log", required=True, help="Path to stdout.log")
    parser.add_argument("--refresh", type=float, default=5.0, help="Refresh seconds")
    parser.add_argument("--max-lines", type=int, default=2000, help="Max lines to keep in memory")
    return parser.parse_args()


def _clear() -> None:
    os.system("cls")


def main() -> int:
    args = _parse_args()
    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    offset = 0
    lines = []
    days_started = []
    days_completed = []
    last_line = ""

    while True:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(offset)
            chunk = fh.read()
            offset = fh.tell()

        if chunk:
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                last_line = line
                lines.append(line)
                if len(lines) > args.max_lines:
                    lines = lines[-args.max_lines :]

                health_match = HEALTH_RE.match(line)
                if health_match:
                    day = health_match.group(1)
                    if day not in days_started:
                        days_started.append(day)

                pnl_match = PNL_RE.match(line)
                if pnl_match:
                    day = pnl_match.group(1)
                    if day not in days_completed:
                        days_completed.append(day)

        _clear()
        last_started = days_started[-1] if days_started else "-"
        last_completed = days_completed[-1] if days_completed else "-"
        print("Backtest Progress")
        print("------------------")
        print(f"Log: {log_path}")
        print(f"Days started:   {len(days_started)} (last: {last_started})")
        print(f"Days completed: {len(days_completed)} (last: {last_completed})")
        print(f"Last line: {last_line[:200]}")

        time.sleep(args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
