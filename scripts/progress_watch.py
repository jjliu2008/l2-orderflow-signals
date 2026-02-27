import argparse
import os
import re
import time
from pathlib import Path


HEALTH_RE = re.compile(r"^ES (\d{4}-\d{2}-\d{2}) health ")
PNL_RE = re.compile(r"^PNL_DAY ES (\d{4}-\d{2}-\d{2}) ")
OPENCLAW_USER_RE = re.compile(r"\"role\":\"user\".*\"text\":\"([^\"]+)\"")
OPENCLAW_ASSIST_RE = re.compile(r"\"role\":\"assistant\"")
OPENCLAW_ERR_RE = re.compile(r"openclaw:prompt-error")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch backtest log progress.")
    parser.add_argument("--log", help="Path to stdout.log")
    parser.add_argument("--openclaw-log", help="Path to OpenClaw session jsonl")
    parser.add_argument("--refresh", type=float, default=5.0, help="Refresh seconds")
    parser.add_argument("--max-lines", type=int, default=2000, help="Max lines to keep in memory")
    return parser.parse_args()


def _clear() -> None:
    os.system("cls")


def _progress_bar(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        total = 1
    completed = max(0, min(completed, total))
    filled = int(round((completed / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    pct = (completed / total) * 100
    return f"[{bar}] {pct:5.1f}%"


def _watch_backtest(log_path: Path, refresh: float, max_lines: int) -> None:
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
                if len(lines) > max_lines:
                    lines = lines[-max_lines :]

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
        total_days = max(len(days_started), len(days_completed))
        progress = _progress_bar(len(days_completed), total_days)
        print("Backtest Progress")
        print("------------------")
        print(f"Log: {log_path}")
        print(f"Days started:   {len(days_started)} (last: {last_started})")
        print(f"Days completed: {len(days_completed)} (last: {last_completed})")
        print(f"Progress: {progress}")
        print(f"Last line: {last_line[:200]}")

        time.sleep(refresh)


def _watch_openclaw(log_path: Path, refresh: float, max_lines: int) -> None:
    offset = 0
    last_user = ""
    assistant_count = 0
    error_count = 0
    status = "waiting for agents"

    while True:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(offset)
            chunk = fh.read()
            offset = fh.tell()

        if chunk:
            for line in chunk.splitlines():
                if OPENCLAW_USER_RE.search(line):
                    last_user = OPENCLAW_USER_RE.search(line).group(1)
                    status = "agent received a task"
                if OPENCLAW_ASSIST_RE.search(line):
                    assistant_count += 1
                    status = "agent responded"
                if OPENCLAW_ERR_RE.search(line):
                    error_count += 1
                    status = "agent hit an error"

        _clear()
        print("Jayden/BbgnllinbB08_H stuff")
        print("---------------------------")
        print(f"Log: {log_path}")
        print(f"Status: {status}")
        print(f"Agent replies: {assistant_count}")
        print(f"Errors: {error_count}")
        if last_user:
            print(f"Last request: {last_user[:120]}")
        else:
            print("Last request: -")
        print("Mood: caffeinated lobster, still typing.")

        time.sleep(refresh)


def _watch_both(backtest_log: Path, agent_log: Path, refresh: float, max_lines: int) -> None:
    bt_offset = 0
    bt_days_started = []
    bt_days_completed = []
    bt_last_line = ""

    ag_offset = 0
    ag_last_user = ""
    ag_assistant_count = 0
    ag_error_count = 0
    ag_status = "waiting for agents"

    while True:
        with backtest_log.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(bt_offset)
            chunk = fh.read()
            bt_offset = fh.tell()

        if chunk:
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                bt_last_line = line
                health_match = HEALTH_RE.match(line)
                if health_match:
                    day = health_match.group(1)
                    if day not in bt_days_started:
                        bt_days_started.append(day)
                pnl_match = PNL_RE.match(line)
                if pnl_match:
                    day = pnl_match.group(1)
                    if day not in bt_days_completed:
                        bt_days_completed.append(day)

        with agent_log.open("r", encoding="utf-8", errors="ignore") as fh:
            fh.seek(ag_offset)
            chunk = fh.read()
            ag_offset = fh.tell()

        if chunk:
            for line in chunk.splitlines():
                user_match = OPENCLAW_USER_RE.search(line)
                if user_match:
                    ag_last_user = user_match.group(1)
                    ag_status = "agent received a task"
                if OPENCLAW_ASSIST_RE.search(line):
                    ag_assistant_count += 1
                    ag_status = "agent responded"
                if OPENCLAW_ERR_RE.search(line):
                    ag_error_count += 1
                    ag_status = "agent hit an error"

        _clear()
        bt_last_started = bt_days_started[-1] if bt_days_started else "-"
        bt_last_completed = bt_days_completed[-1] if bt_days_completed else "-"
        bt_total_days = max(len(bt_days_started), len(bt_days_completed))
        bt_progress = _progress_bar(len(bt_days_completed), bt_total_days)
        print("Backtest Progress")
        print("------------------")
        print(f"Log: {backtest_log}")
        print(f"Days started:   {len(bt_days_started)} (last: {bt_last_started})")
        print(f"Days completed: {len(bt_days_completed)} (last: {bt_last_completed})")
        print(f"Progress: {bt_progress}")
        print(f"Last line: {bt_last_line[:200]}")
        print("")
        print("Jayden/BbgnllinbB08_H stuff")
        print("---------------------------")
        print(f"Log: {agent_log}")
        print(f"Status: {ag_status}")
        print(f"Agent replies: {ag_assistant_count}")
        print(f"Errors: {ag_error_count}")
        if ag_last_user:
            print(f"Last request: {ag_last_user[:120]}")
        else:
            print("Last request: -")
        print("Mood: caffeinated lobster, still typing.")

        time.sleep(refresh)


def main() -> int:
    args = _parse_args()
    if not args.log and not args.openclaw_log:
        raise SystemExit("Provide --log or --openclaw-log.")

    if args.log and args.openclaw_log:
        log_path = Path(args.log).expanduser().resolve()
        agent_path = Path(args.openclaw_log).expanduser().resolve()
        if not log_path.exists():
            raise SystemExit(f"Log not found: {log_path}")
        if not agent_path.exists():
            raise SystemExit(f"OpenClaw log not found: {agent_path}")
        _watch_both(log_path, agent_path, args.refresh, args.max_lines)
        return 0

    if args.log:
        log_path = Path(args.log).expanduser().resolve()
        if not log_path.exists():
            raise SystemExit(f"Log not found: {log_path}")
        _watch_backtest(log_path, args.refresh, args.max_lines)

    if args.openclaw_log:
        log_path = Path(args.openclaw_log).expanduser().resolve()
        if not log_path.exists():
            raise SystemExit(f"OpenClaw log not found: {log_path}")
        _watch_openclaw(log_path, args.refresh, args.max_lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
