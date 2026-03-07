"""
Reprocess a date range by invoking build_dataset.py per day.

Env vars:
  START_DATE        inclusive YYYY-MM-DD
  END_DATE          inclusive YYYY-MM-DD
  DATES             optional comma list of YYYY-MM-DD (overrides START/END)
  MBP_TEMPLATE      path template for MBP DBN (optional)
  TRADES_TEMPLATE   path template for trades DBN (optional)
  DATA_DIR_RAW      raw data root (default: data/raw)
  OUTPUT_DIR        processed output root (default: data/processed)
  INSTRUMENT        instrument symbol (default: ES)
  FRONT_MONTH_RAW   optional raw symbol to keep (e.g., ESZ5) — required when auto-detect fails (e.g., file starts before midnight UTC)
  FRONT_MONTH_ID    optional instrument_id to keep
  STEP_MS           bar size in ms (default: 100)
  LFP_TICK_SIZE     tick size override (optional)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def _discover_raw_files(raw_root: Path, date_str: str) -> tuple[list[Path], list[Path], dict]:
    date_compact = date_str.replace("-", "")
    tokens = [date_compact, date_str]
    exts = {".dbn", ".parquet", ".csv"}
    candidates: list[Path] = []
    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        name = p.name.lower()
        if any(t in name for t in tokens):
            candidates.append(p)

    def _is_mbp(path: Path) -> bool:
        name = path.name.lower()
        return "mbp" in name and ("10" in name or "mbp-10" in name)

    def _is_trade(path: Path) -> bool:
        return "trade" in path.name.lower()

    mbp = [p for p in candidates if _is_mbp(p)]
    trades = [p for p in candidates if _is_trade(p)]
    report = {
        "tokens": tokens,
        "mbp_matches": len(mbp),
        "trade_matches": len(trades),
        "sample_candidates": [str(p) for p in candidates[:10]],
    }
    return mbp, trades, report


def _resolve_dbn_path(p: Path) -> Path:
    """Databento downloads wrap the actual .dbn inside a same-named directory.

    e.g.  .../foo.mbp-10.dbn/foo.mbp-10.dbn   <- real file inside a dir
    If *p* is a directory and contains a single .dbn file, return that file.
    """
    if p.is_dir():
        candidates = list(p.glob("*.dbn"))
        if len(candidates) == 1:
            return candidates[0]
    return p


def _rank_matches(paths: list[Path], date_str: str) -> list[Path]:
    date_compact = date_str.replace("-", "")
    date_tokens = {date_compact, date_str}

    def _score(p: Path) -> tuple[int, int, int]:
        name = p.name.lower()
        score = 0
        if "glbx-mdp3" in name:
            score += 3
        if any(t in name for t in date_tokens):
            score += 3
        if "mbp-10" in name:
            score += 2
        return (score, -len(name), 0)

    return sorted(paths, key=_score, reverse=True)


def _parse_dates() -> list[str]:
    dates_env = os.environ.get("DATES", "").strip()
    if dates_env:
        return [d.strip() for d in dates_env.split(",") if d.strip()]
    start = os.environ.get("START_DATE", "").strip()
    end = os.environ.get("END_DATE", "").strip()
    if not start or not end:
        raise ValueError("Provide DATES or START_DATE and END_DATE.")
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise ValueError("END_DATE must be >= START_DATE.")
    out = []
    cur = start_dt
    while cur <= end_dt:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def main() -> None:
    dates = _parse_dates()
    mbp_tmpl = os.environ.get("MBP_TEMPLATE", "").strip()
    trades_tmpl = os.environ.get("TRADES_TEMPLATE", "").strip()
    output_dir = os.environ.get("OUTPUT_DIR", "data/processed")
    instrument = os.environ.get("INSTRUMENT", "ES")
    front_month_id = os.environ.get("FRONT_MONTH_ID", "").strip()
    front_month_raw = os.environ.get("FRONT_MONTH_RAW", "").strip()
    step_ms = os.environ.get("STEP_MS", "100")
    lfp_tick_size = os.environ.get("LFP_TICK_SIZE", "").strip()
    raw_root = Path(os.environ.get("DATA_DIR_RAW", "data/raw")).expanduser().resolve()

    for date in dates:
        date_compact = date.replace("-", "")
        mbp_path = mbp_tmpl.format(date=date_compact, date_dash=date) if mbp_tmpl else ""
        trades_path = trades_tmpl.format(date=date_compact, date_dash=date) if trades_tmpl else ""

        mbp_paths: list[Path] = []
        trade_paths: list[Path] = []
        if mbp_path and Path(mbp_path).exists():
            mbp_paths = [_resolve_dbn_path(Path(mbp_path))]
        if trades_path and Path(trades_path).exists():
            trade_paths = [_resolve_dbn_path(Path(trades_path))]

        if (mbp_path and not mbp_paths) or (trades_path and not trade_paths):
            print("Template path missing; falling back to discovery.", flush=True)

        if not mbp_paths or not trade_paths:
            mbp_found, trade_found, report = _discover_raw_files(raw_root, date)
            print(f"Discovery report for {date}: {report}", flush=True)
            if not mbp_paths:
                ranked = _rank_matches(mbp_found, date)
                mbp_paths = [_resolve_dbn_path(p) for p in ranked[:1]]
                if ranked:
                    print(f"Selected MBP: {mbp_paths[0]}", flush=True)
                    if len(ranked) > 1:
                        print(f"Top MBP alternatives: {[str(p) for p in ranked[1:6]]}", flush=True)
            if not trade_paths:
                ranked = _rank_matches(trade_found, date)
                trade_paths = ranked[:1]
                if ranked:
                    print(f"Selected TRADES: {ranked[0]}", flush=True)
                    if len(ranked) > 1:
                        print(f"Top TRADES alternatives: {[str(p) for p in ranked[1:6]]}", flush=True)
        if not mbp_paths:
            print(f"Skipping {date}: no MBP file found.", flush=True)
            continue
        if not trade_paths:
            print(f"No TRADES file for {date}; proceeding MBP-only.", flush=True)
        env = os.environ.copy()
        if mbp_paths:
            env["INPUT_MBP_DBN"] = str(mbp_paths[0])
        if trade_paths:
            env["INPUT_TRADES_DBN"] = str(trade_paths[0])
        env["OUTPUT_DIR"] = output_dir
        env["INSTRUMENT"] = instrument
        env["STEP_MS"] = step_ms
        if front_month_id:
            env["FRONT_MONTH_ID"] = front_month_id
        if front_month_raw:
            env["FRONT_MONTH_RAW"] = front_month_raw
        if lfp_tick_size:
            env["LFP_TICK_SIZE"] = lfp_tick_size
        env["DATE_FILTER"] = date
        print(
            f"Reprocessing {date} using MBP={env.get('INPUT_MBP_DBN','')} "
            f"TRADES={env.get('INPUT_TRADES_DBN','')}",
            flush=True,
        )
        subprocess.check_call([sys.executable, "scripts/build_dataset.py"], env=env)


if __name__ == "__main__":
    main()
