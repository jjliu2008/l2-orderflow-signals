"""
Debug quote pipeline for a single day: processed vs raw checks.

Env vars:
  DATA_DIR_RAW        raw data root (default: data/raw)
  DATA_DIR_PROCESSED  processed data root (default: data/processed)
  INSTRUMENT          instrument symbol (default: ES)
  DATE                YYYY-MM-DD (required)
  FRONT_MONTH         optional exact symbol (e.g., ESU5)
  FRONT_MONTH_RAW     optional raw symbol in DBN metadata (e.g., ESZ5)
  FRONT_MONTH_ID      optional instrument_id to keep
  SESSION_START       optional session start HH:MM:SS
  SESSION_END         optional session end HH:MM:SS
  SESSION_TZ          optional IANA timezone
  STEP_MS             bar size in ms (default: 100)
  EMIT_EMPTY          emit empty intervals (default: 0)
"""

from __future__ import annotations

import os
import heapq
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime

import numpy as np
import pandas as pd


MONTH_CODE = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}


def _parse_time(value: str) -> Optional[datetime.time]:
    if not value:
        return None
    return pd.to_datetime(value).time()


def _in_session(
    ts: pd.Timestamp,
    start: Optional[datetime.time],
    end: Optional[datetime.time],
    tz: Optional[str],
) -> bool:
    if start is None and end is None:
        return True
    if tz:
        ts = ts.tz_convert(tz)
    t = ts.time()
    if start and end:
        if start <= end:
            return start <= t <= end
        return t >= start or t <= end
    if start:
        return t >= start
    return t <= end


def _price_to_float(raw: int) -> float:
    import databento as db
    if raw == db.UNDEF_PRICE:
        return np.nan
    return raw / db.FIXED_PRICE_SCALE


def _parse_mbp10(rec) -> Dict[str, float]:
    levels = getattr(rec, "levels", [])
    if not levels:
        return {}
    bid_px = [_price_to_float(lvl.bid_px) for lvl in levels[:10]]
    ask_px = [_price_to_float(lvl.ask_px) for lvl in levels[:10]]
    bid_sz = [float(lvl.bid_sz) for lvl in levels[:10]]
    ask_sz = [float(lvl.ask_sz) for lvl in levels[:10]]
    bid0 = bid_px[0]
    ask0 = ask_px[0]
    if not np.isfinite(bid0) or not np.isfinite(ask0):
        return {}
    if ask0 < bid0:
        return {}
    return {
        "bid_price_1": bid0,
        "ask_price_1": ask0,
        "bid_size_1": bid_sz[0],
        "ask_size_1": ask_sz[0],
    }


def _auto_front_month_raw(instrument: str, ts0: pd.Timestamp, mappings: Dict[str, list]) -> Optional[str]:
    month_code = MONTH_CODE.get(ts0.month)
    year_digit = str(ts0.year)[-1]
    if not month_code:
        return None
    candidate = f"{instrument}{month_code}{year_digit}"
    if candidate in mappings:
        return candidate
    for k in mappings.keys():
        if k.startswith(instrument) and "-" not in k and k.endswith(year_digit) and k[-2] == month_code:
            return k
    return None


def _search_raw_files(
    raw_root: Path, instrument: str, date_str: str
) -> Tuple[List[Path], List[Path], Dict[str, object]]:
    date_compact = date_str.replace("-", "")
    tokens = [date_compact, date_str]
    exts = {".dbn", ".parquet", ".csv"}
    candidates: List[Path] = []
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


def _rank_matches(paths: List[Path], date_str: str) -> List[Path]:
    date_compact = date_str.replace("-", "")
    date_tokens = {date_compact, date_str}

    def _score(p: Path) -> Tuple[int, int, int]:
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


def _print_processed_stats(processed_path: Path) -> Dict[str, float]:
    df = pd.read_parquet(processed_path)
    bid = pd.to_numeric(df["bid_price_1"], errors="coerce").to_numpy()
    ask = pd.to_numeric(df["ask_price_1"], errors="coerce").to_numpy()
    mid = 0.5 * (bid + ask)
    spread = ask - bid
    mid_diff = np.diff(mid)
    mid_change_pct = float(np.mean(np.isfinite(mid_diff) & (mid_diff != 0))) if mid_diff.size else 0.0
    stats = {
        "nunique_bid": int(pd.Series(bid).nunique(dropna=True)),
        "nunique_ask": int(pd.Series(ask).nunique(dropna=True)),
        "nunique_mid": int(pd.Series(mid).nunique(dropna=True)),
        "mid_change_pct": mid_change_pct,
    }
    print("Processed stats:", stats, flush=True)
    view = df[["Time", "bid_price_1", "ask_price_1"]].copy()
    view["mid"] = mid
    view["spread"] = spread
    print("Head:", flush=True)
    print(view.head(5).to_string(index=False), flush=True)
    print("Tail:", flush=True)
    print(view.tail(5).to_string(index=False), flush=True)
    return stats


def _print_raw_summary(paths: List[Path], label: str) -> None:
    if not paths:
        print(f"No raw files found for {label}.", flush=True)
        return
    for p in paths:
        print(f"{label} file: {p}", flush=True)


def _raw_pipeline_counts(
    mbp_files: List[Path],
    trade_files: List[Path],
    instrument: str,
    front_month: str,
    front_month_raw: str,
    front_month_id: str,
    session_start: Optional[datetime.time],
    session_end: Optional[datetime.time],
    session_tz: Optional[str],
    step_ms: int,
    emit_empty: bool,
) -> None:
    try:
        import databento as db  # type: ignore
    except ImportError as exc:
        raise ImportError("Databento package is required: pip install databento") from exc

    def _iter_store(path: Path):
        store = db.DBNStore.from_file(path)
        for rec in store:
            yield rec

    iterators = []
    for p in mbp_files:
        iterators.append(_iter_store(p))
    for p in trade_files:
        iterators.append(_iter_store(p))

    heap = []
    for idx, it in enumerate(iterators):
        try:
            rec = next(it)
            heapq.heappush(heap, (rec.ts_event, idx, rec, it))
        except StopIteration:
            continue

    if not heap:
        print("No records found in raw inputs.", flush=True)
        return

    meta = None
    if mbp_files:
        try:
            meta = db.DBNStore.from_file(mbp_files[0]).metadata
        except Exception:
            meta = None
    elif trade_files:
        try:
            meta = db.DBNStore.from_file(trade_files[0]).metadata
        except Exception:
            meta = None

    allowed_ids: Optional[set[int]] = None
    if front_month_id:
        allowed_ids = {int(front_month_id)}
    elif front_month_raw and meta and hasattr(meta, "mappings"):
        ids = set()
        for raw, intervals in meta.mappings.items():
            if raw != front_month_raw:
                continue
            for interval in intervals:
                try:
                    ids.add(int(interval.get("symbol")))
                except Exception:
                    continue
        allowed_ids = ids if ids else None
    elif meta and hasattr(meta, "mappings"):
        ts0 = pd.to_datetime(heap[0][0], unit="ns", utc=True)
        auto_raw = _auto_front_month_raw(instrument, ts0, meta.mappings)
        if auto_raw:
            ids = set()
            for raw, intervals in meta.mappings.items():
                if raw != auto_raw:
                    continue
                for interval in intervals:
                    try:
                        ids.add(int(interval.get("symbol")))
                    except Exception:
                        continue
            allowed_ids = ids if ids else None
            if allowed_ids:
                print(f"Auto-selected FRONT_MONTH_RAW={auto_raw} (instrument_id={sorted(allowed_ids)})", flush=True)

    counts = {
        "total": 0,
        "session_kept": 0,
        "instrument_kept": 0,
        "mbp_total": 0,
        "trade_total": 0,
        "mbp_parsed": 0,
        "emitted_rows": 0,
        "emitted_with_book": 0,
    }
    inst_counts: Dict[int, int] = {}
    ts_min = None
    ts_max = None
    book: Dict[str, float] = {}
    next_emit_ns: Optional[int] = None

    while heap:
        ts_event, idx, rec, it = heapq.heappop(heap)
        ts = pd.to_datetime(ts_event, unit="ns", utc=True)
        counts["total"] += 1
        inst_id = getattr(rec, "instrument_id", None)
        if inst_id is not None:
            inst_counts[int(inst_id)] = inst_counts.get(int(inst_id), 0) + 1
        if ts_min is None or ts_event < ts_min:
            ts_min = ts_event
        if ts_max is None or ts_event > ts_max:
            ts_max = ts_event

        if not _in_session(ts, session_start, session_end, session_tz):
            try:
                rec_next = next(it)
                heapq.heappush(heap, (rec_next.ts_event, idx, rec_next, it))
            except StopIteration:
                pass
            continue
        counts["session_kept"] += 1

        if allowed_ids is not None and getattr(rec, "instrument_id", None) not in allowed_ids:
            try:
                rec_next = next(it)
                heapq.heappush(heap, (rec_next.ts_event, idx, rec_next, it))
            except StopIteration:
                pass
            continue
        counts["instrument_kept"] += 1

        if next_emit_ns is None:
            next_emit_ns = int(ts_event // (step_ms * 1_000_000) * (step_ms * 1_000_000) + step_ms * 1_000_000)

        rec_type = rec.__class__.__name__.lower()
        if rec_type.startswith("mbp"):
            counts["mbp_total"] += 1
            parsed = _parse_mbp10(rec)
            if parsed:
                counts["mbp_parsed"] += 1
                book = parsed
        elif rec_type.startswith("trade"):
            counts["trade_total"] += 1

        while next_emit_ns is not None and ts_event >= next_emit_ns:
            counts["emitted_rows"] += 1
            if book:
                counts["emitted_with_book"] += 1
            next_emit_ns += step_ms * 1_000_000

        try:
            rec_next = next(it)
            heapq.heappush(heap, (rec_next.ts_event, idx, rec_next, it))
        except StopIteration:
            pass

    print("Raw counts:", counts, flush=True)
    if ts_min and ts_max:
        print(
            "Raw timespan:",
            pd.to_datetime(ts_min, unit="ns", utc=True),
            "->",
            pd.to_datetime(ts_max, unit="ns", utc=True),
            flush=True,
        )
    if inst_counts:
        top = sorted(inst_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("Top instrument_id counts:", top, flush=True)

    if counts["instrument_kept"] == 0:
        print("No records after instrument filter; check FRONT_MONTH/ID.", flush=True)
    elif counts["session_kept"] == 0:
        print("No records in session; check SESSION_START/END/TZ.", flush=True)
    elif counts["mbp_parsed"] == 0:
        print("No MBP records parsed into book; check raw MBP inputs.", flush=True)


def main() -> None:
    raw_dir = Path(os.environ.get("DATA_DIR_RAW", "data/raw")).expanduser().resolve()
    processed_dir = Path(os.environ.get("DATA_DIR_PROCESSED", "data/processed")).expanduser().resolve()
    instrument = os.environ.get("INSTRUMENT", "ES").strip()
    date = os.environ.get("DATE", "").strip()
    if not date:
        raise ValueError("DATE is required (YYYY-MM-DD).")

    processed_path = processed_dir / f"instrument={instrument}" / f"date={date}" / "features_labels.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    stats = _print_processed_stats(processed_path)

    if (
        stats["nunique_bid"] >= 50
        and stats["nunique_ask"] >= 50
        and stats["mid_change_pct"] >= 0.005
    ):
        print("Processed data looks non-flat; raw checks skipped.", flush=True)
        return

    mbp_files, trade_files, search_report = _search_raw_files(raw_dir, instrument, date)
    print("Raw file search report:", search_report, flush=True)
    mbp_ranked = _rank_matches(mbp_files, date)
    trade_ranked = _rank_matches(trade_files, date)
    if mbp_ranked:
        print("Selected MBP file:", mbp_ranked[0], flush=True)
        if len(mbp_ranked) > 1:
            print("Top MBP alternatives:", [str(p) for p in mbp_ranked[1:6]], flush=True)
    if trade_ranked:
        print("Selected trades file:", trade_ranked[0], flush=True)
        if len(trade_ranked) > 1:
            print("Top trades alternatives:", [str(p) for p in trade_ranked[1:6]], flush=True)
    _print_raw_summary(mbp_ranked[:1], "MBP")
    _print_raw_summary(trade_ranked[:1], "TRADES")

    if not mbp_ranked and not trade_ranked:
        print("No raw files found for date; cannot debug raw pipeline.", flush=True)
        return

    front_month = os.environ.get("FRONT_MONTH", "").strip()
    front_month_raw = os.environ.get("FRONT_MONTH_RAW", "").strip()
    front_month_id = os.environ.get("FRONT_MONTH_ID", "").strip()
    step_ms = int(os.environ.get("STEP_MS", "100"))
    emit_empty = os.environ.get("EMIT_EMPTY", "0").strip() in {"1", "true", "yes", "y"}
    session_start = _parse_time(os.environ.get("SESSION_START", "").strip())
    session_end = _parse_time(os.environ.get("SESSION_END", "").strip())
    session_tz = os.environ.get("SESSION_TZ", "").strip() or None

    _raw_pipeline_counts(
        mbp_ranked[:1],
        trade_ranked[:1],
        instrument=instrument,
        front_month=front_month,
        front_month_raw=front_month_raw,
        front_month_id=front_month_id,
        session_start=session_start,
        session_end=session_end,
        session_tz=session_tz,
        step_ms=step_ms,
        emit_empty=emit_empty,
    )


if __name__ == "__main__":
    main()
