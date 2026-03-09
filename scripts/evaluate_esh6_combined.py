"""
Evaluate combined ESH6 Jan+Feb bid_absorption OOS results against the frozen spec.

Runs tick-level TP/SL sequencing for each bid_absorption fire using full-day MBP-10 DBN
action='T' trade records. Each trade record carries best bid_px/ask_px at that moment.
Excludes NFP/CPI/FOMC dates per the frozen spec exclusion policy.

Usage:
    python scripts/evaluate_esh6_combined.py

Outputs:
    artifacts/absorption_detector_esh6/combined_evaluation_summary.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import databento as db
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DETECTOR_DIR  = PROJECT_ROOT / "artifacts/absorption_detector_esh6"
DBN_ROOT      = Path("c:/Users/majin/Downloads/ESH6-MBP10")
OUTPUT_FILE   = DETECTOR_DIR / "combined_evaluation_summary.json"

TICK     = 0.25
TP_TICKS = 8
SL_TICKS = 1   # SL anchored to absorption level (~1 tick from entry)

# ── Exclusion dates (NFP / CPI / FOMC) ─────────────────────────────────────
EXCLUDED_DATES: dict[str, str] = {
    # January 2026
    "2026-01-09": "NFP",
    "2026-01-15": "CPI",
    "2026-01-29": "FOMC",
    # February 2026
    "2026-02-06": "NFP",
    "2026-02-12": "CPI",   # Jan 2026 CPI release — verify if needed
    # No FOMC in February 2026 (next FOMC: 2026-03-18/19)
}


def _load_ticks(date: str, symbol: str = "ESH6") -> list[dict]:
    """Load full-day MBP-10 DBN trade records (action='T') for one date.

    Each record: {ts, bid, ask} — same approach used in the Jan tick-level evaluation.
    """
    compact  = date.replace("-", "")
    # ESH6 files are flat .dbn files (not nested in a dir)
    dbn_path = DBN_ROOT / f"glbx-mdp3-{compact}.mbp-10.dbn"
    if not dbn_path.exists():
        # Fallback: check if nested (ESZ5 style)
        nested = DBN_ROOT / f"glbx-mdp3-{compact}.mbp-10.dbn" / f"glbx-mdp3-{compact}.mbp-10.dbn"
        if nested.exists():
            dbn_path = nested
        else:
            print(f"  DBN not found for {date}")
            return []

    store = db.DBNStore.from_file(str(dbn_path))

    # Resolve instrument_id from symbology
    target_iid = None
    try:
        sym_map = store.symbology
        for sym, mappings in sym_map.get("mappings", {}).items():
            if sym == symbol:
                target_iid = int(mappings[0]["symbol"])
                break
    except Exception:
        pass
    if target_iid is None:
        print(f"  Could not resolve {symbol} instrument_id for {date}")
        return []

    records = []
    for rec in store:
        if str(getattr(rec, "action", "")) != "T":
            continue
        if rec.instrument_id != target_iid:
            continue
        lvl = rec.levels[0]
        records.append({
            "ts":  pd.Timestamp(rec.ts_event, unit="ns", tz="UTC"),
            "bid": lvl.bid_px / 1e9,
            "ask": lvl.ask_px / 1e9,
        })

    print(f"  {symbol} instrument_id={target_iid}, trade events: {len(records):,}")
    return records


def _evaluate_fire(fire: dict, ticks: list[dict]) -> str:
    """Return 'win', 'loss', or 'hold' for one bid_absorption fire."""
    det_ts_raw = fire.get("detection_ts")
    if isinstance(det_ts_raw, str):
        det_ts = pd.Timestamp(det_ts_raw)
        if det_ts.tzinfo is None:
            det_ts = det_ts.tz_localize("UTC")
    else:
        det_ts = pd.Timestamp(det_ts_raw, unit="ns", tz="UTC")

    absorption_px = fire["bid_absorption_px"]
    direction = fire.get("trade_direction", "short").lower()
    is_long = direction == "long"

    if is_long:
        entry_px = absorption_px + TICK   # buy at ask (1 tick above bid level)
        tp_px    = entry_px + TP_TICKS * TICK
        sl_px    = absorption_px - SL_TICKS * TICK
    else:
        entry_px = absorption_px - TICK   # sell at bid (1 tick below ask level)
        tp_px    = entry_px - TP_TICKS * TICK
        sl_px    = absorption_px + SL_TICKS * TICK

    for t in ticks:
        if t["ts"] < det_ts:
            continue
        if is_long:
            if t["bid"] >= tp_px:
                return "win"
            if t["ask"] <= sl_px:
                return "loss"
        else:
            if t["ask"] <= tp_px:
                return "win"
            if t["bid"] >= sl_px:
                return "loss"
    return "hold"


def evaluate_date(date: str, jsonl_path: Path) -> list[dict]:
    """Return list of outcome records for all bid_absorption fires on one date."""
    fires = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("detection_signature") != "bid_absorption":
                continue
            fires.append(rec)

    if not fires:
        return []

    print(f"  {date}: {len(fires)} bid_absorption fires — loading DBN...")
    ticks = _load_ticks(date)
    if not ticks:
        print(f"  {date}: no ticks loaded — skipping")
        return []

    results = []
    for fire in fires:
        outcome = _evaluate_fire(fire, ticks)
        results.append({"date": date, "outcome": outcome, "fire": fire})

    w = sum(1 for r in results if r["outcome"] == "win")
    l = sum(1 for r in results if r["outcome"] == "loss")
    print(f"  {date}: W={w} L={l} H={len(results)-w-l}")
    return results


def main() -> None:
    jsonl_files = sorted(DETECTOR_DIR.glob("absorption_2026-0*.jsonl"))
    if not jsonl_files:
        print(f"No absorption JSONL files found in {DETECTOR_DIR}")
        return

    per_date: dict[str, dict] = {}
    all_results: list[dict] = []
    excluded: list[str] = []

    for jf in jsonl_files:
        date = jf.stem.replace("absorption_", "")
        if date in EXCLUDED_DATES:
            reason = EXCLUDED_DATES[date]
            print(f"Skipping {date} ({reason})")
            excluded.append(f"{date} ({reason})")
            continue

        print(f"Processing {date}...")
        results = evaluate_date(date, jf)
        if not results:
            continue

        n = len(results)
        w = sum(1 for r in results if r["outcome"] == "win")
        l = sum(1 for r in results if r["outcome"] == "loss")
        per_date[date] = {"n": n, "w": w, "l": l, "wr": round(w / n, 3) if n else 0}
        all_results.extend(results)

    if not all_results:
        print("No fires found.")
        return

    total_n = len(all_results)
    total_w = sum(1 for r in all_results if r["outcome"] == "win")
    total_l = sum(1 for r in all_results if r["outcome"] == "loss")
    total_h = total_n - total_w - total_l
    wr = total_w / total_n
    ev = wr * TP_TICKS - (1 - wr) * SL_TICKS

    long_count  = sum(1 for r in all_results if r["fire"].get("trade_direction", "short") == "long")
    short_count = sum(1 for r in all_results if r["fire"].get("trade_direction", "short") == "short")

    # Wilson 95% CI
    z, n, p = 1.96, total_n, wr
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    ci_lo  = round(max(0.0, centre - margin), 4)
    ci_hi  = round(min(1.0, centre + margin), 4)

    verdict = (
        "PASS"         if wr >= 0.20  else
        "INCONCLUSIVE" if wr >= 0.111 else
        "FAIL"
    )

    summary: dict[str, Any] = {
        "spec_id":        "bid_absorption_oos_v1",
        "evaluation_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "instrument":     "ESH6",
        "period":         "2026-01-02 through 2026-02-24 (combined Jan+Feb)",
        "excluded_dates": excluded,
        "included_dates": len(per_date),
        "methodology":    "tick-level TP/SL sequencing using action='T' bid_px/ask_px from MBP-10 DBN",

        "results": {
            "n_bid_absorption_fires": total_n,
            "long":   long_count,
            "short":  short_count,
            "wins":   total_w,
            "losses": total_l,
            "holds":  total_h,
            "win_rate":      round(wr, 4),
            "ev_ticks":      round(ev, 4),
            "tp_ticks":      TP_TICKS,
            "sl_ticks":      SL_TICKS,
            "ci_95_wilson":  [ci_lo, ci_hi],
        },

        "benchmarks": {
            "breakeven_win_rate":       0.111,
            "in_sample_win_rate_esz5":  0.133,
            "jan_oos_win_rate":         0.14,
            "oos_hypothesis_threshold": 0.20,
        },

        "verdict":        verdict,
        "verdict_detail": (
            f"WR={wr:.1%}, EV={ev:+.3f}t on n={total_n} fires. "
            f"95% CI [{ci_lo:.1%}, {ci_hi:.1%}]. "
            + (
                "Above OOS hypothesis (20%) — PASS." if wr >= 0.20 else
                f"Above breakeven ({0.111:.1%}) and above ESZ5 in-sample ({0.133:.1%}) "
                "but below OOS hypothesis (20%) — INCONCLUSIVE." if wr >= 0.111 else
                f"Below breakeven ({0.111:.1%}) — FAIL."
            )
        ),

        "per_date": per_date,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Combined Jan+Feb ESH6 Result")
    print(f"  n={total_n}  W={total_w}  L={total_l}  H={total_h}")
    print(f"  WR={wr:.1%}  EV={ev:+.3f}t")
    print(f"  95% Wilson CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"  Verdict: {verdict}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
