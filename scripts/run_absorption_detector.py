"""
Absorption Entry Detector — research tool, standalone post-processing step.

For each DSR-gated signal, opens the raw MBP-10 DBN file, scans the 30-second
window after bar close, and detects tick-level absorption signatures that would
define a tighter entry price.

Outputs a JSONL file with full detection metadata for every signal window,
including windows where no signature fires. The diagnostic split between
no_dip / no_signature / genuine_adverse / fired is the primary output.

Usage:
    python scripts/run_absorption_detector.py --date 2025-12-02

Parameters tunable via CLI:
    --dsr-gate          DSR threshold (default 0.135)
    --window-secs       detection window in seconds (default 30)
    --min-dip-ticks     minimum adverse ticks before arm (default 1)
    --bid-sz-retention  fraction of initial bid_sz required (default 0.60)
    --absorption-prints minimum sell prints in burst (default 5)
    --burst-ms          burst window in ms (default 500)
    --flip-window-secs  rolling window for aggressor flip (default 1.0)
    --genuine-adverse   ticks of dip that counts as "kept going" (default 4)
"""

import argparse
import json
import sys
from collections import deque
from datetime import timedelta
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRADES_ROOT   = PROJECT_ROOT / "artifacts/experiment_agent/pflft_v8_regime_v1/runs/A_cd0_prog0_none/artifacts/run_20260304_214204"
PARQUET_ROOT  = PROJECT_ROOT / "data/processed/instrument=ES"
DBN_ROOT      = Path("c:/Users/majin/Downloads/GLBX-20260214-5X9XQHYJXV")
OUTPUT_DIR    = PROJECT_ROOT / "artifacts/absorption_detector"
TICK          = 0.25
EXCL_DATES    = {"2025-12-10", "2025-12-14"}


# ── Load gated trades for one date ───────────────────────────────────────────
def load_trades(date_str: str) -> pd.DataFrame:
    p = TRADES_ROOT / f"W10/mode=side_matched/ES_{date_str}/trades_gated.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "strategy" in df.columns:
        df = df[df["strategy"].astype(str).str.lower() == "gated"]
    if "entry_family" in df.columns:
        df = df[df["entry_family"].astype(str).str.startswith("PFLFT_v8")]
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    return df.dropna(subset=["entry_time"]).reset_index(drop=True)


# ── Attach DSR from parquet ───────────────────────────────────────────────────
def attach_dsr(trades: pd.DataFrame, date_str: str) -> pd.DataFrame:
    p = PARQUET_ROOT / f"date={date_str}/features_labels.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Parquet missing: {p}")
    schema = pq.read_schema(p).names
    if "depth_shape_ratio_bid" not in schema:
        raise ValueError(f"Old schema for {date_str}, rebuild required")

    pdf = pd.read_parquet(p, columns=["Time", "depth_shape_ratio_bid",
                                       "depth_shape_ratio_ask", "bid_price_1",
                                       "ask_price_1"])
    pdf["Time"] = pd.to_datetime(pdf["Time"], utc=True, errors="coerce")
    pdf = pdf.set_index("Time").sort_index()
    pdf["mid"] = (pdf["bid_price_1"] + pdf["ask_price_1"]) / 2.0

    rows = []
    for _, t in trades.iterrows():
        i = pdf.index.get_indexer([t["entry_time"]], method="nearest")[0]
        if i < 0:
            rows.append({"dsr": np.nan, "bar_close_mid": np.nan})
            continue
        bar = pdf.iloc[i]
        is_long = str(t.get("side", "")).lower() == "long"
        rows.append({
            "dsr": float(bar["depth_shape_ratio_ask"] if is_long else bar["depth_shape_ratio_bid"]),
            "bar_close_mid": float(bar["mid"]),
        })

    dsr_df = pd.DataFrame(rows, index=trades.index)
    return pd.concat([trades, dsr_df], axis=1)


# ── Stream trade events from DBN file ────────────────────────────────────────
def load_dbn_trades(date_str: str) -> list[dict]:
    """
    Returns all action='T' records from the DBN file as a list of dicts,
    sorted by ts_event. Loads the full session so we can slice any window.
    """
    compact = date_str.replace("-", "")
    dbn_dir = DBN_ROOT / f"glbx-mdp3-{compact}.mbp-10.dbn"
    candidates = [dbn_dir / f"glbx-mdp3-{compact}.mbp-10.dbn", dbn_dir]
    dbn_path = None
    for c in candidates:
        if c.is_file():
            dbn_path = c
            break
    if dbn_path is None:
        raise FileNotFoundError(f"DBN file not found for {date_str} under {dbn_dir}")

    store = db.DBNStore.from_file(dbn_path)

    # Resolve ESZ5 instrument_id from symbology (varies per date)
    esz5_iid = None
    try:
        sym_map = store.symbology
        for sym, mappings in sym_map.get("mappings", {}).items():
            if sym == "ESZ5":
                esz5_iid = int(mappings[0]["symbol"])
                break
    except Exception:
        pass
    if esz5_iid is None:
        raise RuntimeError(f"Could not resolve ESZ5 instrument_id for {date_str}")

    records = []
    for rec in store:
        if str(getattr(rec, "action", "")) != "T":
            continue
        if rec.instrument_id != esz5_iid:
            continue
        lvl = rec.levels[0]
        records.append({
            "ts":        pd.Timestamp(rec.ts_event, unit="ns", tz="UTC"),
            "price":     rec.price / 1e9,
            "size":      rec.size,
            "side":      str(rec.side),   # 'A'=buy-aggressor, 'B'=sell-aggressor
            "bid_px":    lvl.bid_px / 1e9,
            "ask_px":    lvl.ask_px / 1e9,
            "bid_sz":    lvl.bid_sz,
            "ask_sz":    lvl.ask_sz,
        })
    print(f"  ESZ5 instrument_id={esz5_iid}, trade events: {len(records):,}")
    return records


# ── Core: run absorption detector on one signal window ───────────────────────
def detect_absorption(
    signal: dict,
    dbn_trades: list[dict],
    *,
    window_secs: float = 30.0,
    min_dip_ticks: float = 1.0,
    bid_sz_retention: float = 0.60,
    absorption_prints: int = 5,
    burst_ms: float = 500.0,
    flip_window_secs: float = 1.0,
    genuine_adverse_ticks: float = 4.0,
) -> dict:
    """
    Runs the 4 absorption signatures on the tick stream for one signal window.
    Returns a dict with full detection metadata.
    """
    t0          = signal["entry_time"]          # signal bar close timestamp
    t_end       = t0 + timedelta(seconds=window_secs)
    is_long     = str(signal.get("side", "")).lower() == "long"
    bar_mid     = signal["bar_close_mid"]
    dip_thresh  = bar_mid - min_dip_ticks * TICK if is_long else bar_mid + min_dip_ticks * TICK

    # Slice window from pre-loaded trade list (binary search on sorted list)
    window = [r for r in dbn_trades if t0 <= r["ts"] <= t_end]

    meta = {
        "date":            signal.get("date", ""),
        "entry_time_orig": str(t0),
        "side":            signal.get("side", ""),
        "bar_close_mid":   bar_mid,
        "bar_close_dsr":   signal.get("dsr", np.nan),
        "entry_px_orig":   signal.get("entry_px", np.nan),
        "mfe_ticks_orig":  signal.get("mfe_ticks", np.nan),
        "mae_ticks_orig":  signal.get("mae_ticks", np.nan),
        "n_trades_in_window": len(window),
        # detection results
        "dip_armed":             False,
        "dip_arm_ts":            None,
        "dip_arm_price":         None,
        "dip_arm_bid_sz":        None,
        "max_adverse_ticks":     None,
        "bid_absorption_fired":  False,
        "bid_absorption_ts":     None,
        "bid_absorption_px":     None,
        "bid_absorption_prints": None,
        "ask_absorbed_fired":    False,
        "ask_absorbed_ts":       None,
        "ask_exhausted_fired":   False,
        "ask_exhausted_ts":      None,
        "aggressor_flip_fired":  False,
        "aggressor_flip_ts":     None,
        "aggressor_flip_px":     None,
        "detection_fired":       False,
        "detection_signature":   None,
        "detection_ts":          None,
        "detection_entry_px":    None,
        "entry_dip_ticks":       None,
        "window_outcome":        None,
        # diagnostics
        "max_sell_burst":        0,
        "min_bid_sz_retention":  None,
        "max_ask_sz_drop":       None,
    }

    if not window:
        meta["window_outcome"] = "no_trades_in_window"
        return meta

    # Compute max adverse ticks in window
    prices = [r["price"] for r in window]
    if is_long:
        max_adverse = (bar_mid - min(prices)) / TICK
    else:
        max_adverse = (max(prices) - bar_mid) / TICK
    meta["max_adverse_ticks"] = round(max_adverse, 2)

    # ── ARM the detector once price dips >= min_dip_ticks ────────────────────
    dip_armed    = False
    arm_ts       = None
    arm_bid_sz   = None
    arm_ask_sz   = None

    # absorption state
    burst_deque: deque = deque()   # (ts, price, size) of sell (long) or buy (short) prints
    min_bid_sz_frac    = 1.0
    max_sell_burst     = 0
    ask_sz_baseline    = None
    max_ask_sz_drop    = 0.0

    # aggressor flip state: rolling window of (ts, signed_size)
    flip_deque: deque = deque()    # (ts, signed_size) positive=buy, negative=sell

    detection: dict | None = None  # first signature to fire

    for rec in window:
        ts       = rec["ts"]
        price    = rec["price"]
        side     = rec["side"]       # 'A'=buy, 'B'=sell
        size     = rec["size"]
        bid_px   = rec["bid_px"]
        ask_px   = rec["ask_px"]
        bid_sz   = rec["bid_sz"]
        ask_sz   = rec["ask_sz"]

        # ── Check arm condition ───────────────────────────────────────────────
        if not dip_armed:
            dip_cond = (price <= dip_thresh) if is_long else (price >= dip_thresh)
            if dip_cond:
                dip_armed    = True
                arm_ts       = ts
                arm_bid_sz   = bid_sz if is_long else ask_sz
                arm_ask_sz   = ask_sz if is_long else bid_sz
                ask_sz_baseline = arm_ask_sz
                meta["dip_armed"]     = True
                meta["dip_arm_ts"]    = str(ts)
                meta["dip_arm_price"] = price
                meta["dip_arm_bid_sz"] = arm_bid_sz
            else:
                continue   # not armed yet

        # ── Bid size retention tracking (diagnostic) ─────────────────────────
        cur_bid_sz = bid_sz if is_long else ask_sz
        if arm_bid_sz and arm_bid_sz > 0:
            frac = cur_bid_sz / arm_bid_sz
            min_bid_sz_frac = min(min_bid_sz_frac, frac)

        # ── Ask size drop tracking ────────────────────────────────────────────
        cur_ask_sz = ask_sz if is_long else bid_sz
        if ask_sz_baseline is not None and ask_sz_baseline > 0:
            drop = (ask_sz_baseline - cur_ask_sz) / ask_sz_baseline
            max_ask_sz_drop = max(max_ask_sz_drop, drop)

        # ── Aggressor flip: rolling window ───────────────────────────────────
        signed = size if side == "A" else -size
        flip_deque.append((ts, signed))
        cutoff = ts - timedelta(seconds=flip_window_secs)
        while flip_deque and flip_deque[0][0] < cutoff:
            flip_deque.popleft()
        flip_sum = sum(v for _, v in flip_deque)

        # ── Bid absorption: sell burst at same price ──────────────────────────
        # For longs: watch sell-aggressor trades (side='B') hitting the bid
        if (is_long and side == "B") or (not is_long and side == "A"):
            burst_deque.append((ts, price, size))
            burst_cutoff = ts - timedelta(milliseconds=burst_ms)
            while burst_deque and burst_deque[0][0] < burst_cutoff:
                burst_deque.popleft()
            # Count prints at same price level as current
            same_level = [x for x in burst_deque if x[1] == price]
            n_burst = len(same_level)
            max_sell_burst = max(max_sell_burst, n_burst)

            # Bid absorption: N prints at same level, bid hasn't dropped, bid_sz retained
            if (not meta["bid_absorption_fired"]
                    and n_burst >= absorption_prints
                    and bid_px == price   # bid hasn't dropped
                    and arm_bid_sz is not None
                    and arm_bid_sz > 0
                    and (bid_sz / arm_bid_sz) >= bid_sz_retention):
                meta["bid_absorption_fired"]  = True
                meta["bid_absorption_ts"]     = str(ts)
                meta["bid_absorption_px"]     = price
                meta["bid_absorption_prints"] = n_burst
                if detection is None:
                    detection = {"sig": "bid_absorption", "ts": ts,
                                 "entry_px": ask_px if is_long else bid_px}

        # ── Offer size absorbed / exhausted ──────────────────────────────────
        # Long:  watch ask_sz declining at same ask_px while price doesn't advance
        # Short: watch bid_sz declining at same bid_px while price doesn't retreat
        if ask_sz_baseline is not None:
            sz_declined = (cur_ask_sz < ask_sz_baseline * 0.7)   # 30% shrink
            if is_long:
                price_flat  = (ask_px <= (bar_mid + 1 * TICK))   # ask hasn't risen
                aggr_count  = sum(1 for _, v in flip_deque if v > 0)  # buys in window
                entry_ref   = ask_px
            else:
                price_flat  = (bid_px >= (bar_mid - 1 * TICK))   # bid hasn't fallen
                aggr_count  = sum(1 for _, v in flip_deque if v < 0)  # sells in window
                entry_ref   = bid_px
            if sz_declined and price_flat:
                if not meta["ask_absorbed_fired"] and aggr_count >= 2:
                    meta["ask_absorbed_fired"] = True
                    meta["ask_absorbed_ts"]    = str(ts)
                    if detection is None:
                        detection = {"sig": "ask_absorbed", "ts": ts,
                                     "entry_px": entry_ref}
                elif not meta["ask_exhausted_fired"] and aggr_count < 2:
                    meta["ask_exhausted_fired"] = True
                    meta["ask_exhausted_ts"]    = str(ts)
                    if detection is None:
                        detection = {"sig": "ask_exhausted", "ts": ts,
                                     "entry_px": entry_ref}

        # ── Aggressor flip ────────────────────────────────────────────────────
        # Long:  price dipped below bar_mid, rolling buy volume overtakes sells → flip_sum > 0
        # Short: price rose above bar_mid, rolling sell volume overtakes buys  → flip_sum < 0
        flip_price_ok = (price <= bar_mid - TICK) if is_long else (price >= bar_mid + TICK)
        flip_cond     = (flip_sum > 0)             if is_long else (flip_sum < 0)
        if (not meta["aggressor_flip_fired"]
                and flip_cond and flip_price_ok):
            meta["aggressor_flip_fired"] = True
            meta["aggressor_flip_ts"]    = str(ts)
            meta["aggressor_flip_px"]    = price
            # Aggressor flip is highest priority — override earlier detection
            flip_det = {"sig": "aggressor_flip", "ts": ts,
                        "entry_px": ask_px if is_long else bid_px}
            if detection is None or flip_det["ts"] <= detection["ts"]:
                detection = flip_det

    # ── Diagnostics ──────────────────────────────────────────────────────────
    meta["max_sell_burst"]       = max_sell_burst
    meta["min_bid_sz_retention"] = round(min_bid_sz_frac, 3)
    meta["max_ask_sz_drop"]      = round(max_ask_sz_drop, 3)

    # ── Window outcome ────────────────────────────────────────────────────────
    if not dip_armed:
        meta["window_outcome"] = "no_dip"
    elif detection is not None:
        meta["detection_fired"]     = True
        meta["detection_signature"] = detection["sig"]
        meta["detection_ts"]        = str(detection["ts"])
        meta["detection_entry_px"]  = detection["entry_px"]
        meta["entry_dip_ticks"]     = round(
            (bar_mid - detection["entry_px"]) / TICK if is_long
            else (detection["entry_px"] - bar_mid) / TICK, 2
        )
        meta["window_outcome"] = "fired"
    elif max_adverse >= genuine_adverse_ticks:
        meta["window_outcome"] = "genuine_adverse"
    else:
        meta["window_outcome"] = "no_signature"

    return meta


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Absorption entry detector")
    parser.add_argument("--date",               default="2025-12-02")
    parser.add_argument("--dsr-gate",           type=float, default=0.135)
    parser.add_argument("--window-secs",        type=float, default=30.0)
    parser.add_argument("--min-dip-ticks",      type=float, default=1.0)
    parser.add_argument("--bid-sz-retention",   type=float, default=0.60)
    parser.add_argument("--absorption-prints",  type=int,   default=5)
    parser.add_argument("--burst-ms",           type=float, default=500.0)
    parser.add_argument("--flip-window-secs",   type=float, default=1.0)
    parser.add_argument("--genuine-adverse",    type=float, default=4.0)
    args = parser.parse_args()

    date_str = args.date
    print(f"=== Absorption Detector — {date_str} ===")

    # Load signals
    trades = load_trades(date_str)
    if trades.empty:
        print(f"No gated trades for {date_str}")
        sys.exit(0)
    print(f"Gated trades: {len(trades)}")

    trades = attach_dsr(trades, date_str)
    dsr_pass = trades[trades["dsr"] < args.dsr_gate].copy()
    print(f"DSR-passing trades (DSR < {args.dsr_gate}): {len(dsr_pass)}")

    if dsr_pass.empty:
        print("No DSR-passing trades — nothing to detect")
        sys.exit(0)

    # Load full session tick stream (once)
    print(f"Loading tick stream from DBN...")
    dbn_trades = load_dbn_trades(date_str)
    print(f"  Trade events loaded: {len(dbn_trades):,}")

    # Run detector on each signal
    results = []
    for _, signal in dsr_pass.iterrows():
        sig_dict = signal.to_dict()
        meta = detect_absorption(
            sig_dict,
            dbn_trades,
            window_secs=args.window_secs,
            min_dip_ticks=args.min_dip_ticks,
            bid_sz_retention=args.bid_sz_retention,
            absorption_prints=args.absorption_prints,
            burst_ms=args.burst_ms,
            flip_window_secs=args.flip_window_secs,
            genuine_adverse_ticks=args.genuine_adverse,
        )
        results.append(meta)
        fired = "FIRED" if meta["detection_fired"] else meta["window_outcome"].upper()
        print(f"  {signal['entry_time']}  {signal['side']:<5}  "
              f"dsr={signal['dsr']:.4f}  outcome={fired}"
              + (f"  sig={meta['detection_signature']}  "
                 f"entry_px={meta['detection_entry_px']}  "
                 f"dip={meta['entry_dip_ticks']}t"
                 if meta["detection_fired"] else
                 f"  max_adv={meta['max_adverse_ticks']}t")
              )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Summary ({date_str}) ===")
    outcomes = {}
    for r in results:
        outcomes[r["window_outcome"]] = outcomes.get(r["window_outcome"], 0) + 1
    for k, v in sorted(outcomes.items()):
        print(f"  {k:<20}: {v:>3}  ({v/len(results)*100:.0f}%)")

    fired = [r for r in results if r["detection_fired"]]
    if fired:
        sigs = {}
        for r in fired:
            sigs[r["detection_signature"]] = sigs.get(r["detection_signature"], 0) + 1
        print(f"\nFired by signature: {sigs}")
        dips = [r["entry_dip_ticks"] for r in fired if r["entry_dip_ticks"] is not None]
        print(f"Entry dip below bar-close (ticks): "
              f"min={min(dips):.1f}  median={sorted(dips)[len(dips)//2]:.1f}  max={max(dips):.1f}")

    no_dip = [r for r in results if r["window_outcome"] == "no_dip"]
    no_sig = [r for r in results if r["window_outcome"] == "no_signature"]
    gen_adv= [r for r in results if r["window_outcome"] == "genuine_adverse"]
    if no_sig:
        bursts = [r["max_sell_burst"] for r in no_sig]
        retentions = [r["min_bid_sz_retention"] for r in no_sig if r["min_bid_sz_retention"] is not None]
        print(f"\nno_signature diagnostics (n={len(no_sig)}):")
        print(f"  max_sell_burst:      {sorted(bursts)}")
        print(f"  min_bid_sz_retention:{[round(x,2) for x in sorted(retentions)]}")
    if gen_adv:
        adv = [r["max_adverse_ticks"] for r in gen_adv]
        print(f"\ngenuine_adverse max dips: {sorted(adv)}")

    # ── Write output ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"absorption_{date_str}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nWrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    main()
