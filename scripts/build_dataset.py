"""
Build fixed-interval feature rows + labels from Databento DBN (MBP-10 + TRADES).

Output:
  data/processed/instrument=ES/date=YYYY-MM-DD/features_labels.parquet

Env vars:
  INPUT_DBN            path to .dbn file or directory of .dbn files (optional if INPUT_MBP_DBN/TRADES set)
  INPUT_MBP_DBN        path to MBP-10 .dbn file (optional)
  INPUT_TRADES_DBN     path to TRADES .dbn file (optional)
  OUTPUT_DIR           output root (default: data/processed)
  INSTRUMENT           instrument symbol (default: ES)
  FRONT_MONTH          optional exact symbol to keep (e.g., ESU5); others skipped
  STEP_MS              emit interval in ms (default: 100)
  HORIZON_MS           label horizon in ms (default: 1000)
  NEUTRAL_BAND_BPS     neutral band in bps for direction label (default: 0.0)
  REPRICE_BPS          repricing event threshold in bps (default: 4.0)
  FLOW_WINDOW          rolling window (rows) for flow/lag features (default: 20)
  SESSION_START        optional session start (HH:MM:SS) in exchange time (treated as UTC if SESSION_TZ unset)
  SESSION_END          optional session end (HH:MM:SS) in exchange time (treated as UTC if SESSION_TZ unset)
  SESSION_TZ           optional IANA timezone for session filtering (e.g., America/Chicago)
  EMIT_EMPTY           if "1", emit empty intervals even with no events (default: 0)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import datetime

import numpy as np
import pandas as pd
import heapq


@dataclass
class TradeAgg:
    volume: float = 0.0
    signed_volume: float = 0.0
    count: int = 0

    def reset(self):
        self.volume = 0.0
        self.signed_volume = 0.0
        self.count = 0


def _iter_dbn_files(path: Path) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob("*.dbn"))
    if path.is_file() and path.suffix.lower() == ".dbn":
        return [path]
    raise FileNotFoundError(f"INPUT_DBN not found or not a .dbn file/dir: {path}")


def _split_mbp_trades(paths: List[Path]) -> tuple[List[Path], List[Path]]:
    mbp = [p for p in paths if "mbp" in p.name.lower()]
    trades = [p for p in paths if "trade" in p.name.lower()]
    return mbp, trades


def _trade_side_sign(side_val) -> float:
    # Databento side is often 1=buy, 2=sell; some schemas use 'B'/'S'.
    if side_val in (1, "B", "b", "BUY", "buy", "BID", "bid"):
        return 1.0
    if side_val in (2, "S", "s", "SELL", "sell", "ASK", "ask", "A"):
        return -1.0
    return 0.0


def _vwap(prices: List[float], sizes: List[float]) -> Optional[float]:
    tot = sum(sizes)
    if tot == 0:
        return None
    return sum(p * s for p, s in zip(prices, sizes)) / tot


def _sweep_cost(prices: List[float], sizes: List[float], mid: float, qty: float = 1.0) -> Optional[float]:
    remaining = qty
    cost = 0.0
    filled = 0.0
    for p, s in zip(prices, sizes):
        take = min(s, remaining)
        cost += take * p
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled == 0 or mid == 0:
        return None
    avg_price = cost / filled
    return (avg_price - mid) / mid


def _price_to_float(raw: int) -> float:
    import databento as db
    if raw == db.UNDEF_PRICE:
        return np.nan
    return raw / db.FIXED_PRICE_SCALE


def _parse_mbp10(rec) -> Dict[str, float]:
    # Extract top-10 book arrays from MBP-10 record levels.
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

    mid = (bid0 + ask0) / 2
    spread = ask0 - bid0
    top_bid_depth = bid_sz[0]
    top_ask_depth = ask_sz[0]
    denom = top_bid_depth + top_ask_depth
    order_book_imbalance = (top_bid_depth / denom) if denom else np.nan

    depth_bid_top5 = sum(bid_sz[:5])
    depth_ask_top5 = sum(ask_sz[:5])
    depth_denom = depth_bid_top5 + depth_ask_top5
    depth_imbalance_top5 = (depth_bid_top5 - depth_ask_top5) / depth_denom if depth_denom else np.nan

    vwap_bid = _vwap(bid_px[:5], bid_sz[:5])
    vwap_ask = _vwap(ask_px[:5], ask_sz[:5])
    book_slope = (vwap_ask - vwap_bid) / mid if (vwap_bid and vwap_ask and mid) else np.nan

    sweep_cost_buy1 = _sweep_cost(ask_px, ask_sz, mid=mid, qty=1.0)
    sweep_cost_sell1 = _sweep_cost(bid_px, bid_sz, mid=mid, qty=1.0)

    return {
        "mid": mid,
        "spread": spread,
        "top_bid_depth": top_bid_depth,
        "top_ask_depth": top_ask_depth,
        "order_book_imbalance": order_book_imbalance,
        "depth_bid_top5": depth_bid_top5,
        "depth_ask_top5": depth_ask_top5,
        "depth_imbalance_top5": depth_imbalance_top5,
        "book_slope_top5": book_slope,
        "sweep_cost_buy1": sweep_cost_buy1,
        "sweep_cost_sell1": sweep_cost_sell1,
    }


def _emit_row(ts: pd.Timestamp, symbol: str, book: Dict[str, float], trade_agg: TradeAgg) -> Dict[str, float]:
    row = {
        "Time": ts,
        "Symbol": symbol,
        **book,
        "trade_volume": trade_agg.volume,
        "signed_volume": trade_agg.signed_volume,
        "trade_count": trade_agg.count,
    }
    return row


def _label_rows(
    df: pd.DataFrame,
    horizon_steps: int,
    neutral_band_bps: float,
    reprice_bps: float,
) -> Tuple[pd.DataFrame, float]:
    df = df.copy()
    if horizon_steps < 1:
        horizon_steps = 1
    mid = df["mid"].astype(float)
    future = mid.shift(-horizon_steps)
    fwd_ret = (future - mid) / mid
    missing_pct = float(fwd_ret.isna().mean())
    band = neutral_band_bps / 1e4
    label = (fwd_ret > band).astype(int) - (fwd_ret < -band).astype(int)
    df["fwd_ret"] = fwd_ret
    df["label"] = label
    # Cost-aware label (relative)
    if {"spread", "sweep_cost_buy1", "sweep_cost_sell1", "mid"}.issubset(df.columns):
        spread_rel = df["spread"] / df["mid"]
        sweep_mag = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
        cost_proxy = spread_rel / 2 + sweep_mag
        df["fwd_ret_net"] = df["fwd_ret"] - cost_proxy
    # Favorable excursion label (directional)
    if "label" in df.columns:
        dir_sign = df["label"].astype(float).clip(-1, 1)
        fwd = mid.shift(-1)
        rel = (fwd - mid) / mid
        rel_signed = rel * dir_sign
        fav = rel_signed.rolling(horizon_steps, min_periods=horizon_steps).max()
        df["fav_excursion"] = fav
    # Repricing event: directional jump relative to flow sign
    if "signed_volume" in df.columns:
        flow_sign = np.sign(df["signed_volume"].astype(float))
        fwd = mid.shift(-1)
        rel = (fwd - mid) / mid
        rel_signed = rel * flow_sign
        thresh = reprice_bps / 1e4
        max_rel_signed = rel_signed.rolling(horizon_steps, min_periods=horizon_steps).max()
        df["repriced_within_H"] = (max_rel_signed >= thresh).astype(int)
    return df.dropna(subset=["fwd_ret", "label"]), missing_pct


def _add_derived_features(df: pd.DataFrame, step_ms: int, window: int) -> pd.DataFrame:
    df = df.copy()
    if "signed_volume" in df.columns:
        df["abs_signed_volume"] = df["signed_volume"].abs()
    if {"signed_volume", "trade_volume"}.issubset(df.columns):
        denom = df["trade_volume"].astype(float).replace(0.0, np.nan)
        df["trade_imbalance_ratio"] = df["signed_volume"] / denom
    if "trade_volume" in df.columns:
        per_sec = 1000.0 / max(step_ms, 1)
        df["flow_intensity"] = df["trade_volume"] * per_sec
        df["flow_intensity_roll"] = df["flow_intensity"].rolling(window, min_periods=1).mean()
        df["flow_accel"] = df["flow_intensity_roll"].diff()
    if "mid" in df.columns:
        prev = df["mid"].shift(1)
        df["dmid_bps"] = ((df["mid"] - prev) / prev) * 1e4
        prev_w = df["mid"].shift(window)
        df["dmid_bps_roll"] = ((df["mid"] - prev_w) / prev_w) * 1e4
    if "abs_signed_volume" in df.columns and "dmid_bps" in df.columns:
        eps = 1e-9
        df["impact_per_flow"] = df["dmid_bps"] / (df["abs_signed_volume"] + eps)
    if "flow_intensity" in df.columns and "dmid_bps" in df.columns:
        eps = 1e-9
        df["lag_score"] = df["flow_intensity"] / (df["dmid_bps"].abs() + eps)
    if {"depth_bid_top5", "depth_ask_top5"}.issubset(df.columns):
        df["depth_total_top5"] = df["depth_bid_top5"] + df["depth_ask_top5"]
    if {"spread", "mid"}.issubset(df.columns):
        spread_bps = (df["spread"] / df["mid"]) * 1e4
        df["spread_stability"] = spread_bps.rolling(window, min_periods=1).std()
    return df


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


def main():
    try:
        import databento as db  # type: ignore
    except ImportError as exc:
        raise ImportError("Databento Python package is required. Install with `pip install databento`.") from exc

    input_path = Path(os.environ.get("INPUT_DBN", "")).expanduser().resolve() if os.environ.get("INPUT_DBN") else None
    input_mbp = Path(os.environ.get("INPUT_MBP_DBN", "")).expanduser().resolve() if os.environ.get("INPUT_MBP_DBN") else None
    input_trades = Path(os.environ.get("INPUT_TRADES_DBN", "")).expanduser().resolve() if os.environ.get("INPUT_TRADES_DBN") else None
    if input_path is None and input_mbp is None and input_trades is None:
        raise FileNotFoundError("Set INPUT_DBN or INPUT_MBP_DBN/INPUT_TRADES_DBN.")

    output_root = Path(os.environ.get("OUTPUT_DIR", "data/processed")).expanduser().resolve()
    instrument = os.environ.get("INSTRUMENT", "ES")
    front_month = os.environ.get("FRONT_MONTH", "").strip()
    step_ms = int(os.environ.get("STEP_MS", "100"))
    horizon_ms = int(os.environ.get("HORIZON_MS", "1000"))
    neutral_band_bps = float(os.environ.get("NEUTRAL_BAND_BPS", "0.0"))
    reprice_bps = float(os.environ.get("REPRICE_BPS", "4.0"))
    flow_window = int(os.environ.get("FLOW_WINDOW", "20"))
    emit_empty = os.environ.get("EMIT_EMPTY", "0").strip() in {"1", "true", "yes", "y"}
    session_start = _parse_time(os.environ.get("SESSION_START", "").strip())
    session_end = _parse_time(os.environ.get("SESSION_END", "").strip())
    session_tz = os.environ.get("SESSION_TZ", "").strip() or None

    files = _iter_dbn_files(input_path) if input_path is not None else []
    if input_mbp is not None:
        files.append(input_mbp)
    if input_trades is not None:
        files.append(input_trades)
    files = list({p for p in files})
    mbp_files, trade_files = _split_mbp_trades(files)
    if not mbp_files and not trade_files:
        raise FileNotFoundError("No .dbn files found for MBP/TRADES.")

    # Build iterators per file type and merge by ts_event
    def _iter_store(path: Path):
        store = db.DBNStore.from_file(path)
        for rec in store:
            yield rec

    iterators = []
    for p in mbp_files:
        iterators.append(_iter_store(p))
    for p in trade_files:
        iterators.append(_iter_store(p))

    # Merge iterators by timestamp
    heap = []
    for idx, it in enumerate(iterators):
        try:
            rec = next(it)
            heapq.heappush(heap, (rec.ts_event, idx, rec, it))
        except StopIteration:
            continue

    if not heap:
        raise ValueError("No records found in DBN input.")

    rows: List[Dict[str, float]] = []
    book: Dict[str, float] = {}
    agg = TradeAgg()
    next_emit_ns: Optional[int] = None

    while heap:
        ts_event, idx, rec, it = heapq.heappop(heap)
        ts = pd.to_datetime(ts_event, unit="ns", utc=True)

        if not _in_session(ts, session_start, session_end, session_tz):
            book = {}
            agg.reset()
            next_emit_ns = None
            try:
                rec_next = next(it)
                heapq.heappush(heap, (rec_next.ts_event, idx, rec_next, it))
            except StopIteration:
                pass
            continue

        if next_emit_ns is None:
            next_emit_ns = int(ts_event // (step_ms * 1_000_000) * (step_ms * 1_000_000) + step_ms * 1_000_000)

        if front_month:
            rec_symbol = getattr(rec, "symbol", None) or getattr(rec, "instrument", None)
            if rec_symbol and rec_symbol != front_month:
                try:
                    rec = next(it)
                    heapq.heappush(heap, (rec.ts_event, idx, rec, it))
                except StopIteration:
                    pass
                continue

        # Update book or trades
        if rec.__class__.__name__.lower().startswith("mbp"):
            parsed = _parse_mbp10(rec)
            if parsed:
                book = parsed
        elif rec.__class__.__name__.lower().startswith("trade"):
            price = getattr(rec, "price", None)
            size = getattr(rec, "size", None)
            side = getattr(rec, "side", None)
            if price is not None and size is not None:
                if isinstance(price, int):
                    price = _price_to_float(price)
                agg.volume += float(size)
                agg.signed_volume += float(size) * _trade_side_sign(side)
                agg.count += 1

        # Emit rows on clock
        while next_emit_ns is not None and ts_event >= next_emit_ns:
            if book or emit_empty:
                emit_ts = pd.to_datetime(next_emit_ns, unit="ns", utc=True)
                rows.append(_emit_row(emit_ts, instrument, book, agg))
            agg.reset()
            next_emit_ns += step_ms * 1_000_000

        # advance iterator
        try:
            rec_next = next(it)
            heapq.heappush(heap, (rec_next.ts_event, idx, rec_next, it))
        except StopIteration:
            pass

    if not rows:
        print("No rows emitted; check inputs/session filter.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("Time")
    df = _add_derived_features(df, step_ms=step_ms, window=flow_window)
    horizon_steps = max(1, horizon_ms // step_ms)
    df, missing_pct = _label_rows(
        df,
        horizon_steps=horizon_steps,
        neutral_band_bps=neutral_band_bps,
        reprice_bps=reprice_bps,
    )

    # Basic summary diagnostics
    if {"spread", "mid"}.issubset(df.columns):
        spread_bps = (df["spread"] / df["mid"]) * 1e4
        q = spread_bps.quantile([0.05, 0.5, 0.95]).to_dict()
        print(f"Spread bps quantiles (5/50/95): {q}")
    if {"spread", "mid", "sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df.columns):
        sweep_mag = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
        cost_bps = ((df["spread"] / df["mid"]) / 2 + sweep_mag) * 1e4
        q = cost_bps.quantile([0.05, 0.5, 0.95]).to_dict()
        print(f"Cost proxy bps quantiles (5/50/95): {q}")
    print(f"Label missing rate: {missing_pct:.2%}")

    # Dtype tightening for Parquet size/speed
    float32_cols = [
        "mid",
        "spread",
        "order_book_imbalance",
        "depth_imbalance_top5",
        "book_slope_top5",
        "sweep_cost_buy1",
        "sweep_cost_sell1",
        "fwd_ret",
        "fwd_ret_net",
        "fav_excursion",
        "trade_volume",
        "signed_volume",
    ]
    int32_cols = ["trade_count", "top_bid_depth", "top_ask_depth", "depth_bid_top5", "depth_ask_top5"]
    int8_cols = ["label"]
    for col in float32_cols:
        if col in df.columns:
            df[col] = df[col].astype("float32")
    for col in int32_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
    for col in int8_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int8")

    # Partition by date
    df["date"] = df["Time"].dt.date.astype(str)
    for date, df_day in df.groupby("date"):
        out_dir = output_root / f"instrument={instrument}" / f"date={date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "features_labels.parquet"
        df_day.drop(columns=["date"]).to_parquet(out_path, index=False)
        print(f"Wrote {len(df_day)} rows to {out_path}")


if __name__ == "__main__":
    main()
