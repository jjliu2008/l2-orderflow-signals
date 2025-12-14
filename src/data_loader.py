import json
from pathlib import Path
from typing import Iterable, List, Union

import pandas as pd


def _load_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Time"], nrows=max_rows)
    df["Symbol"] = path.stem
    return df


def _parse_coinbase_message(msg: dict, fallback_symbol: str) -> List[dict]:
    """
    Handle Coinbase websocket messages (matches/last_match/ticker).
    """
    mtype = msg.get("type")
    if mtype not in {"match", "last_match", "ticker"}:
        return []

    price = float(msg.get("price", "nan"))
    size = float(msg.get("size", msg.get("last_size", "nan")))
    t = pd.to_datetime(msg.get("time"))
    symbol = msg.get("product_id", fallback_symbol)

    return [
        {
            "Time": t,
            "Symbol": symbol,
            "BidPrice1": price,
            "AskPrice1": price,
            "BidVolume1": 0.0,
            "AskVolume1": 0.0,
            "Volume": size,
        }
    ]


def _best_book_levels(entries: Iterable, side: str) -> tuple[float | None, float | None, pd.Timestamp | None]:
    """
    Extract best price/volume/timestamp from Kraken book arrays.
    side: 'b' or 'a'
    """
    best_price = None
    best_vol = None
    best_ts = None

    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        try:
            price = float(entry[0])
            vol = float(entry[1])
            ts = pd.to_datetime(float(entry[2]), unit="s", utc=True)
        except (ValueError, TypeError):
            continue
        if vol <= 0:
            continue

        if best_price is None:
            best_price, best_vol, best_ts = price, vol, ts
            continue

        if side == "b" and price > best_price:
            best_price, best_vol, best_ts = price, vol, ts
        if side == "a" and price < best_price:
            best_price, best_vol, best_ts = price, vol, ts

    return best_price, best_vol, best_ts


def _update_book_state(state: dict, updates: Iterable, side: str):
    """
    Apply book updates to the in-memory ladder (price -> volume).
    Zero volume removes a level.
    """
    ladder = state.setdefault(side, {})
    for entry in updates:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            price = float(entry[0])
            vol = float(entry[1])
        except (ValueError, TypeError):
            continue
        if vol <= 0:
            ladder.pop(price, None)
        else:
            ladder[price] = vol


def _sorted_ladder(ladder: dict, side: str) -> List[tuple[float, float]]:
    items = list(ladder.items())
    if side == "b":
        items.sort(key=lambda x: -x[0])
    else:
        items.sort(key=lambda x: x[0])
    return items


def _aggregate_depth_features(bids: List[tuple[float, float]], asks: List[tuple[float, float]], mid: float, top_n: int = 5, sweep_qty: float = 1.0) -> dict:
    """
    Compute depth aggregates and simple sweep-cost proxies from ladders.
    """
    bid_top = bids[:top_n]
    ask_top = asks[:top_n]

    depth_bid_top = sum(v for _, v in bid_top)
    depth_ask_top = sum(v for _, v in ask_top)
    depth_denom = depth_bid_top + depth_ask_top
    depth_imb_top = (depth_bid_top - depth_ask_top) / depth_denom if depth_denom else None

    def _vwap(levels: List[tuple[float, float]]) -> float | None:
        vol_sum = sum(v for _, v in levels)
        if vol_sum == 0:
            return None
        return sum(p * v for p, v in levels) / vol_sum

    vwap_bid = _vwap(bid_top)
    vwap_ask = _vwap(ask_top)
    book_slope = None
    if vwap_bid is not None and vwap_ask is not None and mid:
        book_slope = (vwap_ask - vwap_bid) / mid

    def _sweep_cost(levels: List[tuple[float, float]], qty: float) -> float | None:
        remaining = qty
        cost = 0.0
        filled = 0.0
        for price, vol in levels:
            take = min(vol, remaining)
            cost += take * price
            filled += take
            remaining -= take
            if remaining <= 0:
                break
        if filled == 0 or mid == 0:
            return None
        avg_price = cost / filled
        return (avg_price - mid) / mid

    sweep_cost_buy1 = _sweep_cost(asks, sweep_qty)
    sweep_cost_sell1 = _sweep_cost(bids, sweep_qty)

    return {
        "depth_bid_top5": depth_bid_top if depth_bid_top else None,
        "depth_ask_top5": depth_ask_top if depth_ask_top else None,
        "depth_imbalance_top5": depth_imb_top,
        "book_slope_top5": book_slope,
        "sweep_cost_buy1": sweep_cost_buy1,
        "sweep_cost_sell1": sweep_cost_sell1,
    }


def _parse_kraken_message(msg: Iterable, fallback_symbol: str, book_state: dict) -> List[dict]:
    """
    Handle Kraken public websocket messages saved directly from the feed.
    Supports trade messages and book snapshots/deltas (book-N).
    """
    msg_list = list(msg)
    if len(msg_list) < 3:
        return []

    channel = msg_list[-2]
    pair = msg_list[-1] if isinstance(msg_list[-1], str) else fallback_symbol

    rows: List[dict] = []

    # Trade messages
    if isinstance(channel, str) and "trade" in channel:
        trades = msg_list[1] if len(msg_list) > 1 else []
        for trade in trades:
            if not isinstance(trade, (list, tuple)) or len(trade) < 3:
                continue
            try:
                price = float(trade[0])
                size = float(trade[1])
                t = pd.to_datetime(float(trade[2]), unit="s", utc=True)
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "Time": t,
                    "Symbol": pair,
                    "BidPrice1": price,
                    "AskPrice1": price,
                    "BidVolume1": 0.0,
                    "AskVolume1": 0.0,
                    "Volume": size,
                    "depth_bid_top5": None,
                    "depth_ask_top5": None,
                    "depth_imbalance_top5": None,
                    "book_slope_top5": None,
                    "sweep_cost_buy1": None,
                    "sweep_cost_sell1": None,
                }
            )
        return rows

    # Book snapshots / updates
    if not (isinstance(channel, str) and "book" in channel):
        return rows

    payload = msg_list[1] if len(msg_list) > 1 else {}
    if not isinstance(payload, dict):
        return rows

    # Initialize state storage per pair
    state = book_state.setdefault(pair, {"bid": None, "ask": None, "ts": None, "bids": {}, "asks": {}})

    # Snapshot keys: "as" (asks), "bs" (bids)
    asks_snapshot = payload.get("as")
    bids_snapshot = payload.get("bs")
    if asks_snapshot or bids_snapshot:
        if asks_snapshot:
            _update_book_state(state, asks_snapshot, side="asks")
        if asks_snapshot:
            ask_price, ask_vol, ask_ts = _best_book_levels(asks_snapshot, side="a")
            if ask_price is not None:
                state["ask"] = (ask_price, ask_vol)
                state["ts"] = ask_ts
        if bids_snapshot:
            _update_book_state(state, bids_snapshot, side="bids")
            bid_price, bid_vol, bid_ts = _best_book_levels(bids_snapshot, side="b")
            if bid_price is not None:
                state["bid"] = (bid_price, bid_vol)
                state["ts"] = bid_ts or state.get("ts")

    # Update keys: "a" (asks), "b" (bids)
    asks_update = payload.get("a")
    bids_update = payload.get("b")
    if asks_update:
        _update_book_state(state, asks_update, side="asks")
        ask_price, ask_vol, ask_ts = _best_book_levels(asks_update, side="a")
        if ask_price is not None:
            state["ask"] = (ask_price, ask_vol)
            state["ts"] = ask_ts or state.get("ts")
    if bids_update:
        _update_book_state(state, bids_update, side="bids")
        bid_price, bid_vol, bid_ts = _best_book_levels(bids_update, side="b")
        if bid_price is not None:
            state["bid"] = (bid_price, bid_vol)
            state["ts"] = bid_ts or state.get("ts")

    if state["bid"] is None or state["ask"] is None:
        return rows

    bid_price, bid_vol = state["bid"]
    ask_price, ask_vol = state["ask"]
    ts = state.get("ts") or pd.Timestamp.utcnow()
    ladder_bids = _sorted_ladder(state.get("bids", {}), side="b")
    ladder_asks = _sorted_ladder(state.get("asks", {}), side="a")
    mid = (bid_price + ask_price) / 2
    depth_feats = _aggregate_depth_features(ladder_bids, ladder_asks, mid=mid, top_n=5, sweep_qty=1.0)

    rows.append(
        {
            "Time": ts,
            "Symbol": pair,
            "BidPrice1": bid_price,
            "AskPrice1": ask_price,
            "BidVolume1": bid_vol,
            "AskVolume1": ask_vol,
            "Volume": 0.0,
            **depth_feats,
        }
    )
    return rows


def _load_jsonl_messages(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """
    Parse websocket jsonl (Kraken/legacy Coinbase) into tabular form.
    We synthesize minimal columns so feature code can run:
      - Time: message time (datetime)
      - Symbol: product_id/pair (e.g., XBT/USD)
      - BidPrice1/AskPrice1: set to trade/ticker price or derived from book
      - BidVolume1/AskVolume1: best-level depth when available
      - Volume: trade size (0 for book updates)
    """
    rows: List[dict] = []
    book_state: dict = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if isinstance(msg, dict):
                rows.extend(_parse_coinbase_message(msg, fallback_symbol=path.stem))
            elif isinstance(msg, list):
                rows.extend(_parse_kraken_message(msg, fallback_symbol=path.stem, book_state=book_state))
            if max_rows is not None and max_rows > 0 and i + 1 >= max_rows:
                break

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_all_raw_data(raw_dir: Union[str, Path], max_rows_per_file: int | None = None) -> pd.DataFrame:
    """
    Load and combine all CSV or jsonl files in `raw_dir`, adding Symbol from filename when missing.
    Returns a DataFrame indexed by Symbol then Time.
    """
    raw_path = Path(raw_dir).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    csv_files = sorted(raw_path.glob("*.csv"))
    jsonl_files = sorted(raw_path.glob("*.jsonl"))

    if not csv_files and not jsonl_files:
        raise FileNotFoundError(f"No CSV or JSONL files found in {raw_path}")

    frames: List[pd.DataFrame] = []
    frames.extend(_load_csv(f, max_rows=max_rows_per_file) for f in csv_files)
    frames.extend(_load_jsonl_messages(f, max_rows=max_rows_per_file) for f in jsonl_files)
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise ValueError("No usable data parsed from raw files.")

    combined = pd.concat(frames, ignore_index=True)
    # Drop duplicate Symbol/Time rows early; keep the last occurrence to match training script behavior.
    if {"Symbol", "Time"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["Symbol", "Time"], keep="last")
    return combined.set_index(["Symbol", "Time"]).sort_index()
