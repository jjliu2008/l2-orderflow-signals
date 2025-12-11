#!/usr/bin/env python3
"""
Simple Kraken historical downloader (public endpoints only).
- Supports trades or OHLC pulls with pagination via `since`.
- Optional order book snapshots (Depth) at a fixed interval to mimic websocket rows with depth features.
- Outputs CSV compatible with existing loader.
"""
import argparse
import csv
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://api.kraken.com/0/public/"

def _kraken_request(endpoint: str, params: dict, retries: int = 5, backoff: float = 1.0):
    url = BASE_URL + endpoint
    query = urllib.parse.urlencode(params) if params else ""
    full_url = f"{url}?{query}" if query else url
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full_url) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
            continue
        if data.get("error"):
            # retry on rate limits
            if any("rate" in str(err).lower() for err in data["error"]):
                time.sleep(backoff * (2 ** attempt))
                continue
            raise RuntimeError(f"Kraken API error: {data['error']}")
        return data["result"]
    raise RuntimeError("Exhausted retries")

def _sanitize_pair(pair: str) -> tuple[str, str]:
    pair_clean = pair.replace("/", "").upper()
    label = pair.upper()
    return pair_clean, label

def _best_from_levels(levels, side: str):
    best_price = None
    best_vol = None
    ts = None
    for lvl in levels:
        if not isinstance(lvl, (list, tuple)) or len(lvl) < 3:
            continue
        try:
            price = float(lvl[0])
            vol = float(lvl[1])
            t = float(lvl[2])
        except (ValueError, TypeError):
            continue
        if vol <= 0:
            continue
        if best_price is None:
            best_price, best_vol, ts = price, vol, t
            continue
        if side == "b" and price > best_price:
            best_price, best_vol, ts = price, vol, t
        if side == "a" and price < best_price:
            best_price, best_vol, ts = price, vol, t
    return best_price, best_vol, ts

def _agg_depth(levels, side: str, top_n: int = 5):
    cleaned = []
    for lvl in levels:
        if not isinstance(lvl, (list, tuple)) or len(lvl) < 2:
            continue
        try:
            price = float(lvl[0])
            vol = float(lvl[1])
        except (ValueError, TypeError):
            continue
        if vol <= 0:
            continue
        cleaned.append((price, vol))
    cleaned.sort(key=lambda x: -x[0] if side == "b" else x[0])
    top = cleaned[:top_n]
    depth_sum = sum(v for _, v in top)
    return cleaned, depth_sum

def _depth_metrics(bids, asks, top_n: int = 5, sweep_qty: float = 1.0):
    bid_ladder, depth_bid_top = _agg_depth(bids, "b", top_n=top_n)
    ask_ladder, depth_ask_top = _agg_depth(asks, "a", top_n=top_n)
    depth_denom = depth_bid_top + depth_ask_top
    depth_imb_top = (depth_bid_top - depth_ask_top) / depth_denom if depth_denom else None

    def _vwap(levels):
        vol_sum = sum(v for _, v in levels[:top_n])
        if vol_sum == 0:
            return None
        return sum(p * v for p, v in levels[:top_n]) / vol_sum

    vwap_bid = _vwap(bid_ladder)
    vwap_ask = _vwap(ask_ladder)

    bid_best = bid_ladder[0][0] if bid_ladder else None
    ask_best = ask_ladder[0][0] if ask_ladder else None
    mid = (bid_best + ask_best) / 2 if bid_best is not None and ask_best is not None else None

    book_slope = None
    if vwap_bid is not None and vwap_ask is not None and mid:
        book_slope = (vwap_ask - vwap_bid) / mid

    def _sweep(levels, qty, mid_price):
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
        if filled == 0 or mid_price is None or mid_price == 0:
            return None
        avg_price = cost / filled
        return (avg_price - mid_price) / mid_price

    sweep_cost_buy1 = _sweep(ask_ladder, sweep_qty, mid)
    sweep_cost_sell1 = _sweep(bid_ladder, sweep_qty, mid)

    return {
        "depth_bid_top5": depth_bid_top if depth_bid_top else None,
        "depth_ask_top5": depth_ask_top if depth_ask_top else None,
        "depth_imbalance_top5": depth_imb_top,
        "book_slope_top5": book_slope,
        "sweep_cost_buy1": sweep_cost_buy1,
        "sweep_cost_sell1": sweep_cost_sell1,
        "mid": mid,
        "bid_best": bid_best,
        "ask_best": ask_best,
    }

def download_trades(pair: str, out_path: Path, since: int | None, batch: int, sleep: float, max_rows: int | None):
    pair_api, label = _sanitize_pair(pair)
    total = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Time", "Symbol", "BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1", "Volume", "Side"])
        cursor = since
        while True:
            params = {"pair": pair_api, "count": batch}
            if cursor:
                params["since"] = cursor
            res = _kraken_request("Trades", params)
            trades = None
            next_cursor = res.get("last")
            for k, v in res.items():
                if k == "last":
                    continue
                trades = v
                break
            if not trades:
                break
            for trade in trades:
                if not isinstance(trade, (list, tuple)) or len(trade) < 3:
                    continue
                price = trade[0]
                volume = trade[1]
                t = trade[2]
                side = trade[3] if len(trade) > 3 else ""
                ts = dt.datetime.fromtimestamp(float(t), tz=dt.timezone.utc).isoformat()
                w.writerow([ts, label, price, price, 0.0, 0.0, volume, side])
                total += 1
                if max_rows and total >= max_rows:
                    return total
            if next_cursor and len(trades) == batch:
                cursor = next_cursor
                time.sleep(sleep)
                continue
            break
    return total

def download_ohlc(pair: str, out_path: Path, since: int | None, interval: int, sleep: float, max_rows: int | None):
    pair_api, label = _sanitize_pair(pair)
    total = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Time", "Symbol", "Open", "High", "Low", "Close", "Volume", "Count"])
        cursor = since
        while True:
            params = {"pair": pair_api, "interval": interval}
            if cursor:
                params["since"] = cursor
            res = _kraken_request("OHLC", params)
            candles = None
            next_cursor = res.get("last")
            for k, v in res.items():
                if k == "last":
                    continue
                candles = v
                break
            if not candles:
                break
            for candle in candles:
                ts = dt.datetime.fromtimestamp(float(candle[0]), tz=dt.timezone.utc).isoformat()
                open_, high, low, close, vwap, vol, count = candle[1:8]
                w.writerow([ts, label, open_, high, low, close, vol, count])
                total += 1
                if max_rows and total >= max_rows:
                    return total
            if next_cursor and candles and len(candles) > 0:
                cursor = next_cursor
                time.sleep(sleep)
                continue
            break
    return total

def download_book_snapshots(
    pair: str,
    out_path: Path,
    snapshots: int,
    interval_sec: float,
    depth_levels: int,
    sweep_qty: float,
):
    pair_api, label = _sanitize_pair(pair)
    total = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Time",
                "Symbol",
                "BidPrice1",
                "AskPrice1",
                "BidVolume1",
                "AskVolume1",
                "Volume",
                "Side",
                "depth_bid_top5",
                "depth_ask_top5",
                "depth_imbalance_top5",
                "book_slope_top5",
                "sweep_cost_buy1",
                "sweep_cost_sell1",
            ]
        )
        for i in range(snapshots):
            params = {"pair": pair_api, "count": depth_levels}
            res = _kraken_request("Depth", params)
            book = None
            for k, v in res.items():
                if k == "last":
                    continue
                book = v
                break
            if not book:
                time.sleep(interval_sec)
                continue
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid, bid_vol, ts_bid = _best_from_levels(bids, "b")
            best_ask, ask_vol, ts_ask = _best_from_levels(asks, "a")
            ts = ts_bid or ts_ask or time.time()
            ts_iso = dt.datetime.fromtimestamp(float(ts), tz=dt.timezone.utc).isoformat()
            feats = _depth_metrics(bids, asks, top_n=min(5, depth_levels), sweep_qty=sweep_qty)
            w.writerow(
                [
                    ts_iso,
                    label,
                    best_bid,
                    best_ask,
                    bid_vol,
                    ask_vol,
                    0.0,
                    "",
                    feats["depth_bid_top5"],
                    feats["depth_ask_top5"],
                    feats["depth_imbalance_top5"],
                    feats["book_slope_top5"],
                    feats["sweep_cost_buy1"],
                    feats["sweep_cost_sell1"],
                ]
            )
            total += 1
            time.sleep(interval_sec)
    return total

def main():
    parser = argparse.ArgumentParser(description="Download Kraken historical trades or OHLC")
    parser.add_argument("--pair", default="XBT/USD", help="Trading pair, e.g., XBT/USD")
    parser.add_argument("--data-type", choices=["trades", "ohlc", "book"], default="trades")
    parser.add_argument("--interval", type=int, default=1, help="OHLC interval minutes (1,5,15,...)")
    parser.add_argument("--days", type=int, default=3, help="Lookback days (converted to since timestamp)")
    parser.add_argument("--since", type=int, default=None, help="Override since timestamp (unix seconds)")
    parser.add_argument("--batch", type=int, default=5000, help="Batch size per request")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between paged requests")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows to fetch")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument("--book-snapshots", type=int, default=200, help="Number of book snapshots to pull (data-type=book)")
    parser.add_argument("--book-interval", type=float, default=1.0, help="Seconds between book snapshots")
    parser.add_argument("--book-depth", type=int, default=25, help="Depth levels to request from API")
    parser.add_argument("--sweep-qty", type=float, default=1.0, help="Quantity used for sweep cost calc")
    args = parser.parse_args()

    if args.since is None:
        since_ts = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)).timestamp())
    else:
        since_ts = args.since

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        start_tag = dt.datetime.fromtimestamp(since_ts, tz=dt.timezone.utc).strftime("%Y%m%d")
        end_tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
        out_dir = Path("data") / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"kraken_{args.pair.replace('/', '')}_{args.data_type}_{start_tag}_{end_tag}.csv"

    if args.data_type == "trades":
        total = download_trades(args.pair, out_path, since_ts, args.batch, args.sleep, args.max_rows)
    elif args.data_type == "ohlc":
        total = download_ohlc(args.pair, out_path, since_ts, args.interval, args.sleep, args.max_rows)
    else:
        total = download_book_snapshots(
            args.pair,
            out_path,
            snapshots=args.book_snapshots,
            interval_sec=args.book_interval,
            depth_levels=args.book_depth,
            sweep_qty=args.sweep_qty,
        )

    print(f"Saved {total} rows to {out_path}")


if __name__ == "__main__":
    main()
