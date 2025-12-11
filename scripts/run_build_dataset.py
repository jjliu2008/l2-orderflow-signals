import asyncio
import json
import os
from datetime import datetime, UTC
from pathlib import Path

import websockets
from websockets import exceptions

# Default to public Kraken feed; override via KRAKEN_WS_URL for custom pipeline endpoints.
KRAKEN_WS_URL = os.environ.get("KRAKEN_WS_URL", "wss://ws.kraken.com")


def _clear_old_output(out_dir: Path):
    """
    Remove existing kraken_ws_*.jsonl files before starting a new capture.
    To avoid touching historical datasets, we only clear when targeting data/live.
    """
    # Only clear if we're explicitly in data/live to protect other datasets.
    if out_dir.resolve().name != "live":
        return

    archive_dir = out_dir.parent / "processed"
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for f in out_dir.glob("kraken_ws_*.jsonl"):
        try:
            dest = archive_dir / f.name
            f.replace(dest)
            moved += 1
        except OSError:
            continue
    if moved:
        print(f"Moved {moved} old files from {out_dir} to {archive_dir}")


def _choose_output_path() -> Path:
    default_dir = os.environ.get("WS_OUTPUT_DIR", "data/live")
    out_dir = default_dir
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"kraken_ws_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_old_output(out_path.parent)
    print("Writing to:", out_path)
    return out_path


def _parse_pairs() -> list[str]:
    """
    Default to Bitcoin on Kraken (XBT/USD) unless explicitly overridden.
    """
    pair_env = os.environ.get("KRAKEN_PAIRS") or "XBT/USD"
    return [p.strip() for p in pair_env.split(",") if p.strip()]


def _choose_channels() -> list[dict]:
    """
    Build Kraken-compatible subscription payloads from env.
    Example env: KRAKEN_CHANNELS="trade,ticker,book".
    """
    channel_env = os.environ.get("KRAKEN_CHANNELS", os.environ.get("WS_CHANNELS", "trade,ticker,book"))
    base = [c.strip() for c in channel_env.split(",") if c.strip()]
    depth = int(os.environ.get("KRAKEN_BOOK_DEPTH", "10"))

    subs: list[dict] = []
    for name in base or ["trade"]:
        sub = {"name": name}
        if name.startswith("book") or name == "book":
            sub["name"] = "book"
            sub["depth"] = depth
        subs.append(sub)
    return subs


def _extra_headers() -> list[tuple[str, str]] | None:
    """
    Optionally attach an API key/token header for custom pipelines.
    Kraken public WS does not require auth, but we allow Authorization if provided.
    When env var is absent, skip prompting so unattended capture keeps running.
    """
    api_key = os.environ.get("KRAKEN_API_KEY")
    if not api_key:
        return None
    return [("Authorization", f"Bearer {api_key.strip()}")]


async def kraken_ws_record():
    out_path = _choose_output_path()

    while True:
        try:
            async with websockets.connect(
                KRAKEN_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                additional_headers=_extra_headers(),
            ) as ws:
                pairs = _parse_pairs()
                subs = _choose_channels()
                for sub in subs:
                    subscribe_msg = {"event": "subscribe", "pair": pairs, "subscription": sub}
                    await ws.send(json.dumps(subscribe_msg))

                print(
                    f"Subscribed to Kraken WebSocket ({KRAKEN_WS_URL}) -> pairs={pairs}, channels={[s['name'] for s in subs]}"
                )

                # append so we keep data across reconnects; line-buffered to reduce batching
                with out_path.open("a", encoding="utf-8", buffering=1) as f:
                    async for msg in ws:
                        f.write(msg + "\n")
                        f.flush()

        except exceptions.ConnectionClosedError as e:
            print(f"Connection closed: {e}; reconnecting in 2s")
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(kraken_ws_record())
