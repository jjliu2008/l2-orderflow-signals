import asyncio
import json
import websockets
from websockets import exceptions
from datetime import datetime, UTC
from pathlib import Path

COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"

async def coinbase_ws_record():
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = Path("data/raw") / f"coinbase_ws_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing to:", out_path)

    while True:
        try:
            async with websockets.connect(
                COINBASE_WS_URL, ping_interval=20, ping_timeout=20
            ) as ws:
                subscribe_msg = {
                    "type": "subscribe",
                    "product_ids": ["BTC-USD"],
                    "channels": ["level2", "matches"],
                }
                await ws.send(json.dumps(subscribe_msg))
                print("Subscribed to Coinbase WebSocket")

                # append so we keep data across reconnects
                with out_path.open("a", encoding="utf-8") as f:
                    async for msg in ws:
                        f.write(msg + "\n")

        except exceptions.ConnectionClosedError as e:
            print(f"Connection closed: {e}; reconnecting in 2s")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(coinbase_ws_record())
