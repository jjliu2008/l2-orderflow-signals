import asyncio
import json
import websockets
from datetime import datetime

COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"

async def coinbase_ws_record():
    async with websockets.connect(COINBASE_WS_URL) as ws:
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["level2", "matches"]
        }
        await ws.send(json.dumps(subscribe_msg))
        print("Subscribed to Coinbase WebSocket")

        # open a file to record raw messages
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = f"data/raw/coinbase_ws_{ts}.jsonl"
        print("Writing to:", out_path)

        with open(out_path, "w", encoding="utf-8") as f:
            async for msg in ws:
                f.write(msg + "\n")   # each message one line
                # optional: flush periodically if you want
                # f.flush()

if __name__ == "__main__":
    asyncio.run(coinbase_ws_record())