import asyncio
import base64
import json
import os
import secrets
import time
from datetime import datetime, UTC
from pathlib import Path

import jwt
import websockets
from cryptography.hazmat.primitives import serialization
from websockets import exceptions

# Default to Advanced Trade websocket host; override via COINBASE_WS_URL if needed.
COINBASE_WS_URL = os.environ.get(
    "COINBASE_WS_URL", "wss://advanced-trade-ws.coinbase.com"
)


def _get_env_or_prompt(name: str, prompt: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    return input(prompt)


def _normalize_pem(pem: str) -> str:
    # Allow pasting single-line PEM with literal "\n"
    if "\\n" in pem and "\n" not in pem:
        return pem.replace("\\n", "\n")
    return pem


def build_cdp_jwt(key_name: str, key_material: str) -> str:
    """
    Build an ES256 JWT for Coinbase "cdp" auth.
    Requires:
      COINBASE_KEY_NAME: organizations/{org_id}/apiKeys/{key_id}
      COINBASE_KEY_PEM: EC private key PEM (P-256)
    """

    key_bytes = key_material.strip().encode("utf-8")

    private_key = None
    pem_error = None

    # First try PEM
    try:
        private_key = serialization.load_pem_private_key(key_bytes, password=None)
    except ValueError as exc:
        pem_error = exc

    # If PEM failed, try treating the input as raw base64 DER
    if private_key is None:
        try:
            der = base64.b64decode(key_material.strip())
            private_key = serialization.load_der_private_key(der, password=None)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load EC private key. Provide either full PEM with BEGIN/END"
                " lines or a base64-encoded DER EC private key (P-256)."
            ) from (exc if pem_error is None else pem_error)

    now = int(time.time())
    payload = {"sub": key_name, "iss": "cdp", "nbf": now, "exp": now + 120}
    headers = {"kid": key_name, "nonce": secrets.token_hex()}

    return jwt.encode(payload, private_key, algorithm="ES256", headers=headers)


def get_auth_config():
    """
    Decide whether to use authenticated level2 (JWT) or public matches-only.
    Returns None if unauthenticated; otherwise returns (key_name, key_material).
    """
    use_auth_env = os.environ.get("COINBASE_USE_JWT")
    if use_auth_env is not None:
        use_auth = use_auth_env.strip().lower() in {"y", "yes", "true", "1"}
    else:
        reply = input(
            "Do you have an Advanced Trade API key for level2 (requires EC private key)? [y/N]: "
        ).strip().lower()
        use_auth = reply in {"y", "yes"}

    if not use_auth:
        print("Proceeding without auth: subscribing to matches only (no level2).")
        return None

    key_name = _get_env_or_prompt(
        "COINBASE_KEY_NAME",
        "Enter Coinbase key name (organizations/{org_id}/apiKeys/{key_id}): ",
    )
    key_material = _normalize_pem(
        _get_env_or_prompt(
            "COINBASE_KEY_PEM",
            "Paste Coinbase EC private key PEM: ",
        )
    )
    return key_name, key_material


async def coinbase_ws_record():
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = Path("data/raw") / f"coinbase_ws_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Writing to:", out_path)

    auth_config = get_auth_config()

    while True:
        try:
            async with websockets.connect(
                COINBASE_WS_URL, ping_interval=20, ping_timeout=20
            ) as ws:
                if auth_config is None:
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": ["BTC-USD"],
                        "channels": ["matches"],
                    }
                else:
                    key_name, key_material = auth_config
                    jwt_token = build_cdp_jwt(key_name, key_material)
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": ["BTC-USD"],
                        "channels": ["level2", "matches"],
                        "jwt": jwt_token,
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
