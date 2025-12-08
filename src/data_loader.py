import json
from pathlib import Path
from typing import List, Union

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Time"])
    df["Symbol"] = path.stem
    return df


def _load_jsonl_matches(path: Path) -> pd.DataFrame:
    """
    Parse Coinbase websocket jsonl (matches / last_match) into a tabular form.
    We synthesize minimal columns so feature code can run:
      - Time: message time (datetime)
      - Symbol: product_id (e.g., BTC-USD)
      - BidPrice1/AskPrice1: set to trade price
      - BidVolume1/AskVolume1: set to 0 (we don't have depth)
      - Volume: trade size
    """
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            if msg.get("type") not in {"match", "last_match"}:
                continue

            price = float(msg.get("price", "nan"))
            size = float(msg.get("size", "nan"))
            t = pd.to_datetime(msg.get("time"))
            symbol = msg.get("product_id", path.stem)

            rows.append(
                {
                    "Time": t,
                    "Symbol": symbol,
                    "BidPrice1": price,
                    "AskPrice1": price,
                    "BidVolume1": 0.0,
                    "AskVolume1": 0.0,
                    "Volume": size,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def load_all_raw_data(raw_dir: Union[str, Path]) -> pd.DataFrame:
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
    frames.extend(_load_csv(f) for f in csv_files)
    frames.extend(_load_jsonl_matches(f) for f in jsonl_files)
    frames = [f for f in frames if not f.empty]

    if not frames:
        raise ValueError("No usable data parsed from raw files.")

    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index(["Symbol", "Time"]).sort_index()
