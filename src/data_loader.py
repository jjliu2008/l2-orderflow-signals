from pathlib import Path
from typing import Union

import pandas as pd


def _load_single_raw_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Time"])
    df["Symbol"] = path.stem
    return df


def load_all_raw_data(raw_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load and combine all CSVs in `raw_dir`, adding Symbol from filename.
    Returns a DataFrame indexed by Symbol then Time.
    """
    raw_path = Path(raw_dir).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_path}")

    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    frames = [_load_single_raw_file(csv_file) for csv_file in csv_files]
    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index(["Symbol", "Time"]).sort_index()
