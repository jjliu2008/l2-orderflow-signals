"""
Orchestrate all simulations (Monte Carlo + backtest) against the latest data.
- Prefers data/live if present, otherwise falls back to data/raw.
- Suppresses interactive prompts so it can run unattended.
Override the data directory via LIVE_SIM_DATA_DIR.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Local imports after sys.path update
from scripts import run_backtest, run_monte_carlo  # type: ignore


def _has_data_files(path: Path) -> bool:
    """Return True if the directory contains CSV or JSONL files."""
    return any(path.glob("*.csv")) or any(path.glob("*.jsonl"))


def _pick_data_dir() -> Path:
    """
    Choose a data directory in priority order:
    1) LIVE_SIM_DATA_DIR env var
    2) data/live (if it exists and has data)
    3) data/raw
    """
    env_dir = os.environ.get("LIVE_SIM_DATA_DIR")
    if env_dir:
        env_path = Path(env_dir).expanduser().resolve()
        if not env_path.exists():
            raise FileNotFoundError(f"LIVE_SIM_DATA_DIR does not exist: {env_path}")
        if not _has_data_files(env_path):
            raise FileNotFoundError(f"LIVE_SIM_DATA_DIR has no CSV/JSONL files: {env_path}")
        return env_path

    live_dir = PROJECT_ROOT / "data" / "live"
    if live_dir.exists() and _has_data_files(live_dir):
        return live_dir

    raw_dir = PROJECT_ROOT / "data" / "raw"
    if raw_dir.exists() and _has_data_files(raw_dir):
        return raw_dir

    raise FileNotFoundError(
        "No data directory found with CSV/JSONL files. "
        "Populate data/live or data/raw, or set LIVE_SIM_DATA_DIR."
    )


@contextmanager
def _patched_input(default_response: str = ""):
    """Temporarily replace input() to avoid interactive prompts."""
    with patch("builtins.input", lambda prompt="": default_response):
        yield


def run_all(data_dir: Optional[Path] = None):
    """
    Run Monte Carlo and backtest sequentially using the chosen data directory.
    """
    target_dir = data_dir or _pick_data_dir()
    print(f"Running simulations with data from: {target_dir}")

    # Ensure downstream scripts read from the chosen directory and save artifacts.
    os.environ.setdefault("MONTE_CARLO_DATA_DIR", str(target_dir))
    os.environ.setdefault("BT_DATA_DIR", str(target_dir))
    os.environ.setdefault("BACKTEST_SAVE", "1")

    with _patched_input(""):
        print("\n=== Monte Carlo simulation ===")
        run_monte_carlo.main()

    with _patched_input(""):
        print("\n=== Backtest simulation ===")
        run_backtest.main()


if __name__ == "__main__":
    run_all()
