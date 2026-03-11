from __future__ import annotations

from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATA_DIR = APP_ROOT / "data"
EXPORT_DIR = APP_ROOT / "artifacts" / "exports"

TRADES_PATH = DATA_DIR / "trades.parquet"
FEATURES_PATH = DATA_DIR / "features.parquet"
DAILY_STATS_PATH = DATA_DIR / "daily_stats.parquet"
WEEKLY_STATS_PATH = DATA_DIR / "weekly_stats.parquet"
SUMMARY_VARIANTS_PATH = DATA_DIR / "summary_variants.parquet"
REPLAY_INDEX_PATH = DATA_DIR / "replay_index.parquet"

DEFAULT_COMMISSION_TICKS = 0.36
DEFAULT_TICK_VALUE_BY_CONTRACT = {
    "ES": 12.50,
    "MES": 1.25,
}

DEFAULT_MONTE_CARLO_PATHS = 2000
DEFAULT_MONTE_CARLO_HORIZON = None  # use observed trade count
DEFAULT_MC_STARTING_CAPITAL = 2000.0
DEFAULT_MC_DD_THRESHOLD = -800.0

REPLAY_PAD_PRE_MINUTES = 10
REPLAY_PAD_POST_MINUTES = 10

DEFAULT_DATE_GLOB = "date=2026-0[12]-*"
DEFAULT_STEP_MS = 1000
DEFAULT_TICK_SIZE = 0.25
DEFAULT_HOLD_MINUTES = 30
