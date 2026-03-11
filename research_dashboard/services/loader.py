from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without duckdb installed
    duckdb = None

from research_dashboard import config
from research_dashboard.services import bootstrap, filters as flt, metrics


def ensure_materialized_data(force_rebuild: bool = False) -> None:
    ok = bootstrap.ensure_data_files(force_rebuild=force_rebuild)
    if not ok:
        raise RuntimeError(
            "Could not materialize dashboard data. Expected at least one source trade CSV under artifacts/signal_research."
        )


@lru_cache(maxsize=1)
def _conn():
    if duckdb is None:
        return None
    return duckdb.connect(database=":memory:", read_only=False)


@lru_cache(maxsize=16)
def _read_parquet(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    if duckdb is not None:
        c = _conn()
        return c.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetch_df()
    return pd.read_parquet(path)


def load_trades() -> pd.DataFrame:
    ensure_materialized_data()
    return _read_parquet(str(config.TRADES_PATH)).copy()


def load_features() -> pd.DataFrame:
    ensure_materialized_data()
    return _read_parquet(str(config.FEATURES_PATH)).copy()


def load_daily_stats() -> pd.DataFrame:
    ensure_materialized_data()
    return _read_parquet(str(config.DAILY_STATS_PATH)).copy()


def load_weekly_stats() -> pd.DataFrame:
    ensure_materialized_data()
    return _read_parquet(str(config.WEEKLY_STATS_PATH)).copy()


def load_summary_variants() -> pd.DataFrame:
    ensure_materialized_data()
    return _read_parquet(str(config.SUMMARY_VARIANTS_PATH)).copy()


def load_replay_index() -> pd.DataFrame:
    ensure_materialized_data()
    return _read_parquet(str(config.REPLAY_INDEX_PATH)).copy()


def get_filter_options(trades_df: pd.DataFrame | None = None) -> dict:
    df = load_trades() if trades_df is None else trades_df
    if df.empty:
        return {
            "strategy_name": [],
            "variant_name": [],
            "session": [],
            "month": [],
            "weekday": [],
            "date_min": [None],
            "date_max": [None],
        }
    dts = pd.to_datetime(df["date"], errors="coerce")
    return {
        "strategy_name": sorted(df["strategy_name"].dropna().unique().tolist()),
        "variant_name": sorted(df["variant_name"].dropna().unique().tolist()),
        "session": sorted(df["session"].dropna().unique().tolist()),
        "month": sorted(df["month"].dropna().unique().tolist()),
        "weekday": sorted(df["weekday"].dropna().unique().tolist()),
        "date_min": [dts.min().date() if dts.notna().any() else None],
        "date_max": [dts.max().date() if dts.notna().any() else None],
    }


@lru_cache(maxsize=256)
def _filtered_trade_view_cached(filter_key: tuple) -> pd.DataFrame:
    all_trades = load_trades()
    # Rehydrate from cache key into DashboardFilters.
    f = flt.DashboardFilters(
        strategy_name=filter_key[0],
        variant_names=list(filter_key[1]),
        date_start=pd.to_datetime(filter_key[2]).date() if filter_key[2] else None,
        date_end=pd.to_datetime(filter_key[3]).date() if filter_key[3] else None,
        sessions=list(filter_key[4]),
        event_mode=filter_key[5],
        direction_mode=filter_key[6],
        months=list(filter_key[7]),
        weekdays=list(filter_key[8]),
        cost_override_enabled=bool(filter_key[9]),
        cost_override_ticks=float(filter_key[10]),
        contract=filter_key[11],
        oos_only=bool(filter_key[12]),
        exclude_power_hour=bool(filter_key[13]),
    )
    out = flt.apply_trade_filters(all_trades, f)
    out = metrics.add_cost_column(
        out,
        base_cost_ticks=config.DEFAULT_COMMISSION_TICKS,
        override_enabled=f.cost_override_enabled,
        override_ticks=f.cost_override_ticks,
        gross_col="ticks_pnl_gross",
        out_col="ticks_pnl_net",
    )
    return out


def get_filtered_trade_view(filters: flt.DashboardFilters) -> pd.DataFrame:
    return _filtered_trade_view_cached(filters.to_cache_key()).copy()


def get_filtered_feature_view(filters: flt.DashboardFilters) -> pd.DataFrame:
    trades = get_filtered_trade_view(filters)
    features = load_features()
    return flt.apply_feature_filters(features, trades["trade_id"])


def get_filtered_trade_feature_view(filters: flt.DashboardFilters) -> pd.DataFrame:
    t = get_filtered_trade_view(filters)
    f = get_filtered_feature_view(filters)
    if t.empty:
        return t
    if f.empty:
        return t
    return t.merge(f, on="trade_id", how="left")


def get_filtered_daily_view(filters: flt.DashboardFilters) -> pd.DataFrame:
    daily = load_daily_stats()
    trades = get_filtered_trade_view(filters)
    if daily.empty or trades.empty:
        return pd.DataFrame()
    keys = trades[["date", "strategy_name", "variant_name"]].drop_duplicates()
    out = daily.merge(keys, on=["date", "strategy_name", "variant_name"], how="inner")
    return out.sort_values("date")


def get_filtered_weekly_view(filters: flt.DashboardFilters) -> pd.DataFrame:
    weekly = load_weekly_stats()
    trades = get_filtered_trade_view(filters)
    if weekly.empty or trades.empty:
        return pd.DataFrame()
    keys = trades[["strategy_name", "variant_name"]].drop_duplicates()
    out = weekly.merge(keys, on=["strategy_name", "variant_name"], how="inner")
    return out.sort_values("week")
