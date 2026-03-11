from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research_dashboard import config
from research_dashboard.services import metrics


SOURCE_TRADE_CSV_CANDIDATES = [
    "artifacts/signal_research/es_cluster_signedvol_gate_validation_20260310_trades.csv",
    "artifacts/signal_research/es_survivor_powerhour_rerun_20260310_trades.csv",
    "artifacts/signal_research/es_survivor_validation_20260310_trades.csv",
]


def _required_paths() -> list[Path]:
    return [
        config.TRADES_PATH,
        config.FEATURES_PATH,
        config.DAILY_STATS_PATH,
        config.WEEKLY_STATS_PATH,
        config.SUMMARY_VARIANTS_PATH,
        config.REPLAY_INDEX_PATH,
    ]


def ensure_data_files(force_rebuild: bool = False) -> bool:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = all(p.exists() for p in _required_paths())
    if existing and not force_rebuild:
        return True
    return build_data_from_artifacts()


def _pick_source_csv(project_root: Path) -> Path | None:
    for rel in SOURCE_TRADE_CSV_CANDIDATES:
        p = project_root / rel
        if p.exists():
            return p
    return None


def _load_market_frame(project_root: Path) -> pd.DataFrame:
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    import run_es_exhaustion_suite as suite

    df = suite._load_resampled_data(
        root=project_root / "data/processed_esh6/instrument=ES",
        date_glob=config.DEFAULT_DATE_GLOB,
        step_ms=config.DEFAULT_STEP_MS,
    )
    df = suite._add_features(df, tick_size=config.DEFAULT_TICK_SIZE, step_ms=config.DEFAULT_STEP_MS)
    g = df.groupby("date_et", group_keys=False)
    df["signedvol_sum30s"] = g["signed_volume"].transform(lambda s: s.rolling(30, min_periods=10).sum())
    df["ret_1s"] = g["mid"].pct_change()
    df["ret_300s"] = g["mid"].pct_change(300)
    df["rv_300s"] = g["ret_1s"].transform(lambda s: s.rolling(300, min_periods=60).std())
    df["trend_z_300s"] = (df["ret_300s"].abs() / df["rv_300s"]).replace([np.inf, -np.inf], np.nan)
    return df


def _build_trades_table(raw: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "variant" not in df.columns:
        if {"setup", "mode", "filter"}.issubset(df.columns):
            df["variant"] = (
                df["setup"].astype(str) + "|" + df["mode"].astype(str) + "|" + df["filter"].astype(str)
            )
        else:
            df["variant"] = "baseline_no_sv_gate"

    if "setup" not in df.columns:
        df["setup"] = "signal_cluster_first"
    if "filter" not in df.columns:
        df["filter"] = "spread_1tick"
    if "mode" not in df.columns:
        df["mode"] = "delayed_1bar"

    idx_to_time = market["Time"].to_dict()
    if "entry_time_et" in df.columns:
        entry_et = pd.to_datetime(df["entry_time_et"], errors="coerce")
        entry_utc = entry_et.dt.tz_convert("UTC") if entry_et.dt.tz is not None else entry_et.dt.tz_localize("UTC")
    else:
        entry_utc = pd.to_datetime(df["entry_idx_used"].map(idx_to_time), utc=True, errors="coerce")

    exit_utc = pd.to_datetime(df["exit_idx_used"].map(idx_to_time), utc=True, errors="coerce")
    missing_exit = exit_utc.isna()
    exit_utc.loc[missing_exit] = entry_utc.loc[missing_exit] + pd.Timedelta(minutes=config.DEFAULT_HOLD_MINUTES)

    entry_et = entry_utc.dt.tz_convert("America/New_York")
    exit_et = exit_utc.dt.tz_convert("America/New_York")

    out = pd.DataFrame()
    out["trade_id"] = [f"T{i:06d}" for i in range(1, len(df) + 1)]
    out["strategy_name"] = df["setup"].astype(str)
    out["variant_name"] = df["variant"].astype(str)
    out["date"] = entry_et.dt.date.astype(str)
    out["entry_ts"] = entry_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["exit_ts"] = exit_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["session"] = df.get("entry_session_bucket", "unknown").astype(str)
    out["direction"] = np.where(df.get("side", -1).astype(float) > 0, "long", "short")
    out["entry_price"] = np.nan
    out["exit_price"] = np.nan
    out["ticks_pnl_gross"] = pd.to_numeric(df.get("pnl_gross_ticks"), errors="coerce")
    out["ticks_pnl_net"] = out["ticks_pnl_gross"] - config.DEFAULT_COMMISSION_TICKS
    out["hold_seconds"] = (exit_utc - entry_utc).dt.total_seconds().clip(lower=0)
    out["is_event_day"] = df.get("is_event_day", False).astype(bool)
    out["month"] = entry_et.dt.strftime("%Y-%m")
    out["weekday"] = entry_et.dt.day_name()
    fallback_idx = pd.Series(np.arange(len(df)), index=df.index)
    cluster_idx = pd.to_numeric(df.get("entry_idx_orig", fallback_idx), errors="coerce").fillna(fallback_idx).astype(int)
    out["cluster_id"] = entry_et.dt.strftime("%Y-%m-%d") + "_" + cluster_idx.astype(str)
    out["passed_spread_gate"] = df.get("filter", "").astype(str).str.contains("spread_1tick", case=False)
    out["passed_sv_gate"] = df["variant"].astype(str).str.contains("sv_le_", case=False)
    out["passed_powerhour_filter"] = out["session"] != "power_hour"

    # Default OOS: latest month in dataset.
    latest_month = out["month"].max()
    out["oos_flag"] = out["month"] == latest_month
    out["contract"] = "ES"

    # Keep source indices for feature joins.
    out["entry_idx_used"] = pd.to_numeric(df.get("entry_idx_used"), errors="coerce")
    out["entry_idx_orig"] = pd.to_numeric(df.get("entry_idx_orig"), errors="coerce")
    out["signal_signedvol_sum30s"] = pd.to_numeric(df.get("signal_signedvol_sum30s"), errors="coerce")
    out["replay_date"] = entry_et.dt.strftime("%Y-%m-%d")
    return out


def _build_features_table(trades: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    market_idx = market.copy()
    market_idx["row_idx"] = market_idx.index

    day_ctx = (
        market_idx[market_idx["is_rth"]]
        .groupby("date_et", as_index=False)
        .agg(day_high_mid=("mid", "max"), day_low_mid=("mid", "min"), day_open_mid=("mid", "first"), day_close_mid=("mid", "last"))
    )
    day_ctx["day_range_ticks"] = (day_ctx["day_high_mid"] - day_ctx["day_low_mid"]) / config.DEFAULT_TICK_SIZE
    day_ctx["day_ret_ticks"] = (day_ctx["day_close_mid"] - day_ctx["day_open_mid"]) / config.DEFAULT_TICK_SIZE

    select_cols = [
        "row_idx",
        "vwap_dev_ticks",
        "signedvol_sum30s",
        "order_book_imbalance",
        "rv_300s",
        "trend_z_300s",
        "spread_ticks",
    ]
    f = market_idx[select_cols].copy()
    join = trades[["trade_id", "entry_idx_used", "entry_idx_orig", "signal_signedvol_sum30s", "replay_date"]].copy()
    join["entry_idx_used"] = pd.to_numeric(join["entry_idx_used"], errors="coerce")
    join["entry_idx_orig"] = pd.to_numeric(join["entry_idx_orig"], errors="coerce")
    merged = join.merge(f, left_on="entry_idx_used", right_on="row_idx", how="left")

    if "signedvol_sum30s" in merged.columns:
        merged["signedvol_sum30s"] = merged["signedvol_sum30s"].fillna(merged["signal_signedvol_sum30s"])
    else:
        merged["signedvol_sum30s"] = merged["signal_signedvol_sum30s"]

    merged = merged.merge(day_ctx[["date_et", "day_range_ticks", "day_ret_ticks"]], left_on="replay_date", right_on="date_et", how="left")
    merged["bid_size_sum"] = np.nan
    merged["ask_size_sum"] = np.nan

    out = pd.DataFrame()
    out["trade_id"] = merged["trade_id"]
    out["vwap_dev_ticks"] = pd.to_numeric(merged["vwap_dev_ticks"], errors="coerce")
    out["signedvol_sum30s"] = pd.to_numeric(merged["signedvol_sum30s"], errors="coerce")
    out["order_book_imbalance"] = pd.to_numeric(merged["order_book_imbalance"], errors="coerce")
    out["rv_300s"] = pd.to_numeric(merged["rv_300s"], errors="coerce")
    out["trend_z_300s"] = pd.to_numeric(merged["trend_z_300s"], errors="coerce")
    out["day_range_ticks"] = pd.to_numeric(merged["day_range_ticks"], errors="coerce")
    out["day_ret_ticks"] = pd.to_numeric(merged["day_ret_ticks"], errors="coerce")
    out["spread_ticks"] = pd.to_numeric(merged["spread_ticks"], errors="coerce")
    out["bid_size_sum"] = pd.to_numeric(merged["bid_size_sum"], errors="coerce")
    out["ask_size_sum"] = pd.to_numeric(merged["ask_size_sum"], errors="coerce")
    return out


def _build_daily_stats(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, s, v), g in trades.groupby(["date", "strategy_name", "variant_name"]):
        g = g.sort_values("entry_ts")
        pnl = pd.to_numeric(g["ticks_pnl_net"], errors="coerce").fillna(0.0)
        eq = pnl.cumsum()
        dd = eq - eq.cummax()
        rows.append(
            {
                "date": d,
                "strategy_name": s,
                "variant_name": v,
                "n_trades": int(len(g)),
                "net_ticks": float(pnl.sum()),
                "best_trade_ticks": float(pnl.max()) if len(pnl) else None,
                "worst_trade_ticks": float(pnl.min()) if len(pnl) else None,
                "max_intraday_dd_ticks": float(dd.min()) if len(dd) else None,
                "is_event_day": bool(g["is_event_day"].any()),
            }
        )
    return pd.DataFrame(rows)


def _build_weekly_stats(trades: pd.DataFrame) -> pd.DataFrame:
    tmp = trades.copy()
    tmp["week"] = pd.to_datetime(tmp["entry_ts"], utc=True).dt.to_period("W-MON").astype(str)
    rows = []
    for (w, s, v), g in tmp.groupby(["week", "strategy_name", "variant_name"]):
        pnl = pd.to_numeric(g["ticks_pnl_net"], errors="coerce")
        rows.append(
            {
                "week": w,
                "strategy_name": s,
                "variant_name": v,
                "n_trades": int(len(g)),
                "net_ticks": float(pnl.sum()),
                "ev_per_trade": metrics.ev_per_trade(pnl),
                "win_rate": metrics.win_rate(pnl),
                "profit_factor": metrics.profit_factor(pnl),
                "max_dd_ticks": metrics.max_drawdown_ticks(pnl),
            }
        )
    return pd.DataFrame(rows)


def _build_summary_variants(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, v), g in trades.groupby(["strategy_name", "variant_name"]):
        pnl = pd.to_numeric(g["ticks_pnl_net"], errors="coerce")
        contrib = metrics.contribution_shares(pnl)
        jan = g[g["month"] == "2026-01"]["ticks_pnl_net"]
        feb = g[g["month"] == "2026-02"]["ticks_pnl_net"]
        oos = g[g["oos_flag"]]["ticks_pnl_net"]
        rows.append(
            {
                "strategy_name": s,
                "variant_name": v,
                "n_trades": int(len(g)),
                "ev_net": metrics.ev_per_trade(pnl),
                "wr": metrics.win_rate(pnl),
                "pf": metrics.profit_factor(pnl),
                "max_dd_ticks": metrics.max_drawdown_ticks(pnl),
                "top1_share": contrib["top1_share"],
                "top3_share": contrib["top3_share"],
                "top5_share": contrib["top5_share"],
                "oos_ev": metrics.ev_per_trade(pd.to_numeric(oos, errors="coerce")),
                "jan_ev": metrics.ev_per_trade(pd.to_numeric(jan, errors="coerce")),
                "feb_ev": metrics.ev_per_trade(pd.to_numeric(feb, errors="coerce")),
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def _build_replay_index(trades: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    entry = pd.to_datetime(trades["entry_ts"], utc=True)
    exit_ = pd.to_datetime(trades["exit_ts"], utc=True)
    out = pd.DataFrame()
    out["trade_id"] = trades["trade_id"]
    out["replay_source"] = trades["replay_date"].map(
        lambda d: str(project_root / "data/processed_esh6/instrument=ES" / f"date={d}" / "features_labels.parquet")
    )
    out["window_start_ts"] = (entry - pd.Timedelta(minutes=config.REPLAY_PAD_PRE_MINUTES)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["window_end_ts"] = (exit_ + pd.Timedelta(minutes=config.REPLAY_PAD_POST_MINUTES)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def build_data_from_artifacts() -> bool:
    project_root = config.PROJECT_ROOT
    source_csv = _pick_source_csv(project_root)
    if source_csv is None:
        return False

    raw = pd.read_csv(source_csv)
    market = _load_market_frame(project_root)
    trades = _build_trades_table(raw, market)
    features = _build_features_table(trades, market)
    daily = _build_daily_stats(trades)
    weekly = _build_weekly_stats(trades)
    summary = _build_summary_variants(trades)
    replay_index = _build_replay_index(trades, project_root)

    # Strip internal helper columns from public schema.
    trades_out = trades[
        [
            "trade_id",
            "strategy_name",
            "variant_name",
            "date",
            "entry_ts",
            "exit_ts",
            "session",
            "direction",
            "entry_price",
            "exit_price",
            "ticks_pnl_gross",
            "ticks_pnl_net",
            "hold_seconds",
            "is_event_day",
            "month",
            "weekday",
            "cluster_id",
            "passed_spread_gate",
            "passed_sv_gate",
            "passed_powerhour_filter",
            "oos_flag",
            "contract",
        ]
    ].copy()

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    trades_out.to_parquet(config.TRADES_PATH, index=False)
    features.to_parquet(config.FEATURES_PATH, index=False)
    daily.to_parquet(config.DAILY_STATS_PATH, index=False)
    weekly.to_parquet(config.WEEKLY_STATS_PATH, index=False)
    summary.to_parquet(config.SUMMARY_VARIANTS_PATH, index=False)
    replay_index.to_parquet(config.REPLAY_INDEX_PATH, index=False)
    return True
