"""
ES exhaustion suite: structured tests before buying new data.

Implements and tests:
1) Session-window behavior
2) One-trade-per-day formulations (ORB, first-pullback, prior-day reaction)
3) MBP-10 skip filters (use book/flow as veto, not primary alpha)
4) Execution timing variants (immediate vs delayed 1 bar)
5) Event segmentation
6) Independent-trade construction (cluster-first for dense signal streams)
7) OOS-style month split (Jan vs Feb)

Usage:
  python scripts/run_es_exhaustion_suite.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data/processed_esh6/instrument=ES"
OUT_DIR = PROJECT_ROOT / "artifacts/signal_research"
ET_TZ = "America/New_York"


# Major event dates in the currently used research range (Jan/Feb 2026).
# Keep this explicit and editable; do not infer from internet at runtime.
EVENT_DATES = {
    "2026-01-14": "CPI",
    "2026-01-28": "FOMC",
    "2026-02-11": "CPI",
}


@dataclass
class TradeEntry:
    setup: str
    date_et: str
    entry_idx: int
    side: int  # +1 long, -1 short
    note: str = ""


def _safe(x: float | int | None) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _series_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "n": int(len(s)),
        "mean": _safe(s.mean()),
        "std": _safe(s.std()),
        "p10": _safe(q.loc[0.1]),
        "p25": _safe(q.loc[0.25]),
        "p50": _safe(q.loc[0.5]),
        "p75": _safe(q.loc[0.75]),
        "p90": _safe(q.loc[0.9]),
    }


def _trade_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    pnl_g = df["pnl_gross_ticks"]
    pnl_n = df["pnl_net_ticks"]
    win = pnl_n > 0
    out = {
        "n": int(len(df)),
        "win_rate_net": _safe(win.mean()),
        "ev_gross_ticks": _safe(pnl_g.mean()),
        "ev_net_ticks": _safe(pnl_n.mean()),
        "gross_std_ticks": _safe(pnl_g.std()),
        "net_std_ticks": _safe(pnl_n.std()),
        "gross_sharpe": _safe(pnl_g.mean() / pnl_g.std()) if pnl_g.std() not in (0, np.nan) else None,
        "net_sharpe": _safe(pnl_n.mean() / pnl_n.std()) if pnl_n.std() not in (0, np.nan) else None,
        "avg_win_ticks": _safe(pnl_n[win].mean()) if win.any() else None,
        "avg_loss_ticks": _safe(pnl_n[~win].mean()) if (~win).any() else None,
        "gross_dist": _series_stats(pnl_g),
        "net_dist": _series_stats(pnl_n),
    }
    return out


def _time_to_minute(t: pd.Timestamp) -> int:
    return t.hour * 60 + t.minute


def _bucket_from_minute(m: int) -> str:
    if m >= 18 * 60 or m < 4 * 60:
        return "overnight"
    if 4 * 60 <= m < 9 * 60 + 30:
        return "europe_overlap"
    if 9 * 60 + 30 <= m < 10 * 60 + 30:
        return "cash_open"
    if 10 * 60 + 30 <= m < 12 * 60:
        return "late_morning"
    if 12 * 60 <= m < 14 * 60:
        return "lunch"
    if 14 * 60 <= m <= 16 * 60:
        return "power_hour"
    return "other"


def _load_resampled_data(root: Path, date_glob: str, step_ms: int) -> pd.DataFrame:
    cols = [
        "Time",
        "mid",
        "bid_price_1",
        "ask_price_1",
        "trade_volume",
        "signed_volume",
        "stress_ratio",
        "order_book_imbalance",
    ]
    frames: list[pd.DataFrame] = []
    for dd in sorted(root.glob(date_glob)):
        pq = dd / "features_labels.parquet"
        if not pq.exists():
            continue
        raw = pd.read_parquet(pq)
        keep = [c for c in cols if c in raw.columns]
        if "Time" not in keep or "mid" not in keep:
            continue
        df = raw[keep].copy()
        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        df = df.sort_values("Time")
        if step_ms > 0:
            rule = f"{step_ms}ms"
            agg = {
                "mid": "last",
                "bid_price_1": "last",
                "ask_price_1": "last",
                "trade_volume": "sum",
                "signed_volume": "sum",
                "stress_ratio": "last",
                "order_book_imbalance": "last",
            }
            agg = {k: v for k, v in agg.items() if k in df.columns}
            df = (
                df.set_index("Time")
                .resample(rule)
                .agg(agg)
                .dropna(subset=["mid"])
                .reset_index()
            )
        frames.append(df)

    if not frames:
        raise RuntimeError(f"No data loaded from {root} with glob={date_glob}")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("Time").reset_index(drop=True)
    return out


def _add_features(df: pd.DataFrame, tick_size: float, step_ms: int) -> pd.DataFrame:
    out = df.copy()
    out["Time_et"] = out["Time"].dt.tz_convert(ET_TZ)
    out["date_et"] = out["Time_et"].dt.strftime("%Y-%m-%d")
    out["month_et"] = out["Time_et"].dt.strftime("%Y-%m")
    out["minute_et"] = out["Time_et"].map(_time_to_minute)
    out["session_bucket"] = out["minute_et"].map(_bucket_from_minute)
    out["is_event_day"] = out["date_et"].isin(EVENT_DATES.keys())

    if "bid_price_1" in out.columns and "ask_price_1" in out.columns:
        out["spread"] = out["ask_price_1"] - out["bid_price_1"]
    else:
        out["spread"] = 0.0
    out["spread_ticks"] = out["spread"] / tick_size

    # RTH features for setups based on open->close behavior.
    rth_mask = (out["minute_et"] >= 9 * 60 + 30) & (out["minute_et"] <= 16 * 60)
    out["is_rth"] = rth_mask

    vol = pd.to_numeric(out.get("trade_volume", 0.0), errors="coerce").fillna(0.0)
    pv = out["mid"] * vol
    rth_vol = vol.where(rth_mask, 0.0)
    rth_pv = pv.where(rth_mask, 0.0)
    out["rth_cum_vol"] = rth_vol.groupby(out["date_et"]).cumsum()
    out["rth_cum_pv"] = rth_pv.groupby(out["date_et"]).cumsum()
    out["rth_vwap"] = out["rth_cum_pv"] / out["rth_cum_vol"].replace(0.0, np.nan)
    out["vwap_dev_ticks"] = (out["mid"] - out["rth_vwap"]) / tick_size

    mid_rth = out["mid"].where(rth_mask, np.nan)
    rth_hi = mid_rth.groupby(out["date_et"]).cummax()
    rth_lo = mid_rth.groupby(out["date_et"]).cummin()
    rth_range = (rth_hi - rth_lo).replace(0.0, np.nan)
    out["range_pos"] = (out["mid"] - rth_lo) / rth_range

    # Vol/flow context for filter tests.
    ret = out.groupby("date_et", group_keys=False)["mid"].pct_change()
    bars_5m = max(1, int((5 * 60_000) // step_ms))
    out["rv_5m"] = out.groupby("date_et", group_keys=False)["mid"].transform(
        lambda s: s.pct_change().rolling(bars_5m, min_periods=max(3, bars_5m // 3)).std()
    )
    if "signed_volume" in out.columns:
        out["flow_3"] = out.groupby("date_et", group_keys=False)["signed_volume"].transform(
            lambda s: s.rolling(3, min_periods=1).sum()
        )
    else:
        out["flow_3"] = 0.0

    return out


def _collect_cluster_first_entries(df: pd.DataFrame, step_ms: int) -> list[TradeEntry]:
    gap_bars = max(1, int((15 * 60_000) // step_ms))
    sig = df["is_rth"] & (df["range_pos"] >= 0.85) & (df["vwap_dev_ticks"] >= 50.0)
    entries: list[TradeEntry] = []
    for d, g in df[sig].groupby("date_et"):
        idx = g.index.to_list()
        if not idx:
            continue
        clusters = [[idx[0]]]
        for i in idx[1:]:
            if i - clusters[-1][-1] <= gap_bars:
                clusters[-1].append(i)
            else:
                clusters.append([i])
        for c in clusters:
            entries.append(TradeEntry(setup="signal_cluster_first", date_et=d, entry_idx=c[0], side=-1))
    return entries


def _first_index(mask_idx: np.ndarray) -> int | None:
    if mask_idx.size == 0:
        return None
    return int(mask_idx[0])


def _collect_orb_entries(df: pd.DataFrame, tick: float) -> list[TradeEntry]:
    entries: list[TradeEntry] = []
    for d, g in df.groupby("date_et"):
        gm = g["minute_et"].values
        px = g["mid"].values
        idx = g.index.values
        open_m = (gm >= 9 * 60 + 30) & (gm < 9 * 60 + 45)
        trade_m = (gm >= 9 * 60 + 45) & (gm < 11 * 60)
        if open_m.sum() < 5 or trade_m.sum() < 5:
            continue
        orb_high = float(np.nanmax(px[open_m]))
        orb_low = float(np.nanmin(px[open_m]))
        ti = idx[trade_m]
        tp = px[trade_m]
        long_hits = ti[tp >= orb_high + tick]
        short_hits = ti[tp <= orb_low - tick]
        first_long = _first_index(long_hits)
        first_short = _first_index(short_hits)
        if first_long is None and first_short is None:
            continue
        if first_short is None or (first_long is not None and first_long < first_short):
            entries.append(TradeEntry(setup="orb_15m_breakout", date_et=d, entry_idx=first_long, side=1))
        else:
            entries.append(TradeEntry(setup="orb_15m_breakout", date_et=d, entry_idx=first_short, side=-1))
    return entries


def _collect_pullback_entries(df: pd.DataFrame, tick: float) -> list[TradeEntry]:
    """
    First pullback after open drive:
    - Measure 09:30 -> 09:40 drive.
    - Require abs(drive) >= 4 ticks.
    - Seek first >=2 tick pullback, then 1 tick resumption.
    - One trade/day.
    """
    entries: list[TradeEntry] = []
    for d, g in df.groupby("date_et"):
        gm = g["minute_et"].values
        px = g["mid"].values
        idx = g.index.values
        open_m = (gm >= 9 * 60 + 30) & (gm < 9 * 60 + 40)
        search_m = (gm >= 9 * 60 + 40) & (gm < 11 * 60 + 30)
        if open_m.sum() < 5 or search_m.sum() < 5:
            continue
        p0 = float(px[open_m][0])
        p1 = float(px[open_m][-1])
        drive_ticks = (p1 - p0) / tick
        if abs(drive_ticks) < 4.0:
            continue
        side = 1 if drive_ticks > 0 else -1
        s_idx = idx[search_m]
        s_px = px[search_m]
        if side > 0:
            run_hi = s_px[0]
            pulled = False
            pull_low = np.nan
            entry = None
            for i in range(len(s_px)):
                p = float(s_px[i])
                run_hi = max(run_hi, p)
                if not pulled and (run_hi - p) >= 2.0 * tick:
                    pulled = True
                    pull_low = p
                elif pulled and p >= pull_low + 1.0 * tick:
                    entry = int(s_idx[i])
                    break
            if entry is not None:
                entries.append(TradeEntry(setup="first_pullback_open_drive", date_et=d, entry_idx=entry, side=1))
        else:
            run_lo = s_px[0]
            pulled = False
            pull_high = np.nan
            entry = None
            for i in range(len(s_px)):
                p = float(s_px[i])
                run_lo = min(run_lo, p)
                if not pulled and (p - run_lo) >= 2.0 * tick:
                    pulled = True
                    pull_high = p
                elif pulled and p <= pull_high - 1.0 * tick:
                    entry = int(s_idx[i])
                    break
            if entry is not None:
                entries.append(TradeEntry(setup="first_pullback_open_drive", date_et=d, entry_idx=entry, side=-1))
    return entries


def _collect_prev_day_reaction_entries(df: pd.DataFrame, tick: float, step_ms: int) -> list[TradeEntry]:
    entries: list[TradeEntry] = []
    rth = df[df["is_rth"]]
    by_day = {d: g for d, g in df.groupby("date_et")}
    rth_levels = {}
    for d, g in rth.groupby("date_et"):
        rth_levels[d] = (float(g["mid"].max()), float(g["mid"].min()))
    dates = sorted(rth_levels.keys())
    lookahead = max(1, int((30 * 60_000) // step_ms))
    for i in range(1, len(dates)):
        d = dates[i]
        prev = dates[i - 1]
        prev_hi, prev_lo = rth_levels[prev]
        g = by_day[d]
        gm = g["minute_et"].values
        px = g["mid"].values
        idx = g.index.values
        search = (gm >= 9 * 60 + 30) & (gm < 11 * 60 + 30)
        s_idx = idx[search]
        s_px = px[search]
        if len(s_idx) < 20:
            continue

        cand_entries: list[tuple[int, int, str]] = []  # (entry_idx, side, note)

        # Prior-day high reaction -> short if reversal >= 2 ticks.
        hi_touch = np.where(s_px >= prev_hi)[0]
        if hi_touch.size > 0:
            t0 = int(hi_touch[0])
            end = min(len(s_px), t0 + lookahead + 1)
            rev = np.where(s_px[t0:end] <= prev_hi - 2.0 * tick)[0]
            if rev.size > 0:
                cand_entries.append((int(s_idx[t0 + int(rev[0])]), -1, "pdh_reject"))

        # Prior-day low reaction -> long if reversal >= 2 ticks.
        lo_touch = np.where(s_px <= prev_lo)[0]
        if lo_touch.size > 0:
            t0 = int(lo_touch[0])
            end = min(len(s_px), t0 + lookahead + 1)
            rev = np.where(s_px[t0:end] >= prev_lo + 2.0 * tick)[0]
            if rev.size > 0:
                cand_entries.append((int(s_idx[t0 + int(rev[0])]), 1, "pdl_reject"))

        if not cand_entries:
            continue
        cand_entries.sort(key=lambda x: x[0])
        e_idx, side, note = cand_entries[0]
        entries.append(TradeEntry(setup="prior_day_level_reaction", date_et=d, entry_idx=e_idx, side=side, note=note))
    return entries


def _build_entries(df: pd.DataFrame, tick: float, step_ms: int) -> pd.DataFrame:
    all_entries: list[TradeEntry] = []
    all_entries.extend(_collect_cluster_first_entries(df, step_ms))
    all_entries.extend(_collect_orb_entries(df, tick))
    all_entries.extend(_collect_pullback_entries(df, tick))
    all_entries.extend(_collect_prev_day_reaction_entries(df, tick, step_ms))
    if not all_entries:
        return pd.DataFrame(columns=["setup", "date_et", "entry_idx", "side", "note"])
    edf = pd.DataFrame([e.__dict__ for e in all_entries])
    edf = edf.sort_values(["setup", "date_et", "entry_idx"]).reset_index(drop=True)
    return edf


def _apply_filter(row: pd.Series, side: int, filter_name: str, med_rv5: float, med_stress: float) -> bool:
    if filter_name == "none":
        return True
    if filter_name == "spread_1tick":
        return bool(row.get("spread_ticks", np.nan) <= 1.0)
    if filter_name == "low_vol":
        rv = row.get("rv_5m", np.nan)
        return bool(np.isfinite(rv) and rv <= med_rv5)
    if filter_name == "book_stable":
        st = row.get("stress_ratio", np.nan)
        return bool(np.isfinite(st) and st <= med_stress)
    if filter_name == "aligned_flow":
        return bool(side * float(row.get("flow_3", 0.0)) > 0.0)
    if filter_name == "combined":
        return (
            bool(row.get("spread_ticks", np.nan) <= 1.0)
            and bool(np.isfinite(row.get("rv_5m", np.nan)) and row.get("rv_5m", np.nan) <= med_rv5)
            and bool(np.isfinite(row.get("stress_ratio", np.nan)) and row.get("stress_ratio", np.nan) <= med_stress)
            and bool(side * float(row.get("flow_3", 0.0)) > 0.0)
        )
    return True


def _entry_fill(row: pd.Series, side: int) -> float:
    bid = row.get("bid_price_1", np.nan)
    ask = row.get("ask_price_1", np.nan)
    mid = float(row["mid"])
    spr = float(row.get("spread", 0.0) or 0.0)
    if np.isfinite(bid) and np.isfinite(ask):
        return float(ask if side > 0 else bid)
    return float(mid + (spr / 2.0 if side > 0 else -spr / 2.0))


def _exit_fill(row: pd.Series, side: int) -> float:
    bid = row.get("bid_price_1", np.nan)
    ask = row.get("ask_price_1", np.nan)
    mid = float(row["mid"])
    spr = float(row.get("spread", 0.0) or 0.0)
    if np.isfinite(bid) and np.isfinite(ask):
        return float(bid if side > 0 else ask)
    return float(mid - (spr / 2.0 if side > 0 else -spr / 2.0))


def _simulate(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    tick_size: float,
    step_ms: int,
    hold_minutes: int,
    commission_ticks: float,
    mode: str,
    filter_name: str,
) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()
    hold_bars = max(1, int((hold_minutes * 60_000) // step_ms))
    # Force intraday flatten by 15:55 ET.
    cutoff_minute = 15 * 60 + 55
    day_cutoff_idx = (
        df[df["minute_et"] <= cutoff_minute]
        .groupby("date_et")
        .tail(1)[["date_et"]]
        .assign(cut_idx=lambda x: x.index)
        .set_index("date_et")["cut_idx"]
        .to_dict()
    )

    med_rv5 = float(df["rv_5m"].median(skipna=True))
    med_stress = float(df["stress_ratio"].median(skipna=True)) if "stress_ratio" in df.columns else 0.0
    rows = []
    for r in entries.itertuples(index=False):
        setup = r.setup
        date_et = r.date_et
        e_idx = int(r.entry_idx)
        side = int(r.side)
        note = r.note
        idx_use = e_idx + 1 if mode == "delayed_1bar" else e_idx
        if idx_use >= len(df):
            continue
        if str(df.at[idx_use, "date_et"]) != date_et:
            continue
        e_row = df.iloc[idx_use]
        if not _apply_filter(e_row, side, filter_name, med_rv5, med_stress):
            continue

        cut_idx = day_cutoff_idx.get(date_et, None)
        if cut_idx is None:
            continue
        x_idx = min(idx_use + hold_bars, int(cut_idx))
        if x_idx <= idx_use:
            continue
        x_row = df.iloc[x_idx]
        entry_px = _entry_fill(e_row, side)
        exit_px = _exit_fill(x_row, side)
        gross = side * (exit_px - entry_px) / tick_size
        net = gross - commission_ticks
        rows.append(
            {
                "setup": setup,
                "date_et": date_et,
                "month_et": e_row["month_et"],
                "entry_time_et": str(e_row["Time_et"]),
                "entry_session_bucket": e_row["session_bucket"],
                "is_event_day": bool(e_row["is_event_day"]),
                "side": side,
                "mode": mode,
                "filter": filter_name,
                "note": note,
                "entry_idx_used": idx_use,
                "exit_idx_used": x_idx,
                "pnl_gross_ticks": float(gross),
                "pnl_net_ticks": float(net),
            }
        )
    return pd.DataFrame(rows)


def _summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"global": {"n": 0}, "by_setup": [], "by_session": [], "by_month": [], "by_event": []}
    by_setup = []
    for (setup, mode, flt), g in trades.groupby(["setup", "mode", "filter"]):
        st = _trade_stats(g)
        st.update({"setup": setup, "mode": mode, "filter": flt})
        by_setup.append(st)

    by_session = []
    for (setup, mode, flt, sess), g in trades.groupby(["setup", "mode", "filter", "entry_session_bucket"]):
        st = _trade_stats(g)
        by_session.append(
            {
                "setup": setup,
                "mode": mode,
                "filter": flt,
                "session": sess,
                "n": st.get("n", 0),
                "ev_net_ticks": st.get("ev_net_ticks"),
                "win_rate_net": st.get("win_rate_net"),
            }
        )

    by_month = []
    for (setup, mode, flt, month), g in trades.groupby(["setup", "mode", "filter", "month_et"]):
        st = _trade_stats(g)
        by_month.append(
            {
                "setup": setup,
                "mode": mode,
                "filter": flt,
                "month": month,
                "n": st.get("n", 0),
                "ev_net_ticks": st.get("ev_net_ticks"),
                "win_rate_net": st.get("win_rate_net"),
            }
        )

    by_event = []
    for (setup, mode, flt, ev), g in trades.groupby(["setup", "mode", "filter", "is_event_day"]):
        st = _trade_stats(g)
        by_event.append(
            {
                "setup": setup,
                "mode": mode,
                "filter": flt,
                "is_event_day": bool(ev),
                "n": st.get("n", 0),
                "ev_net_ticks": st.get("ev_net_ticks"),
                "win_rate_net": st.get("win_rate_net"),
            }
        )

    top = sorted(
        [r for r in by_setup if r.get("n", 0) >= 20],
        key=lambda x: (x.get("ev_net_ticks") is None, -(x.get("ev_net_ticks") or -1e9)),
    )[:15]
    return {
        "global": _trade_stats(trades),
        "by_setup": by_setup,
        "by_session": by_session,
        "by_month": by_month,
        "by_event": by_event,
        "top_candidates_n_ge_20": top,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ES exhaustion suite with existing ESH6 data.")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--date-glob", default="date=2026-0[12]-*")
    ap.add_argument("--step-ms", type=int, default=1000)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--hold-minutes", type=int, default=30)
    ap.add_argument("--commission-ticks", type=float, default=0.36)
    ap.add_argument("--output-json", default=str(OUT_DIR / "es_exhaustion_suite_20260310.json"))
    ap.add_argument("--output-trades-csv", default=str(OUT_DIR / "es_exhaustion_suite_20260310_trades.csv"))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    df = _load_resampled_data(data_root, args.date_glob, args.step_ms)
    df = _add_features(df, tick_size=args.tick_size, step_ms=args.step_ms)
    entries = _build_entries(df, tick=args.tick_size, step_ms=args.step_ms)

    filters = ["none", "spread_1tick", "low_vol", "book_stable", "aligned_flow", "combined"]
    modes = ["market", "delayed_1bar"]
    trade_frames = []
    for mode in modes:
        for flt in filters:
            tdf = _simulate(
                df=df,
                entries=entries,
                tick_size=args.tick_size,
                step_ms=args.step_ms,
                hold_minutes=args.hold_minutes,
                commission_ticks=args.commission_ticks,
                mode=mode,
                filter_name=flt,
            )
            if not tdf.empty:
                trade_frames.append(tdf)

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary = _summarize(trades)

    meta = {
        "data_root": str(data_root),
        "date_glob": args.date_glob,
        "rows_loaded": int(len(df)),
        "dates_loaded": sorted(df["date_et"].unique().tolist()),
        "step_ms": int(args.step_ms),
        "tick_size": float(args.tick_size),
        "hold_minutes": int(args.hold_minutes),
        "commission_ticks": float(args.commission_ticks),
        "events_used": EVENT_DATES,
        "setups_built": sorted(entries["setup"].unique().tolist()) if not entries.empty else [],
        "entry_count": int(len(entries)),
    }

    out = {"meta": meta, "summary": summary}
    out_json = Path(args.output_json)
    out_csv = Path(args.output_trades_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    if not trades.empty:
        trades.to_csv(out_csv, index=False)

    print(f"Wrote suite summary: {out_json}")
    if not trades.empty:
        print(f"Wrote trade ledger: {out_csv}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Dates loaded: {len(meta['dates_loaded'])}")
    print(f"Entries built: {len(entries):,}")
    print(f"Simulated trades: {len(trades):,}")
    top = summary.get("top_candidates_n_ge_20", [])
    if top:
        print("\nTop candidates (n>=20, by EV net):")
        for r in top[:10]:
            print(
                f"  {r['setup']} | {r['mode']} | {r['filter']} | "
                f"n={r['n']} ev_net={r.get('ev_net_ticks'):.4f} wr={r.get('win_rate_net'):.2%}"
            )


if __name__ == "__main__":
    main()

