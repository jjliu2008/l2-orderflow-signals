"""
Final validation pass for ES survivor candidates.

Survivors:
  1) orb_15m_breakout | filter=none
  2) signal_cluster_first | filter=spread_1tick

Stress tests:
  A) Cost sensitivity: commission/slippage ticks in [0.36, 0.60, 0.90, 1.25]
  B) Session dependence
  C) Event-day contamination
  D) Delay robustness: market, delayed_1bar, delayed_2bar, delayed_spread_norm

Outputs:
  artifacts/signal_research/es_survivor_validation_20260310.json
  artifacts/signal_research/es_survivor_validation_20260310_trades.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import run_es_exhaustion_suite as suite


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "artifacts/signal_research"
DATA_ROOT = PROJECT_ROOT / "data/processed_esh6/instrument=ES"


def _safe(x):
    try:
        v = float(x)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _py_scalar(x):
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


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


def _choose_entry_index(
    df: pd.DataFrame,
    e_idx: int,
    date_et: str,
    mode: str,
    spread_norm_max_ticks: float = 1.0,
    max_wait_bars: int = 10,
) -> int | None:
    if mode == "market":
        idx = e_idx
    elif mode == "delayed_1bar":
        idx = e_idx + 1
    elif mode == "delayed_2bar":
        idx = e_idx + 2
    elif mode == "delayed_spread_norm":
        lo = e_idx + 1
        hi = min(len(df) - 1, e_idx + max_wait_bars)
        idx = None
        for i in range(lo, hi + 1):
            if str(df.at[i, "date_et"]) != date_et:
                break
            s = df.at[i, "spread_ticks"]
            if np.isfinite(s) and float(s) <= spread_norm_max_ticks:
                idx = int(i)
                break
        if idx is None:
            return None
    else:
        return None

    if idx >= len(df):
        return None
    if str(df.at[idx, "date_et"]) != date_et:
        return None
    return int(idx)


def _simulate_survivors(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    tick_size: float,
    step_ms: int,
    hold_minutes: int,
    modes: list[str],
) -> pd.DataFrame:
    hold_bars = max(1, int((hold_minutes * 60_000) // step_ms))
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

        # Survivor filters fixed by setup.
        if setup == "orb_15m_breakout":
            fixed_filter = "none"
        elif setup == "signal_cluster_first":
            fixed_filter = "spread_1tick"
        else:
            continue

        for mode in modes:
            idx_use = _choose_entry_index(
                df=df,
                e_idx=e_idx,
                date_et=date_et,
                mode=mode,
                spread_norm_max_ticks=1.0,
                max_wait_bars=10,
            )
            if idx_use is None:
                continue
            e_row = df.iloc[idx_use]

            if not suite._apply_filter(
                e_row,
                side=side,
                filter_name=fixed_filter,
                med_rv5=med_rv5,
                med_stress=med_stress,
            ):
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

            rows.append(
                {
                    "setup": setup,
                    "filter": fixed_filter,
                    "mode": mode,
                    "date_et": date_et,
                    "month_et": e_row["month_et"],
                    "entry_time_et": str(e_row["Time_et"]),
                    "entry_session_bucket": e_row["session_bucket"],
                    "is_event_day": bool(e_row["is_event_day"]),
                    "side": side,
                    "note": note,
                    "entry_idx_orig": e_idx,
                    "entry_idx_used": idx_use,
                    "delay_bars": int(idx_use - e_idx),
                    "exit_idx_used": x_idx,
                    "pnl_gross_ticks": float(gross),
                }
            )
    return pd.DataFrame(rows)


def _stats_with_cost(df: pd.DataFrame, cost_ticks: float) -> dict:
    if df.empty:
        return {"n": 0}
    net = df["pnl_gross_ticks"] - cost_ticks
    return {
        "n": int(len(df)),
        "ev_net_ticks": _safe(net.mean()),
        "win_rate_net": _safe((net > 0).mean()),
        "avg_win_ticks": _safe(net[net > 0].mean()) if (net > 0).any() else None,
        "avg_loss_ticks": _safe(net[net <= 0].mean()) if (net <= 0).any() else None,
        "net_std_ticks": _safe(net.std()),
        "net_sharpe": _safe(net.mean() / net.std()) if net.std() not in (0, np.nan) else None,
    }


def _group_breakdown(df: pd.DataFrame, by_cols: list[str], cost_ticks: float) -> list[dict]:
    out = []
    if df.empty:
        return out
    for key, g in df.groupby(by_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {k: _py_scalar(v) for k, v in zip(by_cols, key_tuple)}
        row.update(_stats_with_cost(g, cost_ticks))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate ES survivor setups.")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--date-glob", default="date=2026-0[12]-*")
    ap.add_argument("--step-ms", type=int, default=1000)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--hold-minutes", type=int, default=30)
    ap.add_argument("--cost-grid", default="0.36,0.60,0.90,1.25")
    ap.add_argument("--output-json", default=str(OUT_DIR / "es_survivor_validation_20260310.json"))
    ap.add_argument("--output-trades-csv", default=str(OUT_DIR / "es_survivor_validation_20260310_trades.csv"))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    df = suite._load_resampled_data(data_root, args.date_glob, args.step_ms)
    df = suite._add_features(df, tick_size=args.tick_size, step_ms=args.step_ms)
    entries = suite._build_entries(df, tick=args.tick_size, step_ms=args.step_ms)
    entries = entries[entries["setup"].isin(["orb_15m_breakout", "signal_cluster_first"])].copy()

    modes = ["market", "delayed_1bar", "delayed_2bar", "delayed_spread_norm"]
    trades = _simulate_survivors(
        df=df,
        entries=entries,
        tick_size=args.tick_size,
        step_ms=args.step_ms,
        hold_minutes=args.hold_minutes,
        modes=modes,
    )

    costs = [float(x.strip()) for x in args.cost_grid.split(",") if x.strip()]
    by_cost = {}
    for c in costs:
        rows = []
        for (setup, flt, mode), g in trades.groupby(["setup", "filter", "mode"]):
            st = _stats_with_cost(g, c)
            st.update({"setup": setup, "filter": flt, "mode": mode})
            rows.append(st)
        rows = sorted(rows, key=lambda r: (r["ev_net_ticks"] is None, -(r["ev_net_ticks"] or -1e9)))
        by_cost[f"{c:.2f}"] = rows

    # Breakdowns at reference cost (first in grid).
    ref_cost = costs[0] if costs else 0.36
    month_rows = _group_breakdown(trades, ["setup", "filter", "mode", "month_et"], ref_cost)
    session_rows = _group_breakdown(trades, ["setup", "filter", "mode", "entry_session_bucket"], ref_cost)
    event_rows = _group_breakdown(trades, ["setup", "filter", "mode", "is_event_day"], ref_cost)

    # Compact survivor summary for user-facing interpretation.
    survivor_focus = []
    for setup, flt in [("orb_15m_breakout", "none"), ("signal_cluster_first", "spread_1tick")]:
        block = {"setup": setup, "filter": flt, "delay_robustness": []}
        for mode in modes:
            g = trades[(trades["setup"] == setup) & (trades["filter"] == flt) & (trades["mode"] == mode)]
            st = _stats_with_cost(g, ref_cost)
            st.update({"mode": mode})
            block["delay_robustness"].append(st)
        survivor_focus.append(block)

    out = {
        "meta": {
            "data_root": str(data_root),
            "date_glob": args.date_glob,
            "step_ms": int(args.step_ms),
            "tick_size": float(args.tick_size),
            "hold_minutes": int(args.hold_minutes),
            "cost_grid_ticks": costs,
            "reference_cost_for_breakdowns": ref_cost,
            "rows_loaded": int(len(df)),
            "dates_loaded": sorted(df["date_et"].unique().tolist()),
            "entries_survivor_setups": int(len(entries)),
            "trades_simulated": int(len(trades)),
        },
        "survivor_focus": survivor_focus,
        "ranked_by_cost": by_cost,
        "breakdowns": {
            "by_month": month_rows,
            "by_session": session_rows,
            "by_event": event_rows,
        },
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_trades_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    if not trades.empty:
        trades.to_csv(out_csv, index=False)

    print(f"Wrote survivor validation summary: {out_json}")
    if not trades.empty:
        print(f"Wrote survivor validation trades: {out_csv}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Dates loaded: {len(out['meta']['dates_loaded'])}")
    print(f"Survivor entries: {len(entries):,}")
    print(f"Simulated trades: {len(trades):,}")

    for c in costs:
        key = f"{c:.2f}"
        top = out["ranked_by_cost"].get(key, [])[:6]
        print(f"\nTop @ cost={key} ticks:")
        for r in top:
            if r.get("n", 0) < 10:
                continue
            print(
                f"  {r['setup']} | {r['mode']} | {r['filter']} | "
                f"n={r['n']} ev_net={r.get('ev_net_ticks'):.4f} wr={r.get('win_rate_net'):.2%}"
            )


if __name__ == "__main__":
    main()
