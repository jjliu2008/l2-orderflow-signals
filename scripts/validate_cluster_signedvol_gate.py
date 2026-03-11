"""
Validate the refined ES cluster branch with a causal pre-entry signed-volume gate.

Frozen branch under test:
  - setup: signal_cluster_first
  - mode: delayed_1bar
  - base filter: spread_1tick
  - session exclusion: no power_hour

Additional gate under test:
  - signedvol_sum30s <= threshold at signal bar (causal, pre-entry)

Primary comparison:
  - baseline (no signedvol gate)
  - threshold 200
  - threshold 250
  - threshold 300

Optional live candidate:
  - setup: signal_cluster_live_candidate
  - entry gate: cash_open signedvol_sum30s <= 100, otherwise <= 250
  - exit: fail-fast invalidation in first 3 minutes + 30m max hold

Outputs:
  - artifacts/signal_research/es_cluster_signedvol_gate_validation_*.json
  - artifacts/signal_research/es_cluster_signedvol_gate_validation_*_trades.csv
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


def _stats_with_cost(df: pd.DataFrame, cost_ticks: float) -> dict:
    if df.empty:
        return {"n": 0}
    net = df["pnl_gross_ticks"] - cost_ticks
    wins = net[net > 0]
    losses = net[net <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else None
    return {
        "n": int(len(df)),
        "ev_net_ticks": _safe(net.mean()),
        "win_rate_net": _safe((net > 0).mean()),
        "total_net_ticks": _safe(net.sum()),
        "avg_win_ticks": _safe(wins.mean()) if len(wins) else None,
        "avg_loss_ticks": _safe(losses.mean()) if len(losses) else None,
        "net_std_ticks": _safe(net.std()),
        "net_sharpe": _safe(net.mean() / net.std()) if net.std() not in (0, np.nan) else None,
        "profit_factor": _safe(pf) if pf is not None else None,
    }


def _concentration_with_cost(df: pd.DataFrame, cost_ticks: float) -> dict:
    if df.empty:
        return {
            "total_net_ticks": 0.0,
            "top1_share": None,
            "top3_share": None,
            "top5_share": None,
            "top1_ticks": None,
            "top3_ticks": None,
            "top5_ticks": None,
        }
    net = (df["pnl_gross_ticks"] - cost_ticks).sort_values(ascending=False).reset_index(drop=True)
    total = float(net.sum())
    top1 = float(net.head(1).sum())
    top3 = float(net.head(3).sum())
    top5 = float(net.head(5).sum())
    if total > 0:
        top1_share = top1 / total
        top3_share = top3 / total
        top5_share = top5 / total
    else:
        top1_share = None
        top3_share = None
        top5_share = None
    return {
        "total_net_ticks": total,
        "top1_share": _safe(top1_share),
        "top3_share": _safe(top3_share),
        "top5_share": _safe(top5_share),
        "top1_ticks": top1,
        "top3_ticks": top3,
        "top5_ticks": top5,
    }


def _simulate_variant(
    df: pd.DataFrame,
    entries: pd.DataFrame,
    tick_size: float,
    step_ms: int,
    hold_minutes: int,
    sv_threshold: float | None,
    variant_name: str | None = None,
    setup_name: str = "signal_cluster_first",
    live_policy: dict | None = None,
) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame()

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

    rows: list[dict] = []
    for r in entries.itertuples(index=False):
        date_et = r.date_et
        e_idx = int(r.entry_idx)
        side = int(r.side)
        note = r.note

        idx_use = e_idx + 1  # delayed_1bar frozen
        if idx_use >= len(df):
            continue
        if str(df.at[idx_use, "date_et"]) != date_et:
            continue

        sig_row = df.iloc[e_idx]
        e_row = df.iloc[idx_use]

        # Frozen session exclusion: no power_hour.
        if str(e_row.get("session_bucket", "")) == "power_hour":
            continue

        # Frozen base filter: spread_1tick at actual entry bar.
        if not suite._apply_filter(
            e_row,
            side=side,
            filter_name="spread_1tick",
            med_rv5=med_rv5,
            med_stress=med_stress,
        ):
            continue

        # New causal gate: pre-entry signed volume sum over 30s at signal bar.
        sv = sig_row.get("signedvol_sum30s", np.nan)
        if live_policy is not None:
            if not np.isfinite(sv):
                continue
            sess = str(e_row.get("session_bucket", ""))
            sv_max = (
                float(live_policy["cash_open_sv_max"])
                if sess == "cash_open"
                else float(live_policy["non_cash_sv_max"])
            )
            if float(sv) > sv_max:
                continue
        else:
            if sv_threshold is not None:
                if not np.isfinite(sv):
                    continue
                if float(sv) > float(sv_threshold):
                    continue

        cut_idx = day_cutoff_idx.get(date_et, None)
        if cut_idx is None:
            continue
        x_idx_natural = min(idx_use + hold_bars, int(cut_idx))
        if x_idx_natural <= idx_use:
            continue

        # Optional fast thesis-invalidation exit for live candidate.
        x_idx = x_idx_natural
        exit_reason = "TIME_30M"
        if live_policy is not None:
            fail_bars = max(1, int((float(live_policy["failfast_seconds"]) * 1000.0) // float(step_ms)))
            flow_thresh = float(live_policy["failfast_adverse_flow30"])
            obi_thresh = float(live_policy["failfast_adverse_obi"])
            min_prog = float(live_policy["failfast_min_progress_ticks"])

            entry_mid = float(e_row["mid"])
            path = df.iloc[idx_use : x_idx_natural + 1].copy()
            flow30 = (
                pd.to_numeric(path.get("signed_volume", 0.0), errors="coerce")
                .fillna(0.0)
                .rolling(30, min_periods=5)
                .sum()
                .to_numpy()
            )
            obi = pd.to_numeric(path.get("order_book_imbalance", 0.0), errors="coerce").fillna(0.0).to_numpy()
            mid = pd.to_numeric(path["mid"], errors="coerce").to_numpy()

            mfe_mid = 0.0
            rel_exit = len(path) - 1
            for j in range(1, len(path)):
                mid_pnl_ticks = side * (mid[j] - entry_mid) / tick_size
                mfe_mid = max(mfe_mid, float(mid_pnl_ticks))
                if j > fail_bars:
                    continue
                no_progress = mfe_mid < min_prog
                adverse_flow = (side * float(flow30[j])) < -flow_thresh
                adverse_obi = (side * float(obi[j])) < -obi_thresh
                if no_progress and adverse_flow and adverse_obi:
                    rel_exit = j
                    exit_reason = "FAIL_FAST"
                    break
            x_idx = idx_use + int(rel_exit)

        x_row = df.iloc[x_idx]
        entry_px = suite._entry_fill(e_row, side)
        exit_px = suite._exit_fill(x_row, side)
        gross = side * (exit_px - entry_px) / tick_size

        rows.append(
            {
                "variant": variant_name
                if variant_name is not None
                else (f"sv_le_{int(sv_threshold)}" if sv_threshold is not None else "baseline_no_sv_gate"),
                "sv_threshold": sv_threshold,
                "setup": setup_name,
                "filter": "spread_1tick",
                "mode": "delayed_1bar",
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
                "exit_reason": exit_reason,
                "signal_signedvol_sum30s": _safe(sig_row.get("signedvol_sum30s", np.nan)),
                "pnl_gross_ticks": float(gross),
            }
        )
    return pd.DataFrame(rows)


def _group_stats(df: pd.DataFrame, by_cols: list[str], cost_ticks: float) -> list[dict]:
    out: list[dict] = []
    if df.empty:
        return out
    for key, g in df.groupby(by_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {k: v for k, v in zip(by_cols, key_tuple)}
        row.update(_stats_with_cost(g, cost_ticks))
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate signedvol gate on frozen ES cluster branch.")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--date-glob", default="date=2026-0[12]-*")
    ap.add_argument("--step-ms", type=int, default=1000)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--hold-minutes", type=int, default=30)
    ap.add_argument("--commission-ticks", type=float, default=0.36)
    ap.add_argument("--thresholds", default="200,250,300")
    ap.add_argument("--oos-month", default="2026-02")
    ap.add_argument("--include-live-candidate", type=int, default=1)
    ap.add_argument("--live-variant-name", default="live_candidate_v1")
    ap.add_argument("--live-setup-name", default="signal_cluster_live_candidate")
    ap.add_argument("--live-cash-open-sv-max", type=float, default=100.0)
    ap.add_argument("--live-non-cash-sv-max", type=float, default=250.0)
    ap.add_argument("--live-failfast-seconds", type=float, default=180.0)
    ap.add_argument("--live-failfast-min-progress-ticks", type=float, default=1.0)
    ap.add_argument("--live-failfast-adverse-flow30", type=float, default=200.0)
    ap.add_argument("--live-failfast-adverse-obi", type=float, default=0.03)
    ap.add_argument(
        "--output-json",
        default=str(OUT_DIR / "es_cluster_signedvol_gate_validation_20260310.json"),
    )
    ap.add_argument(
        "--output-trades-csv",
        default=str(OUT_DIR / "es_cluster_signedvol_gate_validation_20260310_trades.csv"),
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    df = suite._load_resampled_data(data_root, args.date_glob, args.step_ms)
    df = suite._add_features(df, tick_size=args.tick_size, step_ms=args.step_ms)
    # causal pre-entry 30-second signed volume sum on 1-second bars.
    df["signedvol_sum30s"] = df.groupby("date_et", group_keys=False)["signed_volume"].transform(
        lambda s: s.rolling(30, min_periods=10).sum()
    )

    entries = suite._build_entries(df, tick=args.tick_size, step_ms=args.step_ms)
    entries = entries[entries["setup"] == "signal_cluster_first"].copy()

    variants = [None] + thresholds
    trade_frames: list[pd.DataFrame] = []
    for th in variants:
        tdf = _simulate_variant(
            df=df,
            entries=entries,
            tick_size=args.tick_size,
            step_ms=args.step_ms,
            hold_minutes=args.hold_minutes,
            sv_threshold=th,
        )
        if not tdf.empty:
            trade_frames.append(tdf)

    if bool(int(args.include_live_candidate)):
        live_policy = {
            "cash_open_sv_max": float(args.live_cash_open_sv_max),
            "non_cash_sv_max": float(args.live_non_cash_sv_max),
            "failfast_seconds": float(args.live_failfast_seconds),
            "failfast_min_progress_ticks": float(args.live_failfast_min_progress_ticks),
            "failfast_adverse_flow30": float(args.live_failfast_adverse_flow30),
            "failfast_adverse_obi": float(args.live_failfast_adverse_obi),
        }
        live_df = _simulate_variant(
            df=df,
            entries=entries,
            tick_size=args.tick_size,
            step_ms=args.step_ms,
            hold_minutes=args.hold_minutes,
            sv_threshold=None,
            variant_name=str(args.live_variant_name),
            setup_name=str(args.live_setup_name),
            live_policy=live_policy,
        )
        if not live_df.empty:
            trade_frames.append(live_df)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    # Global stats by variant.
    global_by_variant = _group_stats(trades, ["variant"], args.commission_ticks)

    # Month splits and OOS subset.
    by_month = _group_stats(trades, ["variant", "month_et"], args.commission_ticks)
    by_session = _group_stats(trades, ["variant", "month_et", "entry_session_bucket"], args.commission_ticks)

    oos = trades[trades["month_et"] == args.oos_month].copy()
    oos_by_variant = _group_stats(oos, ["variant"], args.commission_ticks)
    oos_concentration = []
    for v, g in oos.groupby("variant"):
        row = {"variant": v}
        row.update(_concentration_with_cost(g, args.commission_ticks))
        oos_concentration.append(row)

    all_concentration = []
    for v, g in trades.groupby("variant"):
        row = {"variant": v}
        row.update(_concentration_with_cost(g, args.commission_ticks))
        all_concentration.append(row)

    # Robustness around threshold choice.
    robustness_rows = []
    for v, g in trades.groupby("variant"):
        st_all = _stats_with_cost(g, args.commission_ticks)
        g_oos = g[g["month_et"] == args.oos_month]
        st_oos = _stats_with_cost(g_oos, args.commission_ticks)
        robustness_rows.append(
            {
                "variant": v,
                "all_n": st_all.get("n", 0),
                "all_ev_net_ticks": st_all.get("ev_net_ticks"),
                "all_wr_net": st_all.get("win_rate_net"),
                "oos_month": args.oos_month,
                "oos_n": st_oos.get("n", 0),
                "oos_ev_net_ticks": st_oos.get("ev_net_ticks"),
                "oos_wr_net": st_oos.get("win_rate_net"),
            }
        )

    out = {
        "meta": {
            "data_root": str(data_root),
            "date_glob": args.date_glob,
            "step_ms": int(args.step_ms),
            "tick_size": float(args.tick_size),
            "hold_minutes": int(args.hold_minutes),
            "commission_ticks": float(args.commission_ticks),
            "thresholds": thresholds,
            "oos_month": args.oos_month,
            "include_live_candidate": bool(int(args.include_live_candidate)),
            "live_candidate": {
                "variant_name": str(args.live_variant_name),
                "setup_name": str(args.live_setup_name),
                "cash_open_sv_max": float(args.live_cash_open_sv_max),
                "non_cash_sv_max": float(args.live_non_cash_sv_max),
                "failfast_seconds": float(args.live_failfast_seconds),
                "failfast_min_progress_ticks": float(args.live_failfast_min_progress_ticks),
                "failfast_adverse_flow30": float(args.live_failfast_adverse_flow30),
                "failfast_adverse_obi": float(args.live_failfast_adverse_obi),
            },
            "frozen_branch": {
                "setup": "signal_cluster_first",
                "mode": "delayed_1bar",
                "base_filter": "spread_1tick",
                "exclude_session": "power_hour",
            },
            "rows_loaded": int(len(df)),
            "dates_loaded": sorted(df["date_et"].unique().tolist()),
            "cluster_entries_total": int(len(entries)),
            "trades_simulated_total": int(len(trades)),
        },
        "global_by_variant": global_by_variant,
        "month_split_by_variant": by_month,
        "session_split_by_variant_month": by_session,
        "oos_by_variant": oos_by_variant,
        "concentration_all": all_concentration,
        "concentration_oos": oos_concentration,
        "threshold_robustness": robustness_rows,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_trades_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if not trades.empty:
        trades.to_csv(out_csv, index=False)

    print(f"Wrote signedvol gate validation summary: {out_json}")
    if not trades.empty:
        print(f"Wrote signedvol gate validation trades: {out_csv}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Dates loaded: {len(out['meta']['dates_loaded'])}")
    print(f"Cluster entries: {len(entries):,}")
    print(f"Simulated trades: {len(trades):,}")

    print("\nOOS comparison:")
    for row in sorted(oos_by_variant, key=lambda r: (r.get("ev_net_ticks") is None, -(r.get("ev_net_ticks") or -1e9))):
        print(
            f"  {row['variant']}: "
            f"n={row.get('n', 0)} ev_net={row.get('ev_net_ticks')} wr={row.get('win_rate_net')}"
        )


if __name__ == "__main__":
    main()
