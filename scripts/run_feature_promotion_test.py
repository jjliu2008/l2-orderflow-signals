from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _discover_trade_files(root: Path, pattern: str) -> List[Path]:
    paths = [Path(p) for p in glob.glob(str(root / pattern), recursive=True)]
    # keep aggregate files; skip per-day folders like .../ES_2025-12-01/trades_gated.csv
    out: List[Path] = []
    for p in paths:
        try:
            parent = p.parent.name
        except Exception:
            continue
        if parent.startswith("ES_20"):
            continue
        out.append(p)
    return sorted(out)


def _aligned_depth_imbalance(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df.get("ec_depth_imbalance_l1"), errors="coerce")
    if not isinstance(s, pd.Series):
        s = pd.Series(np.nan, index=df.index, dtype=float)
    side = df.get("side", pd.Series("", index=df.index)).astype(str).str.lower()
    side_sign = pd.Series(np.where(side == "long", 1.0, np.where(side == "short", -1.0, np.nan)), index=df.index)
    return s * side_sign


def _coerce_series(df: pd.DataFrame, col: str) -> pd.Series:
    s = pd.to_numeric(df.get(col), errors="coerce")
    if isinstance(s, pd.Series):
        return s
    return pd.Series(np.nan, index=df.index, dtype=float)


def _score_candidate(df: pd.DataFrame, signal: pd.Series, direction: str, min_rows: int = 40) -> Dict[str, float | int | None]:
    y = pd.to_numeric(df.get("pnl_ticks"), errors="coerce")
    mask = y.notna() & signal.notna()
    if int(mask.sum()) < min_rows:
        return {
            "rows_scored": int(mask.sum()),
            "q_spread_ev": None,
            "top_q_ev": None,
            "bottom_q_ev": None,
            "overall_ev": float(y.mean()) if y.notna().any() else None,
        }

    sub = pd.DataFrame({"x": signal[mask].astype(float), "y": y[mask].astype(float)})
    if sub["x"].nunique() < 4:
        return {
            "rows_scored": int(mask.sum()),
            "q_spread_ev": None,
            "top_q_ev": None,
            "bottom_q_ev": None,
            "overall_ev": float(sub["y"].mean()) if not sub.empty else None,
        }
    try:
        sub["q"] = pd.qcut(sub["x"], 4, duplicates="drop")
    except Exception:
        return {
            "rows_scored": int(mask.sum()),
            "q_spread_ev": None,
            "top_q_ev": None,
            "bottom_q_ev": None,
            "overall_ev": float(sub["y"].mean()) if not sub.empty else None,
        }
    g = sub.groupby("q", observed=False)["y"].mean()
    if len(g) < 2:
        return {
            "rows_scored": int(mask.sum()),
            "q_spread_ev": None,
            "top_q_ev": None,
            "bottom_q_ev": None,
            "overall_ev": float(sub["y"].mean()) if not sub.empty else None,
        }

    low = float(g.iloc[0])
    high = float(g.iloc[-1])
    q_spread = (high - low) if direction == "high_better" else (low - high)
    return {
        "rows_scored": int(mask.sum()),
        "q_spread_ev": float(q_spread),
        "top_q_ev": float(high),
        "bottom_q_ev": float(low),
        "overall_ev": float(sub["y"].mean()),
    }


def run(args: argparse.Namespace) -> int:
    files = _discover_trade_files(Path(args.root), args.pattern)
    if not files:
        print("No trades_gated.csv files found.")
        return 1

    candidates = {
        "ec_flow_zscore_50": {
            "direction": "high_better",
            "hypothesis": "Higher flow z-score improves follow-through quality.",
        },
        "ec_flow_persistence_20": {
            "direction": "high_better",
            "hypothesis": "More persistent flow sign improves continuation reliability.",
        },
        "ec_abs_dmid_ticks_10": {
            "direction": "low_better",
            "hypothesis": "Lower absolute dmid avoids overextended entries.",
        },
        "ec_abs_dmid_accel_10": {
            "direction": "low_better",
            "hypothesis": "Lower absolute dmid acceleration avoids unstable impulses.",
        },
        "ec_spread_regime_pct_100": {
            "direction": "low_better",
            "hypothesis": "Lower spread percentile reduces friction and slippage tax.",
        },
        "ec_depth_imbalance_aligned": {
            "direction": "high_better",
            "hypothesis": "Depth imbalance aligned with side should improve entry quality.",
        },
        "ec_depth_replenish_rate_20": {
            "direction": "high_better",
            "hypothesis": "Higher replenishment rate suggests stronger local liquidity support.",
        },
        "ec_entry_toxicity_3": {
            "direction": "low_better",
            "hypothesis": "Lower immediate adverse move should map to better outcomes.",
        },
    }

    per: Dict[str, Dict[str, object]] = {
        k: {
            "runs_scored": 0,
            "coverage_rows": 0,
            "total_rows": 0,
            "q_spreads": [],
            "top_evs": [],
            "overall_evs": [],
            "pass_count": 0,
        }
        for k in candidates
    }

    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "pnl_ticks" not in df.columns:
            continue
        n = len(df)
        for name, meta in candidates.items():
            if name == "ec_abs_dmid_ticks_10":
                sig = _coerce_series(df, "ec_dmid_ticks_10").abs()
            elif name == "ec_abs_dmid_accel_10":
                sig = _coerce_series(df, "ec_dmid_accel_10").abs()
            elif name == "ec_depth_imbalance_aligned":
                sig = _aligned_depth_imbalance(df)
            else:
                sig = _coerce_series(df, name)
            pr = per[name]
            pr["total_rows"] = int(pr["total_rows"]) + n
            pr["coverage_rows"] = int(pr["coverage_rows"]) + int(sig.notna().sum())
            sc = _score_candidate(df, sig, str(meta["direction"]), min_rows=args.min_rows)
            if sc["q_spread_ev"] is None:
                continue
            pr["runs_scored"] = int(pr["runs_scored"]) + 1
            pr["q_spreads"].append(float(sc["q_spread_ev"]))
            pr["top_evs"].append(float(sc["top_q_ev"]))
            pr["overall_evs"].append(float(sc["overall_ev"]))
            if float(sc["q_spread_ev"]) > args.min_q_spread:
                pr["pass_count"] = int(pr["pass_count"]) + 1

    summary = []
    for name, meta in candidates.items():
        pr = per[name]
        runs = int(pr["runs_scored"])
        total_rows = int(pr["total_rows"])
        coverage = float(pr["coverage_rows"]) / total_rows if total_rows else 0.0
        q_spreads = np.array(pr["q_spreads"], dtype=float) if pr["q_spreads"] else np.array([])
        top_evs = np.array(pr["top_evs"], dtype=float) if pr["top_evs"] else np.array([])
        pass_rate = (float(pr["pass_count"]) / runs) if runs else 0.0
        med_q = float(np.median(q_spreads)) if q_spreads.size else None
        med_top = float(np.median(top_evs)) if top_evs.size else None
        promote = bool(
            runs >= args.min_runs
            and coverage >= args.min_coverage
            and pass_rate >= args.min_pass_rate
            and (med_q is not None and med_q > 0.0)
        )
        summary.append(
            {
                "candidate": name,
                "direction": meta["direction"],
                "hypothesis": meta["hypothesis"],
                "coverage_ratio": round(coverage, 4),
                "runs_scored": runs,
                "median_q_spread_ev_lift_ticks": None if med_q is None else round(med_q, 4),
                "median_top_quartile_ev_ticks": None if med_top is None else round(med_top, 4),
                "pass_rate_q_spread_gt_threshold": round(pass_rate, 4),
                "promote_for_strategy_hypothesis": promote,
            }
        )

    # Session phase is categorical; report phase means.
    phase_rows = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "pnl_ticks" not in df.columns or "ec_session_phase" not in df.columns:
            continue
        d = df[["ec_session_phase", "pnl_ticks"]].copy()
        d["pnl_ticks"] = pd.to_numeric(d["pnl_ticks"], errors="coerce")
        d["ec_session_phase"] = d["ec_session_phase"].astype(str)
        d = d[d["pnl_ticks"].notna() & d["ec_session_phase"].ne("")]
        if d.empty:
            continue
        g = d.groupby("ec_session_phase")["pnl_ticks"].mean()
        phase_rows.append({k: float(v) for k, v in g.items()})
    phase_summary = {}
    if phase_rows:
        keys = sorted({k for row in phase_rows for k in row.keys()})
        for k in keys:
            vals = [row[k] for row in phase_rows if k in row]
            phase_summary[k] = {
                "runs": len(vals),
                "median_ev_ticks": float(np.median(vals)),
                "mean_ev_ticks": float(np.mean(vals)),
            }

    summary = sorted(
        summary,
        key=lambda x: (
            x["promote_for_strategy_hypothesis"],
            x["median_q_spread_ev_lift_ticks"] if x["median_q_spread_ev_lift_ticks"] is not None else -999.0,
        ),
        reverse=True,
    )

    out = {
        "files_scanned": len(files),
        "criteria": {
            "min_rows_per_run": args.min_rows,
            "min_q_spread_for_pass": args.min_q_spread,
            "min_runs": args.min_runs,
            "min_coverage": args.min_coverage,
            "min_pass_rate": args.min_pass_rate,
        },
        "summary": summary,
        "session_phase_summary": phase_summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    for row in summary:
        print(
            f"{row['candidate']}: promote={row['promote_for_strategy_hypothesis']} "
            f"runs={row['runs_scored']} coverage={row['coverage_ratio']} "
            f"median_lift={row['median_q_spread_ev_lift_ticks']} "
            f"median_top_ev={row['median_top_quartile_ev_ticks']}"
        )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature promotion scoring for trades_gated.csv artifacts.")
    p.add_argument("--root", default="artifacts/experiment_orchestrator", help="Artifact root directory.")
    p.add_argument(
        "--pattern",
        default="**/trades_gated.csv",
        help="Glob pattern under root (recursive).",
    )
    p.add_argument("--out", default="artifacts/feature_promotion_report.json")
    p.add_argument("--min-rows", type=int, default=40)
    p.add_argument("--min-q-spread", type=float, default=0.25)
    p.add_argument("--min-runs", type=int, default=6)
    p.add_argument("--min-coverage", type=float, default=0.6)
    p.add_argument("--min-pass-rate", type=float, default=0.55)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
