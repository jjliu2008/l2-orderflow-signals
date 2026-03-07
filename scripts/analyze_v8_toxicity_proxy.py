from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    x2 = x[mask]
    y2 = y[mask]
    pearson = float(x2.corr(y2, method="pearson"))
    spearman = float(x2.corr(y2, method="spearman"))
    return pearson, spearman, n


def _rate(df: pd.DataFrame, outcome_col: str) -> float:
    if df.empty:
        return float("nan")
    vals = pd.to_numeric(df[outcome_col], errors="coerce")
    if vals.notna().sum() == 0:
        return float("nan")
    return float(vals.mean())


def _best_threshold_low(
    x: pd.Series, y: pd.Series, min_selected: int = 10
) -> Dict[str, float | int]:
    mask = x.notna() & y.notna()
    x2 = x[mask]
    y2 = y[mask]
    if len(x2) == 0:
        return {"threshold": float("nan"), "selected": 0, "rate": float("nan")}
    qs = np.linspace(0.1, 0.9, 17)
    best_thr = float("nan")
    best_rate = float("-inf")
    best_n = 0
    for q in qs:
        thr = float(np.nanquantile(x2, q))
        sel = y2[x2 <= thr]
        n = int(sel.shape[0])
        if n < min_selected:
            continue
        rate = float(np.nanmean(sel))
        if rate > best_rate or (rate == best_rate and n > best_n):
            best_rate = rate
            best_thr = thr
            best_n = n
    if best_n == 0:
        return {"threshold": float("nan"), "selected": 0, "rate": float("nan")}
    return {"threshold": best_thr, "selected": best_n, "rate": best_rate}


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate v8 causal toxicity proxy.")
    ap.add_argument("--trades", required=True, help="Path to trades CSV.")
    ap.add_argument(
        "--out",
        default="artifacts/v8_toxicity_proxy_validation_20260305.json",
        help="Output report JSON path.",
    )
    args = ap.parse_args()

    trades_path = Path(args.trades)
    if not trades_path.exists():
        raise FileNotFoundError(trades_path)

    df = pd.read_csv(trades_path)
    if "strategy" in df.columns:
        df = df[df["strategy"].astype(str).str.lower() == "gated"].copy()
    if "entry_family" in df.columns:
        df = df[df["entry_family"].astype(str) == "PFLFT_v8"].copy()

    if df.empty:
        raise RuntimeError("No gated PFLFT_v8 rows found in trades file.")

    side = df["side"].astype(str).str.lower() if "side" in df.columns else pd.Series("", index=df.index)
    side_sign = np.where(side == "long", 1.0, np.where(side == "short", -1.0, np.nan))
    side_sign_s = pd.Series(side_sign, index=df.index, dtype=float)

    flow_sum = _to_num(df, "pflft_flow_sum_signed")
    flow_int = _to_num(df, "pflft_flow_intensity")
    aligned_flow = flow_sum * side_sign_s
    opp_mag = (flow_int - aligned_flow) / 2.0
    opp_mag = np.clip(opp_mag, 0.0, None)
    opp_flow_frac_3_derived = opp_mag / np.clip(flow_int, 1e-9, None)

    comp_opp = _to_num(df, "pflft_v8_opp_flow_frac_3")
    if comp_opp.notna().sum() == 0:
        comp_opp = opp_flow_frac_3_derived

    comp_imb = _to_num(df, "pflft_v8_imb_mismatch_3")
    comp_thin = _to_num(df, "pflft_v8_thin_stress_3")
    comp_proxy = _to_num(df, "pflft_v8_toxicity_proxy_pre3")
    comp_stress = _to_num(df, "pflft_v8_stress_mean_3")
    comp_depth = _to_num(df, "pflft_v8_depth_mean_3")
    comp_imb_aligned = _to_num(df, "pflft_v8_imb_aligned_3")

    tox_future = _to_num(df, "ec_entry_toxicity_3")
    mfe = _to_num(df, "mfe_ticks")
    bars_to_plus1 = _to_num(df, "bars_to_plus1")
    follow_plus1 = (mfe >= 1.0).astype(float)
    follow_2bar = ((bars_to_plus1 >= 0) & (bars_to_plus1 <= 2)).astype(float)

    base_plus1_rate = float(np.nanmean(follow_plus1))
    base_2bar_rate = float(np.nanmean(follow_2bar))
    n = int(df.shape[0])

    comps: Dict[str, pd.Series] = {
        "opp_flow_frac_3": comp_opp,
        "imb_mismatch_3": comp_imb,
        "thin_stress_3": comp_thin,
        "toxicity_proxy_pre3": comp_proxy,
        "stress_mean_3": comp_stress,
        "depth_mean_3": comp_depth,
        "imb_aligned_3": comp_imb_aligned,
    }

    corr_report = []
    for name, s in comps.items():
        pearson, spearman, n_pair = _corr(s, tox_future)
        corr_report.append(
            {
                "feature": name,
                "pearson_vs_ec_entry_toxicity_3": pearson,
                "spearman_vs_ec_entry_toxicity_3": spearman,
                "n_pairs": n_pair,
            }
        )

    min_sel = max(5, int(np.ceil(n * 0.15)))
    threshold_candidates = {}
    for name in ["opp_flow_frac_3", "imb_mismatch_3", "thin_stress_3", "toxicity_proxy_pre3"]:
        s = comps[name]
        best = _best_threshold_low(s, follow_plus1, min_selected=min_sel)
        if np.isfinite(best["threshold"]):
            thr = float(best["threshold"])
            selected = df[s <= thr]
            threshold_candidates[name] = {
                "direction": "<=",
                "threshold": thr,
                "n_selected": int(selected.shape[0]),
                "follow_plus1_rate_selected": _rate(selected.assign(follow_plus1=follow_plus1.loc[selected.index]), "follow_plus1"),
                "follow_2bar_rate_selected": _rate(selected.assign(follow_2bar=follow_2bar.loc[selected.index]), "follow_2bar"),
                "base_follow_plus1_rate": base_plus1_rate,
                "base_follow_2bar_rate": base_2bar_rate,
            }
        else:
            threshold_candidates[name] = {
                "direction": "<=",
                "threshold": float("nan"),
                "n_selected": 0,
                "follow_plus1_rate_selected": float("nan"),
                "follow_2bar_rate_selected": float("nan"),
                "base_follow_plus1_rate": base_plus1_rate,
                "base_follow_2bar_rate": base_2bar_rate,
            }

    tox_cmp = {}
    if tox_future.notna().sum() > 0:
        tox_thr = 0.156
        sel_tox = df[tox_future <= tox_thr]
        sel_proxy = df[comp_proxy <= float(threshold_candidates["toxicity_proxy_pre3"]["threshold"])] if np.isfinite(
            threshold_candidates["toxicity_proxy_pre3"]["threshold"]
        ) else df.iloc[0:0]
        tox_cmp = {
            "future_toxicity_threshold": tox_thr,
            "future_toxicity_n_selected": int(sel_tox.shape[0]),
            "future_toxicity_follow_plus1_rate": _rate(sel_tox.assign(follow_plus1=follow_plus1.loc[sel_tox.index]), "follow_plus1"),
            "future_toxicity_follow_2bar_rate": _rate(sel_tox.assign(follow_2bar=follow_2bar.loc[sel_tox.index]), "follow_2bar"),
            "proxy_threshold": threshold_candidates["toxicity_proxy_pre3"]["threshold"],
            "proxy_n_selected": int(sel_proxy.shape[0]),
            "proxy_follow_plus1_rate": _rate(sel_proxy.assign(follow_plus1=follow_plus1.loc[sel_proxy.index]), "follow_plus1"),
            "proxy_follow_2bar_rate": _rate(sel_proxy.assign(follow_2bar=follow_2bar.loc[sel_proxy.index]), "follow_2bar"),
        }

    out = {
        "trades_csv": str(trades_path),
        "n_gated_v8_trades": n,
        "follow_metrics": {
            "follow_plus1_definition": "mfe_ticks >= 1.0",
            "follow_2bar_definition": "0 <= bars_to_plus1 <= 2",
            "base_follow_plus1_rate": base_plus1_rate,
            "base_follow_2bar_rate": base_2bar_rate,
        },
        "component_vs_future_toxicity": corr_report,
        "threshold_candidates": threshold_candidates,
        "proxy_vs_future_toxicity_comparison": tox_cmp,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

