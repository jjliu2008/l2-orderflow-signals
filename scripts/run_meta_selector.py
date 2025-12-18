"""
Meta selector (Option B): fit a lightweight model on saved preds + realized returns to learn a better gate.

Inputs:
  - artifacts/hgb_test_predictions.csv (must include Symbol, Time, model outputs)
  - artifacts/backtest_results.csv (must include Symbol, Time, return)

Outputs:
  - artifacts/selector_scores.csv (selector_score, selector_pass, Symbol, Time)
  - artifacts/selector_ridge.json (params/metrics, reused by backtest)

Usage:
  python scripts/run_meta_selector.py

Env overrides:
  META_PREDS_PATH       (default: artifacts/hgb_test_predictions.csv)
  META_BACKTEST_PATH    (default: artifacts/backtest_results.csv)
  SELECTOR_OUTPUT_CSV   (default: artifacts/selector_scores.csv)
  SELECTOR_OUTPUT_JSON  (default: artifacts/selector_ridge.json)
  SELECTOR_FIXED_TAU    (optional tau override)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDS_PATH = Path(os.environ.get("META_PREDS_PATH", PROJECT_ROOT / "artifacts" / "hgb_test_predictions.csv"))
BACKTEST_PATH = Path(os.environ.get("META_BACKTEST_PATH", PROJECT_ROOT / "artifacts" / "backtest_results.csv"))
OUTPUT_CSV = Path(os.environ.get("SELECTOR_OUTPUT_CSV", PROJECT_ROOT / "artifacts" / "selector_scores.csv"))
OUTPUT_JSON = Path(os.environ.get("SELECTOR_OUTPUT_JSON", PROJECT_ROOT / "artifacts" / "selector_ridge.json"))
FIXED_TAU = os.environ.get("SELECTOR_FIXED_TAU")

# Candidate features to pull if present
FEATURE_COLS = [
    "ev_combined",
    "ev_net",
    "expected_value_dir",
    "mag_pred_p75",
    "spread",
    "sweep_cost_buy1",
    "sweep_cost_sell1",
    "order_book_imbalance",
    "depth_imbalance_top5",
    "book_slope_top5",
    "dir_conf",
]


def load_frames() -> pd.DataFrame:
    if not PREDS_PATH.exists():
        raise FileNotFoundError(f"Preds file not found: {PREDS_PATH}")
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Backtest results not found: {BACKTEST_PATH}")

    preds = pd.read_csv(PREDS_PATH)
    if not {"Symbol", "Time"}.issubset(preds.columns):
        raise ValueError("Preds file must include Symbol and Time columns.")
    preds["Time"] = pd.to_datetime(preds["Time"]).dt.tz_localize(None)
    if {"proba_1", "proba_-1"}.issubset(preds.columns):
        preds["dir_conf"] = preds[["proba_1", "proba_-1"]].max(axis=1)

    bt = pd.read_csv(BACKTEST_PATH, parse_dates=["Time"])
    if not {"Symbol", "Time", "return"}.issubset(bt.columns):
        raise ValueError("Backtest file must include Symbol, Time, and return columns.")
    bt["Time"] = bt["Time"].dt.tz_localize(None)

    merged = preds.merge(bt[["Symbol", "Time", "return"]], on=["Symbol", "Time"], how="inner")
    if merged.empty:
        raise ValueError("No overlap between preds and backtest on Symbol/Time.")
    return merged


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLS if c in df.columns]
    if not cols:
        raise ValueError("No meta features available.")
    return df[cols], cols


def fit_model(X: pd.DataFrame, y: pd.Series, alphas: List[float]) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas)),
        ]
    )
    model.fit(X, y)
    return model


def eval_taus(scores: pd.Series, y_true: pd.Series, taus: List[float]) -> pd.DataFrame:
    rows = []
    for tau in taus:
        mask = scores > tau
        cnt = int(mask.sum())
        pass_rate = cnt / len(scores) if len(scores) else 0.0
        if cnt == 0:
            rows.append({"tau": tau, "count": cnt, "pass_rate": pass_rate, "mean": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan})
            continue
        sel = y_true[mask]
        rows.append(
            {
                "tau": tau,
                "count": cnt,
                "pass_rate": pass_rate,
                "mean": sel.mean(),
                "median": sel.median(),
                "p05": sel.quantile(0.05),
                "p95": sel.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def main():
    df = load_frames()
    X, feature_cols = select_features(df)
    y = df["return"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    alpha_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
    model = fit_model(X_train, y_train, alpha_grid)

    scores_all = pd.Series(model.predict(X), index=df.index, name="selector_score")

    # Tau grid from quantiles of scores_all
    quantiles = [0.90, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
    tau_grid = [scores_all.quantile(q) for q in quantiles]

    metrics = eval_taus(scores_all.loc[y_val.index], y_val, tau_grid)
    metrics_sorted = metrics.sort_values(["mean", "count"], ascending=[False, False])
    best_tau_auto = float(metrics_sorted.iloc[0]["tau"]) if not metrics_sorted.empty else 0.0

    chosen_tau = best_tau_auto
    tau_source = "auto"
    if FIXED_TAU:
        try:
            chosen_tau = float(FIXED_TAU)
            tau_source = "env"
        except ValueError as exc:
            raise ValueError(f"Invalid SELECTOR_FIXED_TAU={FIXED_TAU}") from exc

    df_out = df[["Symbol", "Time"]].copy()
    df_out["selector_score"] = scores_all
    df_out["selector_pass"] = df_out["selector_score"] > chosen_tau

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    ridge_step = model.named_steps["ridge"]
    scaler = model.named_steps["scaler"]
    info = {
        "features": feature_cols,
        "alphas": alpha_grid,
        "alpha_chosen": float(ridge_step.alpha_),
        "coef": dict(zip(feature_cols, ridge_step.coef_.tolist())),
        "intercept": float(ridge_step.intercept_),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "best_tau": best_tau_auto,
        "chosen_tau": chosen_tau,
        "tau_source": tau_source,
        "metrics": metrics.to_dict(orient="records"),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(info, indent=2))

    print(f"Fitted meta selector (alpha={info['alpha_chosen']:.4g}) on {len(X_train)} train / {len(X_val)} val.")
    print(f"Best tau (by val mean target): {best_tau_auto:.6g}")
    print(f"Chosen tau ({tau_source}): {chosen_tau:.6g}")
    print("Validation metrics per tau:")
    print(metrics_sorted)
    print(f"Wrote selector scores to {OUTPUT_CSV}")
    print(f"Wrote selector params to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
