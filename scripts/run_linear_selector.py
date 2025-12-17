"""
Train a lightweight ridge selector on saved predictions to replace hand-tuned gates.

Inputs:
  - artifacts/hgb_test_predictions.csv (produced by run_train_model.py)

Outputs:
  - artifacts/selector_ridge.json   # model params and chosen threshold
  - artifacts/selector_scores.csv   # per-row score and pass flags for reference

Usage:
  python scripts/run_linear_selector.py

Env overrides:
  SELECTOR_PREDS_PATH   path to preds CSV (default: artifacts/hgb_test_predictions.csv)
  SELECTOR_OUTPUT_CSV   path for scores CSV (default: artifacts/selector_scores.csv)
  SELECTOR_OUTPUT_JSON  path for params JSON (default: artifacts/selector_ridge.json)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default locations
PREDS_PATH = Path(os.environ.get("SELECTOR_PREDS_PATH", PROJECT_ROOT / "artifacts" / "hgb_test_predictions.csv"))
OUTPUT_CSV = Path(os.environ.get("SELECTOR_OUTPUT_CSV", PROJECT_ROOT / "artifacts" / "selector_scores.csv"))
OUTPUT_JSON = Path(os.environ.get("SELECTOR_OUTPUT_JSON", PROJECT_ROOT / "artifacts" / "selector_ridge.json"))
FIXED_TAU = os.environ.get("SELECTOR_FIXED_TAU")

# Features to use if present in the preds file
BASE_FEATURES = [
    "dir_conf",
    "ev_combined",
    "expected_value_dir",
    "mag_pred_p75",
    "spread",
    "sweep_cost_buy1",
    "sweep_cost_sell1",
    "depth_imbalance_top5",
    "book_slope_top5",
    "order_book_imbalance",
    "sweep_mag",
]


def _build_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    df = pd.read_csv(path)
    # Drop obvious index columns if present
    for col in ["Unnamed: 0", "index"]:
        if col in df:
            df = df.drop(columns=[col])

    # Dir confidence: max prob across classes (if probs exist)
    if {"proba_1", "proba_-1"}.issubset(df.columns):
        df["dir_conf"] = df[["proba_1", "proba_-1"]].max(axis=1)
    elif "dir_conf" not in df.columns:
        raise ValueError("dir_conf/proba columns not found in preds CSV.")

    # Sweep magnitude proxy
    if {"sweep_cost_buy1", "sweep_cost_sell1"}.issubset(df.columns):
        df["sweep_mag"] = df[["sweep_cost_buy1", "sweep_cost_sell1"]].abs().max(axis=1)
    else:
        df["sweep_mag"] = 0.0

    # Target: realized EV_net if present, else fall back to ev_combined
    if "ev_net" in df.columns:
        df["target"] = df["ev_net"]
    elif "ev_combined" in df.columns:
        df["target"] = df["ev_combined"]
    else:
        raise ValueError("No target found (ev_net or ev_combined) in preds CSV.")

    return df


def _select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    cols = [c for c in BASE_FEATURES if c in df.columns]
    if not cols:
        raise ValueError("No selector features available in preds CSV.")
    return df[cols], cols


def _fit_selector(X: pd.DataFrame, y: pd.Series, alphas: List[float]) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=alphas)),
        ]
    )
    model.fit(X, y)
    return model


def _eval_thresholds(scores: pd.Series, y_true: pd.Series, taus: List[float]) -> pd.DataFrame:
    rows = []
    for tau in taus:
        mask = scores > tau
        cnt = mask.sum()
        if cnt == 0:
            rows.append({"tau": tau, "count": 0, "mean": np.nan, "median": np.nan, "p05": np.nan, "p95": np.nan})
            continue
        sel = y_true[mask]
        rows.append(
            {
                "tau": tau,
                "count": int(cnt),
                "pass_rate": float(cnt) / float(len(y_true)),
                "mean": sel.mean(),
                "median": sel.median(),
                "p05": sel.quantile(0.05),
                "p95": sel.quantile(0.95),
            }
        )
    return pd.DataFrame(rows)


def main():
    df = _build_frame(PREDS_PATH)
    X, feature_cols = _select_features(df)
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    alpha_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.5, 1.0, 2.0, 5.0]
    model = _fit_selector(X_train, y_train, alpha_grid)

    # Predict on full set
    scores = pd.Series(model.predict(X), index=df.index, name="selector_score")
    df_out = df.copy()
    df_out["selector_score"] = scores

    # Threshold grid based on score quantiles (higher score = better)
    quantiles = [0.90, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999]
    tau_grid = [scores.quantile(q) for q in quantiles]

    metrics = _eval_thresholds(scores.loc[y_val.index], y_val, tau_grid)
    metrics_sorted = metrics.sort_values(["mean", "count"], ascending=[False, False])
    best = metrics_sorted.iloc[0]
    best_tau_auto = float(best["tau"])

    chosen_tau = best_tau_auto
    tau_source = "auto"
    if FIXED_TAU:
        try:
            chosen_tau = float(FIXED_TAU)
            tau_source = "env"
        except ValueError as exc:
            raise ValueError(f"Invalid SELECTOR_FIXED_TAU={FIXED_TAU}") from exc

    # Add pass mask for chosen_tau
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

    print(f"Fitted ridge selector (alpha={info['alpha_chosen']:.4g}) on {len(X_train)} train / {len(X_val)} val.")
    print(f"Best tau (by val mean target): {best_tau_auto:.6g}")
    print(f"Chosen tau ({tau_source}): {chosen_tau:.6g}")
    print("Validation metrics per tau:")
    print(metrics)
    print(f"Wrote scores to {OUTPUT_CSV}")
    print(f"Wrote selector params to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
