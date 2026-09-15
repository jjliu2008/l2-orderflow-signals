#!/usr/bin/env python
"""Run the public pipeline end to end on deterministic synthetic quotes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import add_basic_features, add_orderflow_features
from src.labels import make_labels


FEATURES = [
    "spread",
    "order_book_imbalance",
    "ret_1",
    "ret_5",
    "ret_10",
    "volatility_20",
    "signed_volume",
    "cvd_20",
    "volume_zscore_20",
    "imbalance_persist",
    "rv_short",
]


def generate_quotes(rows: int = 1_500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-02T14:30:00Z", periods=rows, freq="s")
    latent_flow = rng.normal(size=rows)
    noise = rng.normal(scale=0.025, size=rows)
    mid = 5_000 + np.cumsum(0.012 * latent_flow + noise)
    spread = np.where(rng.random(rows) < 0.92, 0.25, 0.50)

    frame = pd.DataFrame(
        {
            "Symbol": "ES-SYNTH",
            "Time": times,
            "BidPrice1": mid - spread / 2,
            "AskPrice1": mid + spread / 2,
            "BidVolume1": rng.lognormal(3.1 + 0.10 * latent_flow, 0.45),
            "AskVolume1": rng.lognormal(3.1 - 0.10 * latent_flow, 0.45),
            "Volume": rng.lognormal(2.0 + 0.12 * np.abs(latent_flow), 0.55),
        }
    )
    return frame.set_index(["Symbol", "Time"]).sort_index()


def main() -> None:
    quotes = generate_quotes()
    featured = add_orderflow_features(add_basic_features(quotes))
    labels, valid_index = make_labels(featured, price_col="mid", horizon=5, neutral_threshold=0.000005)
    matrix = featured.loc[valid_index, FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    split = int(len(matrix) * 0.75)
    x_train, x_test = matrix.iloc[:split], matrix.iloc[split:]
    y_train, y_test = labels.iloc[:split], labels.iloc[split:]
    if y_train.nunique() < 2 or y_test.empty:
        raise RuntimeError("Synthetic smoke data did not produce a usable chronological split")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1_000, class_weight="balanced"))
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)

    print("Synthetic order-flow smoke test")
    print(f"rows={len(quotes)} features={matrix.shape[1]} train={len(x_train)} holdout={len(x_test)}")
    print(f"accuracy={accuracy_score(y_test, prediction):.3f}")
    print(f"balanced_accuracy={balanced_accuracy_score(y_test, prediction):.3f}")
    print("Smoke pipeline complete: synthetic data -> features -> labels -> model -> holdout metrics")


if __name__ == "__main__":
    main()
