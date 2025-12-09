import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Allow running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data
from src.feature_engineering import (
    add_basic_features,
    add_orderflow_features,
    ensure_multiindex,
    make_feature_matrix,
)
from src.labels import make_labels


def forward_returns(mid: pd.Series, horizon: int = 1) -> pd.Series:
    future = mid.groupby(level=0).shift(-horizon)
    return (future - mid) / mid


def make_dir_label(fwd: pd.Series, neutral_band: float = 0.0) -> pd.Series:
    band = abs(neutral_band)
    def _lab(x: float) -> float:
        if x > band:
            return 1.0
        if x < -band:
            return -1.0
        return 0.0
    return fwd.apply(_lab).rename("label")


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    counts = y.value_counts()
    total = len(y)
    weights = y.map(lambda cls: total / (len(counts) * counts.get(cls, 1)))
    return weights.to_numpy()


def compute_realized_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Realized short-horizon volatility of log returns.
    """
    log_price = np.log(df["mid"].astype(float))
    log_ret = log_price.groupby(level=0).diff()
    vol = log_ret.groupby(level=0).transform(lambda s: s.rolling(window, min_periods=max(5, window // 2)).std())
    return vol.rename("realized_vol")


def _load_multiple_dirs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if p.exists():
            print(f"Loading data from {p} ...")
            frames.append(load_all_raw_data(p))
        else:
            print(f"Skipping missing data dir: {p}")
    if not frames:
        raise FileNotFoundError("No training data directories contained usable files.")
    # Preserve MultiIndex; load_all_raw_data already sets (Symbol, Time) index.
    combined = pd.concat(frames)
    return ensure_multiindex(combined)


def main():
    default_dirs = os.environ.get(
        "TRAIN_DATA_DIRS",
        ",".join(
            [
                str(PROJECT_ROOT / "data" / "raw"),
                str(PROJECT_ROOT / "data" / "processed"),
            ]
        ),
    )
    user_input = input(f"Training data directories (comma-separated) [{default_dirs}]: ").strip()
    dir_list = (
        [Path(d.strip()) for d in user_input.split(",") if d.strip()]
        if user_input
        else [Path(d.strip()) for d in default_dirs.split(",") if d.strip()]
    )

    df = _load_multiple_dirs(dir_list)
    # Drop duplicate Symbol/Time to avoid leaking duplicate samples into training (jsonl files can overlap time ranges).
    if df.index.duplicated().any():
        before = len(df)
        df = df.loc[~df.index.duplicated(keep="last")]
        print(f"Deduped Symbol/Time rows: {before - len(df)} removed, {len(df)} remaining.")

    #print("Computing features ...")
    df_feat = add_basic_features(df)
    df_feat = add_orderflow_features(df_feat)
    # Keep rows even if they contain NaNs; HGB can handle missing values.
    X_df, _ = make_feature_matrix(df_feat, drop_na=False)

    # Direction label from forward returns with neutral band
    neutral_band = float(os.environ.get("TRAIN_NEUTRAL_BAND", "0.0001"))
    fwd_ret = forward_returns(df_feat["mid"], horizon=1).rename("fwd_ret")
    dir_label = make_dir_label(fwd_ret, neutral_band=neutral_band)
    # Magnitude target: realized volatility over a short window
    vol_window = int(os.environ.get("TRAIN_VOL_WINDOW", "20"))
    realized_vol = compute_realized_vol(df_feat, window=vol_window)

    # Align features, direction label, and magnitude
    X_reset = X_df.reset_index()
    fwd_reset = fwd_ret.reset_index()
    dir_reset = dir_label.reset_index()
    vol_reset = realized_vol.reset_index()

    merged = X_reset.merge(fwd_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.merge(dir_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.merge(vol_reset, on=["Symbol", "Time"], how="inner")
    merged = merged.dropna(subset=["fwd_ret", "label", "realized_vol"])

    if merged.empty:
        raise ValueError("No overlapping rows between features and forward returns after merge.")

    feature_cols = [c for c in merged.columns if c not in {"Symbol", "Time", "label", "fwd_ret"}]
    x_small = merged[feature_cols]
    y_dir = merged["label"]
    y_mag = merged["realized_vol"].abs()

    # Default cap to keep runs lightweight; override with TRAIN_MAX_ROWS=0 to disable.
    max_rows = int(os.environ.get("TRAIN_MAX_ROWS", "50000"))
    if max_rows > 0 and len(x_small) > max_rows:
        sampled_idx = x_small.index.to_series().sample(
            n=max_rows, random_state=42, replace=False
        ).index
        x_small = x_small.loc[sampled_idx]
        y_dir = y_dir.loc[sampled_idx]
        y_mag = y_mag.loc[sampled_idx]
        print(f"Sampled down to {len(x_small)} rows for training via TRAIN_MAX_ROWS={max_rows}")

    if len(y_dir) == 0 or len(x_small) == 0:
        raise ValueError(
            "No training samples after feature/label alignment. "
            "Likely the raw data lacked usable fields or all rows were dropped. "
            "Check data/raw contents and feature construction."
        )

    #print(f"Total samples: {len(y_small)}, features: {x_small.shape[1]}")

    X_train, X_test, y_train, y_test, ymag_train, ymag_test = train_test_split(
        x_small, y_dir, y_mag, test_size=0.2, random_state=42, stratify=y_dir
    )
    sample_weight_train = compute_sample_weights(y_train)
    sample_weight_test = compute_sample_weights(y_test)

    # Simple hyperparameter sweep on a validation split (using a subset for speed)
    tune_frac = 0.3
    X_tune = X_train.sample(frac=tune_frac, random_state=42)
    y_tune = y_train.loc[X_tune.index]
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_tune, y_tune, test_size=0.2, random_state=42, stratify=y_tune
    )

    candidates = [
        {"learning_rate": 0.05, "max_depth": 6, "min_samples_leaf": 50},
        {"learning_rate": 0.1, "max_depth": 6, "min_samples_leaf": 50},
        {"learning_rate": 0.05, "max_depth": 8, "min_samples_leaf": 30},
    ]

    best = None
    best_f1 = -1.0
    for params in candidates:
        base = HistGradientBoostingClassifier(
            max_iter=200,
            random_state=42,
            class_weight="balanced",
            **params,
        )
        model = CalibratedClassifierCV(estimator=base, cv=3, method="isotonic")
        model.fit(X_train_sub, y_train_sub, sample_weight=compute_sample_weights(y_train_sub))
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        print(f"Params {params} -> val macro F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best = params

    print(f"Best params: {best} (val macro F1={best_f1:.4f})")

    dir_base = HistGradientBoostingClassifier(
        max_iter=300, random_state=42, class_weight="balanced", **best
    )
    dir_model = CalibratedClassifierCV(estimator=dir_base, cv=3, method="isotonic")
    dir_model.fit(X_train, y_train, sample_weight=sample_weight_train)

    # Magnitude quantile regressors on realized volatility
    mag_median = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.5,
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
    )
    mag_p75 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.75,
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=30,
        random_state=42,
    )
    mag_median.fit(X_train, ymag_train)
    mag_p75.fit(X_train, ymag_train)

    #print("Evaluating ...")
    y_pred = dir_model.predict(X_test)
    y_proba = dir_model.predict_proba(X_test)
    classes = dir_model.classes_.astype(float)
    expected = (y_proba * classes.reshape(1, -1)).sum(axis=1)
    mag_med = np.clip(mag_median.predict(X_test), 0, None)
    mag_hi = np.clip(mag_p75.predict(X_test), 0, None)
    mag_used = np.minimum(mag_hi, mag_med * 2)  # cap median by upper quantile (or 2x median to avoid zero hi)

    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nPredicted class order:", classes)
    print("Probability preview (first 5 rows):")
    print(y_proba[:5])
    print("Expected value of prediction (mean over test set):", expected.mean())
    print("Expected value quantiles (5/50/95):", pd.Series(expected).quantile([0.05, 0.5, 0.95]).to_dict())
    print("Magnitude median preview (first 5 rows):", mag_med[:5])
    print("Magnitude p75 preview (first 5 rows):", mag_hi[:5])

    save_artifacts = os.environ.get("TRAIN_SAVE_ARTIFACTS", "1").strip().lower() in {"1", "true", "yes", "y"}
    if save_artifacts:
        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        model_path = artifacts_dir / "hgb_combo_model.joblib"
        joblib.dump(
            {
                "direction_model": dir_model,
                "magnitude_median": mag_median,
                "magnitude_p75": mag_p75,
                "classes": classes,
                "feature_cols": feature_cols,
            },
            model_path,
        )

        preds_df = pd.DataFrame(index=X_test.index)
        preds_df["y_true"] = y_test
        preds_df["y_pred"] = y_pred
        preds_df["expected_value_dir"] = expected
        preds_df["mag_pred_med"] = mag_med
        preds_df["mag_pred_p75"] = mag_hi
        preds_df["ev_combined"] = mag_used * (y_proba[:, list(classes).index(1.0)] - y_proba[:, list(classes).index(-1.0)])
        preds_df["sample_weight"] = sample_weight_test
        for i, cls in enumerate(classes):
            preds_df[f"proba_{int(cls)}"] = y_proba[:, i]
        preds_path = artifacts_dir / "hgb_test_predictions.csv"
        preds_df.to_csv(preds_path)

        print(f"\nSaved combo model to {model_path}")
        print(f"Saved test predictions to {preds_path}")
    else:
        print("\nSkipping artifact save (TRAIN_SAVE_ARTIFACTS is falsy).")

if __name__ == "__main__":
    main()
