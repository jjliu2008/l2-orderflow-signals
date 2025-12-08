import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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


def main():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    print(f"Loading data from {raw_dir} ...")
    df = ensure_multiindex(load_all_raw_data(raw_dir))

    #print("Computing features ...")
    df_feat = add_basic_features(df)
    df_feat = add_orderflow_features(df_feat)
    # Keep rows even if they contain NaNs; HGB can handle missing values.
    X_df, _ = make_feature_matrix(df_feat, drop_na=False)

   # print("Building labels ...")
    y, _ = make_labels(df_feat, price_col="mid", horizon=1, neutral_threshold=0.0)
    y = y.rename("label")

    # Align features and labels via merge on Symbol/Time to avoid duplicate-index issues.
    X_reset = X_df.reset_index()
    y_reset = y.reset_index()
    merged = X_reset.merge(y_reset, on=["Symbol", "Time"], how="inner")

    if merged.empty:
        raise ValueError("No overlapping rows between features and labels after merge.")

    feature_cols = [c for c in merged.columns if c not in {"Symbol", "Time", "label"}]
    x_small = merged[feature_cols]
    y_small = merged["label"]

    # Default cap to keep runs lightweight; override with TRAIN_MAX_ROWS=0 to disable.
    max_rows = int(os.environ.get("TRAIN_MAX_ROWS", "50000"))
    if max_rows > 0 and len(x_small) > max_rows:
        sampled_idx = x_small.index.to_series().sample(
            n=max_rows, random_state=42, replace=False
        ).index
        x_small = x_small.loc[sampled_idx]
        y_small = y_small.loc[sampled_idx]
        print(f"Sampled down to {len(x_small)} rows for training via TRAIN_MAX_ROWS={max_rows}")

    if len(y_small) == 0 or len(x_small) == 0:
        raise ValueError(
            "No training samples after feature/label alignment. "
            "Likely the raw data lacked usable fields or all rows were dropped. "
            "Check data/raw contents and feature construction."
        )

    #print(f"Total samples: {len(y_small)}, features: {x_small.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        x_small, y_small, test_size=0.2, random_state=42, stratify=y_small
    )

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
        model = HistGradientBoostingClassifier(
            max_iter=200,
            random_state=42,
            **params,
        )
        model.fit(X_train_sub, y_train_sub)
        val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, val_pred, average="macro")
        print(f"Params {params} -> val macro F1: {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best = params

    print(f"Best params: {best} (val macro F1={best_f1:.4f})")

    model = HistGradientBoostingClassifier(
        max_iter=300, random_state=42, **best
    )

    #print("Training model ...")
    model.fit(X_train, y_train)

    #print("Evaluating ...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_.astype(float)
    expected = (y_proba * classes.reshape(1, -1)).sum(axis=1)

    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nPredicted class order:", model.classes_)
    print("Probability preview (first 5 rows):")
    print(y_proba[:5])
    print("Expected value of prediction (mean over test set):", expected.mean())
    print("Expected value quantiles (5/50/95):", pd.Series(expected).quantile([0.05, 0.5, 0.95]).to_dict())

    save_artifacts = os.environ.get("TRAIN_SAVE_ARTIFACTS", "1").strip().lower() in {"1", "true", "yes", "y"}
    if save_artifacts:
        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        model_path = artifacts_dir / "hgb_model.joblib"
        joblib.dump(model, model_path)

        preds_df = pd.DataFrame(index=X_test.index)
        preds_df["y_true"] = y_test
        preds_df["y_pred"] = y_pred
        preds_df["expected_value"] = expected
        for i, cls in enumerate(model.classes_):
            preds_df[f"proba_{int(cls)}"] = y_proba[:, i]
        preds_path = artifacts_dir / "hgb_test_predictions.csv"
        preds_df.to_csv(preds_path)

        print(f"\nSaved model to {model_path}")
        print(f"Saved test predictions to {preds_path}")
    else:
        print("\nSkipping artifact save (TRAIN_SAVE_ARTIFACTS is falsy).")

if __name__ == "__main__":
    main()
