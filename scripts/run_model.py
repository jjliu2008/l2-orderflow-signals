import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, add_orderflow_features, ensure_multiindex, make_feature_matrix


def _choose_data_dir() -> Path:
    default_dir = os.environ.get("INFER_DATA_DIR", str(PROJECT_ROOT / "data" / "raw"))
    choice = input(f"Inference data directory [{default_dir}]: ").strip() or default_dir
    data_dir = Path(choice).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return data_dir


def main():
    # Defaults point to current artifacts; can override via env.
    model_path = Path(os.environ.get("INFER_MODEL_PATH", PROJECT_ROOT / "artifacts" / "hgb_model.joblib"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    data_dir = _choose_data_dir()

    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading data from: {data_dir}")
    df = ensure_multiindex(load_all_raw_data(data_dir))
    df_feat = add_basic_features(df)
    df_feat = add_orderflow_features(df_feat)
    X_df, _ = make_feature_matrix(df_feat, drop_na=False)

    # Align feature columns to the model if available
    if hasattr(model, "feature_names_in_"):
        missing = [c for c in model.feature_names_in_ if c not in X_df.columns]
        if missing:
            raise ValueError(f"Missing required features for model: {missing}")
        X = X_df[model.feature_names_in_]
    else:
        X = X_df

    print(f"Running inference on {len(X):,} rows ...")
    proba = model.predict_proba(X)
    preds = model.predict(X)

    classes = model.classes_.astype(float)
    expected = (proba * classes.reshape(1, -1)).sum(axis=1)

    out_df = pd.DataFrame(index=X.index)
    out_df["pred"] = preds
    out_df["expected_value"] = expected
    for i, cls in enumerate(model.classes_):
        out_df[f"proba_{int(cls)}"] = proba[:, i]

    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "inference_predictions.csv"
    out_df.reset_index().to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")
    # Percent breakdown by predicted class
    pred_counts = pd.Series(preds).value_counts(normalize=True).sort_index() * 100
    print("Prediction class distribution (%):", pred_counts.to_dict())
    print("Preview:")
    print(out_df.head())


if __name__ == "__main__":
    main()
