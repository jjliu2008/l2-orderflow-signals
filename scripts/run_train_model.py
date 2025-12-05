import os
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Allow running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, ensure_multiindex, make_feature_matrix
from src.labels import make_labels


def main():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    print(f"Loading data from {raw_dir} ...")
    df = ensure_multiindex(load_all_raw_data(raw_dir))

    #print("Computing features ...")
    df_feat = add_basic_features(df)
    X_df, _ = make_feature_matrix(df_feat)

   # print("Building labels ...")
    y, _ = make_labels(df_feat, price_col="mid", horizon=1, neutral_threshold=0.0)

    # Align features and labels on shared MultiIndex and name them for clarity
    common_idx = X_df.index.intersection(y.index)
    x_small = X_df.loc[common_idx]
    y_small = y.loc[common_idx]

    #print(f"Total samples: {len(y_small)}, features: {x_small.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        x_small, y_small, test_size=0.2, random_state=42, stratify=y_small
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200, n_jobs=-1, solver="lbfgs"),
    )

    #print("Training model ...")
    model.fit(X_train, y_train)

    #print("Evaluating ...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nPredicted class order:", model.classes_)
    print("Probability preview (first 5 rows):")
    print(y_proba[:5])

if __name__ == "__main__":
    main()
