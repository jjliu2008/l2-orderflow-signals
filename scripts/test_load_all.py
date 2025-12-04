import os
import sys

# Allow running the script directly by adding project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, make_feature_matrix
from src.labels import make_labels


df = load_all_raw_data("data/raw")
df_feat = add_basic_features(df)
X, idx = make_feature_matrix(df_feat)
y, y_idx = make_labels(df_feat, price_col="mid", horizon=1, neutral_threshold=0.0)

print(df_feat[["mid", "spread", "top_bid_depth", "top_ask_depth", "order_book_imbalance"]].head())
print("\nFeature matrix preview:")
print(X.head())
print("X shape:", X.shape)
print("\nLabels preview:")
print(y.head())
print("y length:", len(y))
