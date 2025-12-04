import os
import sys

# Allow running the script directly by adding project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features, make_feature_matrix


df = load_all_raw_data("data/raw")
df_feat = add_basic_features(df)
X, idx = make_feature_matrix(df_feat)

print(df_feat[["mid", "spread", "top_bid_depth", "top_ask_depth", "order_book_imbalance"]].head())
print("\nFeature matrix preview:")
print(X.head())
print("X shape:", X.shape)
