import os
import sys

# Allow running the script directly by adding project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all_raw_data
from src.feature_engineering import add_basic_features


df = load_all_raw_data("data/raw")
df_feat = add_basic_features(df)

print(df_feat[["mid", "spread", "top_bid_depth", "top_ask_depth", "order_book_imbalance"]].head())
