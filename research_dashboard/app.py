from __future__ import annotations

import streamlit as st

from research_dashboard.services import loader


st.set_page_config(
    page_title="Strategy Research Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Strategy Research Dashboard")
st.caption("Local research control panel for strategy health, regime behavior, outlier risk, replay, and path risk.")

try:
    loader.ensure_materialized_data(force_rebuild=False)
    trades = loader.load_trades()
    st.success(f"Data ready. Loaded {len(trades):,} trades from local Parquet.")
except Exception as exc:
    st.error(f"Data bootstrap failed: {exc}")
    st.stop()

st.markdown(
    """
Use the pages in the left sidebar:

1. `Overview`
2. `Regime Analysis`
3. `Filter Comparison`
4. `Outlier Kill Test`
5. `Trade Replay`
6. `Monte Carlo`
"""
)
