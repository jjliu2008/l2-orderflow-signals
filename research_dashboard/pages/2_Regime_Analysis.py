from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from research_dashboard.components import feature_boxplots
from research_dashboard.services import aggregates
from research_dashboard.services import ui as ui_state
from research_dashboard.services import loader


st.set_page_config(page_title="Regime Analysis", layout="wide")
st.title("Regime Analysis")

filters = ui_state.get_global_filters(st)
merged = loader.get_filtered_trade_feature_view(filters)

feature_candidates = [
    "signedvol_sum30s",
    "order_book_imbalance",
    "vwap_dev_ticks",
    "rv_300s",
    "day_range_ticks",
    "day_ret_ticks",
]
feature_candidates = [c for c in feature_candidates if c in merged.columns]

bucket_feature = st.selectbox("Bucket Variable", options=feature_candidates, index=0 if feature_candidates else None)
bins = st.slider("Number of bins", min_value=3, max_value=8, value=5)
winners_only = st.toggle("Winners only", value=False)
losers_only = st.toggle("Losers only", value=False)

data = merged.copy()
if winners_only:
    data = data[data["ticks_pnl_net"] > 0]
elif losers_only:
    data = data[data["ticks_pnl_net"] <= 0]

row1 = st.columns(2)
with row1[0]:
    session_df = aggregates.grouped_ev(data, "session")
    if not session_df.empty:
        st.plotly_chart(
            px.bar(session_df, x="session", y="ev_per_trade", hover_data=["n", "win_rate"], title="EV by Session", template="plotly_dark"),
            use_container_width=True,
        )
with row1[1]:
    weekday_df = aggregates.grouped_ev(data, "weekday")
    if not weekday_df.empty:
        st.plotly_chart(
            px.bar(weekday_df, x="weekday", y="ev_per_trade", hover_data=["n", "win_rate"], title="EV by Weekday", template="plotly_dark"),
            use_container_width=True,
        )

row2 = st.columns(2)
with row2[0]:
    event_df = aggregates.grouped_ev(data.assign(event_bucket=data["is_event_day"].map({True: "event", False: "non-event"})), "event_bucket")
    if not event_df.empty:
        st.plotly_chart(
            px.bar(event_df, x="event_bucket", y="ev_per_trade", hover_data=["n", "win_rate"], title="EV by Event/Non-Event", template="plotly_dark"),
            use_container_width=True,
        )
with row2[1]:
    if bucket_feature:
        bucket_df = aggregates.feature_bucket_summary(data, bucket_feature, bins=bins)
        if not bucket_df.empty:
            st.plotly_chart(
                px.bar(bucket_df, x="bucket", y="ev_per_trade", hover_data=["n", "win_rate"], title=f"EV by {bucket_feature} bucket", template="plotly_dark"),
                use_container_width=True,
            )

st.subheader("Winners vs Losers Feature Distributions")
box_cols = st.columns(3)
for i, feat in enumerate(feature_candidates[:3]):
    with box_cols[i]:
        st.plotly_chart(feature_boxplots.feature_boxplot(data, feat), use_container_width=True)

if len(feature_candidates) > 3:
    box_cols2 = st.columns(min(3, len(feature_candidates) - 3))
    for i, feat in enumerate(feature_candidates[3:6]):
        with box_cols2[i]:
            st.plotly_chart(feature_boxplots.feature_boxplot(data, feat), use_container_width=True)

st.subheader("Bucket Summary Table")
if bucket_feature:
    st.dataframe(aggregates.feature_bucket_summary(data, bucket_feature, bins=bins), use_container_width=True, hide_index=True)
