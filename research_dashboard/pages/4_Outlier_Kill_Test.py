from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from research_dashboard.components import contribution_chart, kpi_cards
from research_dashboard.services import metrics
from research_dashboard.services import ui as ui_state
from research_dashboard.services import loader


st.set_page_config(page_title="Outlier Kill Test", layout="wide")
st.title("Outlier Kill Test")

filters = ui_state.get_global_filters(st)
trades = loader.get_filtered_trade_view(filters)
if trades.empty:
    st.warning("No trades for current filters.")
    st.stop()

variants = sorted(trades["variant_name"].dropna().unique().tolist())
variant = st.selectbox("Variant", options=variants, index=0)
vdf = trades[trades["variant_name"] == variant].copy().sort_values("entry_ts")

col_a, col_b, col_c, col_d = st.columns(4)
remove_top_n = col_a.number_input("Remove Top Winners", min_value=0, max_value=20, value=0, step=1)
remove_bottom_n = col_b.number_input("Remove Top Losers", min_value=0, max_value=20, value=0, step=1)
winsor_pct = col_c.slider("Winsorize % (both tails)", min_value=0, max_value=20, value=0, step=1)
recompute = col_d.button("Recompute", type="primary")

if not recompute and "outlier_cached_variant" in st.session_state:
    if st.session_state["outlier_cached_variant"] == variant:
        before_df = st.session_state["outlier_before_df"]
        after_df = st.session_state["outlier_after_df"]
    else:
        before_df = vdf.copy()
        after_df = vdf.copy()
else:
    before_df = vdf.copy()
    after_df = vdf.copy()

    if winsor_pct > 0 and not after_df.empty:
        s = pd.to_numeric(after_df["ticks_pnl_net"], errors="coerce")
        lo = s.quantile(winsor_pct / 100.0)
        hi = s.quantile(1.0 - winsor_pct / 100.0)
        after_df["ticks_pnl_net"] = s.clip(lo, hi)

    if remove_top_n > 0 and len(after_df) > 0:
        top_idx = after_df["ticks_pnl_net"].nlargest(int(remove_top_n)).index
        after_df = after_df.drop(index=top_idx)
    if remove_bottom_n > 0 and len(after_df) > 0:
        bot_idx = after_df["ticks_pnl_net"].nsmallest(int(remove_bottom_n)).index
        after_df = after_df.drop(index=bot_idx)

    st.session_state["outlier_cached_variant"] = variant
    st.session_state["outlier_before_df"] = before_df
    st.session_state["outlier_after_df"] = after_df

st.subheader("Before / After KPIs")
left, right = st.columns(2)
with left:
    st.caption("Before")
    kpi_cards.render_kpi_cards(st, metrics.compute_core_metrics(before_df, pnl_col="ticks_pnl_net"))
with right:
    st.caption("After")
    kpi_cards.render_kpi_cards(st, metrics.compute_core_metrics(after_df, pnl_col="ticks_pnl_net"))

st.subheader("Before / After Equity Curves")
fig = go.Figure()
if not before_df.empty:
    b = before_df.sort_values("entry_ts").copy()
    b["cum"] = b["ticks_pnl_net"].cumsum()
    fig.add_trace(go.Scatter(x=b["entry_ts"], y=b["cum"], mode="lines", name="Before"))
if not after_df.empty:
    a = after_df.sort_values("entry_ts").copy()
    a["cum"] = a["ticks_pnl_net"].cumsum()
    fig.add_trace(go.Scatter(x=a["entry_ts"], y=a["cum"], mode="lines", name="After"))
fig.update_layout(template="plotly_dark", yaxis_title="Ticks", xaxis_title="Entry Time")
st.plotly_chart(fig, use_container_width=True)

row2 = st.columns(2)
with row2[0]:
    st.plotly_chart(contribution_chart.pnl_histogram(after_df["ticks_pnl_net"], "After: Trade Contribution Histogram"), use_container_width=True)
with row2[1]:
    st.plotly_chart(contribution_chart.contribution_bar(after_df["ticks_pnl_net"], "After: Concentration Shares"), use_container_width=True)

st.subheader("Outlier Removal Metrics")
table = []
for k in [0, 1, 3, 5]:
    m = metrics.outlier_removed_metrics(before_df["ticks_pnl_net"], remove_top_n=k, remove_bottom_n=0)
    table.append(
        {
            "remove_top_n": k,
            "n": m["n"],
            "ev_per_trade": m["ev_per_trade"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "max_drawdown_ticks": m["max_drawdown_ticks"],
            "top3_share": m["concentration"]["top3_share"] if m.get("concentration") else None,
        }
    )
st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
