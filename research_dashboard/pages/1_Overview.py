from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from research_dashboard.components import equity_curve, kpi_cards, warnings_panel
from research_dashboard.services import metrics
from research_dashboard.services import ui as ui_state
from research_dashboard.services import loader


st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")

filters = ui_state.get_global_filters(st)
trades = loader.get_filtered_trade_view(filters)

rolling_window = st.number_input("Rolling Window (trades)", min_value=5, max_value=100, value=10, step=1)
show_gross = st.toggle("Show Gross PnL", value=False)

pnl_col = "ticks_pnl_gross" if show_gross else "ticks_pnl_net"
analysis = trades.copy()
analysis["analysis_pnl"] = pd.to_numeric(analysis[pnl_col], errors="coerce")
kpis = metrics.compute_core_metrics(analysis, pnl_col="analysis_pnl")
oos_ev = metrics.ev_per_trade(pd.to_numeric(analysis.loc[analysis["oos_flag"] == True, "analysis_pnl"], errors="coerce"))  # noqa: E712

col_left, col_right = st.columns([2, 1])
with col_left:
    kpi_cards.render_kpi_cards(st, kpis)
with col_right:
    roll = metrics.rolling_ev(pd.to_numeric(analysis["analysis_pnl"], errors="coerce"), window=int(rolling_window))
    latest_rolling = float(roll.dropna().iloc[-1]) if not roll.dropna().empty else None
    monthly = metrics.monthly_breakdown(analysis, pnl_col="analysis_pnl")
    warnings = warnings_panel.build_warnings(kpis, latest_rolling, monthly, oos_ev)
    warnings_panel.render_oos_tag(st, oos_ev)
    warnings_panel.render_warnings_panel(st, warnings)

plot_df = analysis[["entry_ts", "analysis_pnl"]].rename(columns={"analysis_pnl": "ticks_pnl_net"})

st.plotly_chart(equity_curve.equity_curve_figure(plot_df), use_container_width=True)
st.plotly_chart(equity_curve.rolling_ev_figure(plot_df, window=int(rolling_window)), use_container_width=True)

bottom_left, bottom_mid, bottom_right = st.columns(3)
monthly = metrics.monthly_breakdown(analysis, pnl_col="analysis_pnl")
session = metrics.session_breakdown(analysis, pnl_col="analysis_pnl")

with bottom_left:
    if not monthly.empty:
        fig = px.bar(monthly, x="month", y="ev_per_trade", title="EV by Month", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly data for current filters.")

with bottom_mid:
    if not session.empty:
        fig = px.bar(session, x="session", y="ev_per_trade", title="EV by Session", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No session data for current filters.")

with bottom_right:
    if not session.empty:
        fig = px.bar(session, x="session", y="n", title="Trade Count by Session", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trade count data for current filters.")
