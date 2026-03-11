from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from research_dashboard.services import metrics
from research_dashboard.services import ui as ui_state
from research_dashboard.services import loader


st.set_page_config(page_title="Filter Comparison", layout="wide")
st.title("Filter Comparison")

filters = ui_state.get_global_filters(st)
trades = loader.get_filtered_trade_view(filters)

if trades.empty:
    st.warning("No trades for current filters.")
    st.stop()

all_variants = sorted(trades["variant_name"].dropna().unique().tolist())
selected_variants = st.multiselect("Variants to compare", options=all_variants, default=all_variants)
if not selected_variants:
    st.info("Select at least one variant.")
    st.stop()

subset = trades[trades["variant_name"].isin(selected_variants)].copy()

rows = []
for v, g in subset.groupby("variant_name"):
    core = metrics.compute_core_metrics(g, pnl_col="ticks_pnl_net")
    rows.append(
        {
            "variant_name": v,
            "n_trades": core["n_trades"],
            "ev_net": core["ev_per_trade"],
            "wr": core["win_rate"],
            "pf": core["profit_factor"],
            "max_dd_ticks": core["max_drawdown_ticks"],
            "top1_share": core["top1_share"],
            "top3_share": core["top3_share"],
            "top5_share": core["top5_share"],
            "oos_ev": metrics.ev_per_trade(g.loc[g["oos_flag"] == True, "ticks_pnl_net"]),  # noqa: E712
        }
    )

comp = pd.DataFrame(rows).sort_values("ev_net", ascending=False)
st.subheader("Comparison Table")
st.dataframe(comp, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(
        px.bar(comp, x="variant_name", y="ev_net", title="EV by Variant", template="plotly_dark"),
        use_container_width=True,
    )
with right:
    st.plotly_chart(
        px.bar(comp, x="variant_name", y="n_trades", title="Trade Count by Variant", template="plotly_dark"),
        use_container_width=True,
    )

left2, right2 = st.columns(2)
with left2:
    melt = comp.melt(
        id_vars=["variant_name"],
        value_vars=["top1_share", "top3_share", "top5_share"],
        var_name="share_type",
        value_name="share",
    )
    st.plotly_chart(
        px.bar(melt, x="variant_name", y="share", color="share_type", barmode="group", title="Concentration Comparison", template="plotly_dark"),
        use_container_width=True,
    )
with right2:
    ev_order = comp.sort_values("ev_net", ascending=False).copy()
    ev_order["ev_delta"] = ev_order["ev_net"] - ev_order["ev_net"].iloc[0]
    st.plotly_chart(
        px.bar(ev_order, x="variant_name", y="ev_delta", title="EV Delta vs Best Variant", template="plotly_dark"),
        use_container_width=True,
    )
