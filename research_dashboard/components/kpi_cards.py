from __future__ import annotations

from research_dashboard.services import formatting


def render_kpi_cards(st, data: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Trades", data.get("n_trades", 0))
    cols[1].metric("EV / Trade", formatting.fmt_ticks(data.get("ev_per_trade")))
    cols[2].metric("Win Rate", formatting.fmt_pct(data.get("win_rate")))
    cols[3].metric("Profit Factor", formatting.fmt_num(data.get("profit_factor")))

    cols2 = st.columns(4)
    cols2[0].metric("Max DD", formatting.fmt_ticks(data.get("max_drawdown_ticks")))
    cols2[1].metric("Top1 Share", formatting.fmt_pct(data.get("top1_share")))
    cols2[2].metric("Top3 Share", formatting.fmt_pct(data.get("top3_share")))
    cols2[3].metric("Top5 Share", formatting.fmt_pct(data.get("top5_share")))
