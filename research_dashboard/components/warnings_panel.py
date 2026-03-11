from __future__ import annotations

from research_dashboard.services import formatting


def build_warnings(
    kpis: dict,
    latest_rolling_ev: float | None,
    monthly_df,
    oos_ev: float | None,
) -> list[str]:
    warnings = []
    if (kpis.get("n_trades") or 0) < 20:
        warnings.append("Low sample size: fewer than 20 trades.")
    if (kpis.get("top3_share") or 0) > 0.80:
        warnings.append("High concentration: top3 winners > 80% of total PnL.")
    if oos_ev is not None and oos_ev <= 0:
        warnings.append("OOS EV is non-positive.")
    if latest_rolling_ev is not None and latest_rolling_ev < 0:
        warnings.append("Latest rolling EV is negative.")
    if monthly_df is not None and not monthly_df.empty and (monthly_df["ev_per_trade"] <= 0).any():
        warnings.append("At least one month has non-positive EV.")
    if (kpis.get("profit_factor") or 0) < 1.20:
        warnings.append("Profit factor is below 1.20.")
    return warnings


def render_warnings_panel(st, warnings: list[str]) -> None:
    st.subheader("Warnings")
    if not warnings:
        st.success("No warning flags on current subset.")
        return
    for w in warnings:
        st.warning(w)


def render_oos_tag(st, oos_ev: float | None) -> None:
    if oos_ev is None:
        st.info("OOS EV: -")
        return
    text = f"OOS EV: {formatting.fmt_ticks(oos_ev)}"
    if oos_ev > 0:
        st.success(text)
    else:
        st.error(text)
