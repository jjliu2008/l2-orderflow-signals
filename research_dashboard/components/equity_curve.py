from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from research_dashboard.services import metrics


def _coerce_numeric_series(values) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] == 0:
            return pd.Series(dtype=float)
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce")


def equity_curve_figure(trades_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(title="Cumulative PnL", template="plotly_dark")
        return fig
    df = trades_df.sort_values("entry_ts").copy()
    df["cum_ticks"] = _coerce_numeric_series(df["ticks_pnl_net"]).fillna(0.0).cumsum()
    fig.add_trace(go.Scatter(x=df["entry_ts"], y=df["cum_ticks"], mode="lines", name="Equity (ticks)"))
    fig.update_layout(title="Cumulative PnL", xaxis_title="Entry Time", yaxis_title="Ticks", template="plotly_dark")
    return fig


def rolling_ev_figure(trades_df: pd.DataFrame, window: int = 10) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(title=f"Rolling {window}-Trade EV", template="plotly_dark")
        return fig
    df = trades_df.sort_values("entry_ts").copy()
    roll = metrics.rolling_ev(_coerce_numeric_series(df["ticks_pnl_net"]), window=window)
    fig.add_trace(go.Scatter(x=df["entry_ts"], y=roll, mode="lines", name=f"Rolling EV ({window})"))
    fig.update_layout(title=f"Rolling {window}-Trade EV", xaxis_title="Entry Time", yaxis_title="Ticks", template="plotly_dark")
    return fig


def rolling_dd_figure(trades_df: pd.DataFrame, window: int = 10) -> go.Figure:
    fig = go.Figure()
    if trades_df.empty:
        fig.update_layout(title=f"Rolling {window}-Trade Drawdown", template="plotly_dark")
        return fig
    df = trades_df.sort_values("entry_ts").copy()
    dd = metrics.rolling_drawdown(_coerce_numeric_series(df["ticks_pnl_net"]), window=window)
    fig.add_trace(go.Scatter(x=df["entry_ts"], y=dd, mode="lines", name=f"Rolling DD ({window})"))
    fig.update_layout(title=f"Rolling {window}-Trade Drawdown", xaxis_title="Entry Time", yaxis_title="Ticks", template="plotly_dark")
    return fig
