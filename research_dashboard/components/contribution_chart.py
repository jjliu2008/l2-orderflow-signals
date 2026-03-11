from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def contribution_bar(pnl_series: pd.Series, title: str = "Trade Contribution Shares") -> go.Figure:
    s = pd.to_numeric(pnl_series, errors="coerce").dropna().sort_values(ascending=False)
    fig = go.Figure()
    if s.empty:
        fig.update_layout(title=title, template="plotly_dark")
        return fig
    total = float(s.sum())
    top1 = float(s.head(1).sum())
    top3 = float(s.head(3).sum())
    top5 = float(s.head(5).sum())
    shares = [top1 / total if total > 0 else 0.0, top3 / total if total > 0 else 0.0, top5 / total if total > 0 else 0.0]
    fig.add_trace(go.Bar(x=["Top1", "Top3", "Top5"], y=shares, text=[f"{x:.1%}" for x in shares], textposition="outside"))
    fig.update_layout(title=title, yaxis_title="Share of Total PnL", template="plotly_dark")
    return fig


def pnl_histogram(pnl_series: pd.Series, title: str = "Trade Contribution Histogram") -> go.Figure:
    s = pd.to_numeric(pnl_series, errors="coerce").dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=s, nbinsx=40))
    fig.update_layout(title=title, xaxis_title="Net Ticks", yaxis_title="Count", template="plotly_dark")
    return fig

