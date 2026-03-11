from __future__ import annotations

import pandas as pd
import plotly.express as px


def feature_boxplot(merged_df: pd.DataFrame, feature_col: str):
    if merged_df.empty or feature_col not in merged_df.columns:
        return px.box(title=f"{feature_col}: winners vs losers")
    df = merged_df.copy()
    df["outcome"] = (pd.to_numeric(df["ticks_pnl_net"], errors="coerce") > 0).map({True: "winner", False: "loser"})
    fig = px.box(
        df,
        x="outcome",
        y=feature_col,
        color="outcome",
        points="outliers",
        title=f"{feature_col}: winners vs losers",
        template="plotly_dark",
    )
    return fig

