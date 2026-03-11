from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from research_dashboard import config
from research_dashboard.services import monte_carlo, ui as ui_state
from research_dashboard.services import loader


st.set_page_config(page_title="Monte Carlo", layout="wide")
st.title("Monte Carlo")

filters = ui_state.get_global_filters(st)
trades = loader.get_filtered_trade_view(filters).sort_values("entry_ts")
if trades.empty:
    st.warning("No trades for current filters.")
    st.stop()

variants = sorted(trades["variant_name"].dropna().unique().tolist())
variant = st.selectbox("Variant", options=variants, index=0)
vdf = trades[trades["variant_name"] == variant].copy()

col1, col2, col3 = st.columns(3)
mc_type = col1.selectbox("Simulation Type", options=["bootstrap", "shuffle"], index=0)
num_paths = int(col2.number_input("Number of Paths", min_value=100, max_value=10000, value=config.DEFAULT_MONTE_CARLO_PATHS, step=100))
horizon = int(col3.number_input("Horizon (trades, 0=observed)", min_value=0, max_value=5000, value=0, step=10))
horizon = None if horizon == 0 else horizon

col4, col5, col6 = st.columns(3)
starting_capital = float(col4.number_input("Starting Capital ($)", min_value=0.0, value=config.DEFAULT_MC_STARTING_CAPITAL, step=100.0))
contract = col5.selectbox("Contract", options=["ES", "MES"], index=0 if filters.contract == "ES" else 1)
slippage_stress_ticks = float(col6.number_input("Slippage Stress (ticks)", min_value=0.0, value=0.0, step=0.05))

run = st.button("Run Monte Carlo", type="primary")
if not run and "mc_last_result" not in st.session_state:
    st.info("Set parameters and click `Run Monte Carlo`.")
    st.stop()

if run:
    pnl_ticks = pd.to_numeric(vdf["ticks_pnl_net"], errors="coerce").dropna().to_numpy()
    pnl_ticks = pnl_ticks - slippage_stress_ticks
    paths = monte_carlo.run_mc_cached(
        pnl_ticks,
        mc_type=mc_type,
        num_paths=num_paths,
        horizon=horizon,
        seed=42,
    )
    tick_value = config.DEFAULT_TICK_VALUE_BY_CONTRACT[contract]
    summary = monte_carlo.compute_mc_summary(
        paths,
        tick_value=tick_value,
        starting_capital=starting_capital,
        dd_threshold_currency=config.DEFAULT_MC_DD_THRESHOLD,
    )
    pct = monte_carlo.percentile_equity_paths(paths)
    max_dd_ticks = monte_carlo.compute_drawdowns(paths)
    drawdown_paths = paths - np.maximum.accumulate(paths, axis=1) if paths.size else np.zeros((0, 0))
    st.session_state["mc_last_result"] = {
        "paths": paths,
        "drawdown_paths": drawdown_paths,
        "summary": summary,
        "percentiles": pct,
        "max_dd_ticks": max_dd_ticks,
        "tick_value": tick_value,
        "variant": variant,
        "contract": contract,
        "mc_type": mc_type,
    }

res = st.session_state["mc_last_result"]
paths = res["paths"]
drawdown_paths = res["drawdown_paths"]
summary = res["summary"]
pct = res["percentiles"]
tick_value = res["tick_value"]
max_dd_ticks = res["max_dd_ticks"]

st.subheader("Summary")
st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)

final_pnl_currency = paths[:, -1] * tick_value if paths.size else np.array([])
max_dd_currency = max_dd_ticks * tick_value if max_dd_ticks.size else np.array([])

row = st.columns(2)
with row[0]:
    fig = px.histogram(final_pnl_currency, nbins=50, title="Final PnL Distribution ($)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
with row[1]:
    fig = px.histogram(max_dd_currency, nbins=50, title="Max Drawdown Distribution ($)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Percentile Equity Paths")
if not pct.empty:
    st.caption("This chart intentionally shows only 3 summary lines (P05/P50/P95). Use the sampled paths chart below for many individual paths.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pct["step"], y=pct["p05"] * tick_value, mode="lines", name="P05"))
    fig.add_trace(go.Scatter(x=pct["step"], y=pct["p50"] * tick_value, mode="lines", name="P50"))
    fig.add_trace(go.Scatter(x=pct["step"], y=pct["p95"] * tick_value, mode="lines", name="P95"))
    fig.update_layout(template="plotly_dark", xaxis_title="Trade Step", yaxis_title="PnL ($)")
    st.plotly_chart(fig, use_container_width=True)

if paths.size:
    max_sample = int(min(paths.shape[0], 200))
    sample_n = int(st.slider("Sample Individual Paths", min_value=3, max_value=max_sample, value=min(25, max_sample), step=1))
    sample_idx = np.linspace(0, paths.shape[0] - 1, num=sample_n, dtype=int)
    sample = paths[sample_idx] * tick_value
    x = np.arange(1, sample.shape[1] + 1)
    sample_fig = go.Figure()
    for i in range(sample.shape[0]):
        sample_fig.add_trace(
            go.Scatter(
                x=x,
                y=sample[i],
                mode="lines",
                line=dict(width=1),
                opacity=0.18,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    sample_fig.update_layout(
        template="plotly_dark",
        title=f"Sampled Individual Monte Carlo Paths (n={sample_n})",
        xaxis_title="Trade Step",
        yaxis_title="PnL ($)",
    )
    st.plotly_chart(sample_fig, use_container_width=True)

st.subheader("Percentile Drawdown Paths")
if drawdown_paths.size:
    st.caption("Drawdown percentile lines are summaries; sample-path replay below shows a single full path over time.")
    dd_q = np.quantile(drawdown_paths, [0.05, 0.50, 0.95], axis=0)
    dd_df = pd.DataFrame(
        {
            "step": np.arange(1, drawdown_paths.shape[1] + 1),
            "p05": dd_q[0] * tick_value,
            "p50": dd_q[1] * tick_value,
            "p95": dd_q[2] * tick_value,
        }
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd_df["step"], y=dd_df["p05"], mode="lines", name="P05 DD"))
    fig.add_trace(go.Scatter(x=dd_df["step"], y=dd_df["p50"], mode="lines", name="P50 DD"))
    fig.add_trace(go.Scatter(x=dd_df["step"], y=dd_df["p95"], mode="lines", name="P95 DD"))
    fig.update_layout(template="plotly_dark", xaxis_title="Trade Step", yaxis_title="Drawdown ($)")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Simulation Replay")
if paths.size:
    sim_idx = st.number_input("Simulation Path #", min_value=1, max_value=int(paths.shape[0]), value=1, step=1)
    path_i = int(sim_idx) - 1

    step_key = f"mc_replay_step_{variant}_{res['contract']}_{res['mc_type']}"
    state_key = f"{step_key}_state"
    slider_key = f"{step_key}_slider"
    play_key = f"{step_key}_play"
    speed_key = f"{step_key}_speed"
    max_frame = int(paths.shape[1])

    if state_key not in st.session_state:
        st.session_state[state_key] = max_frame
    if slider_key not in st.session_state:
        st.session_state[slider_key] = int(st.session_state[state_key])

    state = int(st.session_state.get(slider_key, st.session_state[state_key]))
    state = min(max(1, state), max_frame)
    st.session_state[state_key] = state

    if st.session_state.get(play_key, False) and state < max_frame:
        next_state = state + 1
        st.session_state[state_key] = next_state
        st.session_state[slider_key] = next_state
        time.sleep(float(st.session_state.get(speed_key, 300)) / 1000.0)
        st.rerun()

    control_cols = st.columns(5)
    step_jump = control_cols[0].select_slider("Jump", options=[1, 5, 10, 25, 50], value=10, key=f"{step_key}_jump")
    if control_cols[1].button("<<", key=f"{step_key}_rw"):
        state = max(1, state - int(step_jump))
    if control_cols[2].button("<", key=f"{step_key}_b"):
        state = max(1, state - 1)
    if control_cols[3].button(">", key=f"{step_key}_f"):
        state = min(max_frame, state + 1)
    if control_cols[4].button(">>", key=f"{step_key}_ff"):
        state = min(max_frame, state + int(step_jump))

    st.session_state[state_key] = state
    st.session_state[slider_key] = state

    extra_cols = st.columns(2)
    extra_cols[0].toggle("Play", key=play_key)
    extra_cols[1].select_slider("Speed (ms)", options=[100, 200, 300, 500, 800, 1000], value=300, key=speed_key)

    st.slider("Replay Step", min_value=1, max_value=max_frame, key=slider_key)
    frame = int(st.session_state[slider_key])
    st.session_state[state_key] = frame

    if st.session_state.get(play_key, False) and frame >= max_frame:
        st.caption("Replay reached the end. Toggle `Play` off or move the replay step back.")

    eq_full = paths[path_i] * tick_value
    dd_full = drawdown_paths[path_i] * tick_value
    eq = eq_full[:frame]
    dd = dd_full[:frame]
    x = np.arange(1, frame + 1)
    current_eq = float(eq[-1]) if len(eq) else 0.0
    current_dd = float(dd[-1]) if len(dd) else 0.0

    st.caption(
        f"Path {path_i + 1}/{paths.shape[0]} | Step {frame}/{paths.shape[1]} | "
        f"Current PnL: ${current_eq:,.2f} | Current DD: ${current_dd:,.2f}"
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.4])
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(eq_full) + 1),
            y=eq_full,
            mode="lines",
            line=dict(color="rgba(160,160,160,0.25)"),
            name="Equity (full)",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=x, y=eq, mode="lines", name="Equity Replay"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(dd_full) + 1),
            y=dd_full,
            mode="lines",
            line=dict(color="rgba(160,160,160,0.25)"),
            name="Drawdown (full)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=x, y=dd, mode="lines", name="Drawdown Replay"), row=2, col=1)
    fig.add_vline(x=frame, line_color="yellow", line_dash="dash", opacity=0.7)
    fig.update_layout(template="plotly_dark", xaxis2_title="Trade Step", yaxis_title="PnL ($)", yaxis2_title="Drawdown ($)", height=750)
    st.plotly_chart(fig, use_container_width=True)
