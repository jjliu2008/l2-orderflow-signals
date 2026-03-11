from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from research_dashboard.services import replay, ui as ui_state
from research_dashboard.services import loader


st.set_page_config(page_title="Trade Replay", layout="wide")
st.title("Trade Replay")

filters = ui_state.get_global_filters(st)
trades = loader.get_filtered_trade_view(filters).sort_values("entry_ts")
replay_index = loader.load_replay_index()

if trades.empty:
    st.warning("No trades for current filters.")
    st.stop()

mode = st.radio("Trade list mode", options=["Top Winners", "Top Losers", "Random", "All"], horizontal=True)
if mode == "Top Winners":
    table = trades.nlargest(25, "ticks_pnl_net").copy()
elif mode == "Top Losers":
    table = trades.nsmallest(25, "ticks_pnl_net").copy()
elif mode == "Random":
    n = min(25, len(trades))
    table = trades.sample(n=n, random_state=42).copy()
else:
    table = trades.tail(100).copy()

st.subheader("Trade Selector")
st.dataframe(
    table[["trade_id", "date", "session", "direction", "ticks_pnl_net", "hold_seconds", "variant_name"]],
    use_container_width=True,
    hide_index=True,
)
trade_ids = table["trade_id"].tolist()
default_id = trade_ids[0] if trade_ids else trades.iloc[0]["trade_id"]
selected_trade_id = st.selectbox("Trade ID", options=trade_ids if trade_ids else trades["trade_id"].tolist(), index=0)

trade_row = trades[trades["trade_id"] == selected_trade_id].iloc[0]
idx_row = replay_index[replay_index["trade_id"] == selected_trade_id]
if idx_row.empty:
    st.error("Replay index row not found for selected trade.")
    st.stop()
idx_row = idx_row.iloc[0]

replay_df = replay.load_trade_replay(idx_row)
chart_df = replay.build_replay_chart_data(trade_row, replay_df)
if chart_df.empty:
    st.warning("Replay data not available for selected trade window.")
    st.stop()

entry_ts = pd.to_datetime(trade_row["entry_ts"], utc=True).tz_convert("America/New_York")
exit_ts = pd.to_datetime(trade_row["exit_ts"], utc=True).tz_convert("America/New_York")
entry_px = chart_df.loc[(chart_df["Time_et"] - entry_ts).abs().idxmin(), "mid"]
exit_px = chart_df.loc[(chart_df["Time_et"] - exit_ts).abs().idxmin(), "mid"]
side = 1 if str(trade_row["direction"]).lower() == "long" else -1

chart_df = chart_df.copy()
chart_df["mtm_ticks"] = side * (pd.to_numeric(chart_df["mid"], errors="coerce") - float(entry_px)) / 0.25
chart_df["trade_dd_ticks"] = chart_df["mtm_ticks"] - chart_df["mtm_ticks"].cummax()

st.subheader("Replay Controls")
replay_mode = st.toggle("Replay Mode (scrub)", value=True)
step_size = st.select_slider("Step Size", options=[1, 5, 10, 30, 60], value=10)

frame_key = f"replay_frame_{selected_trade_id}"
state_key = f"{frame_key}_state"
slider_key = f"{frame_key}_slider"
play_key = f"{frame_key}_play"
speed_key = f"{frame_key}_speed"
max_frame = len(chart_df)

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

ctrl = st.columns([1, 1, 1, 1, 1, 1.2, 1.8])
if ctrl[0].button("<<", key=f"{frame_key}_rew"):
    state = max(1, state - int(step_size))
if ctrl[1].button("<", key=f"{frame_key}_back"):
    state = max(1, state - 1)
if ctrl[2].button(">", key=f"{frame_key}_fwd"):
    state = min(max_frame, state + 1)
if ctrl[3].button(">>", key=f"{frame_key}_ffwd"):
    state = min(max_frame, state + int(step_size))
if ctrl[4].button("Reset", key=f"{frame_key}_reset"):
    state = 1

st.session_state[state_key] = state
st.session_state[slider_key] = state

ctrl[5].toggle("Play", key=play_key)
ctrl[6].select_slider("Speed (ms)", options=[100, 200, 300, 500, 800, 1000], value=300, key=speed_key)

st.slider("Replay Progress", min_value=1, max_value=max_frame, key=slider_key)
frame = int(st.session_state[slider_key])
st.session_state[state_key] = frame

if st.session_state.get(play_key, False) and frame >= max_frame:
    st.caption("Replay reached the end. Toggle `Play` off or click `Reset`.")

plot_df = chart_df.iloc[:frame].copy() if replay_mode else chart_df.copy()
current_ts = plot_df["Time_et"].iloc[-1]
current_mtm = float(plot_df["mtm_ticks"].iloc[-1])
current_dd = float(plot_df["trade_dd_ticks"].iloc[-1])
progress_pct = frame / len(chart_df)

st.caption(
    f"Frame {frame}/{len(chart_df)} ({progress_pct:.1%}) | "
    f"Current MTM: {current_mtm:.2f} ticks | Current Trade DD: {current_dd:.2f} ticks"
)

fig = make_subplots(
    rows=5,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.42, 0.14, 0.14, 0.14, 0.16],
)

if replay_mode:
    fig.add_trace(
        go.Scatter(
            x=chart_df["Time_et"],
            y=chart_df["mid"],
            mode="lines",
            name="Price (full)",
            line=dict(color="rgba(180,180,180,0.25)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
fig.add_trace(go.Scatter(x=plot_df["Time_et"], y=plot_df["mid"], mode="lines", name="Price"), row=1, col=1)
if "vwap" in plot_df.columns:
    if replay_mode:
        fig.add_trace(
            go.Scatter(
                x=chart_df["Time_et"],
                y=chart_df["vwap"],
                mode="lines",
                name="VWAP (full)",
                line=dict(color="rgba(80,160,255,0.20)"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(go.Scatter(x=plot_df["Time_et"], y=plot_df["vwap"], mode="lines", name="VWAP"), row=1, col=1)

fig.add_trace(go.Scatter(x=[entry_ts], y=[entry_px], mode="markers", name="Entry", marker=dict(size=10, symbol="triangle-up")), row=1, col=1)
fig.add_trace(go.Scatter(x=[exit_ts], y=[exit_px], mode="markers", name="Exit", marker=dict(size=10, symbol="x")), row=1, col=1)
fig.add_vline(x=current_ts, line_color="yellow", line_dash="dash", opacity=0.7)

if "signed_volume" in plot_df.columns:
    fig.add_trace(go.Bar(x=plot_df["Time_et"], y=plot_df["signed_volume"], name="Signed Vol"), row=2, col=1)
if "order_book_imbalance" in plot_df.columns:
    fig.add_trace(go.Scatter(x=plot_df["Time_et"], y=plot_df["order_book_imbalance"], mode="lines", name="OB Imbalance"), row=3, col=1)
if "spread_ticks" in plot_df.columns:
    fig.add_trace(go.Scatter(x=plot_df["Time_et"], y=plot_df["spread_ticks"], mode="lines", name="Spread (ticks)"), row=4, col=1)

# Per-trade mark-to-market drawdown replay.
fig.add_trace(go.Scatter(x=plot_df["Time_et"], y=plot_df["trade_dd_ticks"], mode="lines", name="Trade DD (ticks)"), row=5, col=1)

fig.update_layout(height=980, template="plotly_dark", title=f"Trade Replay: {selected_trade_id}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Trade Metadata")
meta = {
    "trade_id": trade_row["trade_id"],
    "date": trade_row["date"],
    "session": trade_row["session"],
    "direction": trade_row["direction"],
    "gross_ticks": trade_row["ticks_pnl_gross"],
    "net_ticks": trade_row["ticks_pnl_net"],
    "hold_seconds": trade_row["hold_seconds"],
    "variant_name": trade_row["variant_name"],
    "passed_spread_gate": trade_row["passed_spread_gate"],
    "passed_sv_gate": trade_row["passed_sv_gate"],
    "passed_powerhour_filter": trade_row["passed_powerhour_filter"],
    "replay_frame": f"{frame}/{len(chart_df)}",
    "current_mtm_ticks": round(current_mtm, 3),
    "current_trade_dd_ticks": round(current_dd, 3),
}
st.json(meta)
