from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class OBRAParams:
    N: int = 20
    BRK_PAD: int = 1
    IMPULSE_MIN_RANGE: int = 2
    OFI_TH: float = 20.0
    TRADES_MIN: int = 5
    SPREAD_MAX: int = 2
    RETEST_MAX_WAIT: int = 15
    ACCEPT_TOL: int = 1
    FAIL_TICKS: int = 2
    SPREAD_CANCEL: int = 3
    RETEST_OFI_FRAC: float = 0.25
    STOP_FROM_LVL: int = 3
    TP1_TICKS: int = 4
    TP2_TICKS: int = 8
    TIME_STOP_BARS: int = 40


@dataclass
class APBParams:
    N: int = 20
    M: int = 6
    OFI_TH: float = 20.0
    SPREAD_MAX: int = 2
    TRADES_MIN: int = 5
    STOP_FROM_LVL: int = 3
    TP1_TICKS: int = 4
    TP2_TICKS: int = 8
    TIME_STOP_BARS: int = 40


@dataclass
class PBRAParams:
    N: int = 20
    BREAK_MIN_RANGE: int = 2
    PULLBACK_TICKS: int = 2
    PULLBACK_MAX_BARS: int = 10
    LOCAL_HIGH_BARS: int = 5
    OFI_TH: float = 0.0
    TRADES_MIN: int = 5
    SPREAD_MAX: int = 2
    STOP_FROM_LVL: int = 3
    TP1_TICKS: int = 4
    TP2_TICKS: int = 8
    TIME_STOP_BARS: int = 40


@dataclass
class PFLFTParams:
    FLOW_WIN_BARS: int = 20
    SPREAD_MAX: int = 1
    DEPTH_MIN: float = 0.0
    DMID_ABS_MAX_TICKS: float = 1.0
    FLOW_INTENSITY_MIN: float = 0.0
    LAG_SCORE_MIN: float = 0.0
    LAG_DMID_FLOOR_TICKS: float = 1.0
    # v2 structural filters
    V2_ENABLE: bool = False
    V2_FLOW_Z_MIN: float = 0.5
    V2_FLOW_PERSIST_MIN: float = 0.55
    V2_SPREAD_REGIME_MAX: float = 0.95
    V2_DEPTH_IMB_MIN: float = -0.2
    V2_DEPTH_REPLENISH_MIN: float = 0.9
    V2_DMID_ACCEL_MAX: float = 2.5
    # v3 long-only alpha candidate
    V3_ENABLE: bool = False
    V3_LONG_ONLY: bool = True
    V3_SPREAD_MAX: int = 1
    V3_FLOW_INTENSITY_MIN: float = 210.0
    V3_LAG_SCORE_MAX: float = 401.0
    V3_DMID_MIN_TICKS: float = 0.0
    V3_DMID_MAX_TICKS: float = 1.0
    V3_DEPTH_IMB_MIN: float = -1.0
    V3_DEPTH_REPLENISH_MIN: float = 0.0
    V3_STOP_TICKS: int = 2
    V3_TP_TICKS: int = 2
    V3_TIME_STOP_BARS: int = 12
    # v4 continuation burst (CB1)
    V4_ENABLE: bool = False
    V4_SPREAD_MAX: int = 1
    V4_FLOW_Z_MIN: float = 0.6
    V4_FLOW_PERSIST_MIN: float = 0.65
    V4_DMID_MIN_TICKS: float = 0.25
    V4_DMID_MAX_TICKS: float = 2.0
    V4_DMID_ACCEL_MAX: float = 2.5
    V4_DEPTH_IMB_MIN: float = 0.1
    V4_STOP_TICKS: int = 2
    V4_TP_TICKS: int = 3
    V4_TIME_STOP_BARS: int = 14
    # v5 pullback-resume (PR1)
    V5_ENABLE: bool = False
    V5_LONG_ONLY: bool = True
    V5_SPREAD_MAX: int = 1
    V5_SPREAD_REGIME_MAX: float = 0.98
    V5_FLOW_Z_MIN: float = 0.20
    V5_IMPULSE_BARS: int = 10
    V5_IMPULSE_MIN_TICKS: float = 1.25
    V5_PULLBACK_MAX_BARS: int = 8
    V5_PULLBACK_MIN_TICKS: float = 0.25
    V5_PULLBACK_MAX_TICKS: float = 4.0
    V5_RESUME_TICKS: float = 1.0
    V5_RESUME_CONFIRM_BARS: int = 1
    V5_FLOW_PERSIST_MIN: float = 0.45
    V5_DEPTH_IMB_MIN: float = 0.0
    V5_TOXICITY_MAX: float = 2.0
    V5_PROOF_BARS: int = 3
    V5_PROOF_TICKS: int = 1
    V5_PROOF_MIN_FLOW: float = 0.0
    V5_STOP_TICKS: int = 2
    V5_TP_TICKS: int = 4
    V5_TIME_STOP_BARS: int = 18
    # v6 two-stage continuation-resume (long-first)
    V6_ENABLE: bool = False
    V6_LONG_ONLY: bool = True
    V6_SPREAD_MAX: int = 1
    V6_SPREAD_REGIME_MAX: float = 0.96
    V6_FLOW_Z_MIN: float = 0.25
    V6_FLOW_PERSIST_MIN: float = 0.50
    V6_DEPTH_IMB_MIN: float = 0.00
    V6_TOXICITY_MAX: float = 1.5
    V6_IMPULSE_BARS: int = 10
    V6_IMPULSE_MIN_TICKS: float = 1.5
    V6_PULLBACK_MAX_BARS: int = 8
    V6_PULLBACK_MIN_TICKS: float = 0.25
    V6_PULLBACK_MAX_TICKS: float = 3.5
    V6_RESUME_TICKS: float = 1.0
    V6_CONFIRM_BARS: int = 2
    # no-trade band: reject middling dmid regimes
    V6_NO_TRADE_DMID_LOW: float = 0.75
    V6_NO_TRADE_DMID_HIGH: float = 1.5
    # staged proof (entry confirmation gate)
    V6_PROOF_BARS: int = 4
    V6_PROOF_TICKS: int = 1
    V6_PROOF_MIN_FLOW: float = 0.0
    # adaptive exits by conviction
    V6_HIGH_TP_TICKS: int = 5
    V6_HIGH_TIME_STOP_BARS: int = 24
    V6_BASE_TP_TICKS: int = 3
    V6_BASE_TIME_STOP_BARS: int = 12
    V6_STOP_TICKS: int = 2
    # v7 pre-burst (pre-bars energy + proof)
    V7_ENABLE: bool = False
    V7_SPREAD_MAX: int = 1
    V7_FLOWINT_PRE3_MIN: float = 15.0
    V7_SV_PRE3_MIN: float = 12.0
    V7_DMID1_MAX: float = 1.0
    V7_PROOF_BARS: int = 2
    V7_PROOF_TICKS: int = 1
    V7_PROOF_MIN_FLOW: float = 0.0
    V7_TIME_STOP_BARS: int = 5
    V7_TP_TICKS: int = 2
    V7_SL_TICKS: int = 1
    V7_FLOWINT_PRE3_P90: float = 25.0
    V7_RUNNER_TP_TICKS: int = 4
    V7_PREV_RANGE_MAX_TICKS: float = 15.0
    V7_PREV_ABS_FLOW_MAX: float = 20000.0
    V7_ALLOW_MISMATCH: bool = False
    # v8 pre-burst + microstructure regime gate
    V8_ENABLE: bool = False
    V8_LFP_ALIGNED_10_MIN: float = 0.58
    V8_STRESS_RATIO_MIN: float = 0.18
    V8_DEPTH_TOTAL_TOP5_MAX: float = 640.0
    V8_ALIGNED_IMB_DELTA_MIN: float = 0.05
    V8_TOXICITY_MAX: float = 0.156
    # v8 causal toxicity proxy from pre-entry (t-3..t-1) bars
    V8_TOX_PROXY_ENABLE: bool = False
    V8_TOX_PROXY_MAX: float = 0.55
    V8_TOX_PROXY_OPPFLOW_W: float = 0.55
    V8_TOX_PROXY_IMB_W: float = 0.30
    V8_TOX_PROXY_THINSTRESS_W: float = 0.15
    STOP_TICKS: int = 3
    TP_TICKS: int = 4
    TIME_STOP_BARS: int = 20


@dataclass
class EntryAlphaParams:
    obra: OBRAParams = field(default_factory=OBRAParams)
    apb: APBParams = field(default_factory=APBParams)
    pbra: PBRAParams = field(default_factory=PBRAParams)
    pflft: PFLFTParams = field(default_factory=PFLFTParams)


@dataclass
class OBRAState:
    armed: bool = False
    side: int = 0
    lvl: float = 0.0
    arm_bar: int = -1
    ofi_arm: float = 0.0


@dataclass
class APBState:
    armed: bool = False
    side: int = 0
    lvl: float = 0.0
    arm_bar: int = -1
    expiry_bar: int = -1


@dataclass
class PBRAState:
    armed: bool = False
    side: int = 0
    lvl: float = 0.0
    break_price: float = 0.0
    arm_bar: int = -1
    pullback_seen: bool = False


@dataclass
class PFLFTState:
    last_signal_bar: int = -1


@dataclass
class EntryDecision:
    signal: int = 0
    family: str = ""
    reason: str = ""
    setup_bar: int | None = None
    entry_lvl: float | None = None
    stop_price: float | None = None
    tp1_price: float | None = None
    tp2_price: float | None = None
    time_stop_bars: int | None = None
    flow_sum_signed: float | None = None
    flow_intensity: float | None = None
    lag_score: float | None = None
    spread_ticks: float | None = None
    dmid_ticks: float | None = None
    depth_min: float | None = None
    proof_bars: int | None = None
    proof_ticks: int | None = None
    proof_min_flow: float | None = None
    confirm_min_abs_ofid: float | None = None
    confirm_min_flow_sum: float | None = None
    v8_opp_flow_frac_3: float | None = None
    v8_imb_mismatch_3: float | None = None
    v8_thin_stress_3: float | None = None
    v8_toxicity_proxy_pre3: float | None = None
    v8_imb_aligned_3: float | None = None
    v8_stress_mean_3: float | None = None
    v8_depth_mean_3: float | None = None


class EntryAlphaEngine:
    def __init__(self, params: EntryAlphaParams, tick_size: float) -> None:
        self.params = params
        self.tick_size = float(tick_size)
        self.obra_state = OBRAState()
        self.apb_state = APBState()
        self.pbra_state = PBRAState()
        self.pflft_state = PFLFTState()
        self.pflft_stats = {
            "candidates": 0,
            "blocked_spread": 0,
            "blocked_depth": 0,
            "blocked_dmid": 0,
            "blocked_lag": 0,
        }

    def _px(self, ticks: float) -> float:
        return float(ticks) * self.tick_size

    def _add_ticks(self, price: float, ticks: float) -> float:
        return float(price) + self._px(ticks)

    def _sub_ticks(self, price: float, ticks: float) -> float:
        return float(price) - self._px(ticks)

    def compute(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        decision = self._obra_v1(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._apb_v1(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._pflft_v8(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._pflft_v7(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._pflft_v6(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._pflft_v5(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._pflft_v4(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._pflft_v3(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._pflft_v2(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        decision = self._pflft_v1(i, hist, gates_ok_long, gates_ok_short)
        if decision.signal != 0:
            return decision
        return self._pbra_v1(i, hist, gates_ok_long, gates_ok_short)

    def _rolling_high_low(self, arr: np.ndarray, i: int, n: int) -> Tuple[float, float]:
        if i <= 0 or n <= 0 or i < n:
            return float("nan"), float("nan")
        window = arr[i - n : i]
        if window.size == 0:
            return float("nan"), float("nan")
        return float(np.nanmax(window)), float(np.nanmin(window))

    def _obra_v1(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.obra
        last = float(hist["last"][i])
        high = hist["high"]
        low = hist["low"]
        spread_ticks = float(hist["spread_ticks"][i])
        range_ticks = float(hist["range_ticks"][i])
        ofid = float(hist["ofid"][i])
        trades = float(hist["trades"][i])
        decision = EntryDecision()

        H, L = self._rolling_high_low(high, i, p.N)
        if np.isnan(H) or np.isnan(L):
            return decision

        st = self.obra_state
        if st.armed:
            if i - st.arm_bar > p.RETEST_MAX_WAIT:
                self.obra_state = OBRAState()
                return decision
            if spread_ticks > p.SPREAD_CANCEL:
                self.obra_state = OBRAState()
                return decision
            if st.side > 0 and last < self._sub_ticks(st.lvl, p.FAIL_TICKS):
                self.obra_state = OBRAState()
                return decision
            if st.side < 0 and last > self._add_ticks(st.lvl, p.FAIL_TICKS):
                self.obra_state = OBRAState()
                return decision
            if st.side > 0:
                if low[i] <= self._add_ticks(st.lvl, p.ACCEPT_TOL) and last >= st.lvl:
                    if ofid >= 0 and ofid >= p.RETEST_OFI_FRAC * st.ofi_arm and gates_ok_long:
                        entry_px = last
                        decision.signal = 1
                        decision.family = "OBRA_v1"
                        decision.reason = "OBRA_LONG_RETEST_ACCEPT"
                        decision.setup_bar = st.arm_bar
                        decision.entry_lvl = st.lvl
                        decision.stop_price = self._sub_ticks(st.lvl, p.STOP_FROM_LVL)
                        decision.tp1_price = self._add_ticks(entry_px, p.TP1_TICKS)
                        decision.tp2_price = self._add_ticks(entry_px, p.TP2_TICKS)
                        decision.time_stop_bars = p.TIME_STOP_BARS
                        self.obra_state = OBRAState()
                        return decision
            if st.side < 0:
                if high[i] >= self._sub_ticks(st.lvl, p.ACCEPT_TOL) and last <= st.lvl:
                    if ofid <= 0 and abs(ofid) >= p.RETEST_OFI_FRAC * abs(st.ofi_arm) and gates_ok_short:
                        entry_px = last
                        decision.signal = -1
                        decision.family = "OBRA_v1"
                        decision.reason = "OBRA_SHORT_RETEST_ACCEPT"
                        decision.setup_bar = st.arm_bar
                        decision.entry_lvl = st.lvl
                        decision.stop_price = self._add_ticks(st.lvl, p.STOP_FROM_LVL)
                        decision.tp1_price = self._sub_ticks(entry_px, p.TP1_TICKS)
                        decision.tp2_price = self._sub_ticks(entry_px, p.TP2_TICKS)
                        decision.time_stop_bars = p.TIME_STOP_BARS
                        self.obra_state = OBRAState()
                        return decision
            return decision
        if spread_ticks > p.SPREAD_MAX:
            return decision
        if range_ticks < p.IMPULSE_MIN_RANGE:
            return decision
        if trades < p.TRADES_MIN:
            return decision
        if last >= self._add_ticks(H, p.BRK_PAD) and ofid >= p.OFI_TH and gates_ok_long:
            self.obra_state = OBRAState(
                armed=True,
                side=1,
                lvl=H,
                arm_bar=i,
                ofi_arm=ofid,
            )
        elif last <= self._sub_ticks(L, p.BRK_PAD) and ofid <= -p.OFI_TH and gates_ok_short:
            self.obra_state = OBRAState(
                armed=True,
                side=-1,
                lvl=L,
                arm_bar=i,
                ofi_arm=ofid,
            )
        return decision

    def _apb_v1(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.apb
        last = float(hist["last"][i])
        high = hist["high"]
        low = hist["low"]
        spread_ticks = float(hist["spread_ticks"][i])
        ofid = float(hist["ofid"][i])
        trades = float(hist["trades"][i])
        depth_bid = hist["depth_bid"]
        depth_ask = hist["depth_ask"]
        decision = EntryDecision()

        H, L = self._rolling_high_low(high, i, p.N)
        if np.isnan(H) or np.isnan(L):
            return decision
        st = self.apb_state
        if st.armed:
            if i > st.expiry_bar:
                self.apb_state = APBState()
                return decision
            if st.side > 0 and last >= self._add_ticks(st.lvl, 1) and ofid >= 0 and gates_ok_long:
                entry_px = last
                decision.signal = 1
                decision.family = "APB_v1"
                decision.reason = "APB_LONG_BREAK"
                decision.setup_bar = st.arm_bar
                decision.entry_lvl = st.lvl
                decision.stop_price = self._sub_ticks(st.lvl, p.STOP_FROM_LVL)
                decision.tp1_price = self._add_ticks(entry_px, p.TP1_TICKS)
                decision.tp2_price = self._add_ticks(entry_px, p.TP2_TICKS)
                decision.time_stop_bars = p.TIME_STOP_BARS
                self.apb_state = APBState()
                return decision
            if st.side < 0 and last <= self._sub_ticks(st.lvl, 1) and ofid <= 0 and gates_ok_short:
                entry_px = last
                decision.signal = -1
                decision.family = "APB_v1"
                decision.reason = "APB_SHORT_BREAK"
                decision.setup_bar = st.arm_bar
                decision.entry_lvl = st.lvl
                decision.stop_price = self._add_ticks(st.lvl, p.STOP_FROM_LVL)
                decision.tp1_price = self._sub_ticks(entry_px, p.TP1_TICKS)
                decision.tp2_price = self._sub_ticks(entry_px, p.TP2_TICKS)
                decision.time_stop_bars = p.TIME_STOP_BARS
                self.apb_state = APBState()
                return decision
            return decision
        if spread_ticks > p.SPREAD_MAX or trades < p.TRADES_MIN:
            return decision
        if i >= 3:
            ret_3 = (last - float(hist["last"][i - 3])) / self.tick_size
        else:
            ret_3 = 0.0
        depth_ask_med = float(np.nanmedian(depth_ask[max(0, i - 20) : i + 1])) if depth_ask.size else float("nan")
        depth_bid_med = float(np.nanmedian(depth_bid[max(0, i - 20) : i + 1])) if depth_bid.size else float("nan")
        if gates_ok_long and last >= self._sub_ticks(H, 1) and last <= H and ofid >= p.OFI_TH and ret_3 <= 0:
            if np.isfinite(depth_ask_med) and depth_ask[i] >= depth_ask_med:
                self.apb_state = APBState(armed=True, side=1, lvl=H, arm_bar=i, expiry_bar=i + p.M)
        if gates_ok_short and last <= self._add_ticks(L, 1) and last >= L and ofid <= -p.OFI_TH and ret_3 >= 0:
            if np.isfinite(depth_bid_med) and depth_bid[i] >= depth_bid_med:
                self.apb_state = APBState(armed=True, side=-1, lvl=L, arm_bar=i, expiry_bar=i + p.M)
        return decision

    def _pbra_v1(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pbra
        last = float(hist["last"][i])
        high = hist["high"]
        low = hist["low"]
        spread_ticks = float(hist["spread_ticks"][i])
        range_ticks = float(hist["range_ticks"][i])
        ofid = float(hist["ofid"][i])
        trades = float(hist["trades"][i])
        decision = EntryDecision()

        H, L = self._rolling_high_low(high, i, p.N)
        if np.isnan(H) or np.isnan(L):
            return decision
        st = self.pbra_state
        if st.armed:
            if i - st.arm_bar > p.PULLBACK_MAX_BARS:
                self.pbra_state = PBRAState()
                return decision
            if st.side > 0:
                if last <= self._sub_ticks(st.break_price, p.PULLBACK_TICKS) and last >= self._sub_ticks(st.lvl, 1):
                    st.pullback_seen = True
                if st.pullback_seen and gates_ok_long:
                    local_start = max(0, i - p.LOCAL_HIGH_BARS)
                    if local_start >= i:
                        return decision
                    if (
                        last >= float(np.nanmax(hist["last"][local_start:i]))
                        and ofid >= p.OFI_TH
                        and trades >= p.TRADES_MIN
                    ):
                        entry_px = last
                        decision.signal = 1
                        decision.family = "PBRA_v1"
                        decision.reason = "PBRA_LONG_REACCEL"
                        decision.setup_bar = st.arm_bar
                        decision.entry_lvl = st.lvl
                        decision.stop_price = self._sub_ticks(st.lvl, p.STOP_FROM_LVL)
                        decision.tp1_price = self._add_ticks(entry_px, p.TP1_TICKS)
                        decision.tp2_price = self._add_ticks(entry_px, p.TP2_TICKS)
                        decision.time_stop_bars = p.TIME_STOP_BARS
                        self.pbra_state = PBRAState()
                        return decision
            if st.side < 0:
                if last >= self._add_ticks(st.break_price, p.PULLBACK_TICKS) and last <= self._add_ticks(st.lvl, 1):
                    st.pullback_seen = True
                if st.pullback_seen and gates_ok_short:
                    local_start = max(0, i - p.LOCAL_HIGH_BARS)
                    if local_start >= i:
                        return decision
                    if (
                        last <= float(np.nanmin(hist["last"][local_start:i]))
                        and ofid <= -p.OFI_TH
                        and trades >= p.TRADES_MIN
                    ):
                        entry_px = last
                        decision.signal = -1
                        decision.family = "PBRA_v1"
                        decision.reason = "PBRA_SHORT_REACCEL"
                        decision.setup_bar = st.arm_bar
                        decision.entry_lvl = st.lvl
                        decision.stop_price = self._add_ticks(st.lvl, p.STOP_FROM_LVL)
                        decision.tp1_price = self._sub_ticks(entry_px, p.TP1_TICKS)
                        decision.tp2_price = self._sub_ticks(entry_px, p.TP2_TICKS)
                        decision.time_stop_bars = p.TIME_STOP_BARS
                        self.pbra_state = PBRAState()
                        return decision
            return decision

        if spread_ticks > p.SPREAD_MAX or trades < p.TRADES_MIN:
            return decision
        if range_ticks < p.BREAK_MIN_RANGE:
            return decision
        if gates_ok_long and last >= self._add_ticks(H, 1):
            self.pbra_state = PBRAState(
                armed=True,
                side=1,
                lvl=H,
                break_price=last,
                arm_bar=i,
                pullback_seen=False,
            )
        elif gates_ok_short and last <= self._sub_ticks(L, 1):
            self.pbra_state = PBRAState(
                armed=True,
                side=-1,
                lvl=L,
                break_price=last,
                arm_bar=i,
                pullback_seen=False,
            )
        return decision

    def _pflft_v1(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pflft
        last = float(hist["last"][i])
        spread_ticks = float(hist["spread_ticks"][i])
        depth_bid = hist.get("depth_bid", np.array([], dtype=float))
        depth_ask = hist.get("depth_ask", np.array([], dtype=float))
        ofid = hist["ofid"]
        price_ref = hist.get("mid", hist["last"])
        decision = EntryDecision()

        if spread_ticks > p.SPREAD_MAX:
            self.pflft_stats["blocked_spread"] += 1
            return decision

        depth_min = float("nan")
        if depth_bid.size and depth_ask.size:
            bid_depth = float(depth_bid[i])
            ask_depth = float(depth_ask[i])
            if np.isfinite(bid_depth) and np.isfinite(ask_depth):
                depth_min = float(min(bid_depth, ask_depth))
                if p.DEPTH_MIN > 0.0 and depth_min < p.DEPTH_MIN:
                    self.pflft_stats["blocked_depth"] += 1
                    return decision

        w = int(p.FLOW_WIN_BARS)
        if w <= 0 or i < w:
            return decision

        start = max(0, i - w + 1)
        flow_window = ofid[start : i + 1]
        if flow_window.size == 0:
            return decision
        flow_sum_signed = float(np.nansum(flow_window))
        flow_intensity = float(np.nansum(np.abs(flow_window)))

        dmid_ticks = float("nan")
        if i - w >= 0:
            dmid_ticks = (float(price_ref[i]) - float(price_ref[i - w])) / self.tick_size
        if np.isfinite(dmid_ticks) and abs(dmid_ticks) > p.DMID_ABS_MAX_TICKS:
            self.pflft_stats["blocked_dmid"] += 1
            return decision

        lag_score = float("nan")
        if np.isfinite(dmid_ticks):
            lag_dmid_floor = max(float(p.LAG_DMID_FLOOR_TICKS), 1e-9)
            lag_denom = max(abs(dmid_ticks), lag_dmid_floor)
            lag_score = flow_intensity / lag_denom
        if flow_intensity < p.FLOW_INTENSITY_MIN or (
            np.isfinite(lag_score) and lag_score < p.LAG_SCORE_MIN
        ):
            self.pflft_stats["blocked_lag"] += 1
            return decision

        if flow_sum_signed > 0 and gates_ok_long:
            entry_px = last
            decision.signal = 1
            decision.family = "PFLFT_v1"
            decision.reason = "PFLFT_LONG_LAG"
            decision.setup_bar = i
            decision.entry_lvl = entry_px
            decision.stop_price = self._sub_ticks(entry_px, p.STOP_TICKS)
            decision.tp1_price = self._add_ticks(entry_px, p.TP_TICKS)
            decision.tp2_price = self._add_ticks(entry_px, p.TP_TICKS * 2)
            decision.time_stop_bars = p.TIME_STOP_BARS
        elif flow_sum_signed < 0 and gates_ok_short:
            entry_px = last
            decision.signal = -1
            decision.family = "PFLFT_v1"
            decision.reason = "PFLFT_SHORT_LAG"
            decision.setup_bar = i
            decision.entry_lvl = entry_px
            decision.stop_price = self._add_ticks(entry_px, p.STOP_TICKS)
            decision.tp1_price = self._sub_ticks(entry_px, p.TP_TICKS)
            decision.tp2_price = self._sub_ticks(entry_px, p.TP_TICKS * 2)
            decision.time_stop_bars = p.TIME_STOP_BARS
        else:
            return decision

        decision.flow_sum_signed = flow_sum_signed
        decision.flow_intensity = flow_intensity
        decision.lag_score = lag_score
        decision.spread_ticks = spread_ticks
        decision.dmid_ticks = dmid_ticks
        decision.depth_min = depth_min
        self.pflft_stats["candidates"] += 1
        return decision

    def _pflft_v2(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pflft
        if not bool(p.V2_ENABLE):
            return EntryDecision()

        last = float(hist["last"][i])
        spread_ticks = float(hist["spread_ticks"][i])
        ofid = hist["ofid"]
        price_ref = hist.get("mid", hist["last"])
        depth_bid = hist.get("depth_bid", np.array([], dtype=float))
        depth_ask = hist.get("depth_ask", np.array([], dtype=float))
        decision = EntryDecision()

        if spread_ticks > p.SPREAD_MAX:
            return decision

        w = int(p.FLOW_WIN_BARS)
        if w <= 1 or i < w:
            return decision

        start = max(0, i - w + 1)
        flow_window = ofid[start : i + 1]
        if flow_window.size == 0:
            return decision
        flow_sum_signed = float(np.nansum(flow_window))
        flow_intensity = float(np.nansum(np.abs(flow_window)))

        # dmid and accel
        dmid_ticks = float("nan")
        dmid_accel = float("nan")
        if i - w >= 0:
            dmid_ticks = (float(price_ref[i]) - float(price_ref[i - w])) / self.tick_size
        if i - w - 1 >= 0:
            dmid_prev = (float(price_ref[i - 1]) - float(price_ref[i - 1 - w])) / self.tick_size
            dmid_accel = float(dmid_ticks - dmid_prev) if np.isfinite(dmid_ticks) else float("nan")
        if np.isfinite(dmid_ticks) and abs(dmid_ticks) > p.DMID_ABS_MAX_TICKS:
            return decision
        if np.isfinite(dmid_accel) and abs(dmid_accel) > p.V2_DMID_ACCEL_MAX:
            return decision

        # lag score
        lag_score = float("nan")
        if np.isfinite(dmid_ticks):
            lag_dmid_floor = max(float(p.LAG_DMID_FLOOR_TICKS), 1e-9)
            lag_denom = max(abs(dmid_ticks), lag_dmid_floor)
            lag_score = flow_intensity / lag_denom
        if flow_intensity < p.FLOW_INTENSITY_MIN or (np.isfinite(lag_score) and lag_score < p.LAG_SCORE_MIN):
            return decision

        # flow z-score
        z_start = max(0, i - 49)
        z_window = ofid[z_start : i + 1]
        flow_z = float("nan")
        if z_window.size >= 10:
            z_mu = float(np.nanmean(z_window))
            z_sd = float(np.nanstd(z_window))
            if np.isfinite(z_sd) and z_sd > 1e-9:
                flow_z = float((float(ofid[i]) - z_mu) / z_sd)
        if np.isfinite(flow_z) and abs(flow_z) < p.V2_FLOW_Z_MIN:
            return decision

        # persistence
        sgn = np.sign(flow_window)
        nz = sgn != 0
        flow_persist = float("nan")
        if np.any(nz):
            flow_persist = float(np.mean(sgn[nz] == np.sign(flow_sum_signed)))
        if np.isfinite(flow_persist) and flow_persist < p.V2_FLOW_PERSIST_MIN:
            return decision

        # spread regime percentile in rolling window
        sr_start = max(0, i - 99)
        sr_w = hist["spread_ticks"][sr_start : i + 1].astype(float)
        spread_regime = float("nan")
        if sr_w.size >= 10 and np.isfinite(sr_w[-1]):
            spread_regime = float(np.mean(sr_w <= sr_w[-1]))
        if np.isfinite(spread_regime) and spread_regime > p.V2_SPREAD_REGIME_MAX:
            return decision

        # depth state filters
        depth_min = float("nan")
        depth_imb = float("nan")
        depth_repl = float("nan")
        if depth_bid.size and depth_ask.size:
            bid_depth = float(depth_bid[i])
            ask_depth = float(depth_ask[i])
            if np.isfinite(bid_depth) and np.isfinite(ask_depth):
                depth_min = float(min(bid_depth, ask_depth))
                denom = bid_depth + ask_depth
                if abs(denom) > 1e-9:
                    depth_imb = float((bid_depth - ask_depth) / denom)
            ds = max(1, i - 19)
            tot = depth_bid[ds - 1 : i + 1] + depth_ask[ds - 1 : i + 1]
            if tot.size >= 3:
                d = np.diff(tot.astype(float))
                up = float(np.nansum(np.clip(d, 0, None)))
                down = float(np.nansum(np.clip(-d, 0, None)))
                depth_repl = float(up / max(down, 1e-9))
        if p.DEPTH_MIN > 0.0 and np.isfinite(depth_min) and depth_min < p.DEPTH_MIN:
            return decision
        if np.isfinite(depth_imb) and depth_imb < p.V2_DEPTH_IMB_MIN:
            return decision
        if np.isfinite(depth_repl) and depth_repl < p.V2_DEPTH_REPLENISH_MIN:
            return decision

        # Side selection: require dmid/flow agreement where dmid exists
        side = 0
        if flow_sum_signed > 0 and gates_ok_long:
            side = 1
            if np.isfinite(dmid_ticks) and dmid_ticks < 0:
                side = 0
        elif flow_sum_signed < 0 and gates_ok_short:
            side = -1
            if np.isfinite(dmid_ticks) and dmid_ticks > 0:
                side = 0
        if side == 0:
            return decision

        entry_px = last
        decision.signal = side
        decision.family = "PFLFT_v2"
        decision.reason = "PFLFT_V2_LONG" if side > 0 else "PFLFT_V2_SHORT"
        decision.setup_bar = i
        decision.entry_lvl = entry_px
        if side > 0:
            decision.stop_price = self._sub_ticks(entry_px, p.STOP_TICKS)
            decision.tp1_price = self._add_ticks(entry_px, p.TP_TICKS)
            decision.tp2_price = self._add_ticks(entry_px, p.TP_TICKS * 2)
        else:
            decision.stop_price = self._add_ticks(entry_px, p.STOP_TICKS)
            decision.tp1_price = self._sub_ticks(entry_px, p.TP_TICKS)
            decision.tp2_price = self._sub_ticks(entry_px, p.TP_TICKS * 2)
        decision.time_stop_bars = p.TIME_STOP_BARS
        decision.flow_sum_signed = flow_sum_signed
        decision.flow_intensity = flow_intensity
        decision.lag_score = lag_score
        decision.spread_ticks = spread_ticks
        decision.dmid_ticks = dmid_ticks
        decision.depth_min = depth_min
        self.pflft_stats["candidates"] += 1
        return decision

    def _pflft_v4(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pflft
        if not bool(p.V4_ENABLE):
            return EntryDecision()

        last = float(hist["last"][i])
        spread_ticks = float(hist["spread_ticks"][i])
        ofid = hist["ofid"]
        price_ref = hist.get("mid", hist["last"])
        depth_bid = hist.get("depth_bid", np.array([], dtype=float))
        depth_ask = hist.get("depth_ask", np.array([], dtype=float))
        decision = EntryDecision()
        if spread_ticks > float(p.V4_SPREAD_MAX):
            return decision

        w = int(p.FLOW_WIN_BARS)
        if w <= 1 or i < w:
            return decision
        start = max(0, i - w + 1)
        flow_window = ofid[start : i + 1]
        if flow_window.size == 0:
            return decision
        flow_sum_signed = float(np.nansum(flow_window))
        flow_intensity = float(np.nansum(np.abs(flow_window)))
        if flow_intensity <= 0:
            return decision

        # z-score
        z_start = max(0, i - 49)
        z_window = ofid[z_start : i + 1]
        flow_z = float("nan")
        if z_window.size >= 10:
            z_mu = float(np.nanmean(z_window))
            z_sd = float(np.nanstd(z_window))
            if np.isfinite(z_sd) and z_sd > 1e-9:
                flow_z = float((float(ofid[i]) - z_mu) / z_sd)
        if not np.isfinite(flow_z):
            return decision

        # persistence
        sgn = np.sign(flow_window)
        nz = sgn != 0
        flow_persist = float("nan")
        if np.any(nz):
            flow_persist = float(np.mean(sgn[nz] == np.sign(flow_sum_signed)))
        if not np.isfinite(flow_persist) or flow_persist < float(p.V4_FLOW_PERSIST_MIN):
            return decision

        # dmid + accel
        dmid_ticks = float("nan")
        dmid_accel = float("nan")
        if i - w >= 0:
            dmid_ticks = (float(price_ref[i]) - float(price_ref[i - w])) / self.tick_size
        if i - w - 1 >= 0:
            dmid_prev = (float(price_ref[i - 1]) - float(price_ref[i - 1 - w])) / self.tick_size
            dmid_accel = float(dmid_ticks - dmid_prev) if np.isfinite(dmid_ticks) else float("nan")
        if not np.isfinite(dmid_ticks):
            return decision
        if np.isfinite(dmid_accel) and abs(dmid_accel) > float(p.V4_DMID_ACCEL_MAX):
            return decision

        # depth imbalance
        depth_imb = float("nan")
        if depth_bid.size and depth_ask.size:
            bid_depth = float(depth_bid[i])
            ask_depth = float(depth_ask[i])
            denom = bid_depth + ask_depth
            if np.isfinite(denom) and abs(denom) > 1e-9:
                depth_imb = float((bid_depth - ask_depth) / denom)

        side = 0
        if (
            flow_sum_signed > 0
            and flow_z >= float(p.V4_FLOW_Z_MIN)
            and dmid_ticks > float(p.V4_DMID_MIN_TICKS)
            and dmid_ticks <= float(p.V4_DMID_MAX_TICKS)
            and (not np.isfinite(depth_imb) or depth_imb >= float(p.V4_DEPTH_IMB_MIN))
            and gates_ok_long
        ):
            side = 1
        elif (
            flow_sum_signed < 0
            and flow_z <= -float(p.V4_FLOW_Z_MIN)
            and dmid_ticks < -float(p.V4_DMID_MIN_TICKS)
            and dmid_ticks >= -float(p.V4_DMID_MAX_TICKS)
            and (not np.isfinite(depth_imb) or depth_imb <= -float(p.V4_DEPTH_IMB_MIN))
            and gates_ok_short
        ):
            side = -1
        if side == 0:
            return decision

        decision.signal = side
        decision.family = "PFLFT_v4"
        decision.reason = "PFLFT_V4_LONG" if side > 0 else "PFLFT_V4_SHORT"
        decision.setup_bar = i
        decision.entry_lvl = last
        if side > 0:
            decision.stop_price = self._sub_ticks(last, p.V4_STOP_TICKS)
            decision.tp1_price = self._add_ticks(last, p.V4_TP_TICKS)
            decision.tp2_price = self._add_ticks(last, p.V4_TP_TICKS)
        else:
            decision.stop_price = self._add_ticks(last, p.V4_STOP_TICKS)
            decision.tp1_price = self._sub_ticks(last, p.V4_TP_TICKS)
            decision.tp2_price = self._sub_ticks(last, p.V4_TP_TICKS)
        decision.time_stop_bars = int(p.V4_TIME_STOP_BARS)
        decision.flow_sum_signed = flow_sum_signed
        decision.flow_intensity = flow_intensity
        decision.spread_ticks = spread_ticks
        decision.dmid_ticks = dmid_ticks
        self.pflft_stats["candidates"] += 1
        return decision

    def _pflft_v5(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pflft
        if not bool(p.V5_ENABLE):
            return EntryDecision()

        last = float(hist["last"][i])
        spread_ticks = float(hist["spread_ticks"][i])
        ofid = hist["ofid"]
        price_ref = hist.get("mid", hist["last"])
        depth_bid = hist.get("depth_bid", np.array([], dtype=float))
        depth_ask = hist.get("depth_ask", np.array([], dtype=float))
        decision = EntryDecision()
        if spread_ticks > float(p.V5_SPREAD_MAX):
            return decision

        w = int(p.FLOW_WIN_BARS)
        ib = int(p.V5_IMPULSE_BARS)
        pb = int(p.V5_PULLBACK_MAX_BARS)
        if min(w, ib, pb) <= 1 or i < max(w, ib, pb):
            return decision

        start = max(0, i - w + 1)
        flow_window = ofid[start : i + 1]
        if flow_window.size == 0:
            return decision
        flow_sum_signed = float(np.nansum(flow_window))
        flow_intensity = float(np.nansum(np.abs(flow_window)))

        # flow z-score for regime quality
        z_start = max(0, i - 49)
        z_window = ofid[z_start : i + 1]
        flow_z = float("nan")
        if z_window.size >= 10:
            z_mu = float(np.nanmean(z_window))
            z_sd = float(np.nanstd(z_window))
            if np.isfinite(z_sd) and z_sd > 1e-9:
                flow_z = float((float(ofid[i]) - z_mu) / z_sd)
        if not np.isfinite(flow_z):
            return decision
        if abs(flow_z) < float(p.V5_FLOW_Z_MIN):
            return decision

        sgn = np.sign(flow_window)
        nz = sgn != 0
        flow_persist = float("nan")
        if np.any(nz):
            flow_persist = float(np.mean(sgn[nz] == np.sign(flow_sum_signed)))
        if not np.isfinite(flow_persist) or flow_persist < float(p.V5_FLOW_PERSIST_MIN):
            return decision

        # spread regime filter (current spread percentile in trailing 100 bars)
        sr_start = max(0, i - 99)
        sr_w = hist["spread_ticks"][sr_start : i + 1].astype(float)
        spread_regime = float("nan")
        if sr_w.size >= 10 and np.isfinite(sr_w[-1]):
            spread_regime = float(np.mean(sr_w <= sr_w[-1]))
        if np.isfinite(spread_regime) and spread_regime > float(p.V5_SPREAD_REGIME_MAX):
            return decision

        impulse = (float(price_ref[i]) - float(price_ref[i - ib])) / self.tick_size
        recent = price_ref[max(0, i - pb + 1) : i + 1].astype(float)
        if recent.size < 2:
            return decision

        depth_imb = float("nan")
        if depth_bid.size and depth_ask.size:
            bid_depth = float(depth_bid[i])
            ask_depth = float(depth_ask[i])
            denom = bid_depth + ask_depth
            if np.isfinite(denom) and abs(denom) > 1e-9:
                depth_imb = float((bid_depth - ask_depth) / denom)

        # toxicity proxy from recent adverse move magnitude (past only)
        t_start = max(0, i - 2)
        past_window = price_ref[t_start : i + 1].astype(float)
        toxicity_long = float("nan")
        toxicity_short = float("nan")
        if past_window.size >= 2 and np.all(np.isfinite(past_window)):
            toxicity_long = float((np.max(past_window) - past_window[-1]) / self.tick_size)
            toxicity_short = float((past_window[-1] - np.min(past_window)) / self.tick_size)

        rc = max(1, int(p.V5_RESUME_CONFIRM_BARS))
        if i - rc + 1 < 0:
            return decision
        resume_window = price_ref[i - rc + 1 : i + 1].astype(float)
        if resume_window.size < rc or not np.all(np.isfinite(resume_window)):
            return decision

        side = 0
        # Long pullback-resume
        if flow_sum_signed > 0 and impulse >= float(p.V5_IMPULSE_MIN_TICKS) and gates_ok_long:
            recent_high = float(np.nanmax(recent))
            recent_low = float(np.nanmin(recent))
            pullback = (recent_high - recent_low) / self.tick_size
            resume = (float(price_ref[i]) - recent_low) / self.tick_size
            resume_floor = recent_low + self.tick_size * float(p.V5_RESUME_TICKS)
            resume_confirm = bool(np.all(resume_window >= resume_floor))
            if (
                pullback >= float(p.V5_PULLBACK_MIN_TICKS)
                and pullback <= float(p.V5_PULLBACK_MAX_TICKS)
                and resume >= float(p.V5_RESUME_TICKS)
                and resume_confirm
                and (not np.isfinite(depth_imb) or depth_imb >= float(p.V5_DEPTH_IMB_MIN))
                and (not np.isfinite(toxicity_long) or toxicity_long <= float(p.V5_TOXICITY_MAX))
            ):
                side = 1
        # Short pullback-resume
        elif (
            not bool(p.V5_LONG_ONLY)
            and flow_sum_signed < 0
            and impulse <= -float(p.V5_IMPULSE_MIN_TICKS)
            and gates_ok_short
        ):
            recent_high = float(np.nanmax(recent))
            recent_low = float(np.nanmin(recent))
            pullback = (recent_high - recent_low) / self.tick_size
            resume = (recent_high - float(price_ref[i])) / self.tick_size
            resume_ceiling = recent_high - self.tick_size * float(p.V5_RESUME_TICKS)
            resume_confirm = bool(np.all(resume_window <= resume_ceiling))
            if (
                pullback >= float(p.V5_PULLBACK_MIN_TICKS)
                and pullback <= float(p.V5_PULLBACK_MAX_TICKS)
                and resume >= float(p.V5_RESUME_TICKS)
                and resume_confirm
                and (not np.isfinite(depth_imb) or depth_imb <= -float(p.V5_DEPTH_IMB_MIN))
                and (not np.isfinite(toxicity_short) or toxicity_short <= float(p.V5_TOXICITY_MAX))
            ):
                side = -1
        if side == 0:
            return decision

        decision.signal = side
        decision.family = "PFLFT_v5"
        decision.reason = "PFLFT_V5_LONG" if side > 0 else "PFLFT_V5_SHORT"
        decision.setup_bar = i
        decision.entry_lvl = last
        if side > 0:
            decision.stop_price = self._sub_ticks(last, p.V5_STOP_TICKS)
            decision.tp1_price = self._add_ticks(last, p.V5_TP_TICKS)
            decision.tp2_price = self._add_ticks(last, p.V5_TP_TICKS)
        else:
            decision.stop_price = self._add_ticks(last, p.V5_STOP_TICKS)
            decision.tp1_price = self._sub_ticks(last, p.V5_TP_TICKS)
            decision.tp2_price = self._sub_ticks(last, p.V5_TP_TICKS)
        decision.time_stop_bars = int(p.V5_TIME_STOP_BARS)
        decision.proof_bars = int(p.V5_PROOF_BARS)
        decision.proof_ticks = int(p.V5_PROOF_TICKS)
        decision.proof_min_flow = float(p.V5_PROOF_MIN_FLOW)
        decision.flow_sum_signed = flow_sum_signed
        decision.flow_intensity = flow_intensity
        decision.lag_score = flow_z
        decision.spread_ticks = spread_ticks
        decision.dmid_ticks = impulse
        self.pflft_stats["candidates"] += 1
        return decision

    def _pflft_v6(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pflft
        if not bool(p.V6_ENABLE):
            return EntryDecision()

        last = float(hist["last"][i])
        spread_ticks = float(hist["spread_ticks"][i])
        ofid = hist["ofid"]
        price_ref = hist.get("mid", hist["last"])
        depth_bid = hist.get("depth_bid", np.array([], dtype=float))
        depth_ask = hist.get("depth_ask", np.array([], dtype=float))
        decision = EntryDecision()

        if spread_ticks > float(p.V6_SPREAD_MAX):
            return decision

        w = int(p.FLOW_WIN_BARS)
        ib = int(p.V6_IMPULSE_BARS)
        pb = int(p.V6_PULLBACK_MAX_BARS)
        cb = max(1, int(p.V6_CONFIRM_BARS))
        if min(w, ib, pb) <= 1 or i < max(w, ib, pb, cb):
            return decision

        # Stage A: regime + setup
        start = max(0, i - w + 1)
        flow_window = ofid[start : i + 1]
        if flow_window.size == 0:
            return decision
        flow_sum_signed = float(np.nansum(flow_window))
        flow_intensity = float(np.nansum(np.abs(flow_window)))

        z_start = max(0, i - 49)
        z_window = ofid[z_start : i + 1]
        flow_z = float("nan")
        if z_window.size >= 10:
            z_mu = float(np.nanmean(z_window))
            z_sd = float(np.nanstd(z_window))
            if np.isfinite(z_sd) and z_sd > 1e-9:
                flow_z = float((float(ofid[i]) - z_mu) / z_sd)
        if not np.isfinite(flow_z) or abs(flow_z) < float(p.V6_FLOW_Z_MIN):
            return decision

        sgn = np.sign(flow_window)
        nz = sgn != 0
        flow_persist = float("nan")
        if np.any(nz):
            flow_persist = float(np.mean(sgn[nz] == np.sign(flow_sum_signed)))
        if not np.isfinite(flow_persist) or flow_persist < float(p.V6_FLOW_PERSIST_MIN):
            return decision

        sr_start = max(0, i - 99)
        sr_w = hist["spread_ticks"][sr_start : i + 1].astype(float)
        spread_regime = float("nan")
        if sr_w.size >= 10 and np.isfinite(sr_w[-1]):
            spread_regime = float(np.mean(sr_w <= sr_w[-1]))
        if np.isfinite(spread_regime) and spread_regime > float(p.V6_SPREAD_REGIME_MAX):
            return decision

        impulse = (float(price_ref[i]) - float(price_ref[i - ib])) / self.tick_size
        recent = price_ref[max(0, i - pb + 1) : i + 1].astype(float)
        if recent.size < 2:
            return decision

        dmid_ticks = float("nan")
        if i - w >= 0:
            dmid_ticks = (float(price_ref[i]) - float(price_ref[i - w])) / self.tick_size
        if not np.isfinite(dmid_ticks):
            return decision
        # no-trade band around middling dmid where prior runs were unstable
        if float(p.V6_NO_TRADE_DMID_LOW) <= abs(dmid_ticks) <= float(p.V6_NO_TRADE_DMID_HIGH):
            return decision

        depth_imb = float("nan")
        if depth_bid.size and depth_ask.size:
            bid_depth = float(depth_bid[i])
            ask_depth = float(depth_ask[i])
            denom = bid_depth + ask_depth
            if np.isfinite(denom) and abs(denom) > 1e-9:
                depth_imb = float((bid_depth - ask_depth) / denom)

        t_start = max(0, i - 2)
        past_window = price_ref[t_start : i + 1].astype(float)
        toxicity_long = float("nan")
        toxicity_short = float("nan")
        if past_window.size >= 2 and np.all(np.isfinite(past_window)):
            toxicity_long = float((np.max(past_window) - past_window[-1]) / self.tick_size)
            toxicity_short = float((past_window[-1] - np.min(past_window)) / self.tick_size)

        # Stage B: confirmation over 2-4 bars
        resume_window = price_ref[i - cb + 1 : i + 1].astype(float)
        if resume_window.size < cb or not np.all(np.isfinite(resume_window)):
            return decision

        side = 0
        conviction_high = False

        if flow_sum_signed > 0 and impulse >= float(p.V6_IMPULSE_MIN_TICKS) and gates_ok_long:
            recent_high = float(np.nanmax(recent))
            recent_low = float(np.nanmin(recent))
            pullback = (recent_high - recent_low) / self.tick_size
            resume = (float(price_ref[i]) - recent_low) / self.tick_size
            resume_floor = recent_low + self.tick_size * float(p.V6_RESUME_TICKS)
            resume_confirm = bool(np.all(resume_window >= resume_floor))
            if (
                pullback >= float(p.V6_PULLBACK_MIN_TICKS)
                and pullback <= float(p.V6_PULLBACK_MAX_TICKS)
                and resume >= float(p.V6_RESUME_TICKS)
                and resume_confirm
                and (not np.isfinite(depth_imb) or depth_imb >= float(p.V6_DEPTH_IMB_MIN))
                and (not np.isfinite(toxicity_long) or toxicity_long <= float(p.V6_TOXICITY_MAX))
            ):
                side = 1
                conviction_high = (
                    abs(flow_z) >= float(p.V6_FLOW_Z_MIN) + 0.25
                    and flow_persist >= float(p.V6_FLOW_PERSIST_MIN) + 0.10
                    and (not np.isfinite(depth_imb) or depth_imb >= float(p.V6_DEPTH_IMB_MIN) + 0.05)
                )
        elif (
            not bool(p.V6_LONG_ONLY)
            and flow_sum_signed < 0
            and impulse <= -float(p.V6_IMPULSE_MIN_TICKS)
            and gates_ok_short
        ):
            recent_high = float(np.nanmax(recent))
            recent_low = float(np.nanmin(recent))
            pullback = (recent_high - recent_low) / self.tick_size
            resume = (recent_high - float(price_ref[i])) / self.tick_size
            resume_ceiling = recent_high - self.tick_size * float(p.V6_RESUME_TICKS)
            resume_confirm = bool(np.all(resume_window <= resume_ceiling))
            if (
                pullback >= float(p.V6_PULLBACK_MIN_TICKS)
                and pullback <= float(p.V6_PULLBACK_MAX_TICKS)
                and resume >= float(p.V6_RESUME_TICKS)
                and resume_confirm
                and (not np.isfinite(depth_imb) or depth_imb <= -float(p.V6_DEPTH_IMB_MIN))
                and (not np.isfinite(toxicity_short) or toxicity_short <= float(p.V6_TOXICITY_MAX))
            ):
                side = -1
                conviction_high = (
                    abs(flow_z) >= float(p.V6_FLOW_Z_MIN) + 0.25
                    and flow_persist >= float(p.V6_FLOW_PERSIST_MIN) + 0.10
                    and (not np.isfinite(depth_imb) or depth_imb <= -(float(p.V6_DEPTH_IMB_MIN) + 0.05))
                )
        if side == 0:
            return decision

        decision.signal = side
        decision.family = "PFLFT_v6"
        decision.reason = "PFLFT_V6_LONG" if side > 0 else "PFLFT_V6_SHORT"
        decision.setup_bar = i
        decision.entry_lvl = last

        tp_ticks = int(p.V6_HIGH_TP_TICKS if conviction_high else p.V6_BASE_TP_TICKS)
        ts_bars = int(p.V6_HIGH_TIME_STOP_BARS if conviction_high else p.V6_BASE_TIME_STOP_BARS)
        if side > 0:
            decision.stop_price = self._sub_ticks(last, p.V6_STOP_TICKS)
            decision.tp1_price = self._add_ticks(last, tp_ticks)
            decision.tp2_price = self._add_ticks(last, tp_ticks)
        else:
            decision.stop_price = self._add_ticks(last, p.V6_STOP_TICKS)
            decision.tp1_price = self._sub_ticks(last, tp_ticks)
            decision.tp2_price = self._sub_ticks(last, tp_ticks)
        decision.time_stop_bars = ts_bars
        decision.proof_bars = int(p.V6_PROOF_BARS)
        decision.proof_ticks = int(p.V6_PROOF_TICKS)
        decision.proof_min_flow = float(p.V6_PROOF_MIN_FLOW)
        decision.flow_sum_signed = flow_sum_signed
        decision.flow_intensity = flow_intensity
        decision.lag_score = flow_z
        decision.spread_ticks = spread_ticks
        decision.dmid_ticks = dmid_ticks
        self.pflft_stats["candidates"] += 1
        return decision

    def _pflft_v7(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pflft
        if not bool(p.V7_ENABLE):
            return EntryDecision()

        decision = EntryDecision()
        regime_ok = hist.get("regime_ok")
        if regime_ok is not None:
            try:
                if not bool(regime_ok[i]):
                    return decision
            except Exception:
                return decision

        spread_ticks = float(hist["spread_ticks"][i])
        if spread_ticks > float(p.V7_SPREAD_MAX):
            return decision

        ofid = hist["ofid"]
        if i < 3:
            return decision
        pre = ofid[i - 3 : i]
        if pre.size < 3:
            return decision
        sv_pre3 = float(np.nansum(pre))
        flowint_pre3 = float(np.nansum(np.abs(pre)))
        if not np.isfinite(sv_pre3) or not np.isfinite(flowint_pre3):
            return decision
        if abs(sv_pre3) < float(p.V7_SV_PRE3_MIN):
            return decision
        if flowint_pre3 < float(p.V7_FLOWINT_PRE3_MIN):
            return decision

        direction = 1 if sv_pre3 > 0 else -1
        allow_mismatch = bool(getattr(p, "V7_ALLOW_MISMATCH", False))
        if not allow_mismatch:
            if direction > 0 and not gates_ok_long:
                return decision
            if direction < 0 and not gates_ok_short:
                return decision

        price_ref = hist.get("mid", hist.get("last", None))
        if price_ref is None or i < 1:
            return decision
        if not np.isfinite(price_ref[i]) or not np.isfinite(price_ref[i - 1]):
            return decision
        dmid_ticks = (float(price_ref[i]) - float(price_ref[i - 1])) / self.tick_size
        if np.isfinite(dmid_ticks) and abs(dmid_ticks) > float(p.V7_DMID1_MAX):
            return decision

        entry_px = float(hist["last"][i])
        decision.signal = direction
        decision.family = "PFLFT_v7"
        decision.reason = "PFLFT_V7_PREBURST"
        decision.setup_bar = i
        decision.entry_lvl = entry_px
        if direction > 0:
            decision.stop_price = self._sub_ticks(entry_px, p.V7_SL_TICKS)
            decision.tp1_price = self._add_ticks(entry_px, p.V7_TP_TICKS)
            decision.tp2_price = self._add_ticks(entry_px, p.V7_TP_TICKS)
        else:
            decision.stop_price = self._add_ticks(entry_px, p.V7_SL_TICKS)
            decision.tp1_price = self._sub_ticks(entry_px, p.V7_TP_TICKS)
            decision.tp2_price = self._sub_ticks(entry_px, p.V7_TP_TICKS)
        decision.time_stop_bars = int(p.V7_TIME_STOP_BARS)
        decision.proof_bars = int(p.V7_PROOF_BARS)
        decision.proof_ticks = int(p.V7_PROOF_TICKS)
        decision.proof_min_flow = float(p.V7_PROOF_MIN_FLOW)
        decision.flow_sum_signed = sv_pre3
        decision.flow_intensity = flowint_pre3
        decision.spread_ticks = spread_ticks
        decision.dmid_ticks = dmid_ticks
        return decision

    def _pflft_v8(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pflft
        if not bool(p.V8_ENABLE):
            return EntryDecision()

        decision = EntryDecision()
        regime_ok = hist.get("regime_ok")
        if regime_ok is not None:
            try:
                if not bool(regime_ok[i]):
                    return decision
            except Exception:
                return decision

        spread_ticks = float(hist["spread_ticks"][i])
        if spread_ticks > float(p.V7_SPREAD_MAX):
            return decision

        ofid = hist["ofid"]
        if i < 3:
            return decision
        pre = ofid[i - 3 : i]
        if pre.size < 3:
            return decision
        sv_pre3 = float(np.nansum(pre))
        flowint_pre3 = float(np.nansum(np.abs(pre)))
        if not np.isfinite(sv_pre3) or not np.isfinite(flowint_pre3):
            return decision
        if abs(sv_pre3) < float(p.V7_SV_PRE3_MIN):
            return decision
        if flowint_pre3 < float(p.V7_FLOWINT_PRE3_MIN):
            return decision

        direction = 1 if sv_pre3 > 0 else -1
        allow_mismatch = bool(getattr(p, "V7_ALLOW_MISMATCH", False))
        if not allow_mismatch:
            if direction > 0 and not gates_ok_long:
                return decision
            if direction < 0 and not gates_ok_short:
                return decision

        price_ref = hist.get("mid", hist.get("last", None))
        if price_ref is None or i < 1:
            return decision
        if not np.isfinite(price_ref[i]) or not np.isfinite(price_ref[i - 1]):
            return decision
        dmid_ticks = (float(price_ref[i]) - float(price_ref[i - 1])) / self.tick_size
        if np.isfinite(dmid_ticks) and abs(dmid_ticks) > float(p.V7_DMID1_MAX):
            return decision

        # v8 microstructure regime gate
        lfp_buy = hist.get("lfp_event_buy_10")
        lfp_sell = hist.get("lfp_event_sell_10")
        stress_ratio = hist.get("stress_ratio")
        depth_total_top5 = hist.get("depth_total_top5")
        imb_delta = hist.get("imbalance_delta")
        toxicity_3 = hist.get("ec_entry_toxicity_3")
        if (
            lfp_buy is None
            or lfp_sell is None
            or stress_ratio is None
            or depth_total_top5 is None
            or imb_delta is None
        ):
            return decision
        lfp_aligned = float(lfp_buy[i]) if direction > 0 else float(lfp_sell[i])
        stress_val = float(stress_ratio[i])
        depth_val = float(depth_total_top5[i])
        imb_aligned = float(imb_delta[i]) * float(direction)
        if (
            not np.isfinite(lfp_aligned)
            or not np.isfinite(stress_val)
            or not np.isfinite(depth_val)
            or not np.isfinite(imb_aligned)
        ):
            return decision
        if lfp_aligned < float(p.V8_LFP_ALIGNED_10_MIN):
            return decision
        if stress_val < float(p.V8_STRESS_RATIO_MIN):
            return decision
        if depth_val > float(p.V8_DEPTH_TOTAL_TOP5_MAX):
            return decision
        if imb_aligned < float(p.V8_ALIGNED_IMB_DELTA_MIN):
            return decision
        # Causal toxicity proxy from pre-entry bars (t-3..t-1).
        # Goal: approximate short-horizon adverse pressure without future leakage.
        imb_pre = np.asarray(imb_delta[i - 3 : i], dtype=float)
        stress_pre = np.asarray(stress_ratio[i - 3 : i], dtype=float)
        depth_pre = np.asarray(depth_total_top5[i - 3 : i], dtype=float)
        spread_pre = np.asarray(hist["spread_ticks"][i - 3 : i], dtype=float)
        aligned_pre = np.asarray(pre, dtype=float) * float(direction)
        opp_flow_mag = float(np.nansum(np.clip(-aligned_pre, 0.0, None)))
        total_flow_mag = float(np.nansum(np.abs(pre)))
        opp_flow_frac_3 = float(opp_flow_mag / max(total_flow_mag, 1e-9))
        imb_aligned_3 = float(np.nanmean(imb_pre * float(direction))) if imb_pre.size else float("nan")
        imb_mismatch_raw = float(np.nanmean(np.clip(-(imb_pre * float(direction)), 0.0, None))) if imb_pre.size else float("nan")
        imb_scale = max(abs(float(p.V8_ALIGNED_IMB_DELTA_MIN)), 1e-6)
        imb_mismatch_3 = float(np.clip(imb_mismatch_raw / imb_scale, 0.0, 1.0))
        stress_mean_3 = float(np.nanmean(stress_pre)) if stress_pre.size else float("nan")
        depth_mean_3 = float(np.nanmean(depth_pre)) if depth_pre.size else float("nan")
        spread_mean_3 = float(np.nanmean(spread_pre)) if spread_pre.size else float("nan")
        stress_scale = max(float(p.V8_STRESS_RATIO_MIN), 1e-6)
        stress_norm = float(np.clip(stress_mean_3 / stress_scale, 0.0, 3.0) / 3.0)
        depth_thin = float(
            np.clip(
                (float(p.V8_DEPTH_TOTAL_TOP5_MAX) - depth_mean_3)
                / max(float(p.V8_DEPTH_TOTAL_TOP5_MAX), 1e-6),
                0.0,
                1.0,
            )
        )
        spread_wide = float(np.clip((spread_mean_3 - 1.0) / 2.0, 0.0, 1.0))
        thin_stress_3 = float(0.50 * stress_norm + 0.35 * depth_thin + 0.15 * spread_wide)
        w1 = float(max(0.0, p.V8_TOX_PROXY_OPPFLOW_W))
        w2 = float(max(0.0, p.V8_TOX_PROXY_IMB_W))
        w3 = float(max(0.0, p.V8_TOX_PROXY_THINSTRESS_W))
        wsum = max(w1 + w2 + w3, 1e-9)
        tox_proxy_pre3 = float((w1 * opp_flow_frac_3 + w2 * imb_mismatch_3 + w3 * thin_stress_3) / wsum)
        if (
            not np.isfinite(opp_flow_frac_3)
            or not np.isfinite(imb_mismatch_3)
            or not np.isfinite(thin_stress_3)
            or not np.isfinite(tox_proxy_pre3)
        ):
            return decision
        if bool(getattr(p, "V8_TOX_PROXY_ENABLE", False)) and tox_proxy_pre3 > float(p.V8_TOX_PROXY_MAX):
            return decision
        # Optional: only applied when a causal toxicity series is present in runtime hist.
        if toxicity_3 is not None:
            toxicity_val = float(toxicity_3[i])
            if not np.isfinite(toxicity_val):
                return decision
            if toxicity_val > float(p.V8_TOXICITY_MAX):
                return decision

        entry_px = float(hist["last"][i])
        decision.signal = direction
        decision.family = "PFLFT_v8"
        decision.reason = "PFLFT_V8_PREBURST_REGIME"
        decision.setup_bar = i
        decision.entry_lvl = entry_px
        if direction > 0:
            decision.stop_price = self._sub_ticks(entry_px, p.V7_SL_TICKS)
            decision.tp1_price = self._add_ticks(entry_px, p.V7_TP_TICKS)
            decision.tp2_price = self._add_ticks(entry_px, p.V7_TP_TICKS)
        else:
            decision.stop_price = self._add_ticks(entry_px, p.V7_SL_TICKS)
            decision.tp1_price = self._sub_ticks(entry_px, p.V7_TP_TICKS)
            decision.tp2_price = self._sub_ticks(entry_px, p.V7_TP_TICKS)
        decision.time_stop_bars = int(p.V7_TIME_STOP_BARS)
        decision.proof_bars = int(p.V7_PROOF_BARS)
        decision.proof_ticks = int(p.V7_PROOF_TICKS)
        decision.proof_min_flow = float(p.V7_PROOF_MIN_FLOW)
        decision.flow_sum_signed = sv_pre3
        decision.flow_intensity = flowint_pre3
        decision.spread_ticks = spread_ticks
        decision.dmid_ticks = dmid_ticks
        decision.v8_opp_flow_frac_3 = float(opp_flow_frac_3)
        decision.v8_imb_mismatch_3 = float(imb_mismatch_3)
        decision.v8_thin_stress_3 = float(thin_stress_3)
        decision.v8_toxicity_proxy_pre3 = float(tox_proxy_pre3)
        decision.v8_imb_aligned_3 = float(imb_aligned_3)
        decision.v8_stress_mean_3 = float(stress_mean_3)
        decision.v8_depth_mean_3 = float(depth_mean_3)
        return decision

    def _pflft_v3(self, i: int, hist: Dict[str, np.ndarray], gates_ok_long: bool, gates_ok_short: bool) -> EntryDecision:
        p = self.params.pflft
        if not bool(p.V3_ENABLE):
            return EntryDecision()

        last = float(hist["last"][i])
        spread_ticks = float(hist["spread_ticks"][i])
        ofid = hist["ofid"]
        price_ref = hist.get("mid", hist["last"])
        depth_bid = hist.get("depth_bid", np.array([], dtype=float))
        depth_ask = hist.get("depth_ask", np.array([], dtype=float))
        decision = EntryDecision()

        if spread_ticks > float(p.V3_SPREAD_MAX):
            return decision

        w = int(p.FLOW_WIN_BARS)
        if w <= 1 or i < w:
            return decision

        start = max(0, i - w + 1)
        flow_window = ofid[start : i + 1]
        if flow_window.size == 0:
            return decision
        flow_sum_signed = float(np.nansum(flow_window))
        flow_intensity = float(np.nansum(np.abs(flow_window)))
        if flow_intensity < float(p.V3_FLOW_INTENSITY_MIN):
            return decision

        dmid_ticks = float("nan")
        if i - w >= 0:
            dmid_ticks = (float(price_ref[i]) - float(price_ref[i - w])) / self.tick_size
        if not np.isfinite(dmid_ticks):
            return decision
        if dmid_ticks <= float(p.V3_DMID_MIN_TICKS) or dmid_ticks > float(p.V3_DMID_MAX_TICKS):
            return decision

        lag_dmid_floor = max(float(p.LAG_DMID_FLOOR_TICKS), 1e-9)
        lag_score = flow_intensity / max(abs(dmid_ticks), lag_dmid_floor)
        if np.isfinite(lag_score) and lag_score > float(p.V3_LAG_SCORE_MAX):
            return decision

        # Optional depth-state filters
        depth_min = float("nan")
        depth_imb = float("nan")
        depth_repl = float("nan")
        if depth_bid.size and depth_ask.size:
            bid_depth = float(depth_bid[i])
            ask_depth = float(depth_ask[i])
            if np.isfinite(bid_depth) and np.isfinite(ask_depth):
                depth_min = float(min(bid_depth, ask_depth))
                denom = bid_depth + ask_depth
                if abs(denom) > 1e-9:
                    depth_imb = float((bid_depth - ask_depth) / denom)
            ds = max(1, i - 19)
            tot = depth_bid[ds - 1 : i + 1] + depth_ask[ds - 1 : i + 1]
            if tot.size >= 3:
                d = np.diff(tot.astype(float))
                up = float(np.nansum(np.clip(d, 0, None)))
                down = float(np.nansum(np.clip(-d, 0, None)))
                depth_repl = float(up / max(down, 1e-9))
        if np.isfinite(depth_imb) and depth_imb < float(p.V3_DEPTH_IMB_MIN):
            return decision
        if np.isfinite(depth_repl) and depth_repl < float(p.V3_DEPTH_REPLENISH_MIN):
            return decision

        # v3 default is long-only.
        if p.V3_LONG_ONLY:
            if not gates_ok_long or flow_sum_signed <= 0:
                return decision
            side = 1
        else:
            side = 0
            if flow_sum_signed > 0 and gates_ok_long and dmid_ticks > 0:
                side = 1
            elif flow_sum_signed < 0 and gates_ok_short and dmid_ticks < 0:
                side = -1
            if side == 0:
                return decision

        entry_px = last
        decision.signal = side
        decision.family = "PFLFT_v3"
        decision.reason = "PFLFT_V3_LONG" if side > 0 else "PFLFT_V3_SHORT"
        decision.setup_bar = i
        decision.entry_lvl = entry_px
        if side > 0:
            decision.stop_price = self._sub_ticks(entry_px, p.V3_STOP_TICKS)
            decision.tp1_price = self._add_ticks(entry_px, p.V3_TP_TICKS)
            decision.tp2_price = self._add_ticks(entry_px, p.V3_TP_TICKS)
        else:
            decision.stop_price = self._add_ticks(entry_px, p.V3_STOP_TICKS)
            decision.tp1_price = self._sub_ticks(entry_px, p.V3_TP_TICKS)
            decision.tp2_price = self._sub_ticks(entry_px, p.V3_TP_TICKS)
        decision.time_stop_bars = int(p.V3_TIME_STOP_BARS)
        decision.flow_sum_signed = flow_sum_signed
        decision.flow_intensity = flow_intensity
        decision.lag_score = lag_score
        decision.spread_ticks = spread_ticks
        decision.dmid_ticks = dmid_ticks
        decision.depth_min = depth_min
        self.pflft_stats["candidates"] += 1
        return decision


def entry_alpha_self_test(tick_size: float = 0.25) -> None:
    params = EntryAlphaParams()
    eng = EntryAlphaEngine(params, tick_size)
    hist = {
        "last": np.array([1, 2, 3, 4, 5, 6, 7], dtype=float),
        "high": np.array([1, 2, 3, 4, 5, 6, 7], dtype=float),
        "low": np.array([1, 2, 3, 4, 5, 6, 7], dtype=float),
        "spread_ticks": np.zeros(7, dtype=float),
        "range_ticks": np.ones(7, dtype=float) * 3,
        "ofid": np.ones(7, dtype=float) * 50,
        "trades": np.ones(7, dtype=float) * 10,
        "depth_bid": np.ones(7, dtype=float) * 10,
        "depth_ask": np.ones(7, dtype=float) * 10,
    }
    # OBRA cancel on timeout
    eng.params.obra.N = 1
    eng.params.obra.RETEST_MAX_WAIT = 1
    eng.obra_state = OBRAState(armed=True, side=1, lvl=5.0, arm_bar=0, ofi_arm=50.0)
    _ = eng.compute(6, hist, gates_ok_long=False, gates_ok_short=False)
    assert not eng.obra_state.armed
    # OBRA cancel on fail breach
    eng.params.obra.N = 1
    eng.obra_state = OBRAState(armed=True, side=1, lvl=5.0, arm_bar=3, ofi_arm=50.0)
    hist["last"][6] = 0.0
    _ = eng.compute(6, hist, gates_ok_long=False, gates_ok_short=False)
    assert not eng.obra_state.armed
    # gates_ok False still allows cancel
    eng.obra_state = OBRAState(armed=True, side=1, lvl=5.0, arm_bar=0, ofi_arm=50.0)
    _ = eng.compute(6, hist, gates_ok_long=False, gates_ok_short=False)
    assert not eng.obra_state.armed
    # tick conversion sanity
    assert abs(eng._add_ticks(1.0, 4) - (1.0 + 4 * tick_size)) < 1e-9
