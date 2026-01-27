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
class EntryAlphaParams:
    obra: OBRAParams = field(default_factory=OBRAParams)
    apb: APBParams = field(default_factory=APBParams)
    pbra: PBRAParams = field(default_factory=PBRAParams)


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
class EntryDecision:
    signal: int = 0
    family: str = ""
    reason: str = ""
    entry_lvl: float | None = None
    stop_price: float | None = None
    tp1_price: float | None = None
    tp2_price: float | None = None
    time_stop_bars: int | None = None


class EntryAlphaEngine:
    def __init__(self, params: EntryAlphaParams, tick_size: float) -> None:
        self.params = params
        self.tick_size = float(tick_size)
        self.obra_state = OBRAState()
        self.apb_state = APBState()
        self.pbra_state = PBRAState()

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
