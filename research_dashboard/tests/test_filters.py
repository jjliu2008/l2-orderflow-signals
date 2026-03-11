from __future__ import annotations

from datetime import date

import pandas as pd

from research_dashboard.services import filters as flt


def _sample_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": ["T1", "T2", "T3"],
            "strategy_name": ["s", "s", "s2"],
            "variant_name": ["v1", "v2", "v1"],
            "date": ["2026-01-01", "2026-02-01", "2026-02-02"],
            "session": ["cash_open", "lunch", "power_hour"],
            "is_event_day": [False, True, False],
            "direction": ["short", "long", "short"],
            "month": ["2026-01", "2026-02", "2026-02"],
            "weekday": ["Thursday", "Sunday", "Monday"],
            "oos_flag": [False, True, True],
        }
    )


def test_apply_trade_filters_event_and_direction():
    df = _sample_trades()
    f = flt.DashboardFilters(
        strategy_name="s",
        variant_names=["v2"],
        event_mode=flt.EVENT_MODE_EVENT,
        direction_mode=flt.DIRECTION_MODE_LONG,
    )
    out = flt.apply_trade_filters(df, f)
    assert len(out) == 1
    assert out.iloc[0]["trade_id"] == "T2"


def test_apply_trade_filters_date_and_power_hour():
    df = _sample_trades()
    f = flt.DashboardFilters(
        date_start=date(2026, 2, 1),
        date_end=date(2026, 2, 2),
        exclude_power_hour=True,
    )
    out = flt.apply_trade_filters(df, f)
    assert set(out["trade_id"]) == {"T2"}
