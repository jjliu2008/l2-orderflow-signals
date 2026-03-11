from __future__ import annotations

import pandas as pd

from research_dashboard.services import metrics


def test_profit_factor_basic():
    s = pd.Series([2, -1, 3, -2])
    pf = metrics.profit_factor(s)
    assert pf is not None
    assert round(pf, 6) == round((2 + 3) / (1 + 2), 6)


def test_max_drawdown_ticks():
    s = pd.Series([5, -3, 2, -10, 4])
    dd = metrics.max_drawdown_ticks(s)
    assert dd is not None
    # equity path: 5,2,4,-6,-2 => max drawdown is -11 from 5 to -6
    assert round(dd, 6) == -11.0


def test_outlier_removed_metrics_shapes():
    s = pd.Series([10, 8, 5, -2, -3, -4])
    m = metrics.outlier_removed_metrics(s, remove_top_n=1, remove_bottom_n=1)
    assert m["n"] == 4
    assert m["ev_per_trade"] is not None


def test_monthly_breakdown_handles_duplicate_column_names():
    df = pd.DataFrame(
        {
            "month": ["2026-01", "2026-01", "2026-02"],
            "ticks_pnl_net": [1.0, -2.0, 3.0],
            "ticks_pnl_gross": [1.2, -1.8, 3.2],
        }
    )
    # Simulate accidental duplicate target name from rename collisions.
    dup = df.rename(columns={"ticks_pnl_gross": "ticks_pnl_net"})
    out = metrics.monthly_breakdown(dup, pnl_col="ticks_pnl_net")
    assert "month" in out.columns
    assert len(out) == 2
