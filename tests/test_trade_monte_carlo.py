from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import json

from scripts.run_trade_monte_carlo import (
    _cost_ticks_from_config,
    _load_net_ticks,
    _max_drawdown_ticks,
    _resolve_cost_ticks,
    _sample_block_paths,
    _sample_iid_paths,
)


class TradeMonteCarloTests(unittest.TestCase):
    def test_max_drawdown_ticks(self) -> None:
        pnl = np.array([1.0, -3.0, 2.0, -2.0, 4.0], dtype=float)
        # Equity: [1, -2, 0, -2, 2], running peak [1,1,1,1,2], drawdowns [0,3,1,3,0]
        self.assertAlmostEqual(_max_drawdown_ticks(pnl), 3.0, places=9)

    def test_bootstrap_path_shapes(self) -> None:
        base = np.array([1.0, -1.0, 2.0, -2.0], dtype=float)
        rng = np.random.default_rng(123)
        iid = _sample_iid_paths(base, n_paths=100, horizon_trades=7, rng=rng)
        self.assertEqual(iid.shape, (100, 7))
        rng = np.random.default_rng(123)
        block = _sample_block_paths(base, n_paths=100, horizon_trades=7, block_size=3, rng=rng)
        self.assertEqual(block.shape, (100, 7))
        self.assertTrue(np.isin(iid, base).all())
        self.assertTrue(np.isin(block, base).all())

    def test_load_net_ticks_filters_strategy_and_cost(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trades.csv"
            pd.DataFrame(
                [
                    {"strategy": "gated", "pnl_ticks": 2.0},
                    {"strategy": "baseline", "pnl_ticks": 99.0},
                    {"strategy": "gated", "pnl_ticks": -1.0},
                ]
            ).to_csv(path, index=False)
            net = _load_net_ticks(path, strategy="gated", pnl_col="pnl_ticks", cost_ticks=1.1)
            self.assertEqual(net.size, 2)
            self.assertAlmostEqual(float(net[0]), 0.9, places=9)
            self.assertAlmostEqual(float(net[1]), -2.1, places=9)

    def test_cost_ticks_resolves_from_config_commission_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "experiment_agent.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "fixed_env": {
                            "PNL_TICK_VALUE": "12.5",
                            "PNL_COMMISSION_ROUND_TURN": "1.2",
                            "PNL_SLIPPAGE_TICKS": "1",
                        },
                        "cost_mode": "commission_only",
                    }
                ),
                encoding="utf-8",
            )
            ticks, mode = _cost_ticks_from_config(cfg_path)
            self.assertAlmostEqual(ticks, 0.096, places=9)
            self.assertEqual(mode, "commission_only")

            resolved_ticks, source, resolved_mode = _resolve_cost_ticks(
                cost_ticks_arg=None,
                config_path_arg=str(cfg_path),
                cost_mode_arg="auto",
            )
            self.assertAlmostEqual(resolved_ticks, 0.096, places=9)
            self.assertEqual(source, "config")
            self.assertEqual(resolved_mode, "commission_only")

    def test_manual_cost_override_takes_priority(self) -> None:
        ticks, source, mode = _resolve_cost_ticks(
            cost_ticks_arg=0.25,
            config_path_arg="",
            cost_mode_arg="auto",
        )
        self.assertAlmostEqual(ticks, 0.25, places=9)
        self.assertEqual(source, "manual")
        self.assertEqual(mode, "manual")


if __name__ == "__main__":
    unittest.main()
