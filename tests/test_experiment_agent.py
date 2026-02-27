from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.experiment_agent import _compute_cost_ticks, _load_trades


class ExperimentAgentTests(unittest.TestCase):
    def test_compute_cost_ticks_modes(self) -> None:
        env = {
            "PNL_TICK_VALUE": "12.5",
            "PNL_COMMISSION_ROUND_TURN": "1.25",
            "PNL_SLIPPAGE_TICKS": "1",
        }
        self.assertAlmostEqual(_compute_cost_ticks(env, "commission_only"), 0.1, places=9)
        self.assertAlmostEqual(_compute_cost_ticks(env, "round_trip"), 1.1, places=9)
        self.assertAlmostEqual(_compute_cost_ticks(env, "none"), 0.0, places=9)
        with self.assertRaises(ValueError):
            _compute_cost_ticks(env, "bad_mode")

    def test_load_trades_prefers_side_matched_and_filters_gated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)

            non_side = run_dir / "trades_entry_alpha_v1_W05_gate_dummy_baseline_vs_gated.csv"
            side_old = run_dir / "trades_entry_alpha_v1_W10_gate_side_matched_baseline_vs_gated.csv"

            # Create a non-side-matched file and a side-matched file.
            pd.DataFrame(
                [
                    {"strategy": "gated", "pnl_ticks": -9.0},
                    {"strategy": "baseline", "pnl_ticks": 99.0},
                ]
            ).to_csv(non_side, index=False)
            pd.DataFrame(
                [
                    {"strategy": "gated", "pnl_ticks": 1.0},
                    {"strategy": "baseline", "pnl_ticks": 2.0},
                ]
            ).to_csv(side_old, index=False)

            # Touch side file after non-side so selection is deterministic.
            side_old.touch()
            non_side.touch()

            loaded = _load_trades(run_dir)
            self.assertTrue((loaded["strategy"] == "gated").all())
            self.assertEqual(len(loaded), 1)
            self.assertEqual(float(loaded.iloc[0]["pnl_ticks"]), 1.0)


if __name__ == "__main__":
    unittest.main()
