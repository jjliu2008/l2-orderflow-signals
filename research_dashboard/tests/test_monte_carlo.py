from __future__ import annotations

import numpy as np

from research_dashboard.services import monte_carlo


def test_bootstrap_mc_shape():
    arr = np.array([1.0, -1.0, 2.0, -0.5])
    paths = monte_carlo.run_bootstrap_mc(arr, num_paths=100, horizon=20, seed=42)
    assert paths.shape == (100, 20)


def test_shuffle_mc_shape():
    arr = np.array([1.0, -1.0, 2.0, -0.5])
    paths = monte_carlo.run_trade_shuffle_mc(arr, num_paths=50, horizon=4, seed=42)
    assert paths.shape == (50, 4)


def test_mc_summary_fields():
    arr = np.array([1.0, -1.0, 2.0, -0.5])
    paths = monte_carlo.run_bootstrap_mc(arr, num_paths=200, horizon=10, seed=1)
    summary = monte_carlo.compute_mc_summary(paths, tick_value=12.5, starting_capital=2000.0)
    assert summary["n_paths"] == 200
    assert summary["horizon"] == 10
    assert summary["prob_end_positive"] is not None
