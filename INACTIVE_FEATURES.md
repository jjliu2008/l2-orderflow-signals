# Inactive Features (Current Orchestrator Path)

This file documents engineered features that are currently inactive for the live experiment loop:

- `scripts/experiment_orchestrator.py`
- `scripts/experiment_agent.py`
- `scripts/run_lrams_gate_backtest.py`
- `strategy_mode=entry_alpha_v1`
- `family_allowlist=["PFLFT_v1"]`

## Why "inactive"

These features are computed in `src/feature_engineering.py` but are not read by the current entry decision path (`src/entry_alpha.py` + `run_lrams_gate_backtest.py` hist arrays).

## Inactive In Live Entry Decisioning

- `order_book_imbalance`
- `depth_bid_top5`
- `depth_ask_top5`
- `depth_imbalance_top5`
- `book_slope_top5`
- `sweep_cost_buy1`
- `sweep_cost_sell1`
- `ret_1`
- `ret_5`
- `ret_10`
- `ret_20`
- `volatility_20`
- `volatility_50`
- `signed_volume` (name exists in feature-engineering path; live path uses `ofid` arrays)
- `cvd_20`
- `volume_trend_20`
- `volume_zscore_20`
- `vol_regime_score`
- `risk_adj_momentum_10`
- `risk_adj_momentum_20`
- `trend_flip_flag`
- `price_drawdown_50`
- `refill_count`
- `refill_latency`
- `absorption_ratio`
- `imbalance_persist`
- `rv_short`

## Notes

- This does **not** mean the features are globally useless; they can still be used by other scripts (model/training workflows).
- For live orchestrator EV work, prefer features computed and consumed directly in `run_lrams_gate_backtest.py` and `entry_alpha.py`.
