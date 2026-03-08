#!/usr/bin/env bash
# Run frozen PFLFT_v8 backtest over ESH6 Jan 2026 for OOS signal generation.
# Runs in 3 batches to stay within memory limits.
# DATA_DIR points at isolated ESH6-only parquets to avoid loading ESZ5 data.
# After all batches, consolidates per-date trade dirs into artifacts/esh6_oos_trades/
# so the absorption detector can use: --trades-root artifacts/esh6_oos_trades
#
# Run from project root: bash scripts/run_esh6_oos_backtest.sh

set -e
cd "$(dirname "$0")/.."

BASE_ENV=(
  STRATEGY_MODE=entry_alpha_v1
  ENTRY_ALPHA_FAMILY_ALLOWLIST=PFLFT_v8
  GATE_MODE=side_matched
  OUTPUT_DIR=artifacts/esh6_oos
  DATA_DIR=data/processed_esh6
  VALIDATE_DEBUG=0
  ENTRY_CONFIRM_STYLE=price_only
  ENTRY_CONFIRM_BARS=2
  ENTRY_ALPHA_ALLOW_MISMATCH_V7=1
  DISABLE_GATE=0
  PNL_TICK_VALUE=12.5
  PNL_COMMISSION_ROUND_TURN=1.2
  PNL_SLIPPAGE_TICKS=1
  ENTRY_COOLDOWN_BARS=0
  ENTRY_MIN_PROGRESS_TICKS=0
  PFLFT_V7_ENABLE=1
  PFLFT_V7_SPREAD_MAX=1
  PFLFT_V7_FLOWINT_PRE3_MIN=15
  PFLFT_V7_SV_PRE3_MIN=12
  PFLFT_V7_DMID1_MAX=1
  PFLFT_V7_PROOF_BARS=2
  PFLFT_V7_PROOF_TICKS=1
  PFLFT_V7_PROOF_MIN_FLOW=0
  PFLFT_V7_TIME_STOP_BARS=5
  PFLFT_V7_TP_TICKS=1
  PFLFT_V7_SL_TICKS=1
  PFLFT_V7_FLOWINT_PRE3_P90=25
  PFLFT_V7_RUNNER_TP_TICKS=4
  PFLFT_V7_PREV_RANGE_MAX_TICKS=15
  PFLFT_V7_PREV_ABS_FLOW_MAX=20000
  PFLFT_V8_ENABLE=1
  PFLFT_V8_LFP_ALIGNED_10_MIN=0.58
  PFLFT_V8_STRESS_RATIO_MIN=0.18
  PFLFT_V8_DEPTH_TOTAL_TOP5_MAX=640
  PFLFT_V8_ALIGNED_IMB_DELTA_MIN=0.05
  PFLFT_V8_TOXICITY_MAX=0.156
  PFLFT_V8_TOX_PROXY_MAX=0
)

run_batch() {
  local start="$1" end="$2" label="$3"
  echo "=== Batch $label: $start to $end ==="
  env "${BASE_ENV[@]}" START_DATE="$start" END_DATE="$end" \
    python scripts/run_lrams_gate_backtest.py
}

# Three batches to stay within memory limits (~7 days each)
run_batch 2026-01-02 2026-01-09 "1/3"
run_batch 2026-01-12 2026-01-20 "2/3"
run_batch 2026-01-21 2026-01-30 "3/3"

# Consolidate: copy all per-date trade dirs into a single canonical tree
TRADES_DIR="artifacts/esh6_oos_trades"
mkdir -p "$TRADES_DIR/W10/mode=side_matched"
echo ""
echo "=== Consolidating trade dirs into $TRADES_DIR ==="
for src in artifacts/esh6_oos/run_*/W10/mode=side_matched/ES_2026-*; do
  [ -d "$src" ] || continue
  date_dir=$(basename "$src")
  dest="$TRADES_DIR/W10/mode=side_matched/$date_dir"
  if [ -d "$dest" ]; then
    echo "  $date_dir already consolidated, skipping"
  else
    cp -r "$src" "$dest"
    echo "  copied $date_dir"
  fi
done

echo ""
echo "Done. Gated trade dirs in: $TRADES_DIR"
echo ""
echo "Next step — run absorption detector per date, e.g.:"
echo "  python scripts/run_absorption_detector.py --date 2026-01-02 \\"
echo "    --trades-root $TRADES_DIR \\"
echo "    --dbn-root c:/Users/majin/Downloads/ESH6-MBP10 \\"
echo "    --symbol ESH6 \\"
echo "    --output-dir artifacts/absorption_detector_esh6"
