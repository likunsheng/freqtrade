#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   wf_hyperopt.sh <timerange> [epochs]
# Example:
#   wf_hyperopt.sh 20250601-20260101 150

TIMERANGE="${1:-}"
EPOCHS="${2:-150}"

if [[ -z "${TIMERANGE}" ]]; then
  echo "Usage: wf_hyperopt.sh <timerange> [epochs]" >&2
  exit 2
fi

CFG="user_data/config_bt_freqai_5m_mtf.json"
STRAT="PerpPathFreqaiStrategy"
MODEL="XGBoostClassifier"
LOSS="SampleHyperOptLoss"
LOSS_PATH="user_data/hyperopts"

FEE="${FEE:-0.0004}"
WALLET="${WALLET:-1000}"
JOBS="${JOBS:-1}"
MIN_TRADES="${MIN_TRADES:-30}"

echo "[wf_hyperopt] timerange=${TIMERANGE} epochs=${EPOCHS} jobs=${JOBS}"

freqtrade hyperopt -c "${CFG}" -s "${STRAT}" --freqaimodel "${MODEL}" \
  --timerange "${TIMERANGE}" --dry-run-wallet "${WALLET}" --fee "${FEE}" \
  --spaces buy sell -e "${EPOCHS}" -j "${JOBS}" \
  --hyperopt-loss "${LOSS}" --hyperopt-path "${LOSS_PATH}" --min-trades "${MIN_TRADES}"

echo "[wf_hyperopt] Done. Params exported to user_data/strategies/${STRAT}.json"

