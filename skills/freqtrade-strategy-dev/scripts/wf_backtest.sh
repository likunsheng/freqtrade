#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   wf_backtest.sh <timerange> <config> <strategy> <freqaimodel>
# Example:
#   wf_backtest.sh 20250601-20260101 user_data/config_bt_freqai_5m_mtf.json PerpPathFreqaiStrategy XGBoostClassifier

TIMERANGE="${1:-}"
CFG="${2:-}"
STRAT="${3:-}"
MODEL="${4:-}"

if [[ -z "${TIMERANGE}" || -z "${CFG}" || -z "${STRAT}" ]]; then
  echo "Usage: wf_backtest.sh <timerange> <config> <strategy> <freqaimodel>" >&2
  exit 2
fi

FEE="${FEE:-0.0004}"
WALLET="${WALLET:-1000}"

echo "[wf_backtest] timerange=${TIMERANGE} cfg=${CFG} strat=${STRAT} model=${MODEL:-<none>}"

if [[ -n "${MODEL}" ]]; then
  freqtrade backtesting -c "${CFG}" -s "${STRAT}" --freqaimodel "${MODEL}" \
    --timerange "${TIMERANGE}" --dry-run-wallet "${WALLET}" --fee "${FEE}" \
    --export trades --breakdown month
else
  freqtrade backtesting -c "${CFG}" -s "${STRAT}" \
    --timerange "${TIMERANGE}" --dry-run-wallet "${WALLET}" --fee "${FEE}" \
    --export trades --breakdown month
fi

