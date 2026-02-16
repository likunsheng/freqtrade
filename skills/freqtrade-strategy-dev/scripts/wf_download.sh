#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   wf_download.sh <timerange> [pairs...]
# Example:
#   wf_download.sh 20241101-20260101 BTC/USDT:USDT ETH/USDT:USDT

TIMERANGE="${1:-}"
shift || true

if [[ -z "${TIMERANGE}" ]]; then
  echo "Usage: wf_download.sh <timerange> [pairs...]" >&2
  exit 2
fi

CFG="user_data/config_bt_freqai_5m_mtf.json"
PAIRS=("$@")
if [[ ${#PAIRS[@]} -eq 0 ]]; then
  PAIRS=("BTC/USDT:USDT" "ETH/USDT:USDT")
fi

echo "[wf_download] Config: ${CFG}"
echo "[wf_download] Timerange: ${TIMERANGE}"
echo "[wf_download] Pairs: ${PAIRS[*]}"

# More reliable than a single huge request: download per-timeframe.
for tf in 5m 15m 1h; do
  echo "[wf_download] Downloading ${tf} ..."
  freqtrade download-data -c "${CFG}" --trading-mode futures -p "${PAIRS[@]}" -t "${tf}" --timerange "${TIMERANGE}" --prepend
done

echo "[wf_download] Verifying coverage ..."
freqtrade list-data -c "${CFG}" --trading-mode futures --show-timerange

