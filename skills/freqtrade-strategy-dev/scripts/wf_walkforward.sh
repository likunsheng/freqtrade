#!/usr/bin/env bash
set -euo pipefail

# Simple walk-forward runner (backtesting only).
# Usage:
#   wf_walkforward.sh <start_yyyymmdd> <end_yyyymmdd> [step_days]
# Example:
#   wf_walkforward.sh 20250601 20260101 30

START="${1:-}"
END="${2:-}"
STEP_DAYS="${3:-30}"

if [[ -z "${START}" || -z "${END}" ]]; then
  echo "Usage: wf_walkforward.sh <start_yyyymmdd> <end_yyyymmdd> [step_days]" >&2
  exit 2
fi

CFG="user_data/config_bt_freqai_5m_mtf.json"
STRAT="PerpPathFreqaiStrategy"
MODEL="XGBoostClassifier"

FEE="${FEE:-0.0004}"
WALLET="${WALLET:-1000}"

to_ts() {
  python - <<PY
import datetime as d
s="${1}"
dt=d.datetime.strptime(s,"%Y%m%d").replace(tzinfo=d.UTC)
print(int(dt.timestamp()))
PY
}

START_TS="$(to_ts "${START}")"
END_TS="$(to_ts "${END}")"
STEP_SEC="$((STEP_DAYS * 86400))"

ts="${START_TS}"
while [[ "${ts}" -lt "${END_TS}" ]]; do
  nxt="$((ts + STEP_SEC))"
  if [[ "${nxt}" -gt "${END_TS}" ]]; then
    nxt="${END_TS}"
  fi
  sdate="$(python - <<PY
import datetime as d
print(d.datetime.fromtimestamp(${ts}, d.UTC).strftime("%Y%m%d"))
PY
)"
  edate="$(python - <<PY
import datetime as d
print(d.datetime.fromtimestamp(${nxt}, d.UTC).strftime("%Y%m%d"))
PY
)"
  timerange="${sdate}-${edate}"
  echo "=== [wf_walkforward] Backtest ${timerange} ==="
  freqtrade backtesting -c "${CFG}" -s "${STRAT}" --freqaimodel "${MODEL}" \
    --timerange "${timerange}" --dry-run-wallet "${WALLET}" --fee "${FEE}" \
    --export trades --breakdown month || true
  ts="${nxt}"
done

