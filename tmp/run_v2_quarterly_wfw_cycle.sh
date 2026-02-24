#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/freqtrade

STRAT="PerpPathTrendOnlyV1Lev3LongShortStandaloneElasticV2Pyramid"
CONFIG="user_data/config_perppath.json"
STRAT_PATH="user_data/strategies"
PARAM_FILE="user_data/strategies/${STRAT}.json"
ARCHIVE_DIR="user_data/strategies/params_archive"
WFW_CSV="user_data/backtest_results/wf_${STRAT}_6pairs_mot3_sf0.csv"
TS="$(date +%F)"

PAIRS=(
  BTC/USDT:USDT ETH/USDT:USDT BNB/USDT:USDT
  SOL/USDT:USDT XRP/USDT:USDT DOGE/USDT:USDT
)

TRAIN_RANGE="20250701-20251001"
TEST_RANGE="20251001-20260101"
TRAIN_LABEL="2025Q3"
TEST_LABEL="2025Q4"

mkdir -p "$ARCHIVE_DIR" tmp

if [[ ! -x /tmp/run_wf_generic_6pairs_mot3.sh ]]; then
  echo "Missing /tmp/run_wf_generic_6pairs_mot3.sh"
  exit 1
fi

echo "== 1) Quarterly WFW rerun for $STRAT =="
bash /tmp/run_wf_generic_6pairs_mot3.sh "$STRAT" "/tmp/${STRAT}.wf.csv"
cp "/tmp/${STRAT}.wf.csv" "$WFW_CSV"

echo "== 2) WFW summary (OOS 9 segments) =="
awk -F, 'NR>1{sum+=$8; dd+=$13; n++} END{printf("avg_quarterly_oos=%.2f%%\navg_quarterly_dd=%.2f%%\nsegments=%d\n", sum/n, dd/n, n)}' "$WFW_CSV"

echo "== 3) Backup current PROD params =="
PROD_BACKUP="${ARCHIVE_DIR}/${STRAT}__preupdate_backup__${TS}_$(date +%H%M%S).json"
cp "$PARAM_FILE" "$PROD_BACKUP"
echo "$PROD_BACKUP"

echo "== 4) Train latest quarterly candidate params (${TRAIN_LABEL}) =="
freqtrade hyperopt \
  --config "$CONFIG" \
  --strategy-path "$STRAT_PATH" \
  --strategy "$STRAT" \
  --timerange "$TRAIN_RANGE" \
  --max-open-trades 3 \
  -p "${PAIRS[@]}" \
  --spaces buy sell \
  -e 30 -j 4 \
  --random-state 42 \
  --min-trades 10 \
  --hyperopt-loss ProfitDrawDownHyperOptLoss \
  --early-stop 10

CAND_FILE="${ARCHIVE_DIR}/${STRAT}__candidate__train_${TRAIN_LABEL}__val_${TEST_LABEL}__${TS}.json"
cp "$PARAM_FILE" "$CAND_FILE"
echo "Candidate archived: $CAND_FILE"

latest_zip() { ls -1t user_data/backtest_results/backtest-result-*.zip | head -1; }
extract_metrics() {
  local zip="$1"
  local mainjson
  mainjson=$(unzip -Z1 "$zip" | rg '^backtest-result-.*\.json$' | rg -v '_config\.json$' | head -1)
  unzip -p "$zip" "$mainjson" | jq -r '.strategy_comparison[0] | [(.profit_total_pct // 0), ((.max_drawdown_account // 0)*100), (.trades // 0), (.sharpe // 0), (.profit_factor // 0)] | @csv'
}

echo "== 5) Candidate OOS validation on ${TEST_LABEL} (${TEST_RANGE}) =="
freqtrade backtesting \
  --config "$CONFIG" \
  --strategy-path "$STRAT_PATH" \
  --strategy "$STRAT" \
  --timerange "$TEST_RANGE" \
  --max-open-trades 3 \
  -p "${PAIRS[@]}" \
  --export trades >/dev/null
CAND_METRICS="$(extract_metrics "$(latest_zip)")"

echo "== 6) Current PROD OOS validation on same window =="
cp "$PROD_BACKUP" "$PARAM_FILE"
freqtrade backtesting \
  --config "$CONFIG" \
  --strategy-path "$STRAT_PATH" \
  --strategy "$STRAT" \
  --timerange "$TEST_RANGE" \
  --max-open-trades 3 \
  -p "${PAIRS[@]}" \
  --export trades >/dev/null
PROD_METRICS="$(extract_metrics "$(latest_zip)")"

echo "== 7) Compare summary =="
python3 - <<'PY' "$CAND_METRICS" "$PROD_METRICS" "$CAND_FILE" "$PROD_BACKUP" "$TRAIN_LABEL" "$TEST_LABEL"
import sys
c = [float(x) for x in sys.argv[1].split(",")]
p = [float(x) for x in sys.argv[2].split(",")]
cand_file, prod_backup, train_label, test_label = sys.argv[3:7]
print(f"candidate_file={cand_file}")
print(f"prod_backup={prod_backup}")
print(f"train={train_label}  test={test_label}")
print(f"candidate_oos_profit={c[0]:.2f}%  dd={c[1]:.2f}%  trades={int(c[2])}  sharpe={c[3]:.2f}  pf={c[4]:.2f}")
print(f"prod_oos_profit={p[0]:.2f}%       dd={p[1]:.2f}%  trades={int(p[2])}  sharpe={p[3]:.2f}  pf={p[4]:.2f}")
print(f"diff_profit={c[0]-p[0]:+.2f}%  diff_dd={c[1]-p[1]:+.2f}%")
PY

echo
echo "Promote candidate if desired:"
echo "cp \"$CAND_FILE\" \"$PARAM_FILE\""
