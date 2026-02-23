# Single Strategy Baseline

- Date: 2026-02-20
- Mode: Single strategy only (no portfolio / no ensemble)
- Selected strategy: `PerpPathTrendOnlyV1.py`
- Config: `user_data/config_bt_pairarb_eth_btc_5m.json`
- Timerange used for screening: `20220101-20260201`
- Result source: `user_data/backtest_results/single_strategy_screen_202201_202602.csv`

## Screening Result

- `PerpPathTrendOnlyV1`: `+47.819871179%`, trades `222`, max_drawdown `12.811095943051184%`
- Other screened strategies: all negative return.

## Decision

- Keep only `PerpPathTrendOnlyV1.py` in active strategies directory.
- Non-selected strategy files moved to:
  `user_data/strategies/_removed_negative_20260220/`
