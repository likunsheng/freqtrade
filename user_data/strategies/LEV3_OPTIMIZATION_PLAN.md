# Lev3 Optimization Batch Plan

This plan implements a 3-round optimization workflow for:
- `PerpPathTrendOnlyV1Lev3` (long-only baseline)
- `PerpPathTrendOnlyV1Lev3ConservativeShort` (new long+short variant)

## How To Run

```bash
cd /workspaces/freqtrade
./user_data/scripts/optimize_lev3_batches.sh
```

## Round Design

1. Round 1 (`PerpPathTrendOnlyV1Lev3`, spaces `buy sell`, 220 epochs)
- Purpose: Tighten long-entry + exit parameters as a stable baseline.

2. Round 2 (`PerpPathTrendOnlyV1Lev3ConservativeShort`, spaces `buy sell`, 260 epochs)
- Purpose: Fit conservative short filters while retaining long edge.

3. Round 3 (`PerpPathTrendOnlyV1Lev3ConservativeShort`, spaces `sell trailing stoploss`, 180 epochs)
- Purpose: Improve fail-fast/trailing/stop behavior for drawdown control.

## Parameter Grid (what hyperopt explores)

Long-side core (already in strategy):
- `trend_adx`: 14..35
- `trend_spread`: 0.004..0.035
- `max_btc_vol`: 0.004..0.030
- `basis_abs_max`: 0.004..0.040
- `oi_proxy_max`: 1.0..4.0
- `funding_long_max`: -0.002..0.020
- `trend_quick_tp`: 0.008..0.070
- `trend_hold_minutes`: 180..1440

Short-side conservative (new):
- `short_trend_adx`: 16..40
- `short_trend_spread`: 0.008..0.050
- `short_btc_vol_max`: 0.004..0.040
- `short_basis_abs_max`: 0.004..0.040
- `short_oi_proxy_max`: 1.0..4.0
- `short_funding_min`: -0.010..0.020
- `short_rsi_low`: 28..45
- `short_rsi_high`: 46..62
- `short_fail_fast_minutes`: 30..120
- `short_fail_fast_loss`: -0.030..-0.008

## Notes

- Loss function: `SampleHyperOptLoss` from `user_data/hyperopts/sample_hyperopt_loss.py`.
- Timerange: `20220101-20260201` for direct comparability.
- Final decision should prefer robust OOS behavior, not just in-sample total return.
