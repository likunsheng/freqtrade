from __future__ import annotations

from datetime import datetime
from functools import reduce
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from PerpPathTrendOnlyV1 import PerpPathTrendOnlyV1


class PerpPathTrendOnlyV1Lev3ConservativeShort(PerpPathTrendOnlyV1):
    """
    Lev3 variant that keeps the original long logic and adds conservative short entries.
    """

    can_short = True

    # Short-entry specific knobs (kept in buy-space to co-optimize with long filters).
    short_trend_adx = IntParameter(12, 32, default=18, space="buy", optimize=True)
    short_trend_spread = DecimalParameter(0.006, 0.035, default=0.015, decimals=3, space="buy", optimize=True)
    short_btc_vol_max = DecimalParameter(0.010, 0.060, default=0.030, decimals=3, space="buy", optimize=True)
    short_basis_abs_max = DecimalParameter(0.006, 0.060, default=0.030, decimals=3, space="buy", optimize=True)
    short_oi_proxy_max = DecimalParameter(1.5, 6.0, default=3.5, decimals=1, space="buy", optimize=True)
    short_funding_min = DecimalParameter(-0.005, 0.015, default=-0.001, decimals=3, space="buy", optimize=True)
    short_rsi_low = IntParameter(24, 44, default=30, space="buy", optimize=True)
    short_rsi_high = IntParameter(44, 70, default=58, space="buy", optimize=True)

    # Conservative short fail-fast controls.
    short_fail_fast_minutes = IntParameter(25, 120, default=60, space="sell", optimize=True)
    short_fail_fast_loss = DecimalParameter(-0.035, -0.006, default=-0.015, decimals=3, space="sell", optimize=True)

    leverage_value = 3.0

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return min(float(self.leverage_value), float(max_leverage))

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_indicators(dataframe, metadata)
        # Explicit risk-off filter for shorting.
        df["btc_risk_off"] = (
            (df["btc_fast"] < df["btc_slow"])
            & (df["btc_vol"] <= float(self.short_btc_vol_max.value))
            & (df["btc_mom"] < 0.0)
        )
        df["inf1h_trend_down"] = df["inf1h_ema50"] < df["inf1h_ema200"]
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Keep original long logic unchanged.
        df = super().populate_entry_trend(df, metadata)

        spread = (df["ema50"] - df["ema200"]).abs() / df["close"]
        short_rsi_low = int(self.short_rsi_low.value)
        short_rsi_high = int(self.short_rsi_high.value)

        short_cond = [
            df["btc_risk_off"],
            df["inf1h_trend_down"],
            df["ema20"] < df["ema50"],
            df["ema50"] < df["ema200"],
            df["close"] < df["ema20"],
            df["close"] < df["bb_mid"],
            df["adx"] >= int(self.short_trend_adx.value),
            spread >= float(self.short_trend_spread.value),
            df["rsi"].between(short_rsi_low, short_rsi_high),
            df["funding"] >= float(self.short_funding_min.value),
            df["basis"].abs() <= float(self.short_basis_abs_max.value),
            df["oi_proxy"] <= float(self.short_oi_proxy_max.value),
            df["atr_pct"] > 0.002,
        ]
        df.loc[reduce(lambda x, y: x & y, short_cond), ["enter_short", "enter_tag"]] = (1, "trend_short")
        return df

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        # Reuse long behavior for long trades.
        if not trade.is_short:
            return super().custom_exit(
                pair=pair,
                trade=trade,
                current_time=current_time,
                current_rate=current_rate,
                current_profit=current_profit,
                **kwargs,
            )

        hold_min = int((current_time - trade.open_date_utc).total_seconds() // 60)
        if current_profit >= float(self.trend_quick_tp.value):
            return "trend_short_quick_tp"
        if hold_min >= int(self.short_fail_fast_minutes.value) and current_profit < float(self.short_fail_fast_loss.value):
            return "trend_short_fail_fast"
        if hold_min >= int(self.trend_hold_minutes.value):
            return "trend_short_time_stop"
        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float:
        # Symmetric dynamic stop for both long/short based on realized profit.
        if current_profit >= 0.04:
            return -0.010
        if current_profit >= 0.02:
            return -0.005
        if current_profit <= -0.025:
            return -0.025
        return self.stoploss
