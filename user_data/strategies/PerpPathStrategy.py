# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

import talib.abstract as ta
from technical import qtpylib


class PerpPathStrategy(IStrategy):
    """
    BTC/USDT + ETH/USDT perpetual futures, 1m timeframe.
    Uses rolling-window "path" proxy (order of extrema) with tunable x/y/window.
    Includes ATR-based stoploss (capped), partial take-profit, and trailing stop.
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1m"
    startup_candle_count = 200
    process_only_new_candles = True

    # Exit/stop settings
    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = True
    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    # Hard max loss (custom_stoploss will be tighter than this)
    stoploss = -0.03

    # Disable ROI table - exits are handled by custom_exit/stoploss.
    minimal_roi = {"0": 100}

    # --- Hyperoptable parameters ---
    window = IntParameter(5, 30, default=10, space="buy", optimize=True)
    x_profit = DecimalParameter(0.004, 0.02, default=0.008, decimals=4, space="buy", optimize=True)
    y_risk = DecimalParameter(0.004, 0.02, default=0.010, decimals=4, space="buy", optimize=True)

    atr_period = IntParameter(7, 28, default=14, space="protection", optimize=True)
    atr_mult = DecimalParameter(0.8, 2.0, default=1.2, decimals=2, space="protection", optimize=True)
    fixed_sl = DecimalParameter(0.005, 0.02, default=0.010, decimals=4, space="protection", optimize=True)

    tp1 = DecimalParameter(0.004, 0.02, default=0.008, decimals=4, space="sell", optimize=True)
    tp1_ratio = DecimalParameter(0.3, 0.7, default=0.5, decimals=2, space="sell", optimize=True)
    tp2 = DecimalParameter(0.008, 0.04, default=0.012, decimals=4, space="sell", optimize=True)
    trail_fixed = DecimalParameter(0.004, 0.015, default=0.006, decimals=4, space="sell", optimize=True)
    trail_atr_mult = DecimalParameter(0.5, 2.0, default=1.0, decimals=2, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ATR for stoploss/trailing
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=int(self.atr_period.value))

        window = int(self.window.value)
        dataframe["window_high"] = dataframe["high"].rolling(window).max()
        dataframe["window_low"] = dataframe["low"].rolling(window).min()
        dataframe["window_high_idx"] = dataframe["high"].rolling(window).apply(
            lambda s: float(np.argmax(s)), raw=True
        )
        dataframe["window_low_idx"] = dataframe["low"].rolling(window).apply(
            lambda s: float(np.argmin(s)), raw=True
        )

        # Potential move vs current price (risk/reward proxy)
        dataframe["up_move"] = (dataframe["window_high"] - dataframe["close"]) / dataframe["close"]
        dataframe["down_move"] = (dataframe["close"] - dataframe["window_low"]) / dataframe["close"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        x = float(self.x_profit.value)
        y = float(self.y_risk.value)

        long_cond = (
            (dataframe["window_low_idx"] < dataframe["window_high_idx"]) &
            (dataframe["up_move"] >= x) &
            (dataframe["down_move"] <= y)
        )

        short_cond = (
            (dataframe["window_high_idx"] < dataframe["window_low_idx"]) &
            (dataframe["down_move"] >= x) &
            (dataframe["up_move"] <= y)
        )

        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[short_cond, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit trend is handled by custom_exit and custom_stoploss.
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        # Full exit at TP2
        if current_profit >= float(self.tp2.value):
            return "tp2"
        return None

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None | tuple[float | None, str | None]:
        # One-time partial exit at TP1
        if trade.has_open_orders:
            return None

        if current_profit >= float(self.tp1.value) and trade.nr_of_successful_exits == 0:
            part = float(self.tp1_ratio.value)
            return -(trade.stake_amount * part), "tp1"

        return None

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last = dataframe.iloc[-1].squeeze()
        atr = float(last["atr"]) if pd.notnull(last["atr"]) else 0.0

        # Base stoploss: min(ATR * 1.2, fixed 1%)
        atr_dist = atr * float(self.atr_mult.value)
        fixed_dist = float(self.fixed_sl.value) * current_rate
        dist = min(atr_dist, fixed_dist)
        side = 1 if trade.is_short else -1
        stop_price = current_rate + (side * dist)
        base_sl = stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

        # After TP1, switch to trailing stop (max(fixed, ATR))
        if trade.nr_of_successful_exits >= 1:
            trail_dist = max(
                float(self.trail_fixed.value),
                (float(self.trail_atr_mult.value) * atr / current_rate) if current_rate else 0.0
            )
            trail_price = current_rate + (side * current_rate * trail_dist)
            trail_sl = stoploss_from_absolute(
                trail_price,
                current_rate=current_rate,
                is_short=trade.is_short,
                leverage=trade.leverage,
            )
            return trail_sl

        return base_sl
