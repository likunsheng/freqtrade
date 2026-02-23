from __future__ import annotations

from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter


class PerpPathTrendOnlyV1(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 320

    minimal_roi = {"0": 100}
    stoploss = -0.05
    use_exit_signal = True
    use_custom_stoploss = True
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    trend_adx = IntParameter(14, 35, default=24, space="buy", optimize=True)
    trend_spread = DecimalParameter(0.004, 0.035, default=0.018, decimals=3, space="buy", optimize=True)
    max_btc_vol = DecimalParameter(0.004, 0.030, default=0.020, decimals=3, space="buy", optimize=True)
    basis_abs_max = DecimalParameter(0.004, 0.040, default=0.020, decimals=3, space="buy", optimize=True)
    oi_proxy_max = DecimalParameter(1.0, 4.0, default=1.8, decimals=1, space="buy", optimize=True)
    funding_long_max = DecimalParameter(-0.002, 0.020, default=0.012, decimals=3, space="buy", optimize=True)
    trend_quick_tp = DecimalParameter(0.008, 0.070, default=0.035, decimals=3, space="sell", optimize=True)
    trend_hold_minutes = IntParameter(180, 1440, default=960, space="sell", optimize=True)

    @staticmethod
    def _merge_asof(base: DataFrame, inf: DataFrame, prefix: str) -> DataFrame:
        if inf is None or inf.empty:
            for col in ("open", "high", "low", "close", "volume"):
                base[f"{prefix}_{col}"] = np.nan
            return base
        cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in inf.columns]
        inf2 = inf[cols].copy()
        inf2 = inf2.rename(columns={c: f"{prefix}_{c}" for c in cols if c != "date"})
        return pd.merge_asof(base.sort_values("date"), inf2.sort_values("date"), on="date", direction="backward")

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative = []
        for pair in pairs:
            informative.append((pair, "1h"))
            informative.append((pair, "8h", "mark"))
            informative.append((pair, "8h", "funding_rate"))
        informative.append(("BTC/USDT:USDT", "1h"))
        return informative

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        df = dataframe.copy()

        df["ema20"] = ta.EMA(df, timeperiod=20)
        df["ema50"] = ta.EMA(df, timeperiod=50)
        df["ema200"] = ta.EMA(df, timeperiod=200)
        df["adx"] = ta.ADX(df, timeperiod=14)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["atr"] = ta.ATR(df, timeperiod=14)
        df["atr_pct"] = df["atr"] / df["close"]
        df["hl_pct"] = (df["high"] - df["low"]) / df["close"]
        bb = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df["bb_mid"] = bb["middleband"]

        inf_1h = self.dp.get_pair_dataframe(pair=pair, timeframe="1h")
        df = self._merge_asof(df, inf_1h, "inf1h")
        df["inf1h_ema50"] = ta.EMA(df["inf1h_close"], timeperiod=50)
        df["inf1h_ema200"] = ta.EMA(df["inf1h_close"], timeperiod=200)
        df["inf1h_trend_up"] = df["inf1h_ema50"] > df["inf1h_ema200"]

        inf_mark = self.dp.get_pair_dataframe(pair=pair, timeframe="8h", candle_type="mark")
        inf_funding = self.dp.get_pair_dataframe(pair=pair, timeframe="8h", candle_type="funding_rate")
        df = self._merge_asof(df, inf_mark, "mark8h")
        df = self._merge_asof(df, inf_funding, "fund8h")
        df["funding"] = df["fund8h_close"].fillna(0.0)
        df["basis"] = ((df["close"] - df["mark8h_close"]) / df["mark8h_close"].replace(0, np.nan)).fillna(0.0)

        vol_ma = df["volume"].rolling(48).mean()
        df["vol_impulse"] = df["volume"] / (vol_ma + 1e-9)
        df["oi_proxy"] = (df["vol_impulse"] * df["hl_pct"]).fillna(0.0)

        btc_1h = self.dp.get_pair_dataframe(pair="BTC/USDT:USDT", timeframe="1h")
        df = self._merge_asof(df, btc_1h, "btc1h")
        df["btc_fast"] = ta.EMA(df["btc1h_close"], timeperiod=24)
        df["btc_slow"] = ta.EMA(df["btc1h_close"], timeperiod=72)
        df["btc_vol"] = df["btc1h_close"].pct_change().rolling(24).std().fillna(0.0)
        df["btc_mom"] = df["btc1h_close"].pct_change(12).fillna(0.0)
        df["btc_risk_on"] = (
            (df["btc_fast"] > df["btc_slow"])
            & (df["btc_vol"] <= float(self.max_btc_vol.value))
            & (df["btc_mom"] > 0.0)
        )
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        spread = (df["ema50"] - df["ema200"]).abs() / df["close"]
        long_cond = [
            df["btc_risk_on"],
            df["inf1h_trend_up"],
            df["ema20"] > df["ema50"],
            df["ema50"] > df["ema200"],
            df["close"] > df["ema20"],
            df["close"] > df["bb_mid"],
            df["adx"] >= int(self.trend_adx.value),
            spread >= float(self.trend_spread.value),
            df["rsi"].between(52, 68),
            df["funding"] <= float(self.funding_long_max.value),
            df["basis"].abs() <= float(self.basis_abs_max.value),
            df["oi_proxy"] <= float(self.oi_proxy_max.value),
            df["atr_pct"] > 0.002,
        ]
        df.loc[reduce(lambda x, y: x & y, long_cond), ["enter_long", "enter_tag"]] = (1, "trend_long")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
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
        hold_min = int((current_time - trade.open_date_utc).total_seconds() // 60)
        if current_profit >= float(self.trend_quick_tp.value):
            return "trend_quick_tp"
        if hold_min >= 45 and current_profit < -0.012:
            return "trend_fail_fast"
        if hold_min >= int(self.trend_hold_minutes.value):
            return "trend_time_stop"
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
        if current_profit >= 0.04:
            return -0.010
        if current_profit >= 0.02:
            return -0.005
        if current_profit <= -0.025:
            return -0.025
        return self.stoploss
