from __future__ import annotations

from datetime import datetime
from functools import reduce
from typing import Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IStrategy, IntParameter


class PerpPathTrendOnlyV1Lev3LongShortStandaloneElasticV3Pyramid(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
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
    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    # Long side (keep V1/Lev3-compatible tuning surface)
    trend_adx = IntParameter(14, 35, default=24, space="buy", optimize=True)
    trend_spread = DecimalParameter(0.004, 0.035, default=0.018, decimals=3, space="buy", optimize=True)
    max_btc_vol = DecimalParameter(0.004, 0.030, default=0.020, decimals=3, space="buy", optimize=True)
    basis_abs_max = DecimalParameter(0.004, 0.040, default=0.020, decimals=3, space="buy", optimize=True)
    oi_proxy_max = DecimalParameter(1.0, 4.0, default=1.8, decimals=1, space="buy", optimize=True)
    funding_long_max = DecimalParameter(-0.002, 0.020, default=0.012, decimals=3, space="buy", optimize=True)

    # Conservative short side (separate knobs, do not interfere with long tuning)
    short_trend_adx = IntParameter(14, 35, default=21, space="buy", optimize=True)
    short_trend_spread = DecimalParameter(0.004, 0.035, default=0.016, decimals=3, space="buy", optimize=True)
    short_btc_vol_max = DecimalParameter(0.004, 0.080, default=0.059, decimals=3, space="buy", optimize=True)
    short_basis_abs_max = DecimalParameter(0.004, 0.060, default=0.036, decimals=3, space="buy", optimize=True)
    short_oi_proxy_max = DecimalParameter(1.0, 5.0, default=4.1, decimals=1, space="buy", optimize=True)
    short_funding_min = DecimalParameter(-0.010, 0.020, default=0.000, decimals=3, space="buy", optimize=True)
    short_rsi_low = IntParameter(30, 50, default=40, space="buy", optimize=True)
    short_rsi_high = IntParameter(35, 60, default=48, space="buy", optimize=True)
    regime_bull_spread_min = DecimalParameter(0.001, 0.030, default=0.006, decimals=3, space="buy", optimize=True)
    regime_bull_mom_min = DecimalParameter(0.000, 0.050, default=0.008, decimals=3, space="buy", optimize=True)
    regime_bull_vol_max = DecimalParameter(0.004, 0.040, default=0.018, decimals=3, space="buy", optimize=True)

    # Shared / sell space
    trend_quick_tp = DecimalParameter(0.008, 0.070, default=0.035, decimals=3, space="sell", optimize=True)
    trend_hold_minutes = IntParameter(180, 1440, default=960, space="sell", optimize=True)
    short_fail_fast_loss = DecimalParameter(-0.030, -0.004, default=-0.011, decimals=3, space="sell", optimize=True)
    short_fail_fast_minutes = IntParameter(15, 180, default=66, space="sell", optimize=True)

    leverage_value = 3.0

    pyramid_long_add_profit_triggers = (0.015, 0.023, 0.031)
    pyramid_short_add_profit_triggers = (0.012, 0.020, 0.028)
    pyramid_add_stake_fractions = (0.50, 0.50, 0.50)
    pyramid_min_hold_minutes = 20

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
        df["inf1h_trend_down"] = df["inf1h_ema50"] < df["inf1h_ema200"]

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
        df["btc_vol"] = df["btc1h_close"].pct_change(fill_method=None).rolling(24).std().fillna(0.0)
        df["btc_mom"] = df["btc1h_close"].pct_change(12, fill_method=None).fillna(0.0)
        btc_spread = ((df["btc_fast"] - df["btc_slow"]) / df["btc1h_close"].replace(0, np.nan)).fillna(0.0)
        df["btc_regime_strong_up"] = (
            (df["btc_fast"] > df["btc_slow"])
            & (btc_spread >= float(self.regime_bull_spread_min.value))
            & (df["btc_mom"] >= float(self.regime_bull_mom_min.value))
            & (df["btc_vol"] <= float(self.regime_bull_vol_max.value))
        )
        df["btc_risk_on"] = (
            (df["btc_fast"] > df["btc_slow"])
            & (df["btc_vol"] <= float(self.max_btc_vol.value))
            & (df["btc_mom"] > 0.0)
        )
        df["btc_risk_off"] = (
            (df["btc_fast"] < df["btc_slow"])
            & (df["btc_vol"] <= float(self.short_btc_vol_max.value))
            & (df["btc_mom"] < 0.0)
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

        short_rsi_low = int(self.short_rsi_low.value)
        short_rsi_high = int(self.short_rsi_high.value)
        if short_rsi_low > short_rsi_high:
            short_rsi_low, short_rsi_high = short_rsi_high, short_rsi_low

        short_cond = [
            ~df["btc_regime_strong_up"],
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

        last = None
        if self.dp is not None:
            df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            if df is not None and not df.empty:
                last = df.iloc[-1]

        # Strong trend detection used to widen long exits on genuine trend expansion.
        strong_trend_long = False
        if (not trade.is_short) and last is not None:
            close_v = float(last.get("close", 0.0) or 0.0)
            inf1h_close = float(last.get("inf1h_close", 0.0) or 0.0)
            ema_spread_5m = 0.0 if close_v <= 0 else abs(float(last.get("ema50", 0.0)) - float(last.get("ema200", 0.0))) / close_v
            ema_spread_1h = 0.0 if inf1h_close <= 0 else abs(float(last.get("inf1h_ema50", 0.0)) - float(last.get("inf1h_ema200", 0.0))) / inf1h_close
            strong_trend_long = bool(
                bool(last.get("btc_risk_on", False))
                and bool(last.get("inf1h_trend_up", False))
                and float(last.get("adx", 0.0)) >= max(float(self.trend_adx.value) + 5.0, 24.0)
                and ema_spread_5m >= float(self.trend_spread.value) * 1.25
                and ema_spread_1h >= 0.006
                and float(last.get("btc_mom", 0.0)) >= max(float(self.regime_bull_mom_min.value) * 0.8, 0.004)
                and float(last.get("btc_vol", 0.0)) <= float(self.regime_bull_vol_max.value) * 1.15
            )

        if trade.is_short:
            # Relaxed regime flip: only force close on a stronger BTC up-regime confirmation.
            if last is not None:
                btc_strong_up = bool(last.get("btc_regime_strong_up", False))
                btc_mom = float(last.get("btc_mom", 0.0))
                btc_vol = float(last.get("btc_vol", 0.0))
                force_flip = btc_strong_up and btc_mom >= 0.012 and btc_vol <= float(self.regime_bull_vol_max.value)
                if force_flip:
                    return "trend_short_regime_flip_strong"
            if current_profit >= float(self.trend_quick_tp.value):
                return "trend_short_quick_tp"
            if hold_min >= int(self.short_fail_fast_minutes.value) and current_profit < float(self.short_fail_fast_loss.value):
                return "trend_short_fail_fast"
            if hold_min >= int(self.trend_hold_minutes.value):
                return "trend_short_time_stop"
            return None

        if strong_trend_long:
            # Let strong-trend longs run: raise quick TP, loosen fail-fast / time stop.
            if current_profit >= float(self.trend_quick_tp.value) * 1.8:
                return "trend_quick_tp_strong"
            if hold_min >= 90 and current_profit < -0.016:
                return "trend_fail_fast_strong"
            if hold_min >= int(float(self.trend_hold_minutes.value) * 1.8):
                return "trend_time_stop_strong"
            return None

        if current_profit >= float(self.trend_quick_tp.value):
            return "trend_quick_tp"
        if hold_min >= 45 and current_profit < -0.012:
            return "trend_fail_fast"
        if hold_min >= int(self.trend_hold_minutes.value):
            return "trend_time_stop"
        return None

    @staticmethod
    def _adx_ema20_expanding(recent: DataFrame) -> bool:
        if recent is None or len(recent) < 3:
            return False
        r0 = recent.iloc[-1]
        r1 = recent.iloc[-2]
        r2 = recent.iloc[-3]
        adx0 = float(r0.get("adx", 0.0) or 0.0)
        adx1 = float(r1.get("adx", 0.0) or 0.0)
        adx2 = float(r2.get("adx", 0.0) or 0.0)
        ema0 = float(r0.get("ema20", 0.0) or 0.0)
        ema1 = float(r1.get("ema20", 0.0) or 0.0)
        ema2 = float(r2.get("ema20", 0.0) or 0.0)

        adx_expanding = adx0 > adx1 > adx2
        slope_now = ema0 - ema1
        slope_prev = ema1 - ema2
        ema_slope_accel_up = slope_now > slope_prev > 0.0
        ema_slope_accel_down = slope_now < slope_prev < 0.0
        return bool(adx_expanding and (ema_slope_accel_up or ema_slope_accel_down))

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ):
        current_entries = int(trade.nr_of_successful_entries)
        if current_entries >= 4:
            return None

        add_idx = current_entries - 1
        if add_idx < 0 or add_idx >= len(self.pyramid_add_stake_fractions):
            return None

        if trade.is_short:
            required_profit = float(self.pyramid_short_add_profit_triggers[add_idx])
        else:
            required_profit = float(self.pyramid_long_add_profit_triggers[add_idx])
        if current_profit < required_profit:
            return None

        # Avoid repeated adjustment attempts too early in the trade.
        hold_min = int((current_time - trade.open_date_utc).total_seconds() // 60)
        if hold_min < int(self.pyramid_min_hold_minutes):
            return None

        if self.dp is None:
            return None
        df, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        if df is None or len(df) < 3:
            return None
        last = df.iloc[-1]
        recent = df.iloc[-3:]

        close_v = float(last.get("close", 0.0) or 0.0)
        inf1h_close = float(last.get("inf1h_close", 0.0) or 0.0)
        if close_v <= 0.0 or inf1h_close <= 0.0:
            return None
        if not self._adx_ema20_expanding(recent):
            return None

        ema_spread_5m = abs(float(last.get("ema50", 0.0)) - float(last.get("ema200", 0.0))) / close_v
        ema_spread_1h = abs(float(last.get("inf1h_ema50", 0.0)) - float(last.get("inf1h_ema200", 0.0))) / inf1h_close
        strong_trend_long = bool(
            bool(last.get("btc_risk_on", False))
            and bool(last.get("inf1h_trend_up", False))
            and float(last.get("ema20", 0.0)) > float(last.get("ema50", 0.0))
            and float(last.get("adx", 0.0)) >= max(float(self.trend_adx.value) + 4.0, 22.0)
            and ema_spread_5m >= float(self.trend_spread.value) * 1.15
            and ema_spread_1h >= 0.005
            and float(last.get("btc_mom", 0.0)) >= max(float(self.regime_bull_mom_min.value) * 0.75, 0.003)
            and float(last.get("btc_vol", 0.0)) <= float(self.regime_bull_vol_max.value) * 1.20
        )
        strong_trend_short = bool(
            bool(last.get("btc_risk_off", False))
            and (not bool(last.get("btc_regime_strong_up", False)))
            and bool(last.get("inf1h_trend_down", False))
            and float(last.get("ema20", 0.0)) < float(last.get("ema50", 0.0))
            and float(last.get("adx", 0.0)) >= max(float(self.short_trend_adx.value) + 2.0, 20.0)
            and ema_spread_5m >= float(self.short_trend_spread.value) * 1.10
            and ema_spread_1h >= 0.0045
            and float(last.get("funding", 0.0)) > 0.0
            and float(last.get("btc_mom", 0.0)) <= -0.002
        )

        if trade.is_short:
            if not strong_trend_short:
                return None
            if float(last.get("close", 0.0)) >= float(last.get("ema20", 0.0)):
                return None
            if float(last.get("rsi", 100.0)) > float(self.short_rsi_high.value):
                return None
        else:
            if not strong_trend_long:
                return None
            # Require momentum continuation to avoid adding into local stalls.
            if float(last.get("close", 0.0)) <= float(last.get("ema20", 0.0)):
                return None
            if float(last.get("rsi", 0.0)) < 55.0:
                return None

        add_stake = float(trade.stake_amount) * float(self.pyramid_add_stake_fractions[add_idx])
        if min_stake is not None and add_stake < float(min_stake):
            add_stake = float(min_stake)
        if add_stake <= 0:
            return None
        add_stake = min(add_stake, float(max_stake))
        if add_stake <= 0:
            return None
        tag = "pyramid_add_short" if trade.is_short else "pyramid_add_long"
        return add_stake, tag

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
