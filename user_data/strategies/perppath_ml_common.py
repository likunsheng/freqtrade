from __future__ import annotations

import numpy as np
import talib.abstract as ta
from pandas import DataFrame


ML_FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_12",
    "ema20_dist",
    "ema50_dist",
    "ema200_dist",
    "ema_spread",
    "adx",
    "rsi",
    "atr_pct",
    "hl_pct",
    "bb_pos",
    "bb_width",
    "vol_z",
    "mom_12",
    "mom_48",
    "pair_id",
]


def add_ml_features(df: DataFrame, pair_id: float) -> DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_12"] = out["close"].pct_change(12)

    ema20 = ta.EMA(out, timeperiod=20)
    ema50 = ta.EMA(out, timeperiod=50)
    ema200 = ta.EMA(out, timeperiod=200)
    out["ema20_dist"] = (out["close"] - ema20) / out["close"].replace(0, np.nan)
    out["ema50_dist"] = (out["close"] - ema50) / out["close"].replace(0, np.nan)
    out["ema200_dist"] = (out["close"] - ema200) / out["close"].replace(0, np.nan)
    out["ema_spread"] = (ema50 - ema200) / out["close"].replace(0, np.nan)

    out["adx"] = ta.ADX(out, timeperiod=14)
    out["rsi"] = ta.RSI(out, timeperiod=14)
    atr = ta.ATR(out, timeperiod=14)
    out["atr_pct"] = atr / out["close"].replace(0, np.nan)
    out["hl_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)

    bb = ta.BBANDS(out, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
    bb_width = (bb["upperband"] - bb["lowerband"]).replace(0, np.nan)
    out["bb_pos"] = (out["close"] - bb["lowerband"]) / bb_width
    out["bb_width"] = bb_width / bb["middleband"].replace(0, np.nan)

    vol_mean = out["volume"].rolling(48).mean()
    vol_std = out["volume"].rolling(48).std().replace(0, np.nan)
    out["vol_z"] = (out["volume"] - vol_mean) / vol_std

    out["mom_12"] = out["close"].pct_change(12)
    out["mom_48"] = out["close"].pct_change(48)
    out["pair_id"] = float(pair_id)

    for col in ML_FEATURE_COLUMNS:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
    return out
