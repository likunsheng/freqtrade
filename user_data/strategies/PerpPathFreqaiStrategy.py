# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from __future__ import annotations

import logging
from datetime import datetime
from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    Trade,
    stoploss_from_absolute,
)

logger = logging.getLogger(__name__)


class PerpPathFreqaiStrategy(IStrategy):
    """
    FreqAI-native version of the "next-window opportunity" classifier.

    Key points:
    - Features are created via FreqAI feature_engineering_* functions (prefixed with `%`).
    - Target is a 3-class label (prefixed with `&`): "long" / "short" / "none".
    - Label enforces "TP hit before SL", and includes fee + slippage in the label definition.
    - Add ETH into training "the FreqAI way" by setting `include_corr_pairlist`
      in the config to include both BTC/ETH, and by whitelisting both pairs.

    Recommended freqaimodel: `XGBoostClassifier` (handles string labels correctly).

    中文说明（面向策略研发/回测/超参）：
    1) 这是一个“下一窗口机会”分类器策略：模型预测未来一段窗口里更可能出现 long/short/none。
    2) 交易信号不是“预测即交易”，而是“预测 + 过滤 + 风控/退出”的组合：
       - 过滤：波动率过滤（ATR%）、趋势制度过滤（ADX + EMA20/EMA50）。
       - 退出：模型判断 none 优势时退出；同时加入 time-stop（时间止损/时间退出）减少拖延亏损与尾部风险。
    3) 标签（label）是 TP/SL 命中顺序：如果未来 N 根K线中 TP 先于 SL 触达，则标为机会（long/short）。
       这里把 TP/SL 做成 ATR 自适应（随波动扩张），用于提升不同波动环境下标签的可分性。

    注意：
    - 改动 label 相关参数（label_*）会改变训练目标，必须重新训练（更换 freqai.identifier）。
    - 过滤器会减少交易次数，通常换来更低回撤/更稳定分布；需要通过回测与 OOS 验证取舍。
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1m"
    process_only_new_candles = True
    startup_candle_count = 200

    stoploss = -0.03
    minimal_roi = {"0": 100}
    use_exit_signal = True
    use_custom_stoploss = True

    # ---- Research switch (hard constraints) ----
    # “快速止血”阶段：强制启用过滤器（不允许被参数文件/超参覆盖）。
    # 注意：Freqtrade 会从 `user_data/strategies/PerpPathFreqaiStrategy.json` 加载参数，
    # 即使某个 Parameter optimize=False，也可能被该 json 覆盖成 False。
    # 所以这里用硬开关确保策略逻辑层面一定启用过滤器。
    force_entry_filters = True

    # Entry thresholds (operate on predicted probabilities)
    # 入场阈值：基于模型输出的概率列（long/short/none），并做 pair 级别差异化。
    # 这类阈值适合 hyperopt 做“稳健化”：提高阈值通常减少交易与回撤，但也可能降低收益。
    # Global defaults (used for non-overridden pairs).
    p_long_th = DecimalParameter(0.35, 0.85, default=0.60, decimals=3, space="buy", optimize=True)
    # Short signals were weak in backtests - make them harder to trigger by default.
    p_short_th = DecimalParameter(0.35, 0.90, default=0.70, decimals=3, space="buy", optimize=True)
    # 最小优势差（margin）建议不要太低：太低会把“几乎没优势”的噪声信号也放进来，
    # 导致交易次数很多但净期望为负（成本吃掉微弱优势）。
    p_margin = DecimalParameter(0.03, 0.30, default=0.10, decimals=3, space="buy", optimize=True)

    # Pair-specific overrides (FreqAI standard mode trains per-pair models, so this is safe).
    # Defaults are set to match global values; tune these independently if BTC/ETH behave differently.
    p_long_th_btc = DecimalParameter(0.35, 0.95, default=0.60, decimals=3, space="buy", optimize=True)
    p_short_th_btc = DecimalParameter(0.35, 0.95, default=0.70, decimals=3, space="buy", optimize=True)
    p_margin_btc = DecimalParameter(0.03, 0.30, default=0.10, decimals=3, space="buy", optimize=True)

    p_long_th_eth = DecimalParameter(0.35, 0.95, default=0.60, decimals=3, space="buy", optimize=True)
    p_short_th_eth = DecimalParameter(0.35, 0.95, default=0.70, decimals=3, space="buy", optimize=True)
    p_margin_eth = DecimalParameter(0.03, 0.30, default=0.10, decimals=3, space="buy", optimize=True)

    # Exit thresholds (operate on predicted probabilities)
    # Idea: exit when model believes "no opportunity" is most likely.
    p_none_exit_th = DecimalParameter(0.50, 0.99, default=0.85, decimals=3, space="sell", optimize=True)
    p_none_exit_margin = DecimalParameter(
        0.00, 0.30, default=0.05, decimals=3, space="sell", optimize=True
    )

    p_none_exit_th_btc = DecimalParameter(0.50, 0.99, default=0.85, decimals=3, space="sell", optimize=True)
    p_none_exit_margin_btc = DecimalParameter(0.00, 0.30, default=0.05, decimals=3, space="sell", optimize=True)
    p_none_exit_th_eth = DecimalParameter(0.50, 0.99, default=0.85, decimals=3, space="sell", optimize=True)
    p_none_exit_margin_eth = DecimalParameter(0.00, 0.30, default=0.05, decimals=3, space="sell", optimize=True)

    # Time-stop exits (reduce "slow bleed" and long tail exposure)
    # 时间退出（Time-stop）：
    # - max_hold_minutes：任何仓位持有超过该时长后强制退出（降低尾部风险、降低资金占用）。
    # - max_hold_minutes_loss：若仍处于亏损且持有超过该时长，则更早止损退出（减少“慢性失血”）。
    # Exit any trade after max_hold_minutes, and exit losing trades earlier after max_hold_minutes_loss.
    # 提示：你当前的目标是“80–150 笔/月 + 更高月化”，持仓周期需要更短来提高资金周转率。
    # 因此这里把搜索区间从“最长 7 天”收敛到“最长 2 天”，避免 hyperopt 浪费在明显不匹配目标的区域。
    max_hold_minutes = IntParameter(60, 60 * 24 * 2, default=60 * 12, space="sell", optimize=True)
    max_hold_minutes_loss = IntParameter(30, 60 * 12, default=60 * 3, space="sell", optimize=True)

    # Volatility / regime filters (applied to entries)
    # 制度/波动过滤器（用于减少噪声期交易与逆强趋势交易）：
    # - max_atr_pct：ATR% 超过该阈值不交易（高波动往往 stoploss 聚集、滑点与噪声更大）。
    # - min_adx：趋势强度阈值；趋势很强时，避免做“弱势方向”（不要逆趋势硬做）。
    # Filter out very high volatility conditions (often noise / stoploss clusters).
    max_atr_pct = DecimalParameter(0.005, 0.06, default=0.02, decimals=4, space="buy", optimize=True)
    # Use a simple trend-strength filter to avoid fading strong trends.
    min_adx = IntParameter(10, 45, default=18, space="buy", optimize=True)
    # Filter toggles
    # “快速止血”阶段：强制打开过滤器，不允许 hyperopt 关闭。
    # 原因：hyperopt 很容易为了凑交易次数/短期指标，把过滤器关掉，结果就是 OOS 大幅变差。
    enable_atr_filter = BooleanParameter(default=True, space="buy", optimize=False)
    enable_trend_filter = BooleanParameter(default=True, space="buy", optimize=False)

    # Label parameters (changing these requires retraining)
    # 标签（训练目标）参数：
    # label_tp / label_sl：基础 TP/SL 比例；用于定义“未来 N 根K线 TP 是否先于 SL 命中”。
    # 为减少不同波动环境下的标签不可分问题，下面加入 ATR 自适应的 tp/sl 扩张（至少不低于基础值）。
    # Make labels less sparse (especially for BTC) by lowering tp and allowing slightly more sl.
    label_tp = DecimalParameter(0.001, 0.05, default=0.007, decimals=4, space="buy", optimize=False)
    label_sl = DecimalParameter(0.001, 0.05, default=0.012, decimals=4, space="buy", optimize=False)
    label_fee = DecimalParameter(0.0, 0.01, default=0.0004, decimals=6, space="buy", optimize=False)
    label_slippage = DecimalParameter(0.0, 0.01, default=0.0002, decimals=6, space="buy", optimize=False)
    # ATR-adaptive label sizing: tp/sl will be at least the base values above,
    # but can expand with volatility to improve label separability across regimes.
    label_tp_atr_mult = DecimalParameter(0.0, 6.0, default=1.2, decimals=2, space="buy", optimize=False)
    label_sl_atr_mult = DecimalParameter(0.0, 8.0, default=1.8, decimals=2, space="buy", optimize=False)
    label_atr_period = IntParameter(7, 50, default=14, space="buy", optimize=False)

    # Stake sizing / leverage (risk-budgeted)
    # Use account-equity based risk budget and ATR-based stop distance to size positions.
    risk_per_trade = DecimalParameter(0.001, 0.01, default=0.003, decimals=4, space="protection", optimize=True)
    # cap stake to this fraction of total equity (pre-leverage)
    # 单笔使用权益上限：为了“更高频/更稳定”，这里收敛到 25% 以内，避免单笔过度集中导致回撤飙升。
    max_stake_fraction = DecimalParameter(
        0.03, 0.25, default=0.10, decimals=3, space="protection", optimize=True
    )
    # approximate stop distance in ATR multiples for risk sizing (independent from hard stoploss)
    stake_atr_mult = DecimalParameter(0.5, 3.0, default=1.2, decimals=2, space="protection", optimize=True)
    # exposure limit: if there's already an open trade in the same direction (BTC/ETH),
    # reduce the stake for the next same-direction trade.
    same_side_stake_mult = DecimalParameter(
        0.25, 1.0, default=0.6, decimals=2, space="protection", optimize=True
    )

    # Dynamic leverage
    # 杠杆下限不宜太高：低优势/高噪声期保持低杠杆，减少“慢性失血”。
    leverage_min = DecimalParameter(1.0, 1.5, default=1.0, decimals=2, space="protection", optimize=True)
    leverage_max = DecimalParameter(1.5, 3.0, default=3.0, decimals=2, space="protection", optimize=True)
    # increase leverage only when model advantage is strong enough
    # 模型优势门槛：太低会导致大量低质量信号入场（噪声 + 成本 -> 净期望为负）。
    # 提高下限作为“快速止血”，先让策略只吃更强优势的机会。
    min_model_edge = DecimalParameter(0.15, 0.50, default=0.20, decimals=3, space="buy", optimize=True)

    # Profit protection (give-back exit)
    protect_profit_trigger = DecimalParameter(
        0.003, 0.10, default=0.02, decimals=4, space="sell", optimize=True
    )
    protect_profit_giveback = DecimalParameter(
        0.002, 0.08, default=0.01, decimals=4, space="sell", optimize=True
    )

    # Dynamic stoploss (ATR-based, "wider in trends", tighten when model says none dominates)
    sl_atr_mult_wide = DecimalParameter(0.8, 4.0, default=1.8, decimals=2, space="protection", optimize=True)
    sl_atr_mult_tight = DecimalParameter(0.3, 2.5, default=0.9, decimals=2, space="protection", optimize=True)
    sl_tighten_none_th = DecimalParameter(0.50, 0.99, default=0.85, decimals=3, space="protection", optimize=True)
    sl_tighten_none_margin = DecimalParameter(0.00, 0.30, default=0.08, decimals=3, space="protection", optimize=True)
    sl_cap_tight = DecimalParameter(0.002, 0.03, default=0.012, decimals=4, space="protection", optimize=True)
    # Breakeven protection: after reaching some profit, do not allow a trade to become a (small) loser.
    # 目的：减少大量 `trailing_stop_loss` 小亏出场（常见于短周期噪声/反转）。
    sl_be_trigger = DecimalParameter(0.001, 0.03, default=0.006, decimals=4, space="protection", optimize=True)
    sl_be_offset = DecimalParameter(0.0, 0.01, default=0.0005, decimals=5, space="protection", optimize=True)

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Expanded features (FreqAI will expand across:
          indicator_periods_candles * include_timeframes * include_shifted_candles * include_corr_pairs
        """
        # 这里的特征会被 FreqAI “多周期/多窗口/多位移/多相关币”扩展开来。
        # 原则：优先提供“独立信息维度”的特征（波动、趋势、位置、成交量、路径结构），
        # 避免堆叠大量高度相关指标导致训练更慢、更易过拟合、更易 OOM。
        # Past range / volatility
        roll_hi = dataframe["high"].rolling(period).max()
        roll_lo = dataframe["low"].rolling(period).min()
        dataframe["%-range_pct-period"] = (roll_hi - roll_lo) / dataframe["close"]
        dataframe["%-volatility-period"] = dataframe["close"].pct_change().rolling(period).std()

        # Volatility / barrier-related (helps TP-before-SL style problems)
        atr = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr_pct-period"] = atr / dataframe["close"]
        dataframe["%-trange_pct-period"] = ta.TRANGE(dataframe) / dataframe["close"]

        # Distance to rolling extremes (how much room left to move)
        dataframe["%-dist_to_high_pct-period"] = (roll_hi - dataframe["close"]) / dataframe["close"]
        dataframe["%-dist_to_low_pct-period"] = (dataframe["close"] - roll_lo) / dataframe["close"]

        # Bollinger band location/width (regime + mean reversion / breakout)
        # TA-Lib expects float values for nbdevup/nbdevdn.
        bb = ta.BBANDS(dataframe, timeperiod=period, nbdevup=2.0, nbdevdn=2.0)
        bb_upper = bb["upperband"]
        bb_mid = bb["middleband"]
        bb_lower = bb["lowerband"]
        bb_width = (bb_upper - bb_lower) / bb_mid
        dataframe["%-bb_width-period"] = bb_width
        dataframe["%-bb_pos-period"] = (dataframe["close"] - bb_lower) / (bb_upper - bb_lower)

        # Trend strength / direction
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-plus_di-period"] = ta.PLUS_DI(dataframe, timeperiod=period)
        dataframe["%-minus_di-period"] = ta.MINUS_DI(dataframe, timeperiod=period)

        # Volume stats
        vol_mean = dataframe["volume"].rolling(period).mean()
        vol_std = dataframe["volume"].rolling(period).std()
        dataframe["%-vol_rel-period"] = dataframe["volume"] / vol_mean
        dataframe["%-vol_z-period"] = (dataframe["volume"] - vol_mean) / vol_std

        # Extremum order (cheap "path" proxy)
        dataframe["%-win_high_idx-period"] = dataframe["high"].rolling(period).apply(
            lambda s: float(np.argmax(s)), raw=True
        )
        dataframe["%-win_low_idx-period"] = dataframe["low"].rolling(period).apply(
            lambda s: float(np.argmin(s)), raw=True
        )
        dataframe["%-ext_order-period"] = np.where(
            dataframe["%-win_low_idx-period"] < dataframe["%-win_high_idx-period"], 1.0, -1.0
        )

        # Trend-ish helpers
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        # Basic, non-expanded features
        # 基础特征不会跨 period 扩展，通常用于补充“短期动量/形态/时间信息”。
        dataframe["%-ret_1"] = dataframe["close"].pct_change(1)
        dataframe["%-ret_5"] = dataframe["close"].pct_change(5)
        dataframe["%-ret_20"] = dataframe["close"].pct_change(20)

        # Candle anatomy (microstructure proxy)
        oc_max = dataframe[["open", "close"]].max(axis=1)
        oc_min = dataframe[["open", "close"]].min(axis=1)
        dataframe["%-body_pct"] = (dataframe["close"] - dataframe["open"]).abs() / dataframe["close"]
        dataframe["%-upper_wick_pct"] = (dataframe["high"] - oc_max) / dataframe["close"]
        dataframe["%-lower_wick_pct"] = (oc_min - dataframe["low"]) / dataframe["close"]
        dataframe["%-hl_range_pct"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]

        # Momentum / mean-reversion helpers (not expanded)
        macd_df = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["%-macd"] = macd_df["macd"]
        dataframe["%-macdhist"] = macd_df["macdhist"]
        dataframe["%-cci_20"] = ta.CCI(dataframe, timeperiod=20)

        dataframe["%-hour"] = dataframe["date"].dt.hour
        dataframe["%-dow"] = dataframe["date"].dt.dayofweek

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        3-class target for classifier: "long" / "short" / "none".

        Uses the next `label_period_candles` candles for TP/SL hit-order labeling.
        If both TP and SL touch in the same candle, assumes SL first (conservative).
        """
        df = dataframe
        n = int(self.freqai_info["feature_parameters"]["label_period_candles"])
        if n <= 0 or len(df) < n + 2:
            df["&-nw_class"] = np.nan
            return df

        base_tp = float(self.label_tp.value)
        base_sl = float(self.label_sl.value)
        fee = float(self.label_fee.value)
        slip = float(self.label_slippage.value)

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        # ATR-adaptive thresholds (per-row, in ratio terms).
        # This reduces label sparsity in low-vol regimes and avoids "too tight" barriers in high-vol regimes.
        # 中文解释：用 ATR% 来动态放大 TP/SL（但至少不低于基础值）。
        # 好处：不同波动环境下，“能否在 N 根内先触达 TP”这个问题更可分，模型更容易学到稳定模式。
        atr_p = int(self.label_atr_period.value)
        atr = ta.ATR(df, timeperiod=atr_p).to_numpy(dtype=float)
        atr_pct = np.where(close > 0, atr / close, np.nan)
        tp = np.clip(
            np.maximum(base_tp, atr_pct * float(self.label_tp_atr_mult.value)),
            0.0005,
            0.08,
        )
        sl = np.clip(
            np.maximum(base_sl, atr_pct * float(self.label_sl_atr_mult.value)),
            0.0005,
            0.10,
        )

        # Effective entry/exit with adverse slippage + fees (net thresholds).
        entry_cost_long = close * (1.0 + slip) * (1.0 + fee)
        tp_price_long = entry_cost_long * (1.0 + tp) / ((1.0 - slip) * (1.0 - fee))
        sl_price_long = entry_cost_long * (1.0 - sl) / ((1.0 - slip) * (1.0 - fee))

        entry_proceeds_short = close * (1.0 - slip) * (1.0 - fee)
        tp_price_short = entry_proceeds_short * (1.0 - tp) / ((1.0 + slip) * (1.0 + fee))
        sl_price_short = entry_proceeds_short * (1.0 + sl) / ((1.0 + slip) * (1.0 + fee))

        # Future windows (t+1..t+n)
        # 使用滑动窗口视图构造未来 n 根K线的 high/low，用于判断 TP/SL 触达顺序。
        # 注意：这一步对内存较敏感（尤其在 1m + 长区间），因此更推荐用 5m 基线做多周期。
        high_f = high[1:]
        low_f = low[1:]
        rows = len(df) - n

        win_high = np.lib.stride_tricks.sliding_window_view(high_f, n)[:rows]
        win_low = np.lib.stride_tricks.sliding_window_view(low_f, n)[:rows]

        long_tp_hit = win_high >= tp_price_long[:rows, None]
        long_sl_hit = win_low <= sl_price_long[:rows, None]
        short_tp_hit = win_low <= tp_price_short[:rows, None]
        short_sl_hit = win_high >= sl_price_short[:rows, None]

        def _first_hit_idx(hit: np.ndarray) -> np.ndarray:
            any_hit = hit.any(axis=1)
            first = hit.argmax(axis=1)
            return np.where(any_hit, first, n + 1)

        long_tp_i = _first_hit_idx(long_tp_hit)
        long_sl_i = _first_hit_idx(long_sl_hit)
        short_tp_i = _first_hit_idx(short_tp_hit)
        short_sl_i = _first_hit_idx(short_sl_hit)

        long_ok = long_tp_i < long_sl_i
        short_ok = short_tp_i < short_sl_i

        label = np.full(len(df), np.nan, dtype=object)
        base = np.full(rows, "none", dtype=object)
        base[long_ok & ~short_ok] = "long"
        base[short_ok & ~long_ok] = "short"
        label[:rows] = base

        df["&-nw_class"] = label
        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Lightweight regime / volatility indicators for entry filtering and stability.
        # 这些指标用于“交易过滤/制度判断”，不参与 FreqAI 的特征扩展（避免爆内存）。
        # - atr_pct_14：波动率过滤
        # - ema_20/ema_50 + adx_14：趋势制度过滤（强趋势时避免逆势）
        dataframe["atr_14"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct_14"] = dataframe["atr_14"] / dataframe["close"]
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["adx_14"] = ta.ADX(dataframe, timeperiod=14)

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

        # FreqAI will call our feature_engineering_* and add prediction columns to the df.
        return self.freqai.start(dataframe, metadata, self)

    @staticmethod
    def _pair_bucket(pair: str) -> str:
        # "BTC/USDT:USDT" -> "BTC", "ETH/USDT:USDT" -> "ETH"
        if not pair:
            return ""
        return pair.split("/", 1)[0].upper()

    def _thresholds_for_pair(self, pair: str) -> tuple[float, float, float]:
        """
        Returns (p_long_th, p_short_th, p_margin) for the given pair.
        """
        bucket = self._pair_bucket(pair)
        if bucket == "BTC":
            return (
                float(self.p_long_th_btc.value),
                float(self.p_short_th_btc.value),
                float(self.p_margin_btc.value),
            )
        if bucket == "ETH":
            return (
                float(self.p_long_th_eth.value),
                float(self.p_short_th_eth.value),
                float(self.p_margin_eth.value),
            )
        return (float(self.p_long_th.value), float(self.p_short_th.value), float(self.p_margin.value))

    def _none_exit_thresholds_for_pair(self, pair: str) -> tuple[float, float]:
        """
        Returns (p_none_exit_th, p_none_exit_margin) for the given pair.
        """
        bucket = self._pair_bucket(pair)
        if bucket == "BTC":
            return (float(self.p_none_exit_th_btc.value), float(self.p_none_exit_margin_btc.value))
        if bucket == "ETH":
            return (float(self.p_none_exit_th_eth.value), float(self.p_none_exit_margin_eth.value))
        return (float(self.p_none_exit_th.value), float(self.p_none_exit_margin.value))

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        p_long, p_short, margin = self._thresholds_for_pair(metadata.get("pair", ""))
        min_edge = float(self.min_model_edge.value)

        # Regime filters (base timeframe):
        # - avoid extreme volatility
        # - avoid taking the weak side in strong trends
        # 中文解释：
        # 1) atr_ok：高波动期往往噪声更大、止损更密集，先过滤减少回撤尾部。
        # 2) uptrend/downtrend：当趋势很强时，不逆势开仓（例如强上升趋势尽量不做空）。
        atr_ok = df.get("atr_pct_14", 0.0) <= float(self.max_atr_pct.value)
        adx = df.get("adx_14", 0.0)
        ema_fast = df.get("ema_20", 0.0)
        ema_slow = df.get("ema_50", 0.0)
        trend_strong = adx >= float(self.min_adx.value)
        uptrend = trend_strong & (ema_fast > ema_slow)
        downtrend = trend_strong & (ema_fast < ema_slow)

        if bool(self.force_entry_filters):
            atr_filter_ok = atr_ok
            trend_filter_long_ok = ~downtrend
            trend_filter_short_ok = ~uptrend
        else:
            atr_filter_ok = (~bool(self.enable_atr_filter.value)) | atr_ok
            trend_filter_long_ok = (~bool(self.enable_trend_filter.value)) | (~downtrend)
            trend_filter_short_ok = (~bool(self.enable_trend_filter.value)) | (~uptrend)

        # Entry edge filter:
        # Require directional probability to also beat "none" by a minimum edge.
        # This removes many low-confidence/noise trades that often become small losses.
        long_edge_ok = (df.get("long", 0.0) - df.get("none", 0.0)) >= min_edge
        short_edge_ok = (df.get("short", 0.0) - df.get("none", 0.0)) >= min_edge

        # Probability columns are named by class ("long"/"short"/"none") for classifier models.
        enter_long_conditions = [
            df["do_predict"] == 1,
            atr_filter_ok,
            trend_filter_long_ok,
            df.get("long", 0.0) >= p_long,
            df.get("long", 0.0) >= df.get("short", 0.0) + margin,
            long_edge_ok,
        ]
        df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (
            1,
            "ml_long",
        )

        enter_short_conditions = [
            df["do_predict"] == 1,
            atr_filter_ok,
            trend_filter_short_ok,
            df.get("short", 0.0) >= p_short,
            df.get("short", 0.0) >= df.get("long", 0.0) + margin,
            short_edge_ok,
        ]
        df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (
            1,
            "ml_short",
        )

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
        # Time-stop exits (priority before ML exit signals in populate_exit_trend)
        # 时间退出在实盘/回测里非常关键：它能显著减少“长时间不动的亏损单”对回撤的拖累。
        open_dt = trade.open_date_utc
        if open_dt is None:
            return None

        hold_m = (current_time - open_dt).total_seconds() / 60.0

        if current_profit < 0 and hold_m >= float(self.max_hold_minutes_loss.value):
            return "time_stop_loss"
        if hold_m >= float(self.max_hold_minutes.value):
            return "time_stop"

        # Profit protection: exit if we gave back too much from peak profit after reaching a threshold.
        trigger = float(self.protect_profit_trigger.value)
        giveback = float(self.protect_profit_giveback.value)
        peak_profit = 0.0
        if trade.is_short:
            # For shorts, best move is lowest price (min_rate)
            if trade.min_rate:
                peak_profit = (trade.open_rate - trade.min_rate) / trade.open_rate
        else:
            if trade.max_rate:
                peak_profit = (trade.max_rate - trade.open_rate) / trade.open_rate

        if peak_profit >= trigger and current_profit <= (peak_profit - giveback):
            return "profit_giveback"

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
    ) -> float | None:
        """
        ATR 动态止损（偏宽，降低趋势行情被扫概率），并在模型判断 none 优势很强时“收紧止损”。
        同时避免“止损追着价格跑”导致大量小亏（你回测里 `trailing_stop_loss` 很多且为负，就是典型症状）。

        说明：
        - wide：默认使用更宽的 ATR 倍数，更适合趋势行情“少被扫”。
        - tight：当 none 明显占优时（机会消失/噪声期），收紧止损并加 tight cap 限制尾部亏损。
        - 规则：亏损时只使用“基于开仓价”的初始止损（不追价）；盈利后才允许用 peak（max_rate/min_rate）做追踪止损。
        """
        if not hasattr(self, "dp") or self.dp is None:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        # IMPORTANT: Use the candle corresponding to `current_time` to avoid lookahead bias.
        # dp.get_analyzed_dataframe returns the full analyzed dataframe.
        df_now = dataframe.loc[dataframe["date"] <= current_time]
        if df_now.empty:
            return None
        last = df_now.iloc[-1]

        atr = float(last.get("atr_14", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            return None

        p_none = float(last.get("none", 0.0))
        p_long = float(last.get("long", 0.0))
        p_short = float(last.get("short", 0.0))

        none_dom = (p_none >= float(self.sl_tighten_none_th.value)) and (
            (p_none >= p_long + float(self.sl_tighten_none_margin.value))
            and (p_none >= p_short + float(self.sl_tighten_none_margin.value))
        )

        wide_mult = float(self.sl_atr_mult_wide.value)
        tight_mult = float(self.sl_atr_mult_tight.value)
        tight_cap = float(self.sl_cap_tight.value)

        # 1) 初始止损：基于开仓价（亏损时不追价，避免大量小亏）。
        base_dist = atr * wide_mult
        base_stop = (trade.open_rate - base_dist) if not trade.is_short else (trade.open_rate + base_dist)

        stop_price = base_stop

        # 2) 追踪止损：仅在盈利后启用，并用 peak（max_rate/min_rate）作为锚点。
        #    none_dom 时更收紧，同时用 tight_cap 限制“离当前价过远/过近”的极端情况。
        if current_profit > 0:
            # 2.1) 先做保本：达到触发利润后，止损至少移动到开仓价附近（略覆盖费用/滑点）。
            be_trigger = float(self.sl_be_trigger.value)
            be_offset = float(self.sl_be_offset.value)
            if current_profit >= be_trigger:
                if not trade.is_short:
                    stop_price = max(stop_price, trade.open_rate * (1.0 + be_offset))
                else:
                    stop_price = min(stop_price, trade.open_rate * (1.0 - be_offset))

            peak = None
            if trade.is_short:
                if trade.min_rate:
                    peak = float(trade.min_rate)
            else:
                if trade.max_rate:
                    peak = float(trade.max_rate)

            if peak is not None and np.isfinite(peak) and peak > 0:
                mult = tight_mult if none_dom else wide_mult
                dist = atr * mult
                if none_dom:
                    dist = min(dist, tight_cap * current_rate)
                trail_stop = (peak - dist) if not trade.is_short else (peak + dist)

                # 只允许“向有利方向收紧”：
                # - 多单：止损只上移（变大）
                # - 空单：止损只下移（变小）
                if not trade.is_short:
                    stop_price = max(stop_price, trail_stop)
                else:
                    stop_price = min(stop_price, trail_stop)

        # 3) 保护：确保止损在当前价的正确一侧；否则返回 None 保持原止损（避免“追着当前价跑”）。
        if not trade.is_short and stop_price >= current_rate:
            return None
        if trade.is_short and stop_price <= current_rate:
            return None

        return stoploss_from_absolute(
            stop_price,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        动态杠杆（1x~3x）：当模型优势足够强且波动率不过高时提高杠杆，否则保持低杠杆。
        目的：在不显著增加回撤的前提下提高资金效率。
        """
        lev_lo = float(self.leverage_min.value)
        lev_hi = float(self.leverage_max.value)
        lev_hi = min(lev_hi, max_leverage)

        # Default to low leverage if we cannot read latest analyzed df.
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return max(1.0, min(lev_lo, max_leverage))
            last = dataframe.iloc[-1]
        except Exception:
            return max(1.0, min(lev_lo, max_leverage))

        atr_pct = float(last.get("atr_pct_14", np.nan))
        if np.isfinite(atr_pct) and atr_pct > float(self.max_atr_pct.value):
            return max(1.0, min(lev_lo, max_leverage))

        p_long = float(last.get("long", 0.0))
        p_short = float(last.get("short", 0.0))
        p_none = float(last.get("none", 0.0))
        edge = max(p_long, p_short) - p_none
        if edge <= float(self.min_model_edge.value):
            return max(1.0, min(lev_lo, max_leverage))

        # Scale leverage with edge (clipped).
        # edge in [min_edge .. min_edge+0.3] -> lev in [lev_lo .. lev_hi]
        edge_span = 0.30
        x = min(1.0, max(0.0, (edge - float(self.min_model_edge.value)) / edge_span))
        lev = lev_lo + x * (lev_hi - lev_lo)
        return max(1.0, min(lev, max_leverage))

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        风险预算定仓：按账户权益 * risk_per_trade 计算每笔可承受亏损，然后用 ATR 推断止损距离换算仓位。
        同向（BTC/ETH）同时开仓时，降低第二笔同向仓位，减少相关性回撤。
        """
        if self.wallets is None:
            return proposed_stake

        equity = float(self.wallets.get_total(self.stake_currency) or 0.0)
        if equity <= 0:
            return proposed_stake

        # Read latest ATR (base timeframe)
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return proposed_stake
            last = dataframe.iloc[-1]
            atr = float(last.get("atr_14", 0.0))
        except Exception:
            return proposed_stake

        # Stop distance estimation (in price terms)
        stop_dist = max(atr * float(self.stake_atr_mult.value), current_rate * 0.003)  # floor 0.3%
        risk_amt = equity * float(self.risk_per_trade.value)
        if stop_dist <= 0:
            return proposed_stake

        # stake in quote currency; for futures, stake is collateral used.
        stake = risk_amt / (stop_dist / current_rate)  # risk / (stop_dist_pct)

        # Cap stake as fraction of equity (pre-leverage) to avoid concentration.
        stake = min(stake, equity * float(self.max_stake_fraction.value))

        # Same-direction exposure reduction across BTC/ETH
        try:
            open_trades = Trade.get_open_trades()
            same_side_open = any(
                (t.pair != pair)
                and (t.is_short is (side == "short"))
                and (t.pair.split("/", 1)[0] in {"BTC", "ETH"})
                for t in open_trades
            )
            if same_side_open:
                stake *= float(self.same_side_stake_mult.value)
        except Exception:
            pass

        if min_stake is not None:
            stake = max(stake, float(min_stake))
        stake = min(stake, float(max_stake))
        return stake

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Exit when the model becomes confident there's "no opportunity" ahead.
        # This helps avoid holding for days when the next-window edge disappears.
        p_none_th, p_none_margin = self._none_exit_thresholds_for_pair(metadata.get("pair", ""))

        p_none = df.get("none", 0.0)
        p_long = df.get("long", 0.0)
        p_short = df.get("short", 0.0)

        exit_common = df["do_predict"] == 1
        none_dominant = (p_none >= p_none_th) & (p_none >= p_long + p_none_margin) & (
            p_none >= p_short + p_none_margin
        )

        df["exit_long"] = 0
        df["exit_short"] = 0

        df.loc[exit_common & none_dominant, ["exit_long", "exit_short", "exit_tag"]] = (
            1,
            1,
            "ml_none",
        )
        return df
