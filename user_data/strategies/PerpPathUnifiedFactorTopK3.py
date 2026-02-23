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

from PerpPathTrendOnlyV1Lev3ConservativeShort import PerpPathTrendOnlyV1Lev3ConservativeShort


class PerpPathUnifiedFactorTopK3(PerpPathTrendOnlyV1Lev3ConservativeShort):
    """
    Single-bot unified capital pool strategy.
    - Multi-pair / long-short scoring
    - Top-K opportunity gate (K configured via `max_open_trades`, intended 3)
    - Tiered exits: low-score uses quick_tp, high-score disables quick_tp and follows trend exits
    """

    # Entry score thresholds
    long_score_low = DecimalParameter(0.48, 0.70, default=0.56, decimals=2, space="buy", optimize=True)
    long_score_high = DecimalParameter(0.62, 0.88, default=0.72, decimals=2, space="buy", optimize=True)
    short_score_low = DecimalParameter(0.48, 0.70, default=0.56, decimals=2, space="buy", optimize=True)
    short_score_high = DecimalParameter(0.62, 0.88, default=0.72, decimals=2, space="buy", optimize=True)

    # Probability/score flip exit controls
    score_flip_margin = DecimalParameter(0.06, 0.25, default=0.12, decimals=2, space="sell", optimize=True)
    min_hold_for_flip = IntParameter(10, 120, default=30, space="sell", optimize=True)

    # Dynamic stake controls (single pool)
    min_stake_scale = DecimalParameter(0.35, 0.75, default=0.50, decimals=2, space="buy", optimize=True)
    max_stake_scale = DecimalParameter(0.90, 1.60, default=1.20, decimals=2, space="buy", optimize=True)

    @staticmethod
    def _clip01(v):
        return np.clip(v, 0.0, 1.0)

    @staticmethod
    def _safe_div(a, b):
        if hasattr(b, "replace"):
            denom = b.replace(0, np.nan)
        else:
            denom = np.nan if float(b) == 0.0 else float(b)
        return a / denom

    @staticmethod
    def _score_to_tag(prefix: str, score: float, tier: str) -> str:
        return f"{prefix}_{tier}_s{int(round(score * 100))}"

    @staticmethod
    def _parse_score_from_tag(tag: Optional[str]) -> float:
        if not tag:
            return 0.0
        try:
            seg = str(tag).split("_s")[-1]
            return float(seg) / 100.0
        except Exception:
            return 0.0

    @staticmethod
    def _is_high_tier(tag: Optional[str]) -> bool:
        return bool(tag and "_hi_" in str(tag))

    @staticmethod
    def _is_low_tier(tag: Optional[str]) -> bool:
        return bool(tag and "_lo_" in str(tag))

    def _compute_factor_scores(self, df: DataFrame) -> DataFrame:
        # Trend geometry
        spread = (df["ema50"] - df["ema200"]).abs() / df["close"].replace(0, np.nan)
        trend_up_strength = self._clip01((df["ema20"] - df["ema50"]) / df["close"] * 180)
        trend_dn_strength = self._clip01((df["ema50"] - df["ema20"]) / df["close"] * 180)

        adx_score = self._clip01((df["adx"] - 12.0) / 20.0)
        atr_score = self._clip01((df["atr_pct"] - 0.0015) / 0.008)
        spread_score = self._clip01(self._safe_div(spread, float(self.trend_spread.value) + 1e-9))

        rsi_long = self._clip01(1.0 - (np.abs(df["rsi"] - 58.0) / 22.0))
        rsi_short = self._clip01(1.0 - (np.abs(df["rsi"] - 42.0) / 22.0))

        funding_long = self._clip01(
            (float(self.funding_long_max.value) - df["funding"]) / (abs(float(self.funding_long_max.value)) + 0.012)
        )
        funding_short = self._clip01(
            (df["funding"] - float(self.short_funding_min.value)) / (abs(float(self.short_funding_min.value)) + 0.012)
        )

        basis_long = self._clip01(1.0 - (df["basis"].abs() / max(1e-9, float(self.basis_abs_max.value))))
        basis_short = self._clip01(1.0 - (df["basis"].abs() / max(1e-9, float(self.short_basis_abs_max.value))))

        oi_long = self._clip01(1.0 - (df["oi_proxy"] / max(1e-9, float(self.oi_proxy_max.value))))
        oi_short = self._clip01(1.0 - (df["oi_proxy"] / max(1e-9, float(self.short_oi_proxy_max.value))))

        btc_long = df["btc_risk_on"].astype(float)
        btc_short = df["btc_risk_off"].astype(float)
        inf_long = df["inf1h_trend_up"].astype(float)
        inf_short = df["inf1h_trend_down"].astype(float)

        long_components = [
            0.18 * btc_long,
            0.16 * inf_long,
            0.14 * trend_up_strength,
            0.12 * adx_score,
            0.10 * spread_score,
            0.10 * rsi_long,
            0.08 * funding_long,
            0.06 * basis_long,
            0.04 * oi_long,
            0.02 * atr_score,
        ]

        short_components = [
            0.18 * btc_short,
            0.16 * inf_short,
            0.14 * trend_dn_strength,
            0.12 * adx_score,
            0.10 * spread_score,
            0.10 * rsi_short,
            0.08 * funding_short,
            0.06 * basis_short,
            0.04 * oi_short,
            0.02 * atr_score,
        ]

        df["score_long"] = sum(long_components)
        df["score_short"] = sum(short_components)
        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_indicators(dataframe, metadata)
        return self._compute_factor_scores(df)

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Build raw side conditions from base strategy
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

        long_ok = reduce(lambda x, y: x & y, long_cond)
        short_ok = reduce(lambda x, y: x & y, short_cond)

        long_hi = long_ok & (df["score_long"] >= float(self.long_score_high.value)) & (df["score_long"] >= df["score_short"])
        long_lo = (
            long_ok
            & ~long_hi
            & (df["score_long"] >= float(self.long_score_low.value))
            & (df["score_long"] >= df["score_short"])
        )

        short_hi = short_ok & (df["score_short"] >= float(self.short_score_high.value)) & (df["score_short"] > df["score_long"])
        short_lo = (
            short_ok
            & ~short_hi
            & (df["score_short"] >= float(self.short_score_low.value))
            & (df["score_short"] > df["score_long"])
        )

        df.loc[long_hi, ["enter_long", "enter_tag"]] = (
            1,
            self._score_to_tag("factor_long", float(self.long_score_high.value), "hi"),
        )
        df.loc[long_lo, ["enter_long", "enter_tag"]] = (
            1,
            self._score_to_tag("factor_long", float(self.long_score_low.value), "lo"),
        )
        df.loc[short_hi, ["enter_short", "enter_tag"]] = (
            1,
            self._score_to_tag("factor_short", float(self.short_score_high.value), "hi"),
        )
        df.loc[short_lo, ["enter_short", "enter_tag"]] = (
            1,
            self._score_to_tag("factor_short", float(self.short_score_low.value), "lo"),
        )
        return df

    def _global_topk_ok(self, pair: str, side: str, score: float) -> bool:
        """Global candidate ranking across all whitelist pairs and both sides."""
        if self.dp is None:
            return True

        candidates: list[tuple[str, str, float]] = []
        for p in self.dp.current_whitelist():
            dfp, _ = self.dp.get_analyzed_dataframe(pair=p, timeframe=self.timeframe)
            if dfp is None or dfp.empty:
                continue
            last = dfp.iloc[-1]
            s_long = float(last.get("score_long", 0.0))
            s_short = float(last.get("score_short", 0.0))
            if s_long >= float(self.long_score_low.value):
                candidates.append((p, "long", s_long))
            if s_short >= float(self.short_score_low.value):
                candidates.append((p, "short", s_short))

        if not candidates:
            return True

        topk = max(1, int(self.max_open_trades))
        top = sorted(candidates, key=lambda x: x[2], reverse=True)[:topk]
        return any((p == pair and s == side and score >= sc - 1e-12) for p, s, sc in top)

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        score = self._parse_score_from_tag(entry_tag)
        side_norm = "short" if side == "short" else "long"
        return self._global_topk_ok(pair=pair, side=side_norm, score=score)

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        score = self._parse_score_from_tag(entry_tag)
        min_scale = float(self.min_stake_scale.value)
        max_scale = float(self.max_stake_scale.value)
        scale = min_scale + (max_scale - min_scale) * np.clip((score - 0.50) / 0.40, 0.0, 1.0)
        stake = proposed_stake * scale

        if min_stake is not None:
            stake = max(stake, float(min_stake))
        stake = min(stake, float(max_stake))
        return float(stake)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if df is None or df.empty:
            return None

        last = df.iloc[-1]
        score_long = float(last.get("score_long", 0.0))
        score_short = float(last.get("score_short", 0.0))
        hold_min = int((current_time - trade.open_date_utc).total_seconds() // 60)
        tag = trade.enter_tag or ""

        # Tiered quick tp: low-score only
        if self._is_low_tier(tag) and current_profit >= float(self.trend_quick_tp.value):
            return "quick_tp_lo"

        # Probability/score flip exit for high-score trades.
        if hold_min >= int(self.min_hold_for_flip.value):
            flip_margin = float(self.score_flip_margin.value)
            if trade.is_short:
                if (score_long - score_short) >= flip_margin:
                    return "score_flip_short"
            else:
                if (score_short - score_long) >= flip_margin:
                    return "score_flip_long"

        # Preserve baseline safety exits.
        if trade.is_short:
            if hold_min >= int(self.short_fail_fast_minutes.value) and current_profit < float(self.short_fail_fast_loss.value):
                return "trend_short_fail_fast"
            if hold_min >= int(self.trend_hold_minutes.value):
                return "trend_short_time_stop"
        else:
            if hold_min >= 45 and current_profit < -0.012:
                return "trend_fail_fast"
            if hold_min >= int(self.trend_hold_minutes.value):
                return "trend_time_stop"

        return None
