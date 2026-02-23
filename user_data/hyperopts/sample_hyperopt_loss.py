from datetime import datetime
from math import exp

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.optimize.hyperopt import IHyperOptLoss
from freqtrade.util.dry_run_wallet import get_dry_run_wallet


# Define some constants:

# Baseline targets used by the loss function.
# You can override these via config using:
# "hyperopt_loss_params": {
#   "max_drawdown_limit": "12%",
#   "target_trades_per_month": 120,
#   "target_monthly_return": 0.10
# }
#
# Why: hyperopt is run on different timeranges (60d/150d/213d/7mo ...).
# If the loss hardcodes a single TARGET_TRADES / EXPECTED_MAX_PROFIT, it may optimize the wrong thing
# (e.g. chasing trade count or short duration while tolerating negative profit).
DEFAULT_TARGET_TRADES_PER_MONTH = 90
DEFAULT_TARGET_MONTHLY_RETURN = 0.04
DEFAULT_MIN_TRADES_PER_MONTH = 25
DEFAULT_SOFT_DRAWDOWN_LIMIT = 0.08

# max average trade duration in minutes
# if eval ends with higher value, we consider it a failed eval
MAX_ACCEPTED_TRADE_DURATION = 300

# Discrete max-drawdown constraint candidates (as ratios).
# Hyperopt cannot directly optimize the loss-function itself, so this is selected via config.
# See `hyperopt_loss` section in your config file.
ALLOWED_MAX_DRAWDOWN_LIMITS = (0.05, 0.08, 0.10, 0.12)


def _parse_max_dd_choice(value: object) -> float | None:
    """
    Parses a max drawdown choice to a ratio.
    Accepts: 0.08, 8, "8", "8%", "0.08".
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
    elif isinstance(value, str):
        s = value.strip()
        if s.endswith("%"):
            s = s[:-1].strip()
            try:
                return float(s) / 100.0
            except ValueError:
                return None
        try:
            v = float(s)
        except ValueError:
            return None
    else:
        return None

    # Heuristic: values > 1 are likely "percent points" (e.g. 8 -> 8%).
    if v > 1.0:
        v = v / 100.0
    return v


def _select_discrete_max_dd_limit(config: Config) -> float:
    """
    Selects a discrete max drawdown limit from ALLOWED_MAX_DRAWDOWN_LIMITS.
    This is chosen via config (not optimized by hyperopt).
    """
    # Optional config snippet (recommended):
    # "hyperopt_loss_params": { "max_drawdown_limit": "8%" }
    #
    # Note: Freqtrade itself also uses the key "hyperopt_loss" for the loss-class name (string),
    # so we must not rely on "hyperopt_loss" being a dict.
    if isinstance(config, dict):
        loss_cfg = config.get("hyperopt_loss_params", {})
        if not isinstance(loss_cfg, dict):
            loss_cfg = {}
        # Backward compatibility: if user stored params in "hyperopt_loss" and it's a dict, read it.
        legacy = config.get("hyperopt_loss", {})
        if isinstance(legacy, dict):
            loss_cfg = {**legacy, **loss_cfg}
    else:
        loss_cfg = {}

    raw = loss_cfg.get("max_drawdown_limit", None)
    parsed = _parse_max_dd_choice(raw)
    if parsed is None:
        return 0.10
    return min(ALLOWED_MAX_DRAWDOWN_LIMITS, key=lambda x: abs(x - parsed))


def _loss_params(config: Config) -> dict:
    if not isinstance(config, dict):
        return {}
    params = config.get("hyperopt_loss_params", {})
    if isinstance(params, dict):
        return params
    return {}


def _float_param(params: dict, key: str, default: float) -> float:
    try:
        v = params.get(key, default)
        if v is None:
            return float(default)
        if isinstance(v, str):
            s = v.strip()
            if s.endswith("%"):
                return float(s[:-1].strip()) / 100.0
            return float(s)
        return float(v)
    except Exception:
        return float(default)


def _months_in_timerange(min_date: datetime, max_date: datetime) -> float:
    # Use average month length to avoid depending on calendar month boundaries.
    secs = max(0.0, (max_date - min_date).total_seconds())
    days = secs / 86400.0
    return max(1e-9, days / 30.4375)


def _expected_total_return(target_monthly_return: float, months: float) -> float:
    # Compound target to match month-on-month compounding expectation.
    return max(0.0, (1.0 + max(0.0, target_monthly_return)) ** months - 1.0)


def _max_drawdown_ratio(results: DataFrame, starting_balance: float) -> float:
    """
    Computes max drawdown ratio from trade-by-trade equity curve.
    Uses `profit_abs` (preferred) and falls back to 0 when unavailable.
    """
    if results is None or results.empty:
        return 0.0
    if "profit_abs" not in results.columns:
        return 0.0
    try:
        pnl = results["profit_abs"].fillna(0.0).astype(float).reset_index(drop=True)
    except Exception:
        return 0.0
    if len(pnl) == 0:
        return 0.0

    # Guard: starting_balance could be 0 in pathological configs.
    if starting_balance <= 0:
        return 0.0

    equity = starting_balance + pnl.cumsum()
    run_max = equity.cummax()
    dd = (run_max - equity) / run_max
    try:
        mdd = float(dd.max())
    except Exception:
        mdd = 0.0
    return max(0.0, mdd)


def _short_trade_ratio(results: DataFrame) -> float:
    """
    Ratio of trades tagged as short-like.
    Uses `enter_tag` text matching to remain strategy-agnostic.
    """
    if results is None or results.empty or "enter_tag" not in results.columns:
        return 0.0
    tags = results["enter_tag"].fillna("").astype(str).str.lower()
    if len(tags) == 0:
        return 0.0
    short_mask = tags.str.contains("short")
    return float(short_mask.mean())


class SampleHyperOptLoss(IHyperOptLoss):
    """
    Defines the default loss function for hyperopt
    This is intended to give you some inspiration for your own loss function.

    The Function needs to return a number (float) - which becomes smaller for better backtest
    results.
    """

    @staticmethod
    def hyperopt_loss_function(
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        *args,
        **kwargs,
    ) -> float:
        """
        Objective function, returns smaller number for better results
        """
        params = _loss_params(config)

        # Total profit (prefer absolute PnL ratio over Σ% to avoid relying on trade count / sizing).
        start_balance = float(get_dry_run_wallet(config))
        if start_balance > 0 and "profit_abs" in results.columns:
            total_profit = float(results["profit_abs"].fillna(0.0).sum()) / start_balance
        else:
            total_profit = float(results["profit_ratio"].fillna(0.0).sum())

        trade_duration = float(results["trade_duration"].mean()) if "trade_duration" in results.columns else 0.0

        # Risk constraint: reject candidates exceeding the selected max drawdown limit.
        max_dd_limit = _select_discrete_max_dd_limit(config)
        max_dd = _max_drawdown_ratio(results, start_balance)
        if max_dd > max_dd_limit:
            hard_dd_base = _float_param(params, "hard_dd_penalty_base", 50.0)
            hard_dd_scale = _float_param(params, "hard_dd_penalty_scale", 1200.0)
            # Keep a hard constraint, but avoid a penalty so extreme that optimizer
            # ignores all profit information and converges to ultra-conservative sets.
            return hard_dd_base + (max_dd - max_dd_limit) * hard_dd_scale

        # Targets scaled to the timerange length.
        months = _months_in_timerange(min_date, max_date)
        target_trades_per_month = _float_param(params, "target_trades_per_month", DEFAULT_TARGET_TRADES_PER_MONTH)
        min_trades_per_month = _float_param(params, "min_trades_per_month", DEFAULT_MIN_TRADES_PER_MONTH)
        target_monthly_return = _float_param(params, "target_monthly_return", DEFAULT_TARGET_MONTHLY_RETURN)
        soft_drawdown_limit = _float_param(params, "soft_drawdown_limit", DEFAULT_SOFT_DRAWDOWN_LIMIT)

        target_trades = max(1.0, target_trades_per_month * months)
        min_trades = max(1.0, min_trades_per_month * months)
        expected_total = _expected_total_return(target_monthly_return, months)

        # Loss components:
        # - trade_loss: keep activity around the target zone.
        # - min_trade_penalty: hard penalty when coverage is too low.
        # - profit_loss: strong downside penalty when total return is negative.
        # - soft_dd_penalty: continuous drawdown control below the hard cutoff.
        # - duration_loss: mild penalty for very long average holding times.
        trade_loss = 1 - 0.12 * exp(-((trade_count - target_trades) ** 2) / 10**5.6)
        min_trade_penalty = 0.0
        if trade_count < min_trades:
            min_trade_penalty_mult = _float_param(params, "min_trade_penalty_mult", 1.2)
            min_trade_penalty = min_trade_penalty_mult * ((min_trades - trade_count) / min_trades)

        profit_reward_mult = _float_param(params, "profit_reward_mult", 1.6)
        if total_profit < 0:
            profit_loss = 4.0 + min(8.0, 2.0 * abs(total_profit) / max(1e-9, expected_total))
        else:
            # Stronger upside reward once target return is exceeded.
            base = max(0.0, 1.0 - (total_profit / max(1e-9, expected_total)))
            bonus = profit_reward_mult * max(0.0, (total_profit / max(1e-9, expected_total)) - 1.0)
            profit_loss = base - bonus

        short_penalty = 0.0
        min_short_ratio = _float_param(params, "min_short_trade_ratio", 0.0)
        if min_short_ratio > 0.0:
            short_ratio = _short_trade_ratio(results)
            if short_ratio < min_short_ratio:
                short_penalty = 1.5 * ((min_short_ratio - short_ratio) / max(1e-9, min_short_ratio))

        soft_dd_penalty = 0.0
        if max_dd > soft_drawdown_limit:
            soft_dd_penalty_mult = _float_param(params, "soft_dd_penalty_mult", 1.2)
            soft_dd_penalty = soft_dd_penalty_mult * ((max_dd - soft_drawdown_limit) / soft_drawdown_limit)
        duration_loss = 0.20 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1.0)
        result = trade_loss + min_trade_penalty + profit_loss + short_penalty + soft_dd_penalty + duration_loss
        return result
