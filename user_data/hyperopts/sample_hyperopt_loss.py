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
DEFAULT_TARGET_TRADES_PER_MONTH = 120
DEFAULT_TARGET_MONTHLY_RETURN = 0.10

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
    # Compound target to match how "month-on-month" goals are usually stated.
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
            # Large penalty to make these candidates non-competitive.
            # Keep continuous component so hyperopt can still "learn" directionally.
            return 1_000.0 + (max_dd - max_dd_limit) * 10_000.0

        # Targets scaled to the timerange length.
        months = _months_in_timerange(min_date, max_date)
        target_trades_per_month = _float_param(params, "target_trades_per_month", DEFAULT_TARGET_TRADES_PER_MONTH)
        target_monthly_return = _float_param(params, "target_monthly_return", DEFAULT_TARGET_MONTHLY_RETURN)

        target_trades = max(1.0, target_trades_per_month * months)
        expected_total = _expected_total_return(target_monthly_return, months)

        # Loss components:
        # - trade_loss: encourage the chosen activity level (80–150 trades/month typical for 5m).
        # - profit_loss: main driver (penalize negative profit strongly).
        # - duration_loss: mild penalty for very long average holding times.
        trade_loss = 1 - 0.25 * exp(-((trade_count - target_trades) ** 2) / 10**5.6)
        if total_profit < 0:
            profit_loss = 2.0 + min(5.0, abs(total_profit) / max(1e-9, expected_total))
        else:
            profit_loss = max(0.0, 1.0 - (total_profit / max(1e-9, expected_total)))
        duration_loss = 0.25 * min(trade_duration / MAX_ACCEPTED_TRADE_DURATION, 1.0)
        result = trade_loss + profit_loss + duration_loss
        return result
