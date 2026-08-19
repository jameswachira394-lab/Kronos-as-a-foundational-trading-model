from __future__ import annotations
import numpy as np
import pandas as pd


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: float) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
    downside = returns[returns < 0]
    if downside.std() == 0 or len(downside) < 2:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def summarize_trades(trades: pd.DataFrame, equity_curve: pd.Series,
                      periods_per_year: float = 252 * 24 * 12) -> dict:
    if trades.empty:
        return {"n_trades": 0}

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = losses["pnl"].sum()
    period_returns = equity_curve.pct_change().dropna()

    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades) if len(trades) else 0.0,
        "avg_win": wins["pnl"].mean() if len(wins) else 0.0,
        "avg_loss": losses["pnl"].mean() if len(losses) else 0.0,
        "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf"),
        "expectancy": trades["pnl"].mean(),
        "net_pnl": trades["pnl"].sum(),
        "max_drawdown": max_drawdown(equity_curve),
        "sharpe": sharpe_ratio(period_returns, periods_per_year),
        "sortino": sortino_ratio(period_returns, periods_per_year),
        "max_consecutive_losses": _max_consecutive(trades["pnl"] <= 0),
        "avg_holding_periods": trades["holding_periods"].mean() if "holding_periods" in trades else None,
        "turnover_lots": trades["lot_size"].sum() if "lot_size" in trades else None,
    }


def _max_consecutive(bool_series: pd.Series) -> int:
    max_run = run = 0
    for v in bool_series:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run
