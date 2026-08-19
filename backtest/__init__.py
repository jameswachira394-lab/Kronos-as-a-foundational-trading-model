"""
backtest — Event-driven backtester, metrics, and walk-forward evaluation.

Public API:
    run_backtest      — bar-by-bar backtester; returns BacktestResult
    BacktestResult    — dataclass: trades DataFrame, equity curve, signals log
    summarize_trades  — compute Sharpe, Sortino, drawdown, profit factor, etc.
    run_walk_forward  — rolling out-of-sample window evaluation
"""
from .engine import run_backtest, BacktestResult
from .metrics import summarize_trades
from .walk_forward import run_walk_forward

__all__ = ["run_backtest", "BacktestResult", "summarize_trades", "run_walk_forward"]
