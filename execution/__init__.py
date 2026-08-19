"""
execution — Paper and live order execution via MetaTrader 5.

Public API:
    MT5Executor  — sends orders to MT5 terminal (paper or live)
    run_loop     — polling loop: fetch data → build signal → execute
"""
from .mt5_connector import MT5Executor, run_loop

__all__ = ["MT5Executor", "run_loop"]
