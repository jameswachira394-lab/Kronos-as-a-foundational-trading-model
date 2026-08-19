"""
Rolling walk-forward evaluation: split history into N consecutive
train/test windows (train is informational context only — Kronos-base is
pretrained and used zero-shot unless you've separately fine-tuned it; the
"train" window here still matters for calibrating signal thresholds without
peeking at test data). Each test window's results are kept independent, so
you get an honest, non-overlapping out-of-sample performance curve instead
of one lucky global backtest.
"""
from __future__ import annotations
import pandas as pd

from backtest.engine import run_backtest, BacktestResult
from backtest.metrics import summarize_trades


def make_windows(df: pd.DataFrame, n_windows: int, test_fraction: float = 0.2) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(df)
    window_size = n // n_windows
    windows = []
    for w in range(n_windows):
        start = w * window_size
        end = start + window_size if w < n_windows - 1 else n
        segment = df.iloc[start:end].reset_index(drop=True)
        split = int(len(segment) * (1 - test_fraction))
        train, test = segment.iloc[:split], segment.iloc[split:].reset_index(drop=True)
        windows.append((train, test))
    return windows


def run_walk_forward(df: pd.DataFrame, forecaster, cfg: dict, n_windows: int = 4,
                      use_ensemble: bool = True, signal_every_n_bars: int = 12) -> pd.DataFrame:
    windows = make_windows(df, n_windows)
    rows = []
    for idx, (train, test) in enumerate(windows):
        # NOTE: pretrained Kronos-base needs no training here; `train` is
        # reserved for anyone who plugs in threshold-calibration or a
        # fine-tuned checkpoint per window without leaking into `test`.
        if len(test) < cfg["kronos"]["lookback"] + cfg["kronos"]["pred_len"] + 1:
            continue
        result: BacktestResult = run_backtest(test, forecaster, cfg,
                                               use_ensemble=use_ensemble,
                                               signal_every_n_bars=signal_every_n_bars)
        stats = summarize_trades(result.trades, result.equity_curve) if not result.trades.empty else {"n_trades": 0}
        stats["window"] = idx
        stats["test_start"] = test["timestamp"].iloc[0]
        stats["test_end"] = test["timestamp"].iloc[-1]
        rows.append(stats)
    return pd.DataFrame(rows)
