"""
Turns Kronos output into decision-ready features. Never used as a raw
BUY/SELL — see signals/engine.py for how these combine into a score.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class ForecastFeatures:
    current_price: float
    expected_return: float          # (final predicted close / current) - 1
    expected_high_return: float     # max predicted high vs current
    expected_low_return: float      # min predicted low vs current
    forecast_volatility: float      # std of predicted step returns (single path or ensemble mean)
    direction: str                  # "long" / "short" / "flat"
    bullish_probability: float      # fraction of ensemble paths that finish up
    bearish_probability: float
    path_consistency: float         # 1 - (std of terminal returns across ensemble / |mean terminal return|)
    n_paths: int

    def to_dict(self):
        return asdict(self)


def _path_return(pred_df: pd.DataFrame, current_price: float) -> float:
    return float(pred_df["close"].iloc[-1] / current_price - 1.0)


def _path_step_vol(pred_df: pd.DataFrame) -> float:
    rets = pred_df["close"].pct_change().dropna()
    return float(rets.std()) if len(rets) > 1 else 0.0


def extract_single(pred_df: pd.DataFrame, current_price: float) -> ForecastFeatures:
    """Features from one forecast path (no probability info available)."""
    exp_ret = _path_return(pred_df, current_price)
    exp_high = float(pred_df["high"].max() / current_price - 1.0)
    exp_low = float(pred_df["low"].min() / current_price - 1.0)
    vol = _path_step_vol(pred_df)
    direction = "long" if exp_ret > 0 else ("short" if exp_ret < 0 else "flat")
    return ForecastFeatures(
        current_price=current_price,
        expected_return=exp_ret,
        expected_high_return=exp_high,
        expected_low_return=exp_low,
        forecast_volatility=vol,
        direction=direction,
        bullish_probability=float(exp_ret > 0),
        bearish_probability=float(exp_ret < 0),
        path_consistency=1.0,
        n_paths=1,
    )


def extract_ensemble(paths: list[pd.DataFrame], current_price: float) -> ForecastFeatures:
    """Features from a real ensemble of independently-sampled forecast paths.
    This is what gives you an actual bullish/bearish probability, unlike a
    single predict() call."""
    terminal_returns = np.array([_path_return(p, current_price) for p in paths])
    highs = np.array([float(p["high"].max() / current_price - 1.0) for p in paths])
    lows = np.array([float(p["low"].min() / current_price - 1.0) for p in paths])
    vols = np.array([_path_step_vol(p) for p in paths])

    mean_ret = float(terminal_returns.mean())
    std_ret = float(terminal_returns.std())
    bullish_p = float((terminal_returns > 0).mean())
    bearish_p = float((terminal_returns < 0).mean())

    # 1.0 = all paths agree on magnitude/sign, 0.0 = totally scattered
    denom = abs(mean_ret) if abs(mean_ret) > 1e-9 else (std_ret if std_ret > 1e-9 else 1e-9)
    consistency = float(np.clip(1.0 - std_ret / denom, 0.0, 1.0))

    direction = "long" if mean_ret > 0 else ("short" if mean_ret < 0 else "flat")

    return ForecastFeatures(
        current_price=current_price,
        expected_return=mean_ret,
        expected_high_return=float(highs.mean()),
        expected_low_return=float(lows.mean()),
        forecast_volatility=float(vols.mean()),
        direction=direction,
        bullish_probability=bullish_p,
        bearish_probability=bearish_p,
        path_consistency=consistency,
        n_paths=len(paths),
    )
