"""
Everything Kronos doesn't see: trend regime, volatility regime, session,
spread, and cost-vs-expected-edge sanity checks. A LONG forecast can still
be rejected here.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period).mean()


@dataclass
class RegimeSnapshot:
    trend: str              # "bullish" / "bearish" / "neutral"
    atr_value: float
    atr_percentile: float   # 0..1 over trailing history
    session_ok: bool
    spread_ok: bool
    volatility_ok: bool


def compute_regime(df: pd.DataFrame, cfg: dict, current_spread_pips: float,
                    current_hour_utc: int) -> RegimeSnapshot:
    fast = df["close"].rolling(cfg["trend_fast_ma"]).mean().iloc[-1]
    slow = df["close"].rolling(cfg["trend_slow_ma"]).mean().iloc[-1]
    trend = "bullish" if fast > slow else ("bearish" if fast < slow else "neutral")

    atr_series = atr(df, cfg["atr_period"]).dropna()
    atr_value = float(atr_series.iloc[-1]) if len(atr_series) else float("nan")
    atr_pct = float((atr_series <= atr_value).mean()) if len(atr_series) else 0.5
    volatility_ok = atr_pct <= cfg["max_atr_percentile"]

    session_cfg = cfg.get("session_filter", {})
    if session_cfg.get("enabled", False):
        session_ok = current_hour_utc in session_cfg.get("allowed_hours_utc", list(range(24)))
    else:
        session_ok = True

    spread_ok = current_spread_pips <= cfg.get("max_spread_pips", 999)

    return RegimeSnapshot(
        trend=trend, atr_value=atr_value, atr_percentile=atr_pct,
        session_ok=session_ok, spread_ok=spread_ok, volatility_ok=volatility_ok,
    )


def cost_clears_edge(expected_return: float, spread_pips: float, commission_per_lot: float,
                      slippage_pips: float, pip_size: float, current_price: float,
                      contract_size: float, buffer_multiple: float = 1.5) -> bool:
    """Reject trades whose expected move doesn't comfortably clear round-trip costs."""
    spread_cost_ret = (spread_pips + slippage_pips) * pip_size / current_price
    commission_ret = commission_per_lot / (contract_size * current_price)
    total_cost_ret = spread_cost_ret + commission_ret
    return abs(expected_return) >= total_cost_ret * buffer_multiple
