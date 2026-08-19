"""
regime — Market regime classification and hard-filter gate.

Public API:
    RegimeSnapshot    — dataclass: trend, ATR, session, spread, vol-regime flags
    compute_regime    — builds a RegimeSnapshot from OHLCV + config
    cost_clears_edge  — rejects trades where edge doesn't cover round-trip costs
    atr               — ATR utility (Average True Range)
"""
from .filters import RegimeSnapshot, compute_regime, cost_clears_edge, atr

__all__ = ["RegimeSnapshot", "compute_regime", "cost_clears_edge", "atr"]
