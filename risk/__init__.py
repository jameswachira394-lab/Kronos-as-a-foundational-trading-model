"""
risk — Position sizing, SL/TP placement, and daily loss limit tracking.

Public API:
    RiskEngine  — core risk management class
    TradePlan   — dataclass: entry, SL, TP, lot size, dollar risk, R:R
"""
from .risk_engine import RiskEngine, TradePlan

__all__ = ["RiskEngine", "TradePlan"]
