"""
signals — Converts forecast features + regime into a discrete trade decision.

Public API:
    Decision      — LONG / SHORT / NO_TRADE enum
    SignalResult  — dataclass: decision, score, reasons, features, regime
    score_signal  — transparent, config-driven scoring model
"""
from .engine import Decision, SignalResult, score_signal

__all__ = ["Decision", "SignalResult", "score_signal"]
