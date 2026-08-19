"""
kronos_engine — Kronos model forecasting wrappers.

Public API:
    KronosForecaster  — loads real NeoQuasar/Kronos weights from HuggingFace Hub
    MockForecaster    — offline random-walk stand-in with identical interface
    BaseForecaster    — abstract base class (for type hints / custom subclasses)
"""
from .forecaster import BaseForecaster, KronosForecaster, MockForecaster

__all__ = ["BaseForecaster", "KronosForecaster", "MockForecaster"]
