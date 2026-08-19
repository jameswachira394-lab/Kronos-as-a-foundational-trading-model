"""
features — Turns Kronos forecast output into decision-ready feature structs.

Public API:
    ForecastFeatures  — dataclass holding all derived forecast metrics
    extract_single    — features from a single forecast path
    extract_ensemble  — features from a real ensemble of independent paths
                        (provides proper bullish/bearish probabilities)
"""
from .forecast_features import ForecastFeatures, extract_single, extract_ensemble

__all__ = ["ForecastFeatures", "extract_single", "extract_ensemble"]
