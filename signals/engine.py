"""
Converts (ForecastFeatures, RegimeSnapshot, cost check) into a discrete
trade decision + a numeric confidence, via a transparent, tunable scoring
model (not a black box). Thresholds live in config.yaml and must be
calibrated via backtesting/walk-forward — they are NOT assumed profitable
out of the box.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from features.forecast_features import ForecastFeatures
from regime.filters import RegimeSnapshot, cost_clears_edge


class Decision(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"


@dataclass
class SignalResult:
    decision: Decision
    score: int
    reasons: list[str]
    features: ForecastFeatures
    regime: RegimeSnapshot


def score_signal(feat: ForecastFeatures, regime: RegimeSnapshot, cfg: dict,
                  cost_cfg: dict, contract_size: float, pip_size: float) -> SignalResult:
    reasons = []
    score = 0
    w = cfg["weights"]

    long_edge = feat.expected_return > cfg["min_expected_return"]
    short_edge = feat.expected_return < -cfg["min_expected_return"]
    if long_edge:
        score += w["expected_return"]; reasons.append("expected_return>threshold(long)")
    elif short_edge:
        score -= w["expected_return"]; reasons.append("expected_return<-threshold(short)")

    if feat.bullish_probability >= cfg["min_bullish_probability"]:
        score += w["path_probability"]; reasons.append(f"bullish_prob={feat.bullish_probability:.2f}")
    if feat.bearish_probability >= cfg["min_bearish_probability"]:
        score -= w["path_probability"]; reasons.append(f"bearish_prob={feat.bearish_probability:.2f}")

    if feat.forecast_volatility <= cfg["max_forecast_volatility"]:
        score += w["volatility_ok"] if feat.expected_return > 0 else 0
        score -= w["volatility_ok"] if feat.expected_return < 0 else 0
        reasons.append("forecast_volatility_ok")
    else:
        reasons.append("forecast_volatility_too_high(no_bonus)")

    if regime.trend == "bullish" and feat.expected_return > 0:
        score += w["trend_alignment"]; reasons.append("trend_aligned_bullish")
    elif regime.trend == "bearish" and feat.expected_return < 0:
        score -= w["trend_alignment"]; reasons.append("trend_aligned_bearish")

    if feat.path_consistency >= 0.5:
        score += w["path_consistency"] if feat.expected_return > 0 else 0
        score -= w["path_consistency"] if feat.expected_return < 0 else 0
        reasons.append(f"path_consistency={feat.path_consistency:.2f}")

    # hard filters — any failure forces NO_TRADE regardless of score
    hard_fail = []
    if not regime.session_ok:
        hard_fail.append("outside_session")
    if not regime.spread_ok:
        hard_fail.append("spread_too_wide")
    if not regime.volatility_ok:
        hard_fail.append("volatility_regime_extreme")
    if not cost_clears_edge(feat.expected_return, cost_cfg["spread_pips"],
                             cost_cfg["commission_per_lot"], cost_cfg["slippage_pips"],
                             pip_size, feat.current_price, contract_size):
        hard_fail.append("edge_does_not_clear_costs")

    if hard_fail:
        return SignalResult(Decision.NO_TRADE, score, reasons + hard_fail, feat, regime)

    if score >= cfg["score_long_threshold"]:
        decision = Decision.LONG
    elif score <= cfg["score_short_threshold"]:
        decision = Decision.SHORT
    else:
        decision = Decision.NO_TRADE

    return SignalResult(decision, score, reasons, feat, regime)
