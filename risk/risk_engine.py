"""
Risk engine — decides HOW MUCH to risk and WHERE to place SL/TP.
Never decides direction; that's signals/engine.py's job.
"""
from __future__ import annotations
from dataclasses import dataclass

from signals.engine import Decision, SignalResult


@dataclass
class TradePlan:
    decision: Decision
    entry: float
    stop_loss: float
    take_profit: float
    lot_size: float
    dollar_risk: float
    reward_risk_ratio: float
    rejected_reason: str | None = None


class RiskEngine:
    def __init__(self, cfg: dict, pip_size: float, contract_size: float, point_value: float):
        self.cfg = cfg
        self.pip_size = pip_size
        self.contract_size = contract_size
        self.point_value = point_value
        self._equity = cfg["account_size"]
        self._daily_pnl = 0.0
        self._open_positions = 0

    def new_day(self):
        self._daily_pnl = 0.0

    def register_fill_pnl(self, pnl: float):
        self._daily_pnl += pnl
        self._equity += pnl

    def daily_loss_limit_hit(self) -> bool:
        limit = -abs(self.cfg["max_daily_loss_pct"]) / 100.0 * self.cfg["account_size"]
        return self._daily_pnl <= limit

    def _stop_distance(self, sig: SignalResult, atr_value: float) -> float:
        if self.cfg["sl_mode"] == "atr" and atr_value == atr_value:  # not NaN
            return atr_value * self.cfg["atr_sl_multiple"]
        # forecast-based: use predicted adverse excursion
        feat = sig.features
        if sig.decision == Decision.LONG:
            adverse = abs(feat.expected_low_return) * feat.current_price
        else:
            adverse = abs(feat.expected_high_return) * feat.current_price
        return max(adverse, atr_value * 0.5 if atr_value == atr_value else adverse)

    def plan_trade(self, sig: SignalResult, current_price: float, atr_value: float) -> TradePlan:
        if sig.decision == Decision.NO_TRADE:
            return TradePlan(Decision.NO_TRADE, current_price, current_price, current_price,
                              0.0, 0.0, 0.0, rejected_reason="signal=NO_TRADE")

        if self._open_positions >= self.cfg["max_open_positions"]:
            return TradePlan(Decision.NO_TRADE, current_price, current_price, current_price,
                              0.0, 0.0, 0.0, rejected_reason="max_open_positions_reached")

        if self.daily_loss_limit_hit():
            return TradePlan(Decision.NO_TRADE, current_price, current_price, current_price,
                              0.0, 0.0, 0.0, rejected_reason="daily_loss_limit_hit")

        stop_dist = self._stop_distance(sig, atr_value)
        if stop_dist <= 0 or stop_dist != stop_dist:
            return TradePlan(Decision.NO_TRADE, current_price, current_price, current_price,
                              0.0, 0.0, 0.0, rejected_reason="invalid_stop_distance")

        tp_dist = stop_dist * self.cfg["atr_tp_multiple"] / self.cfg["atr_sl_multiple"]
        rr = tp_dist / stop_dist
        if rr < self.cfg["min_reward_risk"]:
            return TradePlan(Decision.NO_TRADE, current_price, current_price, current_price,
                              0.0, 0.0, rr, rejected_reason="reward_risk_below_minimum")

        dollar_risk = self._equity * self.cfg["risk_per_trade_pct"] / 100.0
        stop_pips = stop_dist / self.pip_size
        # dollar_risk = lot_size * stop_pips * point_value  =>  solve for lot_size
        lot_size = dollar_risk / (stop_pips * self.point_value) if stop_pips > 0 else 0.0
        lot_size = max(round(lot_size, 2), 0.01) if lot_size > 0 else 0.0

        if sig.decision == Decision.LONG:
            sl = current_price - stop_dist
            tp = current_price + tp_dist
        else:
            sl = current_price + stop_dist
            tp = current_price - tp_dist

        return TradePlan(
            decision=sig.decision, entry=current_price, stop_loss=sl, take_profit=tp,
            lot_size=lot_size, dollar_risk=dollar_risk, reward_risk_ratio=rr,
        )

    def on_open(self):
        self._open_positions += 1

    def on_close(self):
        self._open_positions = max(0, self._open_positions - 1)
