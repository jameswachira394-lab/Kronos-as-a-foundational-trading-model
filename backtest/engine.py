"""
Event-driven backtester. Walks forward bar by bar, only ever using data up
to "now" (no look-ahead), periodically asks Kronos for a forecast, runs it
through the signal/risk pipeline, and simulates fills including spread,
commission and slippage. Positions are exited on SL/TP touch or forecast
horizon expiry, whichever comes first.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kronos_engine.forecaster import BaseForecaster
from data.loader import to_kronos_frame
from features.forecast_features import extract_ensemble, extract_single
from regime.filters import compute_regime
from signals.engine import score_signal, Decision
from risk.risk_engine import RiskEngine, TradePlan


@dataclass
class OpenPosition:
    decision: Decision
    entry: float
    stop_loss: float
    take_profit: float
    lot_size: float
    entry_index: int
    max_hold_bars: int


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.Series
    signals_log: pd.DataFrame


def _apply_costs_to_entry(price: float, decision: Decision, spread_pips: float,
                           slippage_pips: float, pip_size: float) -> float:
    half_spread = spread_pips * pip_size / 2.0
    slip = slippage_pips * pip_size
    if decision == Decision.LONG:
        return price + half_spread + slip
    else:
        return price - half_spread - slip


def run_backtest(df: pd.DataFrame, forecaster: BaseForecaster, cfg: dict,
                  use_ensemble: bool = True, signal_every_n_bars: int = 12,
                  verbose: bool = False) -> BacktestResult:
    """
    df: cleaned OHLCV with a 'timestamp' column, ascending.
    cfg: the full parsed config.yaml dict.
    """
    inst = cfg["instrument"]
    kcfg = cfg["kronos"]
    lookback = kcfg["lookback"]
    pred_len = kcfg["pred_len"]

    risk_engine = RiskEngine(cfg["risk"], inst["pip_size"], inst["contract_size"], inst["point_value"])

    trades = []
    signals_log = []
    equity = cfg["risk"]["account_size"]
    equity_curve = []
    open_pos: OpenPosition | None = None
    current_day = None

    n = len(df)
    start_idx = lookback

    for i in range(start_idx, n - 1):
        row = df.iloc[i]
        ts = row["timestamp"]

        if current_day != ts.date():
            current_day = ts.date()
            risk_engine.new_day()

        # ---- manage open position first (check SL/TP on this bar) ----
        if open_pos is not None:
            bar = df.iloc[i]
            hit_sl = hit_tp = False
            if open_pos.decision == Decision.LONG:
                hit_sl = bar["low"] <= open_pos.stop_loss
                hit_tp = bar["high"] >= open_pos.take_profit
            else:
                hit_sl = bar["high"] >= open_pos.stop_loss
                hit_tp = bar["low"] <= open_pos.take_profit

            expired = (i - open_pos.entry_index) >= open_pos.max_hold_bars

            if hit_sl or hit_tp or expired:
                if hit_sl and hit_tp:
                    exit_price = open_pos.stop_loss  # conservative: assume SL first
                elif hit_sl:
                    exit_price = open_pos.stop_loss
                elif hit_tp:
                    exit_price = open_pos.take_profit
                else:
                    exit_price = bar["close"]

                direction_mult = 1 if open_pos.decision == Decision.LONG else -1
                price_diff = (exit_price - open_pos.entry) * direction_mult
                commission = cfg["costs"]["commission_per_lot"] * open_pos.lot_size
                pnl = price_diff / inst["pip_size"] * inst["point_value"] * open_pos.lot_size - commission

                trades.append({
                    "entry_time": df.iloc[open_pos.entry_index]["timestamp"],
                    "exit_time": ts,
                    "decision": open_pos.decision.value,
                    "entry": open_pos.entry,
                    "exit": exit_price,
                    "lot_size": open_pos.lot_size,
                    "pnl": pnl,
                    "holding_periods": i - open_pos.entry_index,
                    "exit_reason": "SL" if hit_sl else ("TP" if hit_tp else "EXPIRY"),
                })
                risk_engine.register_fill_pnl(pnl)
                risk_engine.on_close()
                equity += pnl
                open_pos = None

        equity_curve.append((ts, equity))

        # ---- only look for a new signal periodically, and only if flat ----
        if open_pos is not None or (i - start_idx) % signal_every_n_bars != 0:
            continue

        hist = df.iloc[i - lookback:i]
        future_ts_idx = df.iloc[i:i + pred_len]["timestamp"]
        if len(future_ts_idx) < pred_len:
            break  # not enough room left to forecast a full horizon

        x_ts = hist["timestamp"]
        x_df = to_kronos_frame(hist)  # always provides the 6 columns Kronos expects

        current_price = row["close"]

        if use_ensemble:
            paths = forecaster.forecast_ensemble(
                x_df, x_ts, future_ts_idx, pred_len,
                n=kcfg["ensemble_size"], T=kcfg["T"], top_p=kcfg["top_p"], top_k=kcfg["top_k"])
            feat = extract_ensemble(paths, current_price)
        else:
            pred_df = forecaster.forecast(x_df, x_ts, future_ts_idx, pred_len,
                                           T=kcfg["T"], top_p=kcfg["top_p"], top_k=kcfg["top_k"])
            feat = extract_single(pred_df, current_price)

        regime = compute_regime(df.iloc[max(0, i - 300):i], cfg["regime"],
                                 current_spread_pips=cfg["costs"]["spread_pips"],
                                 current_hour_utc=ts.hour)

        sig = score_signal(feat, regime, cfg["signal"], cfg["costs"],
                            inst["contract_size"], inst["pip_size"])

        atr_value = regime.atr_value
        plan: TradePlan = risk_engine.plan_trade(sig, current_price, atr_value)

        signals_log.append({
            "timestamp": ts, "decision": sig.decision.value, "score": sig.score,
            "expected_return": feat.expected_return, "bullish_prob": feat.bullish_probability,
            "bearish_prob": feat.bearish_probability, "path_consistency": feat.path_consistency,
            "trend": regime.trend, "rejected_reason": plan.rejected_reason,
        })

        if plan.decision != Decision.NO_TRADE:
            fill_price = _apply_costs_to_entry(current_price, plan.decision,
                                                cfg["costs"]["spread_pips"],
                                                cfg["costs"]["slippage_pips"],
                                                inst["pip_size"])
            open_pos = OpenPosition(
                decision=plan.decision, entry=fill_price, stop_loss=plan.stop_loss,
                take_profit=plan.take_profit, lot_size=plan.lot_size,
                entry_index=i, max_hold_bars=pred_len,
            )
            risk_engine.on_open()

        if verbose and (i - start_idx) % (signal_every_n_bars * 20) == 0:
            print(f"{ts} equity={equity:.2f} decision={sig.decision.value} score={sig.score}")

    trades_df = pd.DataFrame(trades)
    equity_series = pd.Series(dict(equity_curve))
    signals_df = pd.DataFrame(signals_log)
    return BacktestResult(trades=trades_df, equity_curve=equity_series, signals_log=signals_df)
