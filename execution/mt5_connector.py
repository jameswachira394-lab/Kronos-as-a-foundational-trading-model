"""
Execution layer. Two modes:

  paper  - computes the signal + trade plan on live MT5 data and logs what
           it WOULD do. Sends zero orders. Always safe to run.

  live   - actually sends orders via MetaTrader5. Requires BOTH
           config['execution']['mode'] == 'live' AND
           config['execution']['live'] == True (belt + suspenders), so a
           stray config edit can't silently start real trading.
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timezone

from signals.engine import Decision
from risk.risk_engine import TradePlan

import os

log = logging.getLogger(__name__)


class MT5Executor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.exec_cfg = cfg["execution"]
        self.is_live = self.exec_cfg.get("mode") == "live" and self.exec_cfg.get("live") is True
        self._mt5 = None
        if self.is_live:
            self._connect()

    def _connect(self):
        try:
            import MetaTrader5 as mt5
        except ImportError as e:
            raise RuntimeError("MetaTrader5 package required for live execution.") from e

        m = self.cfg.get("data", {}).get("mt5") or self.cfg.get("execution", {}).get("mt5", {})
        login = m.get("login") or os.getenv("MT5_LOGIN")
        password = m.get("password") or os.getenv("MT5_PASSWORD")
        server = m.get("server") or os.getenv("MT5_SERVER")
        path = m.get("path") or os.getenv("MT5_PATH")

        init_kwargs = {}
        if path:
            init_kwargs["path"] = path
        if login:
            init_kwargs["login"] = int(login)
        if password:
            init_kwargs["password"] = str(password)
        if server:
            init_kwargs["server"] = str(server)

        if not mt5.initialize(**init_kwargs):
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

        if login and password and server:
            if not mt5.login(login=int(login), password=str(password), server=str(server)):
                raise RuntimeError(f"MT5 broker login failed for account {login} on {server}: {mt5.last_error()}")

        self._mt5 = mt5
        acc = mt5.account_info()
        acc_id = acc.login if acc else "Unknown"
        acc_server = acc.server if acc else "Unknown"
        log.warning("LIVE TRADING ENABLED. Real orders will be sent. Account: %s @ %s", acc_id, acc_server)

    def execute(self, plan: TradePlan, symbol: str) -> dict:
        """Returns a fill/log record. Never raises on rejected plans."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "decision": plan.decision.value,
            "entry": plan.entry,
            "sl": plan.stop_loss,
            "tp": plan.take_profit,
            "lot_size": plan.lot_size,
            "mode": "live" if self.is_live else "paper",
        }

        if plan.decision == Decision.NO_TRADE:
            record["status"] = f"no_trade ({plan.rejected_reason})"
            return record

        if not self.is_live:
            record["status"] = "paper_logged_only"
            log.info("PAPER TRADE: %s", record)
            return record

        mt5 = self._mt5
        order_type = mt5.ORDER_TYPE_BUY if plan.decision == Decision.LONG else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": plan.lot_size,
            "type": order_type,
            "price": plan.entry,
            "sl": plan.stop_loss,
            "tp": plan.take_profit,
            "deviation": 10,
            "magic": 991100,
            "comment": "kronos-trading-system",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        record["status"] = str(result.retcode) if result else "order_send_failed"
        record["raw_result"] = result._asdict() if result else None
        log.warning("LIVE ORDER SENT: %s -> %s", request, record["status"])
        return record


def run_loop(cfg: dict, forecaster, data_fetch_fn, build_signal_fn, poll_seconds: int | None = None):
    """Generic paper/live polling loop. `data_fetch_fn()` -> fresh cleaned df.
    `build_signal_fn(df)` -> TradePlan. Kept generic so main.py wires the
    concrete pipeline; this just handles the scheduling + logging + safety."""
    executor = MT5Executor(cfg)
    poll = poll_seconds or cfg["execution"].get("poll_seconds", 30)
    symbol = cfg["instrument"]["symbol"]

    log.info("Starting %s loop for %s, polling every %ss",
              "LIVE" if executor.is_live else "PAPER", symbol, poll)

    while True:
        try:
            df = data_fetch_fn()
            plan = build_signal_fn(df)
            record = executor.execute(plan, symbol)
            log.info(record)
        except Exception:
            log.exception("Error in execution loop iteration")
        time.sleep(poll)
