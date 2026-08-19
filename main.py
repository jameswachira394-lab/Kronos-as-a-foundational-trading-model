"""
CLI entry point.

    python main.py research        # forecast-only: is Kronos even useful here?
    python main.py backtest        # full signal+risk+cost simulation
    python main.py walk-forward    # rolling out-of-sample windows
    python main.py paper           # live MT5 data, log-only, no orders
    python main.py live            # live MT5 data + real orders (guarded)
"""
from __future__ import annotations
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import yaml

from data.loader import load_csv, load_mt5, clean, to_kronos_frame
from kronos_engine.forecaster import KronosForecaster
from features.forecast_features import extract_ensemble
from regime.filters import compute_regime
from signals.engine import score_signal
from risk.risk_engine import RiskEngine
from backtest.engine import run_backtest
from backtest.metrics import summarize_trades
from backtest.walk_forward import run_walk_forward
from execution.mt5_connector import run_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")

TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def load_config(path: str = "config.yaml") -> dict:
    # Resolve relative to this file's directory so the system works
    # regardless of the working directory the user invokes it from.
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    inst = cfg["instrument"]
    if cfg["data"]["source"] == "csv":
        df = load_csv(cfg["data"]["csv_path"])
    else:
        m = cfg["data"]["mt5"]
        df = load_mt5(inst["symbol"], inst["timeframe"], m["n_bars"], m["login"], m["password"], m["server"])
    df, report = clean(df, TIMEFRAME_MINUTES[inst["timeframe"]])
    log.info("Loaded %d bars for %s %s. %s", len(df), inst["symbol"], inst["timeframe"], report)
    return df


def build_forecaster(cfg: dict) -> KronosForecaster:
    k = cfg["kronos"]
    return KronosForecaster(
        tokenizer_repo=k["tokenizer_repo"],
        model_repo=k["model_repo"],
        max_context=k["max_context"],
        device=k["device"],
        vendor_path=k.get("vendor_path", "vendor/Kronos"),
        T=k.get("T", 1.0),
        top_p=k.get("top_p", 0.9),
        top_k=k.get("top_k", 0),
    )


def cmd_research(cfg: dict):
    """Phase 2: does Kronos have any predictive value here, before building
    anything else? Rolls forward through history, forecasts, compares to
    what actually happened, reports RMSE/MAE/hit-rate per horizon step."""
    df = load_data(cfg)
    forecaster = build_forecaster(cfg)
    k = cfg["kronos"]
    lookback, pred_len = k["lookback"], k["pred_len"]

    errors = []
    n_checks = 30
    stride = max(1, (len(df) - lookback - pred_len) // n_checks)
    for i in range(lookback, len(df) - pred_len, stride):
        hist = df.iloc[i - lookback:i]
        future = df.iloc[i:i + pred_len]
        x_df = to_kronos_frame(hist)
        pred_df = forecaster.forecast(x_df, hist["timestamp"], future["timestamp"], pred_len)
        actual = future["close"].values
        predicted = pred_df["close"].values
        err = predicted - actual
        hit = np.sign(predicted[-1] - hist["close"].iloc[-1]) == np.sign(actual[-1] - hist["close"].iloc[-1])
        errors.append({"rmse": float(np.sqrt(np.mean(err ** 2))),
                        "mae": float(np.mean(np.abs(err))),
                        "hit": bool(hit)})

    edf = pd.DataFrame(errors)
    print("\n=== Research summary (forecast-only, no trading) ===")
    print(f"n_checks={len(edf)}  mean_rmse={edf.rmse.mean():.6f}  mean_mae={edf.mae.mean():.6f}  "
          f"directional_hit_rate={edf.hit.mean():.2%}")
    print("A hit rate meaningfully above 50% (with tight error bars, ideally checked across "
          "many more windows) is the minimum bar before building signal/risk/execution around this.")


def cmd_backtest(cfg: dict, mock: bool = False):
    df = load_data(cfg)
    forecaster = _forecaster_or_mock(cfg, mock)
    result = run_backtest(df, forecaster, cfg, use_ensemble=True,
                           signal_every_n_bars=cfg["kronos"]["pred_len"] // 2, verbose=True)
    stats = summarize_trades(result.trades, result.equity_curve) if not result.trades.empty else {"n_trades": 0}
    print("\n=== Backtest summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    result.trades.to_csv("backtest_trades.csv", index=False)
    result.signals_log.to_csv("backtest_signals.csv", index=False)
    log.info("Wrote backtest_trades.csv and backtest_signals.csv")


def cmd_walk_forward(cfg: dict, mock: bool = False):
    df = load_data(cfg)
    forecaster = _forecaster_or_mock(cfg, mock)
    wf = run_walk_forward(df, forecaster, cfg, n_windows=cfg["backtest"]["walk_forward_windows"])
    print("\n=== Walk-forward summary (per out-of-sample window) ===")
    print(wf.to_string(index=False))
    wf.to_csv("walk_forward_results.csv", index=False)


def cmd_paper_or_live(cfg: dict, mode: str):
    cfg["execution"]["mode"] = mode
    if mode == "live" and not cfg["execution"].get("live"):
        log.error("Refusing to run live: set execution.live: true in config.yaml first, "
                   "after paper/backtest results justify it.")
        sys.exit(1)

    forecaster = build_forecaster(cfg)
    inst = cfg["instrument"]

    def data_fetch_fn():
        m = cfg["data"]["mt5"]
        df = load_mt5(inst["symbol"], inst["timeframe"], cfg["kronos"]["lookback"] + 10,
                       m["login"], m["password"], m["server"])
        df, _ = clean(df, TIMEFRAME_MINUTES[inst["timeframe"]])
        return df

    def build_signal_fn(df: pd.DataFrame):
        k = cfg["kronos"]
        lookback, pred_len = k["lookback"], k["pred_len"]
        hist = df.iloc[-lookback:]
        future_ts = pd.date_range(hist["timestamp"].iloc[-1], periods=pred_len + 1,
                                   freq=f"{TIMEFRAME_MINUTES[inst['timeframe']]}min")[1:]
        x_df = to_kronos_frame(hist)
        paths = forecaster.forecast_ensemble(x_df, hist["timestamp"], pd.Series(future_ts),
                                              pred_len, n=k["ensemble_size"], T=k["T"], top_p=k["top_p"])
        current_price = hist["close"].iloc[-1]
        feat = extract_ensemble(paths, current_price)
        regime = compute_regime(hist, cfg["regime"], cfg["costs"]["spread_pips"],
                                 hist["timestamp"].iloc[-1].hour)
        sig = score_signal(feat, regime, cfg["signal"], cfg["costs"],
                            inst["contract_size"], inst["pip_size"])
        risk_engine = RiskEngine(cfg["risk"], inst["pip_size"], inst["contract_size"], inst["point_value"])
        return risk_engine.plan_trade(sig, current_price, regime.atr_value)

    run_loop(cfg, forecaster, data_fetch_fn, build_signal_fn)


def _forecaster_or_mock(cfg: dict, mock: bool):
    if mock:
        from kronos_engine.forecaster import MockForecaster
        return MockForecaster(seed=42)
    return build_forecaster(cfg)


def main():
    parser = argparse.ArgumentParser(description="Kronos trading system")
    parser.add_argument("command", choices=["research", "backtest", "walk-forward", "paper", "live"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mock", action="store_true",
                         help="use MockForecaster instead of real Kronos (offline dev/testing)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "research":
        cmd_research(cfg)
    elif args.command == "backtest":
        cmd_backtest(cfg, mock=args.mock)
    elif args.command == "walk-forward":
        cmd_walk_forward(cfg, mock=args.mock)
    elif args.command in ("paper", "live"):
        cmd_paper_or_live(cfg, args.command)


if __name__ == "__main__":
    main()
