"""
Proves the entire pipeline (data cleaning -> forecast -> features -> regime
filters -> signal scoring -> risk sizing -> cost-aware backtest -> metrics)
runs correctly end to end, using synthetic OHLCV and MockForecaster instead
of real Kronos weights. No internet, no GPU, no MT5 needed.

Run: python run_demo.py
"""
import logging
import numpy as np
import pandas as pd
import yaml

from data.loader import clean
from kronos_engine.forecaster import MockForecaster
from backtest.engine import run_backtest
from backtest.metrics import summarize_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_demo")


def make_synthetic_ohlcv(n_bars: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_bars, freq="5min", tz="UTC")
    # regime-switching random walk so the trend filter has something to do
    drift = np.where((np.arange(n_bars) // 500) % 2 == 0, 0.00003, -0.00002)
    rets = rng.normal(drift, 0.0006, n_bars)
    close = 1.0850 * np.cumprod(1 + rets)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0002, n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0002, n_bars)))
    volume = np.abs(rng.normal(500, 100, n_bars))
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low,
                          "close": close, "volume": volume})


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # smaller/faster settings for a quick demo run
    cfg["kronos"]["lookback"] = 200
    cfg["kronos"]["pred_len"] = 12
    cfg["kronos"]["ensemble_size"] = 15
    cfg["regime"]["session_filter"]["enabled"] = False  # synthetic ts has no real sessions

    df = make_synthetic_ohlcv(6000)
    df, report = clean(df, timeframe_minutes=5)
    log.info("Synthetic data ready: %s", report)

    forecaster = MockForecaster(drift=0.00001, vol=0.0006, seed=123)

    result = run_backtest(df, forecaster, cfg, use_ensemble=True,
                           signal_every_n_bars=6, verbose=True)

    stats = summarize_trades(result.trades, result.equity_curve) if not result.trades.empty else {"n_trades": 0}
    print("\n=== DEMO backtest summary (synthetic data + MockForecaster) ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print(f"\nsignals evaluated: {len(result.signals_log)}")
    if len(result.signals_log):
        print(result.signals_log["decision"].value_counts())

    print("\nPipeline ran end-to-end successfully. Swap MockForecaster for "
          "KronosForecaster (see README) once you have Kronos weights + real data.")


if __name__ == "__main__":
    main()
