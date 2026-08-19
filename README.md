# Kronos Trading System

A full research → backtest → paper → live pipeline built around
[Kronos](https://github.com/shiyu-coder/Kronos) (`NeoQuasar/Kronos-base`) as the
forecasting engine, with an independent signal, regime-filter, risk, and
execution layer around it — Kronos never places a trade by itself.

```
MT5 / CSV  ->  DataEngine  ->  KronosForecaster  ->  ForecastFeatures
   ->  RegimeFilters  ->  SignalEngine  ->  RiskEngine  ->  Backtester / MT5Executor
```

## ⚠️ One correction vs. a common misconception

A lot of write-ups (including the one you pasted) describe `sample_count` in
`KronosPredictor.predict()` as if it returns N distinguishable forecast paths
that you can bucket into "63 bullish / 24 neutral / 13 bearish". **That is not
what the public API does.** Looking at the actual source
(`model/kronos.py`, `auto_regressive_inference`):

```python
z = z.reshape(-1, sample_count, z.size(1), z.size(2))
preds = z.cpu().numpy()
preds = np.mean(preds, axis=1)   # <-- averaged over samples BEFORE returning
return preds
```

`sample_count` generates multiple stochastic rollouts internally and then
**averages them into a single smoother path**. `predict()` never hands you the
individual paths, so you cannot build a bullish/bearish probability histogram
from one call the way the doc implied.

To actually get a forecast *distribution* (which you need for steps 6–7 of the
architecture — "path consistency", "bullish probability", confidence-weighted
sizing), this project calls `predict()` **multiple independent times** with
`sample_count=1` and varying `T`/seed, and builds the ensemble ourselves in
`kronos_engine/forecaster.py::KronosForecaster.forecast_ensemble()`. That's the
correct way to get an actual set of divergent paths out of this model.

Everything else in the architecture you pasted is directionally reasonable —
this project implements it end to end, with working code instead of diagrams.

## Install

```bash
git clone https://github.com/shiyu-coder/Kronos.git vendor/Kronos
pip install -r requirements.txt
pip install -r vendor/Kronos/requirements.txt
```

The model weights (`NeoQuasar/Kronos-base`, `NeoQuasar/Kronos-Tokenizer-base`)
download from Hugging Face Hub the first time you run it — you need normal
internet access to `huggingface.co` for that (this sandbox doesn't have it,
so I could not execute a live download here; the pipeline itself is fully
implemented and tested with a `MockForecaster` — see below).

## Project layout

```
kronos_trading/
├── config.yaml                  # instrument, timeframe, thresholds, risk limits
├── data/
│   └── loader.py                 # CSV + MT5 ingestion, cleaning, resampling
├── kronos_engine/
│   └── forecaster.py             # KronosForecaster (real) + MockForecaster (offline dev/test)
├── features/
│   └── forecast_features.py      # turns a forecast/ensemble into return/vol/direction/consistency
├── regime/
│   └── filters.py                 # trend, volatility, session, spread, cost-vs-edge filters
├── signals/
│   └── engine.py                  # scoring model -> LONG / SHORT / NO_TRADE + confidence
├── risk/
│   └── risk_engine.py             # position sizing, ATR/forecast SL-TP, daily loss cap
├── backtest/
│   ├── engine.py                   # event-driven backtester with spread+commission+slippage
│   ├── metrics.py                  # Sharpe, Sortino, MDD, profit factor, expectancy, etc.
│   └── walk_forward.py             # walk-forward train/test split runner
├── execution/
│   └── mt5_connector.py           # paper + live MT5 execution (guarded, opt-in only)
├── main.py                         # CLI: research / backtest / walk-forward / paper / live
└── run_demo.py                     # runs the whole pipeline end-to-end on synthetic data,
                                     # no Kronos weights or internet required — proves the
                                     # plumbing works before you plug in the real model
```

## Recommended path (matches the phased plan)

1. `python run_demo.py` — sanity-check the whole pipeline with `MockForecaster`
   and synthetic OHLCV. No downloads, no MT5, no GPU needed.
2. Swap `MockForecaster` for `KronosForecaster` in `main.py`, point
   `config.yaml` at a real CSV (`data/loader.py` also reads MT5 directly if
   `MetaTrader5` is installed and a terminal is running).
3. `python main.py research` — forecast-only mode, plots forecast vs. actual,
   prints per-horizon RMSE/MAE/hit-rate so you can tell if Kronos has *any*
   edge on your instrument/timeframe before building anything else.
4. `python main.py backtest` — full signal+risk+cost simulation.
5. `python main.py walk-forward` — rolling train/validate/test windows.
6. `python main.py paper` — connects to live MT5 data, computes signals, logs
   what it *would* have done, sends zero orders.
7. Only after 3–6 look good: flip `execution.live: true` in `config.yaml` and
   run `python main.py live` with a tiny `risk_per_trade_pct`.

Fine-tuning Kronos on your own instrument (Phase 8) is intentionally last —
the finetune scripts already ship in the official repo under
`vendor/Kronos/finetune/`; wire them up only once the pretrained model has
demonstrated a real, cost-adjusted edge in step 3–5 above.
