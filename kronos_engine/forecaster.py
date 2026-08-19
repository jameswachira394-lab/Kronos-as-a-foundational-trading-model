"""
Wraps the official Kronos model. Two classes:

  KronosForecaster  - loads real NeoQuasar/Kronos-base + Kronos-Tokenizer-base
                       from Hugging Face Hub via the vendored `vendor/Kronos`
                       repo's `model` package. Needs torch + internet the
                       first time (weights are cached after that).

  MockForecaster    - drop-in replacement with the identical interface,
                       generates a plausible random-walk-with-drift forecast.
                       Used by run_demo.py so you can validate the entire
                       pipeline (features, filters, signals, risk, backtest)
                       without downloading anything.

Both expose:
    .forecast(df, lookback_timestamps, pred_len) -> single pred_df
    .forecast_ensemble(df, lookback_timestamps, pred_len, n) -> list[pred_df]

`forecast_ensemble` is what you must use if you want an actual forecast
*distribution* (bullish/bearish path counts, path consistency, etc.) — see
the README section on why predict()'s own `sample_count` does NOT give you
that (it averages the samples internally before returning).
"""
from __future__ import annotations
import sys
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class BaseForecaster:
    price_cols = ["open", "high", "low", "close"]

    def forecast(self, df: pd.DataFrame, x_timestamp: pd.Series,
                 y_timestamp: pd.Series, pred_len: int) -> pd.DataFrame:
        raise NotImplementedError

    def forecast_ensemble(self, df: pd.DataFrame, x_timestamp: pd.Series,
                           y_timestamp: pd.Series, pred_len: int,
                           n: int = 20, T: float = 1.0, top_p: float = 0.9,
                           top_k: int = 0) -> list[pd.DataFrame]:
        """Call forecast() n independent times (sample_count=1 each) to build
        a real ensemble of divergent paths, rather than relying on the
        library's internal sample averaging."""
        paths = []
        for i in range(n):
            paths.append(self.forecast(df, x_timestamp, y_timestamp, pred_len))
        return paths


class KronosForecaster(BaseForecaster):
    def __init__(self, tokenizer_repo: str = "NeoQuasar/Kronos-Tokenizer-base",
                 model_repo: str = "NeoQuasar/Kronos-base",
                 max_context: int = 512, device: Optional[str] = None,
                 vendor_path: str = "vendor/Kronos", T: float = 1.0,
                 top_p: float = 0.9, top_k: int = 0):
        if vendor_path not in sys.path and os.path.isdir(vendor_path):
            sys.path.append(vendor_path)
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor  # noqa
        except ImportError as e:
            raise RuntimeError(
                "Could not import Kronos. Clone it next to this project:\n"
                "  git clone https://github.com/shiyu-coder/Kronos.git vendor/Kronos\n"
                "and pip install -r vendor/Kronos/requirements.txt"
            ) from e

        log.info("Loading tokenizer %s and model %s ...", tokenizer_repo, model_repo)
        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_repo)
        self.model = Kronos.from_pretrained(model_repo)
        self.predictor = KronosPredictor(self.model, self.tokenizer,
                                          device=device, max_context=max_context)
        self.default_T = T
        self.default_top_p = top_p
        self.default_top_k = top_k

    def forecast(self, df: pd.DataFrame, x_timestamp: pd.Series,
                 y_timestamp: pd.Series, pred_len: int,
                 T: Optional[float] = None, top_p: Optional[float] = None,
                 top_k: Optional[int] = None, sample_count: int = 1) -> pd.DataFrame:
        return self.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T if T is not None else self.default_T,
            top_p=top_p if top_p is not None else self.default_top_p,
            top_k=top_k if top_k is not None else self.default_top_k,
            sample_count=sample_count,
            verbose=False,
        )

    def forecast_ensemble(self, df, x_timestamp, y_timestamp, pred_len,
                           n: int = 20, T: float = 1.0, top_p: float = 0.9,
                           top_k: int = 0) -> list[pd.DataFrame]:
        paths = []
        for i in range(n):
            # small temperature jitter so paths genuinely diverge rather than
            # relying only on categorical sampling noise
            t_i = T * (1.0 + np.random.uniform(-0.05, 0.05))
            paths.append(self.forecast(df, x_timestamp, y_timestamp, pred_len,
                                        T=t_i, top_p=top_p, top_k=top_k,
                                        sample_count=1))
        return paths


class MockForecaster(BaseForecaster):
    """Random-walk-with-drift stand-in. Same call signature as KronosForecaster
    so the rest of the pipeline is testable offline. Do NOT use this for
    anything but plumbing validation — it has zero real predictive value."""

    def __init__(self, drift: float = 0.0, vol: float = 0.0006, seed: Optional[int] = None):
        self.drift = drift
        self.vol = vol
        self.rng = np.random.default_rng(seed)

    def forecast(self, df: pd.DataFrame, x_timestamp: pd.Series,
                 y_timestamp: pd.Series, pred_len: int, **kwargs) -> pd.DataFrame:
        last_close = df["close"].iloc[-1]
        steps = self.rng.normal(self.drift, self.vol, size=pred_len)
        closes = last_close * np.cumprod(1 + steps)
        opens = np.concatenate([[last_close], closes[:-1]])
        highs = np.maximum(opens, closes) * (1 + np.abs(self.rng.normal(0, self.vol / 2, pred_len)))
        lows = np.minimum(opens, closes) * (1 - np.abs(self.rng.normal(0, self.vol / 2, pred_len)))
        vols = np.abs(self.rng.normal(df["volume"].mean() if "volume" in df else 100, 10, pred_len))
        pred_df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": vols, "amount": vols * closes,
        }, index=y_timestamp)
        return pred_df
