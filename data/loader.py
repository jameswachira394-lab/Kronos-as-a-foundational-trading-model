"""
Data engine: pull OHLCV from CSV or MT5, validate, clean, resample.

Output contract used by the rest of the system:
    DataFrame indexed by nothing special, with columns:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount']
    sorted ascending by timestamp, no duplicates, no gaps silently ignored
    (gaps are reported, not filled, since filling fake candles corrupts
    Kronos's sequence model).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

import os

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close"]


@dataclass
class DataQualityReport:
    n_rows: int
    n_duplicates_removed: int
    n_missing_candles: int
    n_ohlc_violations_fixed: int
    start: pd.Timestamp
    end: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"rows={self.n_rows} dup_removed={self.n_duplicates_removed} "
            f"missing_candles={self.n_missing_candles} "
            f"ohlc_violations_fixed={self.n_ohlc_violations_fixed} "
            f"range=[{self.start} -> {self.end}]"
        )


def load_csv(path: str, tz: str | None = None) -> pd.DataFrame:
    # Resolve relative path against project root if file doesn't exist in current working directory
    if not os.path.isabs(path) and not os.path.exists(path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(project_root, path)
        if os.path.exists(alt_path):
            path = alt_path

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV file not found at '{path}'. Ensure the path is correct or run "
            "run_demo.py to generate sample data."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # be forgiving about common column name variants
    rename_map = {
        "time": "timestamp", "date": "timestamp", "datetime": "timestamp",
        "vol": "volume", "tick_volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=(tz is None))
    if tz:
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz, ambiguous="infer")
    return df


def load_mt5(symbol: str, timeframe: str, n_bars: int,
             login: int | str | None = None, password: str | None = None,
             server: str | None = None, path: str | None = None) -> pd.DataFrame:
    """
    Pull candles directly from a running MT5 terminal with broker credentials.
    Supports environment variables MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH.
    """
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise RuntimeError(
            "MetaTrader5 package not installed. Run `pip install MetaTrader5` "
            "(Windows/Wine only) or set `data.source: csv` in config.yaml."
        ) from e

    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if timeframe not in tf_map:
        raise ValueError(f"Unsupported timeframe {timeframe}")

    # Fallback to env vars if not passed
    login = login or os.getenv("MT5_LOGIN")
    password = password or os.getenv("MT5_PASSWORD")
    server = server or os.getenv("MT5_SERVER")
    path = path or os.getenv("MT5_PATH")

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
        err = mt5.last_error()
        raise RuntimeError(f"MT5 initialize() failed: {err}. Verify broker login details & server in MT5.")

    if login and password and server:
        if not mt5.login(login=int(login), password=str(password), server=str(server)):
            err = mt5.last_error()
            raise RuntimeError(f"MT5 broker login failed for account {login} on {server}: {err}")

    try:
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, n_bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 returned no rates for {symbol}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"tick_volume": "volume"})
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    finally:
        mt5.shutdown()


def clean(df: pd.DataFrame, timeframe_minutes: int) -> tuple[pd.DataFrame, DataQualityReport]:
    """Deduplicate, sort, fix OHLC inconsistencies, detect (but don't fabricate) gaps."""
    df = df.copy()
    n_before = len(df)

    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    n_dupes = n_before - len(df)

    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)

    # enforce high >= max(o,c,l) and low <= min(o,c,h)
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | \
          (df["low"] > df[["open", "close", "high"]].min(axis=1))
    n_bad = int(bad.sum())
    if n_bad:
        df.loc[bad, "high"] = df.loc[bad, ["open", "close", "high", "low"]].max(axis=1)
        df.loc[bad, "low"] = df.loc[bad, ["open", "close", "high", "low"]].min(axis=1)

    df = df.dropna(subset=["open", "high", "low", "close"])

    expected_step = pd.Timedelta(minutes=timeframe_minutes)
    deltas = df["timestamp"].diff().dropna()
    # count gaps larger than 1 step, ignoring weekend closes (>=48h) as expected
    gap_mask = (deltas > expected_step * 1.5) & (deltas < pd.Timedelta(hours=40))
    n_missing = int((gap_mask.sum()))

    report = DataQualityReport(
        n_rows=len(df),
        n_duplicates_removed=n_dupes,
        n_missing_candles=n_missing,
        n_ohlc_violations_fixed=n_bad,
        start=df["timestamp"].iloc[0] if len(df) else None,
        end=df["timestamp"].iloc[-1] if len(df) else None,
    )
    log.info("Data quality: %s", report)
    return df.reset_index(drop=True), report


def to_kronos_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select/order exactly the columns Kronos's predict() expects."""
    cols = ["open", "high", "low", "close", "volume", "amount"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols]
