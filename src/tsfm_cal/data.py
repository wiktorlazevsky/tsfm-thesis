"""yfinance download + caching, log (default) and simple returns.

Gotchas handled (HANDOFF §3.9):
  * ``auto_adjust=True`` set explicitly (adjusted close; silences FutureWarning).
  * yfinance returns come back shape ``(N, 1)`` -> always ``.flatten()``.
  * Short series (EUR/USD ~2003, BTC ~2014) handled gracefully by the caller.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def _cache_path(ticker: str, start: str, end: str) -> Path:
    from .io_utils import base_dir, _safe

    cache = base_dir().parent / "data_cache"
    cache.mkdir(parents=True, exist_ok=True)
    # npz cache (numpy only — no parquet/pyarrow dependency).
    return cache / f"{_safe(ticker)}_{start}_{end}.npz"


def download_prices(ticker: str, start: str, end: str, use_cache: bool = True):
    """Download adjusted close prices as a pandas Series, cached to npz."""
    import pandas as pd

    path = _cache_path(ticker, start, end)
    if use_cache and path.exists():
        z = np.load(path, allow_pickle=False)
        return pd.Series(z["close"], index=pd.to_datetime(z["dates"]), name="close")

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} in [{start}, {end}].")
    close = df["Close"]
    # yfinance may return a (N,1) DataFrame for "Close" with a ticker column.
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()
    if use_cache:
        np.savez_compressed(
            path,
            close=close.values.astype(float),
            dates=np.asarray(close.index, dtype="datetime64[ns]").astype("int64"),
        )
    return pd.Series(close.values, index=close.index, name="close")


def to_returns(prices, kind: str = "log") -> np.ndarray:
    """Convert a price Series to a 1-D returns array (flattened)."""
    import numpy as np

    p = np.asarray(prices, dtype=float).flatten()
    if kind == "log":
        rets = np.diff(np.log(p))
    elif kind == "simple":
        rets = np.diff(p) / p[:-1]
    else:
        raise ValueError(f"kind must be 'log' or 'simple', got {kind!r}")
    return rets[np.isfinite(rets)]


def download_returns(
    ticker: str,
    start: str = config.START,
    end: str = config.END,
    kind: str = config.RETURNS_TYPE,
    use_cache: bool = True,
):
    """Return ``(dates, returns)`` for one ticker.

    ``dates`` aligns with ``returns`` (the date of each realized return, i.e.
    the later endpoint of each diff). ``returns`` is a flat float64 array.
    """
    prices = download_prices(ticker, start, end, use_cache=use_cache)
    rets = to_returns(prices, kind=kind)
    # Dates: returns[i] corresponds to prices.index[i+1].
    dates = np.asarray(prices.index[1:], dtype="datetime64[ns]")
    # to_returns may have dropped non-finite entries; realign by recomputing mask.
    p = np.asarray(prices, dtype=float).flatten()
    if kind == "log":
        raw = np.diff(np.log(p))
    else:
        raw = np.diff(p) / p[:-1]
    mask = np.isfinite(raw)
    return dates[mask], rets
