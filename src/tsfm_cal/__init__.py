"""tsfm_cal — distributional calibration & tail-quantile extraction from
TimesFM 2.5 on financial returns.

Importing the package pulls in only the light modules. ``config``, ``io_utils``,
``eval``, ``plot``, and the PIT math in ``pit`` are CPU-importable with no torch /
timesfm; ``data`` imports pandas/yfinance lazily inside its functions; only the
model-forward path in ``pit`` needs the (Kaggle-side) model object.
"""

from . import config, io_utils  # noqa: F401  (always-light)

__all__ = ["config", "io_utils", "data", "pit", "eval", "plot"]
__version__ = "0.1.0"
