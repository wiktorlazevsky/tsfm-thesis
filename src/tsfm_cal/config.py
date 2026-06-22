"""Central run configuration for the zero-shot PIT pipeline.

This module is deliberately import-light: it pulls in *no* torch / timesfm /
pandas so it stays importable on the CPU-only local Mac. The TimesFM
``ForecastConfig`` is built in the driver notebook from ``FORECAST_FLAGS`` here,
so every flag is explicit and recorded — nothing is silently inherited from the
library defaults (the ``force_flip_invariance=True`` confound, see HANDOFF §3.4).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Asset universe (the 7 assets). SPY first so it leads the figures.            #
# --------------------------------------------------------------------------- #
# Canonical asset KEYS (SPY, TLT, ...) drive filenames + the clean-data flow.
# Each maps to a display label, a data SOURCE, and a source-specific id.
#   source "crsp"     -> CRSP daily total return (RET), pulled by ticker->PERMNO
#   source "wrds_fx"  -> WRDS FRB H.10 daily exchange rate
#   source "crypto"   -> documented non-WRDS source (CRSP/WRDS don't cover crypto)
ASSETS: dict[str, dict[str, str]] = {
    "SPY":    {"display": "Equities (SPY)",     "source": "crsp",    "source_id": "SPY"},
    "TLT":    {"display": "Gov Bonds (TLT)",    "source": "crsp",    "source_id": "TLT"},
    "GLD":    {"display": "Gold (GLD)",         "source": "crsp",    "source_id": "GLD"},
    "USO":    {"display": "Oil (USO)",          "source": "crsp",    "source_id": "USO"},
    "EEM":    {"display": "Emerging Mkts (EEM)", "source": "crsp",   "source_id": "EEM"},
    # FRB H.10 publishes USD-per-EUR daily (series id confirmed at pull time).
    "EURUSD": {"display": "FX EUR/USD",         "source": "wrds_fx", "source_id": "EUR"},
    # Crypto: documented clean source (yfinance BTC-USD fallback), flagged as the
    # one non-WRDS series in the data-quality report + thesis.
    "BTC":    {"display": "Bitcoin",            "source": "crypto",  "source_id": "BTC-USD"},
}
ASSET_KEYS: list[str] = list(ASSETS.keys())
ASSET_SOURCES: dict[str, str] = {k: v["source"] for k, v in ASSETS.items()}


def asset_display(key: str) -> str:
    """Human label for an asset key (falls back to the key itself)."""
    return ASSETS.get(key, {}).get("display", key)


# Legacy yfinance map (Phase-1 fallback + quick tests). Superseded by ASSETS +
# the clean-data flow for the canonical run.
TICKERS: dict[str, str] = {
    "SPY": "Equities (SPY)",
    "TLT": "Gov Bonds (TLT)",
    "GLD": "Gold (GLD)",
    "USO": "Oil (USO)",
    "EEM": "Emerging Mkts (EEM)",
    "EURUSD=X": "FX EUR/USD",
    "BTC-USD": "Bitcoin",
}

# --------------------------------------------------------------------------- #
# Data window & model context                                                 #
# --------------------------------------------------------------------------- #
START = "2000-01-01"
END = "2024-12-31"
CONTEXT = 512          # rolling context length fed to the model
HORIZON = 1            # 1-day-ahead calibration headline

# Returns convention. PROJECT STANDARD = log returns (HANDOFF §2/§8).
# "simple" kept only for a single documented robustness run.
RETURNS_TYPE = "log"   # {"log", "simple"}

# What the native model head actually emits: 9 inner deciles (channels 1..9).
# Channel 0 is the mean; forecast() point output is the MEDIAN (idx 5).
NATIVE_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Phase-3 EVALUATION grid (incl. the tail levels the native head can't emit).
# This is an eval grid only — the continuous heads are queried at ANY tau
# post-hoc; nothing here is baked into a head.
RISK_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# --------------------------------------------------------------------------- #
# Data cleaning / engineering (clean.py)                                      #
# --------------------------------------------------------------------------- #
# Flag |log-return| above this for MANUAL inspection (do NOT auto-winsorize —
# the fat tails are the object of study). 0.5 = a one-day 65% move.
EXTREME_LOGRET_THRESHOLD = 0.5

# Name of the Kaggle Dataset that holds the cleaned returns (mounted at
# /kaggle/input/<KAGGLE_CLEAN_DATASET>/ on Kaggle).
KAGGLE_CLEAN_DATASET = "tsfm-clean-data"

# --------------------------------------------------------------------------- #
# ForecastConfig flags — EXPLICIT, no silent defaults.                        #
# Built into timesfm.ForecastConfig(**FORECAST_FLAGS) in the notebook.        #
# --------------------------------------------------------------------------- #
# CANONICAL Phase-1 config: asymmetry preserved (flip invariance OFF).
FORECAST_FLAGS: dict[str, object] = {
    "max_context": CONTEXT,
    "max_horizon": HORIZON,
    "normalize_inputs": True,
    "use_continuous_quantile_head": True,
    "force_flip_invariance": False,   # CORRECTED — preserve asymmetric tails
    "infer_is_positive": False,       # explicit (harmless for returns)
    "fix_quantile_crossing": True,
}

# COMPARISON config: symmetrized forecast, to quantify how much the old default
# distorted the right-tail spike. Run on SPY alongside the canonical config.
FORECAST_FLAGS_FLIP: dict[str, object] = {
    **FORECAST_FLAGS,
    "force_flip_invariance": True,
}


def run_id(returns_type: str | None = None, flip: bool = False, stamp: str | None = None) -> str:
    """Build a deterministic, descriptive run id.

    Example: ``20260622-143000_log_flipFalse``. ``stamp`` lets the caller pin a
    timestamp (e.g. from datetime.now) so the id is stable within one run.
    """
    import datetime as _dt

    rt = returns_type or RETURNS_TYPE
    stamp = stamp or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{rt}_flip{flip}"
