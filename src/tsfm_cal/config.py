"""Central run configuration for the zero-shot PIT pipeline.

This module is deliberately import-light: it pulls in *no* torch / timesfm /
pandas so it stays importable on the CPU-only local Mac. The TimesFM
``ForecastConfig`` is built in the driver notebook from ``FORECAST_FLAGS`` here,
so every flag is explicit and recorded — nothing is silently inherited from the
library defaults (the ``force_flip_invariance=True`` confound, see HANDOFF §3.4).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Asset universe (the 7 zero-shot assets). SPY first so it leads the figures.  #
# --------------------------------------------------------------------------- #
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
