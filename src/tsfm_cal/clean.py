"""Data cleaning & engineering for financial returns (CPU, pure pandas/numpy).

Turns raw source data (CRSP daily total return, WRDS FX levels, crypto levels)
into canonical cleaned per-asset returns + a data-quality report. Design rules
(see plan Part A):

  * **CRSP ``ret`` is the primitive** for equities/ETFs — it is the holding-period
    return already adjusted for splits, dividends, and delisting. Convert to log
    via ``log(1 + ret)``. Do NOT recompute from price.
  * FX / crypto arrive as price levels -> ``diff(log(level))``.
  * **Never forward-fill non-trading days.** Stale-fill is the root cause of the
    yfinance EURUSD kurtosis≈105 artifact. We keep each market's native calendar.
  * **Tail-preserving error screening.** Flag ``|log-ret| > THRESHOLD`` and price
    glitches for MANUAL inspection; correct/drop only *verified* errors. The fat
    tails are the object of study — no blind winsorizing.

The WRDS pull itself lives in ``wrds_data.py`` (local only). This module is
source-agnostic: it takes a tidy frame of (date, value, kind) and cleans it, so
it is fully unit-testable without WRDS or network.
"""

from __future__ import annotations

import numpy as np

from . import config


def returns_from_levels(levels: np.ndarray) -> np.ndarray:
    """Log returns from a price/level series: ``diff(log(level))``."""
    p = np.asarray(levels, dtype=float)
    return np.diff(np.log(p))


def returns_from_crsp_ret(ret: np.ndarray) -> np.ndarray:
    """Log returns from CRSP holding-period returns: ``log(1 + ret)``."""
    r = np.asarray(ret, dtype=float)
    return np.log1p(r)


def clean_asset(
    dates,
    values: np.ndarray,
    *,
    kind: str,
    source: str,
    threshold: float = config.EXTREME_LOGRET_THRESHOLD,
    aux: dict | None = None,
):
    """Clean one asset's raw series into canonical returns + diagnostics.

    Parameters
    ----------
    dates : array-like of datetime64 — one per row of ``values``.
    values : the raw series. If ``kind="crsp_ret"`` these are CRSP daily ``ret``
        (one return per date). If ``kind="level"`` these are price levels.
    kind : {"crsp_ret", "level"}.
    source : provenance string recorded in the report (e.g. "crsp", "wrds_fx").
    threshold : |log-ret| above which a point is FLAGGED (not dropped).
    aux : optional dict of extra columns for glitch screening, e.g.
        {"prc": array, "vol": array} (CRSP) — zero/negative price or zero volume
        on an extreme day is suspicious.

    Returns
    -------
    out : dict with
        dates, log_returns, simple_returns : aligned cleaned arrays
        flagged : list of {date, log_return, reason} for manual inspection
        report : one-row dict of quality diagnostics
    """
    dates = np.asarray(dates)
    values = np.asarray(values, dtype=float)
    if len(dates) != len(values):
        raise ValueError("dates and values length mismatch")

    # ---- 1. de-duplicate + enforce monotonic increasing dates -------------- #
    order = np.argsort(dates, kind="stable")
    dates, values = dates[order], values[order]
    keep = np.ones(len(dates), dtype=bool)
    keep[1:] = dates[1:] != dates[:-1]   # drop exact-duplicate dates (keep first)
    n_dup = int((~keep).sum())
    dates, values = dates[keep], values[keep]
    aux = {k: np.asarray(v)[order][keep] for k, v in (aux or {}).items()}

    # ---- 2. convert to returns (date-aligned) ------------------------------ #
    if kind == "crsp_ret":
        # values[i] is the return realized ON dates[i]; drop NaN ret rows first.
        finite = np.isfinite(values)
        n_nan = int((~finite).sum())
        dates_r, log_ret = dates[finite], returns_from_crsp_ret(values[finite])
        aux = {k: v[finite] for k, v in aux.items()}
    elif kind == "level":
        finite = np.isfinite(values) & (values > 0)
        n_nan = int((~finite).sum())
        dates_f, lv = dates[finite], values[finite]
        log_ret = returns_from_levels(lv)
        dates_r = dates_f[1:]            # a return is dated at its later endpoint
        aux = {k: v[finite][1:] for k, v in aux.items()}
    else:
        raise ValueError(f"kind must be 'crsp_ret' or 'level', got {kind!r}")

    # drop any non-finite returns produced (e.g. from a zero/negative level)
    rfin = np.isfinite(log_ret)
    dates_r, log_ret = dates_r[rfin], log_ret[rfin]
    aux = {k: v[rfin] for k, v in aux.items()}

    # ---- 3. tail-preserving error screening (FLAG, don't drop) ------------- #
    flagged = []
    extreme = np.abs(log_ret) > threshold
    prc = aux.get("prc")
    vol = aux.get("vol")
    for i in np.where(extreme)[0]:
        reason = f"|log-ret|={abs(log_ret[i]):.3f} > {threshold}"
        if prc is not None and not (np.isfinite(prc[i]) and prc[i] > 0):
            reason += "; bad prc"
        if vol is not None and np.isfinite(vol[i]) and vol[i] == 0:
            reason += "; zero vol"
        flagged.append(
            {"date": str(np.datetime_as_string(dates_r[i], unit="D")),
             "log_return": float(log_ret[i]), "reason": reason}
        )

    simple_ret = np.expm1(log_ret)  # back to simple for the robustness variant

    # ---- 4. quality report ------------------------------------------------- #
    from scipy import stats as _st

    report = {
        "asset": None,  # filled by the caller (asset key)
        "source": source,
        "n_obs": int(len(log_ret)),
        "date_start": str(np.datetime_as_string(dates_r[0], unit="D")) if len(dates_r) else "",
        "date_end": str(np.datetime_as_string(dates_r[-1], unit="D")) if len(dates_r) else "",
        "n_dup_dates": n_dup,
        "n_nan_dropped": n_nan,
        "n_flagged": len(flagged),
        "n_zero_ret": int(np.sum(log_ret == 0.0)),
        "excess_kurtosis": float(_st.kurtosis(log_ret, fisher=True)) if len(log_ret) > 3 else float("nan"),
        "min_log_ret": float(log_ret.min()) if len(log_ret) else float("nan"),
        "max_log_ret": float(log_ret.max()) if len(log_ret) else float("nan"),
        "ann_vol": float(log_ret.std() * np.sqrt(252)) if len(log_ret) else float("nan"),
    }

    return {
        "dates": dates_r,
        "log_returns": log_ret,
        "simple_returns": simple_ret,
        "flagged": flagged,
        "report": report,
    }
