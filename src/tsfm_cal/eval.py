"""Calibration & tail-risk diagnostics, all computed from saved arrays.

Phase 1 uses the distributional + coverage + regime block. The VaR/ES backtest
block (Kupiec, Christoffersen, FZ loss, ES + bootstrap CI) is implemented now so
Phase 2 baselines and Phase 3 heads plug straight in (HANDOFF §9). Everything
here is pure numpy/scipy — no model, CPU-testable.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import config

_LEVELS = np.asarray(config.NATIVE_QUANTILES, dtype=float)  # [0.1..0.9]


# ===========================================================================
# Distributional calibration (PIT-based)
# ===========================================================================
def ks_uniform(pit: np.ndarray) -> tuple[float, float]:
    """KS statistic + p-value against U(0,1)."""
    pit = np.asarray(pit, dtype=float)
    res = stats.kstest(pit, "uniform")
    return float(res.statistic), float(res.pvalue)


def pit_moments(pit: np.ndarray) -> dict:
    """PIT mean (target 0.5) and variance (target 1/12 ≈ 0.0833)."""
    pit = np.asarray(pit, dtype=float)
    return {"pit_mean": float(pit.mean()), "pit_var": float(pit.var())}


def central_coverage(pit: np.ndarray, level: float = 0.80) -> float:
    """Empirical coverage of the central ``level`` interval.

    For 80%: fraction of PIT in [0.1, 0.9] (target 0.80).
    """
    pit = np.asarray(pit, dtype=float)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return float(np.mean((pit >= lo) & (pit <= hi)))


def tail_mass(pit: np.ndarray, edge: float = 0.1) -> dict:
    """Left-tail mass (PIT < edge) and right-tail mass (PIT > 1-edge).

    Both target ``edge`` (0.10). The left/right asymmetry is the signal that
    decides symmetric vs skew head later (HANDOFF §5) — recorded explicitly.
    """
    pit = np.asarray(pit, dtype=float)
    left = float(np.mean(pit < edge))
    right = float(np.mean(pit > 1 - edge))
    return {"tail_left": left, "tail_right": right, "tail_asym": right - left}


def chi2_gof(pit: np.ndarray, bins: int = 10) -> tuple[float, float]:
    """10-bin chi-square goodness-of-fit vs uniform."""
    pit = np.asarray(pit, dtype=float)
    counts, _ = np.histogram(pit, bins=bins, range=(0, 1))
    expected = np.full(bins, len(pit) / bins)
    stat = float(np.sum((counts - expected) ** 2 / expected))
    p = float(stats.chi2.sf(stat, df=bins - 1))
    return stat, p


def conditional_ks(
    pit: np.ndarray,
    realized: np.ndarray,
    n_strata: int = 5,
    vol_window: int = 21,
) -> list[dict]:
    """KS per realized-volatility stratum (regime-conditional calibration).

    Aggregate KS can mask regime-specific collapse (HANDOFF §9, Adler traffic).
    Strata are quantiles of trailing realized vol (rolling std of ``realized``).
    Reuses saved arrays only — no model runs.
    """
    pit = np.asarray(pit, dtype=float)
    realized = np.asarray(realized, dtype=float)
    vol = _rolling_std(realized, vol_window)
    edges = np.nanquantile(vol, np.linspace(0, 1, n_strata + 1))
    out = []
    for k in range(n_strata):
        lo, hi = edges[k], edges[k + 1]
        sel = (vol >= lo) & (vol <= hi if k == n_strata - 1 else vol < hi)
        sel &= np.isfinite(vol)
        if sel.sum() < 30:
            out.append({"stratum": k, "n": int(sel.sum()), "ks": np.nan, "p": np.nan})
            continue
        ks, p = ks_uniform(pit[sel])
        out.append(
            {
                "stratum": k,
                "n": int(sel.sum()),
                "vol_lo": float(lo),
                "vol_hi": float(hi),
                "ks": ks,
                "p": p,
            }
        )
    return out


# ===========================================================================
# Tail risk: VaR / ES backtesting (Phase 2/3 ready)
# ===========================================================================
def var_exceedances(realized: np.ndarray, var: np.ndarray, side: str = "left") -> np.ndarray:
    """Boolean exceedance series of realized vs a VaR forecast.

    ``side="left"`` (loss tail): exceedance when ``realized < var`` (var is the
    low quantile, e.g. q05). ``side="right"``: ``realized > var``.
    """
    realized = np.asarray(realized, dtype=float)
    var = np.asarray(var, dtype=float)
    if side == "left":
        return realized < var
    return realized > var


def kupiec_pof(exceed: np.ndarray, alpha: float) -> dict:
    """Kupiec proportional-of-failures unconditional-coverage test."""
    exceed = np.asarray(exceed, dtype=bool)
    n = exceed.size
    x = int(exceed.sum())
    pi = x / n if n else np.nan
    # LR_uc = -2 ln[ (1-α)^(n-x) α^x / ((1-π)^(n-x) π^x) ]
    if x == 0 or x == n:
        lr = np.nan
    else:
        ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
        ll_alt = (n - x) * np.log(1 - pi) + x * np.log(pi)
        lr = -2 * (ll_null - ll_alt)
    p = float(stats.chi2.sf(lr, df=1)) if np.isfinite(lr) else np.nan
    return {"n": n, "exceed": x, "rate": float(pi), "expected": alpha, "lr_uc": _f(lr), "p_uc": p}


def christoffersen(exceed: np.ndarray, alpha: float) -> dict:
    """Christoffersen independence + conditional-coverage tests."""
    e = np.asarray(exceed, dtype=int)
    # Transition counts n_ij (i=prev state, j=curr state).
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(e[:-1], e[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if e.size > 1 else np.nan

    # LR_ind
    if pi in (0.0, 1.0) or pi01 in (0.0, 1.0) or pi11 in (0.0, 1.0):
        lr_ind = np.nan
    else:
        ll_ind = (
            (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        )
        ll_markov = (
            n00 * np.log(1 - pi01) + n01 * np.log(pi01)
            + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
        )
        lr_ind = -2 * (ll_ind - ll_markov)

    uc = kupiec_pof(exceed, alpha)
    lr_cc = (uc["lr_uc"] + lr_ind) if (np.isfinite(uc["lr_uc"] or np.nan) and np.isfinite(lr_ind)) else np.nan
    return {
        "lr_ind": _f(lr_ind),
        "p_ind": float(stats.chi2.sf(lr_ind, df=1)) if np.isfinite(lr_ind) else np.nan,
        "lr_cc": _f(lr_cc),
        "p_cc": float(stats.chi2.sf(lr_cc, df=2)) if np.isfinite(lr_cc) else np.nan,
    }


def expected_shortfall(realized: np.ndarray, var: np.ndarray, side: str = "left") -> float:
    """Realized ES: mean of returns beyond the VaR forecast (exceedances only)."""
    exceed = var_exceedances(realized, var, side)
    r = np.asarray(realized, dtype=float)[exceed]
    return float(r.mean()) if r.size else np.nan


def es_bootstrap_ci(
    realized: np.ndarray,
    var: np.ndarray,
    side: str = "left",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for realized ES. ES is precision-fragile in finite tail
    samples (HANDOFF §9, ES-precision preprint) — always report with bands."""
    rng = np.random.default_rng(seed)
    exceed = var_exceedances(realized, var, side)
    r = np.asarray(realized, dtype=float)[exceed]
    if r.size < 2:
        return {"es": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n_tail": int(r.size)}
    boots = np.array([rng.choice(r, size=r.size, replace=True).mean() for _ in range(n_boot)])
    a = (1 - ci) / 2
    return {
        "es": float(r.mean()),
        "ci_lo": float(np.quantile(boots, a)),
        "ci_hi": float(np.quantile(boots, 1 - a)),
        "n_tail": int(r.size),
    }


def fz_loss(realized: np.ndarray, var: np.ndarray, es: np.ndarray | float, alpha: float) -> float:
    """Fissler–Ziegel joint VaR–ES loss (the 0-homogeneous AS2014 form).

    Lower is better. ``var``/``es`` may be scalars or per-step arrays (loss tail,
    negative quantiles). Uses the standard FZ0 scoring function.
    """
    r = np.asarray(realized, dtype=float)
    v = np.broadcast_to(np.asarray(var, dtype=float), r.shape)
    e = np.broadcast_to(np.asarray(es, dtype=float), r.shape)
    hit = (r <= v).astype(float)
    # FZ0: (1{r<=v} - α)·v/(α·e) - r·1{r<=v}/(α·e) + v/e + ln(-e) ... use AS form:
    term = (hit - alpha) * v - hit * r
    loss = term / (alpha * (-e)) + np.log(-e)
    return float(np.mean(loss))


def pinball_loss(realized: np.ndarray, quantiles: np.ndarray, levels=_LEVELS) -> float:
    """Mean pinball loss over the provided quantile levels."""
    r = np.asarray(realized, dtype=float)[:, None]
    q = np.asarray(quantiles, dtype=float)
    lv = np.asarray(levels, dtype=float)[None, :]
    diff = r - q
    return float(np.mean(np.maximum(lv * diff, (lv - 1) * diff)))


def crps_from_quantiles(realized: np.ndarray, quantiles: np.ndarray, levels=_LEVELS) -> float:
    """Quantile-approximation to CRPS = 2 × mean pinball over a τ-grid."""
    return 2.0 * pinball_loss(realized, quantiles, levels)


def quantile_crossing_rate(quantiles: np.ndarray) -> float:
    """Fraction of steps where the quantile vector is not non-decreasing."""
    q = np.asarray(quantiles, dtype=float)
    bad = np.any(np.diff(q, axis=1) < -1e-12, axis=1)
    return float(np.mean(bad))


# ===========================================================================
# Head-based tail risk: combined VaR/ES backtest + per-τ reliability
# (pure numpy/scipy — heads produce the var/es/quantile arrays; see heads.py)
# ===========================================================================
def backtest_var_es(
    realized: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    alpha: float,
    side: str = "left",
) -> dict:
    """Full VaR/ES backtest at one level: Kupiec + Christoffersen + ES(±CI) + FZ.

    ``var``/``es`` are per-step forecast series (denormalized) at level ``alpha``.
    For the loss tail use ``side="left"`` with ``alpha`` small (e.g. 0.01, 0.05).
    """
    exceed = var_exceedances(realized, var, side)
    row = {"alpha": alpha, "side": side}
    row.update(kupiec_pof(exceed, alpha if side == "left" else 1 - alpha))
    row.update(christoffersen(exceed, alpha if side == "left" else 1 - alpha))
    es_emp = expected_shortfall(realized, var, side)
    row["es_realized"] = es_emp
    row["es_forecast"] = float(np.mean(es)) if np.ndim(es) else float(es)
    ci = es_bootstrap_ci(realized, var, side)
    row["es_ci_lo"], row["es_ci_hi"], row["n_tail"] = ci["ci_lo"], ci["ci_hi"], ci["n_tail"]
    row["fz_loss"] = fz_loss(realized, var, es, alpha if side == "left" else 1 - alpha)
    return row


def quantile_reliability(realized: np.ndarray, quantiles: np.ndarray, taus) -> list[dict]:
    """Per-τ empirical coverage vs nominal: ``P(realized ≤ Q_τ)`` should equal τ."""
    realized = np.asarray(realized, dtype=float)
    Q = np.asarray(quantiles, dtype=float)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    out = []
    for k, tau in enumerate(taus):
        emp = float(np.mean(realized <= Q[:, k]))
        out.append({"tau": float(tau), "nominal": float(tau), "empirical": emp,
                    "gap": emp - float(tau), "n": realized.size})
    return out


# ===========================================================================
# One-row-per-asset summary
# ===========================================================================
def summarize_asset(
    name: str,
    pit: np.ndarray,
    realized: np.ndarray,
    quantiles: np.ndarray,
    pit_interp: np.ndarray | None = None,
) -> dict:
    """Build the tidy per-asset diagnostic row for pit_summary.csv."""
    ks, p = ks_uniform(pit)
    row = {"asset": name, "n": int(np.asarray(pit).size), "ks": ks, "ks_p": p}
    row.update(pit_moments(pit))
    row["cov80"] = central_coverage(pit, 0.80)
    row.update(tail_mass(pit, 0.1))
    chi, chip = chi2_gof(pit, 10)
    row["chi2"] = chi
    row["chi2_p"] = chip
    row["excess_kurtosis"] = float(stats.kurtosis(np.asarray(realized, dtype=float), fisher=True))
    row["crossing_rate"] = quantile_crossing_rate(quantiles)
    if pit_interp is not None:
        ki, pi_ = ks_uniform(pit_interp)
        row["ks_interp"] = ki
        row["ks_interp_p"] = pi_
    return row


def summarize_run(results_or_dir, from_disk: bool = False) -> list[dict]:
    """Summarize every asset, returning a list of rows (assets x diagnostics).

    ``from_disk=True``: ``results_or_dir`` is a run_dir; load npz per asset.
    Else: ``results_or_dir`` is the in-memory ``{name: result_dict}`` map.
    """
    rows = []
    if from_disk:
        from . import io_utils

        run_dir = results_or_dir
        for asset in io_utils.list_pit_assets(run_dir):
            d = io_utils.load_pit_npz(run_dir, asset)
            rows.append(
                summarize_asset(
                    asset, d["pit"], d["realized"], d["quantiles"],
                    d.get("pit_interp"),
                )
            )
    else:
        for name, d in results_or_dir.items():
            rows.append(
                summarize_asset(
                    name, d["pit"], d["realized"], d["quantiles"],
                    d.get("pit_interp"),
                )
            )
    return rows


# --------------------------------------------------------------------------- #
def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    for i in range(window, len(x)):
        out[i] = x[i - window:i].std()
    return out


def _f(v):
    return float(v) if v is not None and np.isfinite(v) else np.nan
