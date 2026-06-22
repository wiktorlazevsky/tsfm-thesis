"""CPU unit tests for the PIT math + diagnostics + VaR backtests.

Run: ``pytest`` from the repo root (no model, no network).
"""

import numpy as np
import pytest

from tsfm_cal import eval as ev
from tsfm_cal import pit as P


# --------------------------------------------------------------------------- #
# PIT estimators
# --------------------------------------------------------------------------- #
def test_pit_discrete_buckets():
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert P.pit_discrete(q, -1.0) == 0.0          # below all deciles
    assert P.pit_discrete(q, 2.0) == 1.0           # above all deciles
    assert P.pit_discrete(q, 0.45) == pytest.approx(4 / 9)  # 4 deciles crossed


def test_pit_interp_monotone_and_clamped():
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert P.pit_interp(q, 0.5) == pytest.approx(0.5)
    # Clamped, not extrapolated, in the tails.
    assert P.pit_interp(q, -5.0) == pytest.approx(0.1)
    assert P.pit_interp(q, 5.0) == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# KS behaviour: uniform passes, spiked fails
# --------------------------------------------------------------------------- #
def test_ks_uniform_passes():
    rng = np.random.default_rng(0)
    pit = rng.uniform(0, 1, size=5000)
    ks, p = ev.ks_uniform(pit)
    assert ks < 0.05
    assert p > 0.05


def test_ks_spiked_fails():
    rng = np.random.default_rng(1)
    # Right-tail spike like the SPY finding: 30% of mass piled into [0.9, 1].
    body = rng.uniform(0, 0.9, size=3500)
    spike = rng.uniform(0.9, 1.0, size=1500)
    pit = np.concatenate([body, spike])
    ks, p = ev.ks_uniform(pit)
    assert ks > 0.1
    assert p < 1e-3


def test_tail_mass_asymmetry_sign():
    rng = np.random.default_rng(2)
    body = rng.uniform(0, 0.9, size=3500)
    spike = rng.uniform(0.9, 1.0, size=1500)
    pit = np.concatenate([body, spike])
    tm = ev.tail_mass(pit, 0.1)
    assert tm["tail_right"] > tm["tail_left"]
    assert tm["tail_asym"] > 0


def test_pit_moments_uniform():
    rng = np.random.default_rng(3)
    pit = rng.uniform(0, 1, size=20000)
    m = ev.pit_moments(pit)
    assert m["pit_mean"] == pytest.approx(0.5, abs=0.02)
    assert m["pit_var"] == pytest.approx(1 / 12, abs=0.005)


def test_chi2_uniform_vs_spiked():
    rng = np.random.default_rng(4)
    uni = rng.uniform(0, 1, size=5000)
    _, p_uni = ev.chi2_gof(uni, 10)
    spike = np.concatenate([rng.uniform(0, 0.9, 3500), rng.uniform(0.9, 1, 1500)])
    _, p_spk = ev.chi2_gof(spike, 10)
    assert p_uni > 0.05
    assert p_spk < 1e-3


# --------------------------------------------------------------------------- #
# VaR / ES backtests
# --------------------------------------------------------------------------- #
def test_kupiec_well_calibrated():
    rng = np.random.default_rng(5)
    alpha = 0.05
    n = 4000
    realized = rng.normal(0, 1, n)
    var = np.full(n, np.quantile(realized, alpha))  # correct 5% VaR
    exceed = ev.var_exceedances(realized, var, "left")
    res = ev.kupiec_pof(exceed, alpha)
    assert res["rate"] == pytest.approx(alpha, abs=0.02)
    assert res["p_uc"] > 0.05  # should NOT reject correct VaR


def test_kupiec_miscalibrated_rejects():
    rng = np.random.default_rng(6)
    alpha = 0.05
    n = 4000
    realized = rng.normal(0, 1, n)
    var = np.full(n, np.quantile(realized, 0.20))  # far too loose -> over-exceed
    exceed = ev.var_exceedances(realized, var, "left")
    res = ev.kupiec_pof(exceed, alpha)
    assert res["rate"] > alpha
    assert res["p_uc"] < 0.01


def test_es_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(7)
    realized = rng.normal(0, 1, 6000)
    var = np.full(realized.shape, np.quantile(realized, 0.05))
    out = ev.es_bootstrap_ci(realized, var, "left", n_boot=500)
    assert out["ci_lo"] <= out["es"] <= out["ci_hi"]
    assert out["n_tail"] > 0
    assert out["es"] < var[0]  # ES is deeper in the tail than VaR


def test_fz_loss_finite_and_orders():
    rng = np.random.default_rng(8)
    realized = rng.standard_t(5, size=4000) * 0.01
    alpha = 0.05
    good_var = np.quantile(realized, alpha)
    good_es = realized[realized <= good_var].mean()
    bad_var = np.quantile(realized, 0.30)
    bad_es = realized[realized <= bad_var].mean()
    l_good = ev.fz_loss(realized, good_var, good_es, alpha)
    l_bad = ev.fz_loss(realized, bad_var, bad_es, alpha)
    assert np.isfinite(l_good) and np.isfinite(l_bad)
    assert l_good < l_bad  # correct VaR/ES scores better


def test_quantile_crossing_detection():
    ok = np.array([[0.1, 0.2, 0.3], [0.0, 0.5, 0.9]])
    bad = np.array([[0.3, 0.2, 0.1]])
    assert ev.quantile_crossing_rate(ok) == 0.0
    assert ev.quantile_crossing_rate(bad) == 1.0


def test_summarize_asset_row_keys():
    rng = np.random.default_rng(9)
    pit = rng.uniform(0, 1, 2000)
    realized = rng.normal(0, 0.01, 2000)
    quantiles = np.sort(rng.normal(0, 0.01, (2000, 9)), axis=1)
    row = ev.summarize_asset("TEST", pit, realized, quantiles, pit)
    for k in ("asset", "ks", "ks_p", "pit_mean", "cov80", "tail_left",
              "tail_right", "tail_asym", "chi2", "excess_kurtosis"):
        assert k in row
