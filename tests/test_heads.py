"""CPU unit tests for the replacement heads (needs torch, no model)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tsfm_cal import eval as ev
from tsfm_cal import finetune, heads


# --------------------------------------------------------------------------- #
# Hansen skew-t: CDF/PPF round-trips + ES ordering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lam", [-0.5, 0.0, 0.3])
@pytest.mark.parametrize("nu", [4.0, 8.0, 30.0])
def test_skewt_cdf_ppf_roundtrip(nu, lam):
    nu_a = np.full(5, nu)
    lam_a = np.full(5, lam)
    taus = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    z = heads._skewt_ppf_np(taus, nu_a, lam_a)
    back = heads._skewt_cdf_np(z, nu_a, lam_a)
    assert np.allclose(back, taus, atol=1e-4)


def test_skewt_cdf_monotone_and_bounds():
    nu = np.full(200, 6.0)
    lam = np.full(200, 0.4)
    zs = np.linspace(-8, 8, 200)
    c = heads._skewt_cdf_np(zs, nu, lam)
    assert c.min() >= 0.0 and c.max() <= 1.0
    assert np.all(np.diff(c) >= -1e-9)               # non-decreasing


def test_skewt_symmetric_matches_student_t():
    from scipy.stats import t as _t
    nu = np.full(5, 10.0)
    lam = np.zeros(5)
    # standardized: skew-t with lam=0 has unit variance -> scale sqrt((nu-2)/nu) vs t
    taus = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    z = heads._skewt_ppf_np(taus, nu, lam)
    expected = _t.ppf(taus, 10.0) * np.sqrt((10.0 - 2) / 10.0)
    assert np.allclose(z, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Skew-t head module: NLL finite, params in range
# --------------------------------------------------------------------------- #
def test_skewt_head_params_valid():
    head = heads.SkewStudentTHead()
    emb = torch.randn(32, heads.EMB_DIM)
    mu, sigma, nu, lam = head(emb)
    assert torch.all(sigma > 0)
    assert torch.all(nu > 2.0)
    assert torch.all(lam.abs() < 1.0)
    loss = head.nll(emb, torch.randn(32))
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# IQN head: quantiles monotone in tau (no crossing) after eval sort
# --------------------------------------------------------------------------- #
def test_iqn_quantiles_monotone():
    head = heads.IQNHead()
    emb = np.random.default_rng(0).normal(size=(16, heads.EMB_DIM)).astype(np.float32)
    taus = np.linspace(0.05, 0.95, 19)
    Q = head.quantiles_norm(emb, taus)
    assert Q.shape == (16, 19)
    assert np.all(np.diff(Q, axis=1) >= -1e-9)       # sorted -> no crossing


def test_iqn_pinball_finite():
    head = heads.IQNHead()
    emb = torch.randn(16, heads.EMB_DIM)
    loss = head.pinball(emb, torch.randn(16), n_tau=8)
    assert torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# ES deeper in the tail than VaR (shared head_es / head_var)
# --------------------------------------------------------------------------- #
def test_head_es_beyond_var():
    head = heads.SkewStudentTHead()
    n = 64
    cache = {
        "emb": np.random.default_rng(1).normal(size=(n, heads.EMB_DIM)).astype(np.float32),
        "mu": np.zeros(n), "sigma": np.ones(n),
        "target_norm": np.random.default_rng(2).normal(size=n),
        "target_raw": np.random.default_rng(2).normal(size=n),
    }
    var = heads.head_var(head, cache, 0.05)
    es = heads.head_es(head, cache, 0.05, side="left")
    assert np.all(es <= var + 1e-8)                  # ES is at/under VaR (loss tail)


# --------------------------------------------------------------------------- #
# End-to-end: heads trained on a KNOWN conditional fat-tailed skewed target
# recover the distribution + calibration. Signal lives in one embedding dim, the
# rest are zero (so the test probes the head/training/eval loop, not high-dim
# feature selection — which is irrelevant since returns are ~mean-unpredictable).
# --------------------------------------------------------------------------- #
def _synthetic_cache(seed, nu=5.0, lam=0.3, scale=0.7, beta=1.2, n=6000):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n).astype(np.float32)
    emb = np.zeros((n, heads.EMB_DIM), dtype=np.float32)
    emb[:, 0] = s
    z = heads._skewt_ppf_np(rng.uniform(size=n), np.full(n, nu), np.full(n, lam))
    target = beta * s + scale * z
    return {"emb": emb, "mu": np.zeros(n), "sigma": np.ones(n),
            "target_norm": target.astype(np.float64), "target_raw": target.astype(np.float64)}, s


def test_skewt_head_recovers_calibration():
    cache, s = _synthetic_cache(0)
    head = heads.SkewStudentTHead()
    head, hist = finetune.train_head(head, cache, "skewt", epochs=250, lr=2e-3,
                                     batch_size=256, patience=50, verbose=False)
    pit = heads.head_pit(head, cache)
    ks, _ = ev.ks_uniform(pit)
    mu, sigma, nu, lam = head._params_np(cache["emb"])
    assert ks < 0.03                                  # PIT ≈ uniform
    assert np.corrcoef(mu, 1.2 * s)[0, 1] > 0.95      # conditional mean learned
    assert 0.55 < sigma.mean() < 0.85                 # scale ≈ 0.7
    assert nu.mean() < 9.0                            # fat tail captured (true ν=5)
    assert lam.mean() > 0.15                          # positive skew captured


def test_iqn_head_recovers_calibration():
    cache, s = _synthetic_cache(1)
    head = heads.IQNHead()
    head, hist = finetune.train_head(head, cache, "iqn", epochs=250, lr=2e-3,
                                     batch_size=256, patience=50, verbose=False)
    pit = heads.head_pit(head, cache)
    ks, _ = ev.ks_uniform(pit)
    assert ks < 0.06


def test_head_tail_quantiles_available_at_any_tau():
    """The whole point: query 1/5/95/99% post-hoc from a trained head."""
    cache, _ = _synthetic_cache(2)
    head = heads.SkewStudentTHead()
    head, _ = finetune.train_head(head, cache, "skewt", epochs=120, lr=2e-3,
                                  batch_size=256, patience=40, verbose=False)
    taus = [0.01, 0.05, 0.95, 0.99]
    Q = heads.head_quantiles(head, cache, taus)       # (N, 4) denormalized
    assert Q.shape == (len(cache["emb"]), 4)
    assert np.all(np.diff(Q, axis=1) >= -1e-6)        # monotone across the grid
    # tail VaR/ES at 1% are finite and ES is beyond VaR
    var1 = heads.head_var(head, cache, 0.01)
    es1 = heads.head_es(head, cache, 0.01, side="left")
    assert np.all(np.isfinite(var1)) and np.all(es1 <= var1 + 1e-8)
