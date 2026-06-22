"""Phase 3 replacement output heads: Skew-Student-t (Hansen 1994) + IQN.

Both heads model the distribution of the **normalized** 1-step target
``z = (r - ctx_mu) / ctx_sigma`` given the frozen backbone embedding. Callers
denormalize quantiles via ``q_raw = ctx_mu + ctx_sigma * q_norm`` (the ReVIN
reverse). Both expose the same numpy-facing eval API so ``eval.py`` treats them
uniformly:

    head.quantiles_norm(emb_np, taus) -> (N, K) normalized quantiles
    head.cdf_norm(emb_np, z_np)       -> (N,)   PIT values (= CDF of the target)

τ is a QUERY-time argument for both — nothing is baked in at training. Skew-t is
trained by NLL (no τ at all); IQN is trained on randomly sampled τ.

Requires torch (Kaggle / a torch env). scipy is used for the Student-t CDF/inverse
in the skew-t eval path.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

EMB_DIM = 1280


# =========================================================================== #
# Skew-Student-t head (Hansen 1994)
# =========================================================================== #
class SkewStudentTHead(nn.Module):
    """Maps a 1280-dim embedding -> (μ, σ, ν, λ) of a Hansen skew-t.

    μ,σ are the location/scale of the normalized target; ν>2 (df), λ∈(-1,1)
    (skew). Symmetric Student-t is the λ=0 special case.
    """

    def __init__(self, in_dim: int = EMB_DIM, nu_floor: float = 2.05):
        super().__init__()
        self.nu_floor = nu_floor
        self.proj = nn.Linear(in_dim, 4)
        # Initialize near a unit Gaussian: mu=0, sigma~1, nu~8, lam=0. Starting
        # with light tails (large nu) keeps early gradients well-scaled so the
        # head learns location/scale fast (then nu can drop to fit fat tails).
        with torch.no_grad():
            self.proj.bias.copy_(torch.tensor([0.0, 0.5413, 6.0, 0.0]))  # softplus(0.5413)=1, nu=2.05+softplus(6)≈8

    def forward(self, emb: torch.Tensor):
        raw = self.proj(emb)
        mu = raw[..., 0]
        sigma = torch.nn.functional.softplus(raw[..., 1]) + 1e-4
        nu = self.nu_floor + torch.nn.functional.softplus(raw[..., 2])
        lam = 0.99 * torch.tanh(raw[..., 3])
        return mu, sigma, nu, lam

    # --- training loss ------------------------------------------------------ #
    def nll(self, emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood of normalized targets ``y``."""
        mu, sigma, nu, lam = self(emb)
        z = (y - mu) / sigma
        logf = _skewt_logpdf_torch(z, nu, lam) - torch.log(sigma)
        return -logf.mean()

    # --- numpy eval API ----------------------------------------------------- #
    @torch.no_grad()
    def _params_np(self, emb_np: np.ndarray):
        emb = torch.as_tensor(emb_np, dtype=torch.float32, device=self.proj.weight.device)
        mu, sigma, nu, lam = self(emb)
        return (mu.cpu().numpy(), sigma.cpu().numpy(), nu.cpu().numpy(), lam.cpu().numpy())

    def quantiles_norm(self, emb_np: np.ndarray, taus) -> np.ndarray:
        mu, sigma, nu, lam = self._params_np(emb_np)
        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        out = np.empty((len(mu), len(taus)))
        for j, tau in enumerate(taus):
            out[:, j] = mu + sigma * _skewt_ppf_np(tau, nu, lam)
        return out

    def cdf_norm(self, emb_np: np.ndarray, z_np: np.ndarray) -> np.ndarray:
        mu, sigma, nu, lam = self._params_np(emb_np)
        zstd = (np.asarray(z_np, dtype=float) - mu) / sigma
        return _skewt_cdf_np(zstd, nu, lam)


# =========================================================================== #
# Implicit Quantile Network head (Dabney 2018 / FQFormer 2022)
# =========================================================================== #
class IQNHead(nn.Module):
    """τ-conditioned head: Q(τ | emb) = MLP( ψ(emb) ⊙ ReLU(W φ(τ)) ).

    φ(τ) = cosine embedding. Trained with pinball over randomly sampled τ, so any
    τ is available at query time. Monotonicity in τ is enforced (sort) only in the
    numpy eval path, not during training.
    """

    def __init__(self, in_dim: int = EMB_DIM, n_cos: int = 64, hidden: int = 128):
        super().__init__()
        self.n_cos = n_cos
        self.psi = nn.Linear(in_dim, in_dim)
        self.phi = nn.Linear(n_cos, in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("_js", torch.arange(n_cos).float() * np.pi)

    def forward(self, emb: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """emb (B, D); tau (B, K) (or (B,) / scalar). Returns q (B, K) (or (B,))."""
        squeeze = False
        if tau.dim() == 0:
            tau = tau.view(1, 1).expand(emb.shape[0], 1)
        if tau.dim() == 1:
            tau = tau.unsqueeze(-1)  # (B,1)
            squeeze = True
        B, K = tau.shape
        cos = torch.cos(tau.unsqueeze(-1) * self._js)        # (B, K, n_cos)
        phi = torch.relu(self.phi(cos))                       # (B, K, D)
        psi = self.psi(emb).unsqueeze(1)                      # (B, 1, D)
        q = self.mlp(psi * phi).squeeze(-1)                   # (B, K)
        return q.squeeze(-1) if squeeze else q

    # --- training loss ------------------------------------------------------ #
    def pinball(self, emb: torch.Tensor, y: torch.Tensor, n_tau: int = 16) -> torch.Tensor:
        """Mean pinball loss over ``n_tau`` random τ∈(0,1) per step."""
        B = emb.shape[0]
        tau = torch.rand(B, n_tau, device=emb.device).clamp(1e-3, 1 - 1e-3)
        q = self(emb, tau)                                    # (B, n_tau)
        diff = y.unsqueeze(-1) - q                            # (B, n_tau)
        loss = torch.maximum(tau * diff, (tau - 1) * diff)
        return loss.mean()

    # --- numpy eval API ----------------------------------------------------- #
    @torch.no_grad()
    def quantiles_norm(self, emb_np: np.ndarray, taus) -> np.ndarray:
        device = self.psi.weight.device
        emb = torch.as_tensor(emb_np, dtype=torch.float32, device=device)
        taus = np.atleast_1d(np.asarray(taus, dtype=float))
        tau_t = torch.as_tensor(taus, dtype=torch.float32, device=device)
        tau_b = tau_t.unsqueeze(0).expand(emb.shape[0], -1)   # (N, K)
        q = self(emb, tau_b).cpu().numpy()
        return np.sort(q, axis=1)  # enforce monotonicity across the τ grid

    def cdf_norm(self, emb_np: np.ndarray, z_np: np.ndarray, grid: int = 199) -> np.ndarray:
        """PIT by inverting the (sorted) quantile grid: τ s.t. Q(τ)=z."""
        taus = np.linspace(1.0 / (grid + 1), grid / (grid + 1), grid)
        Q = self.quantiles_norm(emb_np, taus)                 # (N, grid), sorted
        z = np.asarray(z_np, dtype=float)
        out = np.empty(len(z))
        for i in range(len(z)):
            out[i] = np.interp(z[i], Q[i], taus, left=0.0, right=1.0)
        return out


# =========================================================================== #
# Shared: arbitrary-τ quantiles, PIT, VaR, ES (denormalized) from any head
# =========================================================================== #
def head_quantiles(head, cache: dict, taus) -> np.ndarray:
    """Denormalized quantiles (N, K) at ``taus`` from a head + embedding cache."""
    qn = head.quantiles_norm(cache["emb"], taus)              # (N, K) normalized
    mu = cache["mu"][:, None]
    sigma = cache["sigma"][:, None]
    return mu + sigma * qn


def head_pit(head, cache: dict) -> np.ndarray:
    """PIT series from a head: CDF of the realized normalized target."""
    return head.cdf_norm(cache["emb"], cache["target_norm"])


def head_var(head, cache: dict, alpha: float) -> np.ndarray:
    """Per-step VaR forecast at level ``alpha`` (denormalized quantile)."""
    return head_quantiles(head, cache, [alpha])[:, 0]


def head_es(head, cache: dict, alpha: float, side: str = "left", grid: int = 200) -> np.ndarray:
    """Per-step Expected Shortfall by numerically integrating the quantile fn.

    Left tail: ES = (1/alpha) ∫_0^alpha Q(p) dp. Right tail (alpha e.g. 0.99):
    ES = (1/(1-alpha)) ∫_alpha^1 Q(p) dp. Uniform across heads (no analytic
    dependence), and robust for both skew-t and IQN.
    """
    if side == "left":
        ps = np.linspace(1e-4, alpha, grid)
        Q = head_quantiles(head, cache, ps)                  # (N, grid)
        return _trapz(Q, ps, axis=1) / alpha
    ps = np.linspace(alpha, 1 - 1e-4, grid)
    Q = head_quantiles(head, cache, ps)
    return _trapz(Q, ps, axis=1) / (1 - alpha)


# numpy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# =========================================================================== #
# Hansen (1994) skew-t internals
# =========================================================================== #
def _hansen_abc_torch(nu, lam):
    log_c = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
             - 0.5 * torch.log(np.pi * (nu - 2)))
    c = torch.exp(log_c)
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = torch.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, log_c


def _skewt_logpdf_torch(z, nu, lam):
    a, b, log_c = _hansen_abc_torch(nu, lam)
    denom = torch.where(z < -a / b, 1 - lam, 1 + lam)
    inner = (b * z + a) / denom
    return (torch.log(b) + log_c
            - ((nu + 1) / 2) * torch.log1p(inner ** 2 / (nu - 2)))


def _hansen_abc_np(nu, lam):
    from scipy.special import gammaln

    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)))
    a = 4 * lam * c * (nu - 2) / (nu - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    return a, b, c


def _skewt_cdf_np(z, nu, lam):
    """CDF of the standardized Hansen skew-t (Jondeau & Rockinger 2003)."""
    from scipy.stats import t as _t

    a, b, _ = _hansen_abc_np(nu, lam)
    scale = np.sqrt(nu / (nu - 2))
    u = (b * z + a) * scale
    left = z < -a / b
    out = np.where(
        left,
        (1 - lam) * _t.cdf(np.where(left, u / (1 - lam), 0.0), nu),
        (1 + lam) * _t.cdf(np.where(left, 0.0, u / (1 + lam)), nu) - lam,
    )
    return np.clip(out, 0.0, 1.0)


def _skewt_ppf_np(tau, nu, lam):
    """Inverse-CDF of the standardized Hansen skew-t."""
    from scipy.stats import t as _t

    a, b, _ = _hansen_abc_np(nu, lam)
    scale = np.sqrt(nu / (nu - 2))
    boundary = (1 - lam) / 2
    tau = np.full_like(np.asarray(nu, dtype=float), tau) if np.isscalar(tau) else np.asarray(tau)
    left = tau < boundary
    u_left = (1 - lam) * _t.ppf(np.where(left, tau / (1 - lam), 0.5), nu)
    u_right = (1 + lam) * _t.ppf(np.where(left, 0.5, (tau + lam) / (1 + lam)), nu)
    u = np.where(left, u_left, u_right)
    return (u / scale - a) / b
