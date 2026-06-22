"""Frozen-backbone embedding extraction + caching (Phase 3, Kaggle/torch env).

The trainable path is the normalization-aware ``decode()`` flow, NOT raw
``forward()`` (verified: timesfm_2p5_torch.py:122-226). This module replicates the
decode *prefill* preprocessing — per-patch ReVIN running stats -> normalize ->
frozen backbone forward -> last-patch ``output_embeddings`` (1280-dim) — and the
matching normalization of the 1-step target.

Because the backbone is frozen, embeddings are fixed inputs to head training, so
we **cache them once per asset** (Adler et al. approach). Heads then train fast on
``(emb, mu, sigma, target_norm)`` tuples without re-running the backbone.

Requires torch + timesfm (Kaggle). Not imported by the package __init__, so the
CPU package stays importable without torch.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config, io_utils


def load_native(torch_compile: bool = False):
    """Load TimesFM 2.5 and return ``(wrapper, native_module)``.

    ``torch_compile=False`` is mandatory for training (avoids torch.compile;
    timesfm_2p5_torch.py:84-90). The native nn.Module is ``wrapper.model``.
    """
    import timesfm

    wrapper = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=torch_compile
    )
    return wrapper, wrapper.model


def freeze_backbone(native, freeze_heads: bool = True) -> None:
    """Set ``requires_grad=False`` on the backbone (tokenizer + transformer stack,
    and the native output heads by default). The replacement head is a separate
    module, so it stays trainable."""
    native.eval()
    for p in native.tokenizer.parameters():
        p.requires_grad = False
    for layer in native.stacked_xf:
        for p in layer.parameters():
            p.requires_grad = False
    if freeze_heads:
        for p in native.output_projection_point.parameters():
            p.requires_grad = False
        for p in native.output_projection_quantiles.parameters():
            p.requires_grad = False


@torch.no_grad()
def embed_batch(native, contexts: torch.Tensor):
    """Embed a batch of contexts via the decode prefill preprocessing.

    Parameters
    ----------
    native : the TimesFM nn.Module (frozen).
    contexts : (B, L) float tensor, L a multiple of patch size p (=32).

    Returns
    -------
    emb : (B, 1280) last-patch backbone embedding (predicts the next step).
    mu, sigma : (B,) last-patch ReVIN stats (normalize the target, denormalize quantiles).
    """
    device = native.device
    p = native.p
    contexts = contexts.to(device).float()
    B, L = contexts.shape
    if L % p != 0:
        raise ValueError(f"context length {L} must be a multiple of patch size {p}")
    num_patches = L // p

    patched = contexts.reshape(B, num_patches, p)
    masks = torch.zeros_like(patched, dtype=torch.bool)

    # Per-patch running stats (cumulative), mirroring decode() prefill.
    n = torch.zeros(B, device=device)
    mu = torch.zeros(B, device=device)
    sigma = torch.zeros(B, device=device)
    patch_mu, patch_sigma = [], []
    for i in range(num_patches):
        (n, mu, sigma), _ = _update_running_stats(n, mu, sigma, patched[:, i], masks[:, i])
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    last_mu, last_sigma = mu, sigma
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)

    normed = _revin(patched, context_mu, context_sigma, reverse=False)
    normed = torch.where(masks, torch.zeros_like(normed), normed)

    (_, output_embeddings, _, _), _ = native(normed, masks, None)
    emb = output_embeddings[:, -1, :]  # last patch -> 1-step-ahead embedding
    return emb, last_mu, last_sigma


def cache_embeddings(
    native,
    returns: np.ndarray,
    context: int = config.CONTEXT,
    batch_size: int = 256,
    label: str = "",
):
    """Roll 1-step windows over ``returns`` and cache backbone embeddings + targets.

    Returns a dict of numpy arrays aligned by forecast step ``t`` in
    ``[context, len-1)``:
      emb (T,1280), mu (T,), sigma (T,), target_raw (T,), target_norm (T,), idx (T,)
    where ``target_norm = (target_raw - mu) / sigma`` is the head's training target.
    """
    returns = np.asarray(returns, dtype=float).flatten()
    n = len(returns)
    starts = list(range(context, n - 1))
    emb_all, mu_all, sig_all, tgt_all, idx_all = [], [], [], [], []

    for b0 in range(0, len(starts), batch_size):
        batch_t = starts[b0:b0 + batch_size]
        ctx = np.stack([returns[t - context:t] for t in batch_t])
        tgt = np.array([returns[t] for t in batch_t])
        emb, mu, sigma = embed_batch(native, torch.from_numpy(ctx))
        emb_all.append(emb.detach().cpu().numpy())
        mu_all.append(mu.detach().cpu().numpy())
        sig_all.append(sigma.detach().cpu().numpy())
        tgt_all.append(tgt)
        idx_all.append(np.array(batch_t))
        if label:
            print(f"    [{label}] cached {min(b0 + batch_size, len(starts))}/{len(starts)}")

    emb = np.concatenate(emb_all).astype(np.float32)
    mu = np.concatenate(mu_all).astype(np.float64)
    sigma = np.concatenate(sig_all).astype(np.float64)
    target_raw = np.concatenate(tgt_all).astype(np.float64)
    idx = np.concatenate(idx_all)
    target_norm = (target_raw - mu) / np.where(sigma < 1e-6, 1.0, sigma)

    return dict(emb=emb, mu=mu, sigma=sigma, target_raw=target_raw,
               target_norm=target_norm, idx=idx)


def save_cache(asset: str, cache: dict) -> "object":
    """Persist an embedding cache under outputs/finetune/<asset>/emb_cache.npz."""
    out = io_utils.finetune_dir(asset) / "emb_cache.npz"
    np.savez_compressed(out, **cache)
    return out


def load_cache(asset: str) -> dict:
    npz = np.load(io_utils.finetune_dir(asset, create=False) / "emb_cache.npz")
    return {k: npz[k] for k in npz.files}


# --- torch ports of the two util fns (kept local so we don't depend on private
#     timesfm internals; identical math to timesfm/torch/util.py) -------------- #
_TOL = 1e-6


def _update_running_stats(n, mu, sigma, x, mask):
    is_legit = torch.logical_not(mask)
    inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)
    inc_n_safe = torch.where(inc_n == 0, torch.ones_like(inc_n), inc_n)
    inc_mu = torch.sum(x * is_legit, dim=-1) / inc_n_safe
    inc_mu = torch.where(inc_n == 0, torch.zeros_like(inc_mu), inc_mu)
    inc_var = torch.sum(((x - inc_mu.unsqueeze(-1)) ** 2) * is_legit, dim=-1) / inc_n_safe
    inc_var = torch.where(inc_n == 0, torch.zeros_like(inc_var), inc_var)
    inc_sigma = torch.sqrt(inc_var)
    new_n = n + inc_n
    new_n_safe = torch.where(new_n == 0, torch.ones_like(new_n), new_n)
    new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
    new_mu = torch.where(new_n == 0, torch.zeros_like(new_mu), new_mu)
    new_var = (n * sigma.pow(2) + inc_n * inc_sigma.pow(2)
               + n * (mu - new_mu).pow(2) + inc_n * (inc_mu - new_mu).pow(2)) / new_n_safe
    new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))
    return (new_n, new_mu, new_sigma), None


def _revin(x, mu, sigma, reverse=False):
    if mu.dim() == x.dim() - 1:
        mu, sigma = mu[..., None], sigma[..., None]
    elif mu.dim() == x.dim() - 2:
        mu, sigma = mu[..., None, None], sigma[..., None, None]
    if reverse:
        return x * sigma + mu
    return (x - mu) / torch.where(sigma < _TOL, torch.ones_like(sigma), sigma)
