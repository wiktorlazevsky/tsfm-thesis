"""Phase 3 — head-only training on cached frozen-backbone embeddings.

Trains a replacement head (``heads.SkewStudentTHead`` or ``heads.IQNHead``) on an
embedding cache from ``backbone.cache_embeddings``. Because the backbone is frozen
and embeddings are cached, this is a fast tabular training loop (no model forward),
so we can run many epochs and iterate head architectures cheaply.

Artifacts under ``outputs/finetune/<ASSET>/`` (schema fixed so eval/plot reuse):
    head.pt            — trained head weights
    train_config.json  — kind, lr, epochs, split, seed, val loss curve
    pit_before.npz     — zero-shot PIT (copied from a zeroshot run, optional)
    pit_after.npz      — PIT after head replacement (io_utils.save_pit_npz schema)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import io_utils


def finetune_paths(asset: str) -> dict[str, Path]:
    """Canonical artifact paths for one asset's head-replacement run."""
    root = io_utils.finetune_dir(asset, create=True)
    return {
        "root": root,
        "head": root / "head.pt",
        "pit_before": root / "pit_before.npz",
        "pit_after": root / "pit_after.npz",
        "train_config": root / "train_config.json",
        "emb_cache": root / "emb_cache.npz",
    }


def chrono_split(n: int, val_frac: float = 0.2):
    """Chronological train/val split indices (no shuffling — time series)."""
    cut = int(n * (1 - val_frac))
    return np.arange(cut), np.arange(cut, n)


def train_head(
    head,
    cache: dict,
    kind: str,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    val_frac: float = 0.2,
    warmup_frac: float = 0.05,
    grad_clip: float = 1.0,
    n_tau: int = 16,
    patience: int = 20,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = True,
):
    """Train ``head`` on a cached embedding dataset.

    ``kind`` ∈ {"skewt", "iqn"} selects the loss (NLL vs sampled-τ pinball).
    Chronological split, AdamW + cosine schedule with linear warmup, gradient
    clipping (representational-shift mitigation), early stopping on val loss.
    Returns the best (lowest-val-loss) head state restored in-place + a history.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    head = head.to(device)

    emb = torch.as_tensor(cache["emb"], dtype=torch.float32)
    y = torch.as_tensor(cache["target_norm"], dtype=torch.float32)
    tr, va = chrono_split(len(emb), val_frac)
    train_ds = TensorDataset(emb[tr], y[tr])
    val_emb, val_y = emb[va].to(device), y[va].to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Weight decay on weight matrices only — never on biases (the skew-t σ/ν/λ
    # shape params live in biases; decaying them toward 0 corrupts the fit).
    decay = [p for n_, p in head.named_parameters() if p.requires_grad and p.dim() > 1]
    no_decay = [p for n_, p in head.named_parameters() if p.requires_grad and p.dim() <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )
    total_steps = epochs * max(1, len(loader))
    warmup_steps = int(total_steps * warmup_frac)

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def loss_fn(e, t):
        return head.nll(e, t) if kind == "skewt" else head.pinball(e, t, n_tau=n_tau)

    best_val, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        head.train()
        for eb, yb in loader:
            eb, yb = eb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(eb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            vloss = float(loss_fn(val_emb, val_y))
        hist.append(vloss)
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  val={vloss:.5f}  best={best_val:.5f}")
        if bad >= patience:
            if verbose:
                print(f"    early stop at epoch {ep} (best val={best_val:.5f})")
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head, {"kind": kind, "best_val": best_val, "val_curve": hist, "n_epochs": len(hist)}


def save_run(asset: str, head, history: dict, cache: dict, pit_after: np.ndarray):
    """Persist trained head + train_config + pit_after for one asset."""
    import torch

    paths = finetune_paths(asset)
    torch.save(head.state_dict(), paths["head"])
    io_utils.save_json(paths["train_config"], history)
    np.savez_compressed(
        paths["pit_after"],
        pit=np.asarray(pit_after),
        realized=np.asarray(cache["target_raw"]),
        target_norm=np.asarray(cache["target_norm"]),
    )
    return paths
