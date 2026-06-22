"""Phase 3 — head/adapter training + saving convention. STUB.

Not built in Phase 1. Exists now to FIX THE SAVING CONVENTION so the next step
has a home and artifacts stay consistent with the zero-shot tree.

Saving convention (HANDOFF §8), under ``outputs/finetune/<ASSET>/``:
    adapter.pt        — LoRA/DoRA adapter weights (if backbone is adapted)
    head.pt           — replacement head weights (IQN / skew-t)
    pit_before.npz    — zero-shot PIT for this asset (copied from the zeroshot run)
    pit_after.npz     — PIT after head replacement (same npz schema as pit.py)
    train_config.json — flags, lr, frozen-vs-LoRA setting, τ-sampling, seed

The ``pit_*.npz`` files use the IDENTICAL schema as ``io_utils.save_pit_npz`` so
``eval.summarize_run`` / ``plot.pit_grid`` work unchanged on before/after.
"""

from __future__ import annotations

from pathlib import Path

from . import io_utils


def finetune_paths(asset: str) -> dict[str, Path]:
    """Return the canonical artifact paths for one asset's head-replacement run."""
    root = io_utils.finetune_dir(asset, create=True)
    return {
        "root": root,
        "adapter": root / "adapter.pt",
        "head": root / "head.pt",
        "pit_before": root / "pit_before.npz",
        "pit_after": root / "pit_after.npz",
        "train_config": root / "train_config.json",
    }


def train_head(*args, **kwargs):  # noqa: D401
    raise NotImplementedError(
        "finetune.train_head is a Phase-3 stub. Implement head-only training on "
        "the native torch module (torch_compile=False) here; see heads.py and "
        "HANDOFF §7 (Phase 3)."
    )
