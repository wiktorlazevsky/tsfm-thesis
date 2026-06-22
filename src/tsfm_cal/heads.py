"""Phase 3 (HEADLINE) — replacement output heads. STUB.

Not built in Phase 1. This file fixes the interface so the head-replacement work
drops in without restructuring. See HANDOFF §5/§6/§7.

Planned classes (all on the NATIVE torch module with ``torch_compile=False``,
hooking backbone output embeddings at ``timesfm_2p5_torch.py:113``):

  IQNHead              — τ-conditioned cosine embedding fused with the backbone
                         embedding; trained with pinball over sampled τ. Yields
                         any τ in one forward pass; no quantile crossing.
                         Precedent: Dabney 2018; FQFormer 2022. (PRIMARY/headline.)

  SkewStudentTHead     — projects embedding -> (μ, σ, ν, skew); analytic CDF gives
                         any τ + analytic Expected Shortfall; stable NLL training.
                         (SECONDARY / fallback; symmetric Student-t = simplest.)

Each head will expose:
  forward(embedding, tau=None) -> quantiles            # τ may be a sampled grid
  cdf(embedding, x)            -> PIT                   # for the same eval.py path
  var(embedding, alpha)        -> VaR                   # analytic where available
  es(embedding, alpha)         -> Expected Shortfall

The training spectrum to sweep (representational-shift mitigation, HANDOFF §6):
  {fully frozen backbone -> LoRA last-2-layers + train head -> head-only on a
   lightly unfrozen body}. Use LR warmup + gradient clipping for the skew-t head.
"""

from __future__ import annotations

raise NotImplementedError(
    "heads.py is a Phase-3 stub. Implement IQNHead / SkewStudentTHead here when "
    "the head-replacement work starts (see HANDOFF §5)."
)
