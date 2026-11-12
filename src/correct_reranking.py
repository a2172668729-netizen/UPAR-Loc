#!/usr/bin/env python3
"""Correct candidate re-ranking primitives.

Two mathematically distinct operations are exposed:

1. ``reorder_candidate_identities`` (historical E4/B2 family): the identity of
   each candidate changes rank, while the multiset of the K base scores is
   preserved and re-assigned in descending order.
2. ``direct_score_fusion``: each candidate receives an absolute fused score
   ``b_i + lambda * std(b) * z(p_i)``.  This does not preserve the base-score
   multiset and is a separate control.

Tie policy
----------
For every score sort in this module the primary key is score descending and the
secondary key is the *original gallery index ascending*.  ``torch.argsort`` is
used only as a stable primitive after a first stable sort by gallery index.
The official CPU ``argsort`` evaluator is left untouched and is reported
separately by the evaluation layer.

No function reads labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _validate_inputs(base: torch.Tensor,
                     cand: torch.Tensor,
                     scores: Optional[torch.Tensor] = None) -> None:
    if base.ndim != 2:
        raise ValueError(f"base must be 2-D, got {tuple(base.shape)}")
    if cand.ndim != 2 or cand.shape[0] != base.shape[0]:
        raise ValueError(f"cand must be (N,K), got {tuple(cand.shape)}")
    if cand.dtype != torch.long:
        raise TypeError("cand must be torch.long")
    if int(cand.min()) < 0 or int(cand.max()) >= base.shape[1]:
        raise ValueError("cand contains an out-of-range gallery index")
    if scores is not None:
        if scores.shape != cand.shape:
            raise ValueError(f"scores must be {tuple(cand.shape)}, got {tuple(scores.shape)}")
        if not scores.is_floating_point():
            raise TypeError("scores must be floating point")
    # A candidate identity can occur only once per query.
    if cand.shape[1] > 1:
        for r in range(cand.shape[0]):
            if torch.unique(cand[r]).numel() != cand.shape[1]:
                raise ValueError(f"duplicate candidate identity in row {r}")


def _stable_order_by_gallery(scores: torch.Tensor,
                             gallery_ids: torch.Tensor) -> torch.Tensor:
    """Return positions sorted by (score desc, gallery index asc).

    First stable-sort positions by gallery id, then stable-sort that permutation
    by descending score.  The second stable sort keeps gallery-id order among
    exact score ties.
    """
    by_gid = torch.argsort(gallery_ids, dim=1, stable=True)
    ordered_scores = scores.gather(1, by_gid)
    by_score = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)
    return by_gid.gather(1, by_score)


def _stable_topk_ids(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k gallery ids under the module tie policy (score desc, gid asc)."""
    n, g = scores.shape
    k = min(int(k), g)
    gids = torch.arange(g, device=scores.device, dtype=torch.long)[None, :].expand(n, -1)
    order = _stable_order_by_gallery(scores, gids)
    return order[:, :k]


def stable_topk_candidates(base: torch.Tensor, k: int) -> torch.Tensor:
    """Select top-k candidates with the declared deterministic tie rule."""
    if base.ndim != 2:
        raise ValueError("base must be 2-D")
    return _stable_topk_ids(base, k)


def reorder_from_order(base: torch.Tensor,
                       cand: torch.Tensor,
                       target_order: torch.Tensor) -> torch.Tensor:
    """Reassign the sorted candidate base scores to a target identity order.

    Parameters
    ----------
    base : (N,G) base retrieval scores.
    cand : (N,K) candidate gallery ids.  The K base scores are the multiset that
        must be preserved.
    target_order : (N,K) positions into ``cand``; entry ``r`` is the candidate
        position that should occupy new rank ``r``.  This is exactly the
        ``perm`` in the task statement.

    Returns
    -------
    (N,G) scores.  Candidate positions receive a permutation of the original K
    base scores; all other gallery entries are bitwise unchanged.
    """
    _validate_inputs(base, cand)
    if target_order.shape != cand.shape or target_order.dtype != torch.long:
        raise ValueError("target_order must be a long tensor with cand's shape")
    cand = cand.to(base.device)
    target_order = target_order.to(base.device)
    n, k = cand.shape
    if k > 1:
        for r in range(n):
            if torch.unique(target_order[r]).numel() != k:
                raise ValueError(f"target_order row {r} is not a permutation")
    b = base.gather(1, cand)
    # Base scores are assigned in descending base-score order, with gallery id
    # as the deterministic tie-break (not the incoming candidate order).
    base_order = _stable_order_by_gallery(b, cand)
    b_sorted = b.gather(1, base_order)
    target_ids = cand.gather(1, target_order)
    out = base.clone()
    out.scatter_(1, target_ids, b_sorted)
    return out


def fusion_target_order(base: torch.Tensor,
                        cand: torch.Tensor,
                        pair_score: torch.Tensor,
                        lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(target_order, fused)`` for the candidate-identity operation.

    ``target_order[r]`` is the position in ``cand`` that should occupy new rank
    ``r``.  Exposed for diagnostics so a report can compare the target identity
    order with the order actually realised in the final score matrix.
    """
    _validate_inputs(base, cand, pair_score)
    if lam < 0:
        raise ValueError("lam must be non-negative for this operation")
    cand = cand.to(base.device)
    pair_score = pair_score.to(base.device)
    b = base.gather(1, cand).float()
    sd = b.std(1, keepdim=True, unbiased=True).clamp_min(1e-6)
    p = pair_score.float()
    pz = (p - p.mean(1, keepdim=True)) / p.std(1, keepdim=True, unbiased=True).clamp_min(1e-6)
    fused = b + float(lam) * sd * pz
    return _stable_order_by_gallery(fused, cand), fused


def fuse_reorder(base: torch.Tensor,
                 cand: torch.Tensor,
                 pair_score: torch.Tensor,
                 lam: float) -> torch.Tensor:
    """Historical E4/B2 candidate-identity reordering, corrected.

    ``f_i = b_i + lam * std_K(b) * z_K(pair_score)`` determines the target
    candidate order.  The original K base scores are then reassigned to the
    target identities in descending order.
    """
    order, _ = fusion_target_order(base, cand, pair_score, lam)
    return reorder_from_order(base, cand, order)


def direct_score_fusion(base: torch.Tensor,
                        cand: torch.Tensor,
                        pair_score: torch.Tensor,
                        lam: float) -> torch.Tensor:
    """Plan-A control: assign absolute fused scores to candidate identities."""
    _validate_inputs(base, cand, pair_score)
    b = base.gather(1, cand).float()
    sd = b.std(1, keepdim=True, unbiased=True).clamp_min(1e-6)
    p = pair_score.float()
    pz = (p - p.mean(1, keepdim=True)) / p.std(1, keepdim=True, unbiased=True).clamp_min(1e-6)
    fused = b + float(lam) * sd * pz
    out = base.clone()
    out.scatter_(1, cand, fused)
    return out


def historical_buggy_reorder(base: torch.Tensor,
                             cand: torch.Tensor,
                             pair_score: torch.Tensor,
                             lam: float) -> torch.Tensor:
    """The defective historical operation, kept only for regression evidence.

    This reproduces ``out.scatter_(1, cand, b.gather(1, new.argsort()))``.  It
    is deliberately retained so the test suite can prove the counterexample
    fails under the old code.  It must never be used to produce results.
    """
    _validate_inputs(base, cand, pair_score)
    b = base.gather(1, cand).float()
    sd = b.std(1, keepdim=True, unbiased=True).clamp_min(1e-6)
    p = pair_score.float()
    pz = (p - p.mean(1, keepdim=True)) / p.std(1, keepdim=True, unbiased=True).clamp_min(1e-6)
    new = b + float(lam) * sd * pz
    out = base.clone()
    out.scatter_(1, cand, b.gather(1, new.argsort(dim=1, descending=True)))
    return out


@dataclass(frozen=True)
class FusionDiagnostics:
    n_rows: int
    n_candidates: int
    outside_changed: int
    multiset_changed: int
    candidate_order_changed: int


def audit_fusion(base: torch.Tensor,
                 cand: torch.Tensor,
                 out: torch.Tensor) -> FusionDiagnostics:
    """Structural checks for a candidate-only re-ranking output."""
    _validate_inputs(base, cand)
    n, k = cand.shape
    mask = torch.zeros_like(base, dtype=torch.bool)
    mask.scatter_(1, cand, True)
    outside_changed = int((out[~mask] != base[~mask]).sum())
    b = base.gather(1, cand)
    o = out.gather(1, cand)
    multiset_changed = int((b.sort(1).values != o.sort(1).values).any(1).sum())
    base_order = _stable_topk_ids(base, k)
    out_order = _stable_topk_ids(out, k)
    candidate_order_changed = int((base_order != out_order).any(1).sum())
    return FusionDiagnostics(n, k, outside_changed, multiset_changed,
                             candidate_order_changed)