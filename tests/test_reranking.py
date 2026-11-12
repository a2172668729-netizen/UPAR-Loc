#!/usr/bin/env python3
"""T1-T8 for the corrected candidate re-ranking implementation.

Run from the repository directory with the reproduction environment:

    python tests/test_reranking.py

The reference implementation below is deliberately plain Python and does not
import or call any helper from ``src.correct_reranking``.
"""
from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import correct_reranking as CR  # noqa: E402


# --------------------------------------------------------------------------- #
# Independent reference implementation (no calls into CR helpers)
# --------------------------------------------------------------------------- #
def reference_order(values: list[float], gids: list[int]) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (-float(values[i]), int(gids[i])))


def reference_reorder(base_row: list[float], cand_row: list[int],
                      order: list[int]) -> list[float]:
    out = [float(x) for x in base_row]
    b = [float(base_row[int(g)]) for g in cand_row]
    b_sorted = [b[i] for i in reference_order(b, cand_row)]
    target = [int(cand_row[i]) for i in order]
    for gid, value in zip(target, b_sorted):
        out[gid] = value
    return out


def reference_fuse(base_row: list[float], cand_row: list[int],
                   pair_row: list[float], lam: float) -> tuple[list[float], list[int]]:
    b = [float(base_row[int(g)]) for g in cand_row]
    n = len(b)
    mean_b = sum(b) / n
    var_b = sum((x - mean_b) ** 2 for x in b) / (n - 1)
    sd_b = max(math.sqrt(var_b), 1e-6)
    mean_p = sum(float(x) for x in pair_row) / n
    var_p = sum((float(x) - mean_p) ** 2 for x in pair_row) / (n - 1)
    sd_p = max(math.sqrt(var_p), 1e-6)
    fused = [b[i] + lam * sd_b * ((float(pair_row[i]) - mean_p) / sd_p) for i in range(n)]
    order = reference_order(fused, cand_row)
    out = reference_reorder(base_row, cand_row, order)
    return out, [int(cand_row[i]) for i in order]


def keyed_base(rng: np.random.Generator, k: int, g: int) -> tuple[torch.Tensor, torch.Tensor]:
    vals = rng.normal(size=g).astype(np.float32)
    vals = vals + np.arange(g, dtype=np.float32) * np.float32(1e-5)
    cand0 = np.argsort(-vals, kind="stable")[:k]
    vals[cand0] -= np.float32(10.0)
    p = rng.permutation(k)
    cand = cand0[p]
    vals[cand] += np.float32(10.0) + np.arange(k, dtype=np.float32) * np.float32(1e-6)
    return torch.from_numpy(vals[None, :]), torch.from_numpy(cand[None, :].astype(np.int64))


def test_three_element_cycle() -> None:
    base = torch.tensor([[3.0, 2.0, 1.0]])
    cand = torch.tensor([[0, 1, 2]], dtype=torch.long)
    order = torch.tensor([[2, 0, 1]], dtype=torch.long)
    out = CR.reorder_from_order(base, cand, order)
    assert out.tolist() == [[2.0, 1.0, 3.0]]
    assert out.argmax(1).item() == 2


def test_historical_bug_fails_counterexample() -> None:
    base = torch.tensor([[3.0, 2.0, 1.0]])
    cand = torch.tensor([[0, 1, 2]], dtype=torch.long)
    # pair z-scores [0, -1.2247, +1.2247], lambda*sd(b)=2.5/0.8165 -> offset 2.5
    q = 1.5 ** 0.5
    pair = torch.tensor([[0.0, -q, q]])
    correct = CR.fuse_reorder(base, cand, pair, 2.5)
    buggy = CR.historical_buggy_reorder(base, cand, pair, 2.5)
    assert correct.tolist() == [[2.0, 1.0, 3.0]]
    assert buggy.tolist() == [[1.0, 3.0, 2.0]]
    assert correct.argmax(1).item() == 2
    assert buggy.argmax(1).item() != 2


def test_random_permutations() -> None:
    for k in (3, 5, 10, 20):
        rng = np.random.default_rng(1000 + k)
        g = k + 7
        for case in range(1000):
            base, cand = keyed_base(rng, k, g)
            pair = torch.from_numpy(rng.normal(size=(1, k)).astype(np.float32))
            out = CR.fuse_reorder(base, cand, pair, lam=0.8)
            ref_out, ref_target = reference_fuse(base[0].tolist(), cand[0].tolist(),
                                                 pair[0].tolist(), 0.8)
            assert np.allclose(out[0].numpy(), ref_out, atol=2e-6, rtol=0), (case, k)
            got_top = CR.stable_topk_candidates(out, k)[0].tolist()
            assert got_top == ref_target, (case, k, got_top, ref_target)


def test_lambda_zero_preserves_unique_ranking() -> None:
    rng = np.random.default_rng(7)
    for k in (3, 5, 10, 20):
        for _ in range(100):
            base, cand = keyed_base(rng, k, k + 5)
            pair = torch.from_numpy(rng.normal(size=(1, k)).astype(np.float32))
            out = CR.fuse_reorder(base, cand, pair, lam=0.0)
            assert torch.equal(CR.stable_topk_candidates(out, k),
                               CR.stable_topk_candidates(base, k))
            assert torch.equal(out.sort(1).values, base.sort(1).values)


def test_candidate_identity_invariance() -> None:
    rng = np.random.default_rng(11)
    for k in (3, 5, 10, 20):
        for _ in range(250):
            base, cand = keyed_base(rng, k, k + 6)
            pair = torch.from_numpy(rng.normal(size=(1, k)).astype(np.float32))
            ref = CR.fuse_reorder(base, cand, pair, 0.8)
            perm = torch.from_numpy(rng.permutation(k).astype(np.int64))
            out = CR.fuse_reorder(base, cand[:, perm], pair[:, perm], 0.8)
            assert torch.equal(ref, out)


def test_outside_candidates_unchanged_per_query() -> None:
    rng = np.random.default_rng(13)
    n, g, k = 20, 31, 10
    base = torch.from_numpy(rng.normal(size=(n, g)).astype(np.float32))
    cand = torch.stack([torch.from_numpy(rng.choice(g, k, replace=False).astype(np.int64))
                        for _ in range(n)])
    pair = torch.from_numpy(rng.normal(size=(n, k)).astype(np.float32))
    out = CR.fuse_reorder(base, cand, pair, 0.8)
    outside_changed = 0
    for q in range(n):
        mask = torch.ones(g, dtype=torch.bool)
        mask[cand[q]] = False
        outside_changed += int((out[q, mask] != base[q, mask]).sum())
    assert outside_changed == 0
    assert CR.audit_fusion(base, cand, out).outside_changed == 0


def test_candidate_score_multiset_preserved() -> None:
    rng = np.random.default_rng(17)
    n, g, k = 128, 40, 20
    base = torch.from_numpy(rng.normal(size=(n, g)).astype(np.float32))
    cand = torch.stack([torch.from_numpy(rng.choice(g, k, replace=False).astype(np.int64))
                        for _ in range(n)])
    pair = torch.from_numpy(rng.normal(size=(n, k)).astype(np.float32))
    out = CR.fuse_reorder(base, cand, pair, 0.8)
    before = base.gather(1, cand).sort(1).values
    after = out.gather(1, cand).sort(1).values
    assert torch.equal(before, after)


def test_tie_policy_is_stable_and_gallery_indexed() -> None:
    base = torch.tensor([[1.0, 1.0, 0.5]])
    cand = torch.tensor([[2, 0, 1]], dtype=torch.long)
    pair = torch.tensor([[5.0, 5.0, 5.0]])
    out = CR.fuse_reorder(base, cand, pair, lam=1.0)
    assert out.tolist() == [[1.0, 1.0, 0.5]]
    assert CR.stable_topk_candidates(out, 3).tolist() == [[0, 1, 2]]
    official = torch.argsort(out, dim=1, descending=True).tolist()
    assert official == [[0, 1, 2]]


def test_independent_reference_full_row() -> None:
    for k in (3, 5, 10, 20):
        rng = np.random.default_rng(2300 + k)
        g = k + 9
        for _ in range(1000):
            base, cand = keyed_base(rng, k, g)
            pair = torch.from_numpy(rng.normal(size=(1, k)).astype(np.float32))
            got = CR.fuse_reorder(base, cand, pair, 0.8)[0].tolist()
            want, _ = reference_fuse(base[0].tolist(), cand[0].tolist(),
                                     pair[0].tolist(), 0.8)
            assert np.allclose(got, want, atol=2e-6, rtol=0)


def test_direct_fusion_is_a_distinct_operation() -> None:
    rng = np.random.default_rng(31)
    base, cand = keyed_base(rng, 5, 12)
    pair = torch.from_numpy(rng.normal(size=(1, 5)).astype(np.float32))
    reordered = CR.fuse_reorder(base, cand, pair, 0.8)
    direct = CR.direct_score_fusion(base, cand, pair, 0.8)
    assert not torch.equal(reordered.gather(1, cand), direct.gather(1, cand))
    assert CR.audit_fusion(base, cand, direct).outside_changed == 0


def test_validation_rejects_duplicate_candidates() -> None:
    base = torch.zeros(1, 5)
    cand = torch.tensor([[0, 1, 1]], dtype=torch.long)
    try:
        CR.fuse_reorder(base, cand, torch.zeros(1, 3), 0.8)
    except ValueError:
        return
    raise AssertionError("duplicate candidate identities were not rejected")


TESTS = [
    test_three_element_cycle,
    test_historical_bug_fails_counterexample,
    test_random_permutations,
    test_lambda_zero_preserves_unique_ranking,
    test_candidate_identity_invariance,
    test_outside_candidates_unchanged_per_query,
    test_candidate_score_multiset_preserved,
    test_tie_policy_is_stable_and_gallery_indexed,
    test_independent_reference_full_row,
    test_direct_fusion_is_a_distinct_operation,
    test_validation_rejects_duplicate_candidates,
]


def main() -> int:
    failures = []
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            failures.append(fn.__name__)
            print(f"FAIL {fn.__name__}", flush=True)
            traceback.print_exc()
    print(f"\n{len(TESTS) - len(failures)}/{len(TESTS)} test groups passed", flush=True)
    if failures:
        print("FAILED:", failures, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
