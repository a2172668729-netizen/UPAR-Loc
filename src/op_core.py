#!/usr/bin/env python3
"""Label-free OP/PCA core for the official-Train source domain.

All functions in this file operate on feature tensors only.  The evaluation
module imports this file; this file never imports labels, split files, paths,
or an evaluator.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# R-MAC weights: levels [1,2], alpha 6.0 -> [1, 1/64*4]
# --------------------------------------------------------------------------- #
def rmac_weights(device, dtype=torch.float32) -> torch.Tensor:
    w = [1.0] + [1.0 / float(2 ** 6)] * 4
    return torch.tensor(w, device=device, dtype=dtype)


# --------------------------------------------------------------------------- #
# PCA / projection
# --------------------------------------------------------------------------- #
def _pca(x: torch.Tensor, dim: int, whiten: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    z = x.reshape(-1, x.shape[-1])
    mean = z.mean(0)
    z = z - mean
    evals, evecs = torch.linalg.eigh(z.T @ z / (len(z) - 1))
    evals = evals.flip(0)[:dim].clamp_min(1e-12)
    evecs = evecs.flip(1)[:, :dim]
    if whiten > 0:
        evecs = evecs / evals.pow(whiten / 2)[None, :]
    return mean, evecs


def _shared(q: torch.Tensor, r: torch.Tensor, dim: int,
            whiten: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.reshape(-1, q.shape[-1])
    r = r.reshape(-1, r.shape[-1])
    mean = (q.mean(0) + r.mean(0)) / 2
    q = q - mean
    r = r - mean
    evals, evecs = torch.linalg.eigh((q.T @ q / len(q) + r.T @ r / len(r)) / 2)
    evals = evals.flip(0)[:dim].clamp_min(1e-12)
    evecs = evecs.flip(1)[:, :dim]
    if whiten > 0:
        evecs = evecs / evals.pow(whiten / 2)[None, :]
    return mean, evecs


def _aggregate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.normalize((F.normalize(x, dim=-1) * (w / w.sum())[None, :, None]).sum(1), dim=1)


def _anchor(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x * (w / w.sum())[None, :, None]).sum(1)


@torch.inference_mode()
def build_projection(q: torch.Tensor, r: torch.Tensor, dim: int = 512,
                     whiten_q: float = 0.55, whiten_r: float = 0.10) -> dict:
    w = rmac_weights(q.device, q.dtype)
    mq, bq = _pca(q, dim, whiten_q)
    mr, br = _pca(r, dim, whiten_r)
    ms, bs = _shared(q, r, dim, whiten_q)
    qa = _anchor((q - mq) @ bq, w)
    ra = _anchor((r - mr) @ br, w)
    qs = _aggregate((q - ms) @ bs, w)
    rs = _aggregate((r - ms) @ bs, w)
    return {
        "query_mean": mq, "query_basis": bq,
        "reference_mean": mr, "reference_basis": br,
        "shared_mean": ms, "shared_basis": bs,
        "weights": w, "dim": dim,
        "whiten_q": whiten_q, "whiten_r": whiten_r,
        "qa": qa, "ra": ra, "qs": qs, "rs": rs,
    }


def fit_view(proj: dict) -> dict:
    return {k: proj[k] for k in ("qa", "ra", "qs", "rs")}


@torch.inference_mode()
def recompute_fit_view(proj: dict, q: torch.Tensor, r: torch.Tensor) -> dict:
    """Rebuild qa/ra/qs/rs from a saved projection and raw region features."""
    w = proj["weights"]
    qa = _anchor((q - proj["query_mean"]) @ proj["query_basis"], w)
    ra = _anchor((r - proj["reference_mean"]) @ proj["reference_basis"], w)
    qs = _aggregate((q - proj["shared_mean"]) @ proj["shared_basis"], w)
    rs = _aggregate((r - proj["shared_mean"]) @ proj["shared_basis"], w)
    return {"qa": qa, "ra": ra, "qs": qs, "rs": rs}


@torch.inference_mode()
def csls_sim(s: torch.Tensor, k: int) -> torch.Tensor:
    kq = min(k, s.shape[1] - 1)
    kr = min(k, s.shape[0] - 1)
    rq = s.topk(kq, dim=1).values.mean(1)
    rr = s.topk(kr, dim=0).values.mean(0)
    return s - 0.5 * rq[:, None] - 0.5 * rr[None, :]


@torch.inference_mode()
def sims(fv: dict, R: Optional[torch.Tensor] = None,
         k: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    if R is None:
        qv = F.normalize(fv["qs"], dim=1)
        rv = F.normalize(fv["rs"], dim=1)
    else:
        qv = F.normalize(fv["qa"] @ R, dim=1)
        rv = F.normalize(fv["ra"], dim=1)
    s = qv @ rv.T
    return s, csls_sim(s, k)


@torch.inference_mode()
def select_hard(inds: torch.Tensor, vals: torch.Tensor, cap: Optional[int]) -> tuple:
    nq = len(inds)
    qi = torch.arange(nq, device=vals.device)
    rj, sc = inds[:, 0], vals[:, 0]
    if cap is not None:
        sel = []
        for j in torch.unique(rj):
            z = torch.where(rj == j)[0]
            sel.append(z[sc[z].argsort(descending=True)[:min(cap, len(z))]])
        sel = torch.cat(sel)
        qi, rj, sc = qi[sel], rj[sel], sc[sel]
        order = torch.argsort(qi, stable=True)
        qi, rj, sc = qi[order], rj[order], sc[order]
    return qi, rj, sc


@torch.inference_mode()
def procrustes(qa: torch.Tensor, ra: torch.Tensor, qi: torch.Tensor,
               rj: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
    A = F.normalize(qa[qi], dim=1)
    B = F.normalize(ra[rj], dim=1)
    if w is not None:
        s = w.float().sqrt()[:, None]
        A, B = A * s, B * s
    u, _, vh = torch.linalg.svd(A.T @ B, full_matrices=False)
    return u @ vh


@torch.inference_mode()
def fit_op(fv: dict, cap: int, iters: int, sim_k: int = 100) -> dict:
    """Hard-assignment label-free iterative OP.

    ``iters`` counts OP updates after the initial shared-space discovery, so
    ``iters=4`` means five Procrustes solves including the initial one.  This
    matches the historical implementation and is recorded explicitly.
    """
    R = None
    hist = []
    pairs = None
    corr = None
    for it in range(int(iters) + 1):
        sim, corr = sims(fv, R, k=sim_k)
        inds = corr.argmax(1)[:, None]
        vals = corr.gather(1, inds)
        qi, rj, sc = select_hard(inds, vals, cap)
        w = torch.ones(len(qi), device=qi.device)
        R = procrustes(fv["qa"], fv["ra"], qi, rj, w)
        pairs = {"qi": qi, "rj": rj, "w": w, "score": sc}
        hist.append({
            "iter": it, "pairs": int(len(qi)),
            "orth_err": float((R.T @ R - torch.eye(R.shape[0], device=R.device)).abs().max()),
            "gallery_used": int(torch.unique(rj).numel()),
            "max_gallery_load": int(torch.bincount(rj, minlength=fv["ra"].shape[0]).max()),
        })
    return {"R": R, "hist": hist, "pairs": pairs, "corr": corr,
            "actual_op_solves": int(iters) + 1}


@torch.inference_mode()
def retrieval_vectors(proj: dict, q_regions: torch.Tensor, r_regions: torch.Tensor,
                      R: torch.Tensor, q_idx: Optional[torch.Tensor] = None) -> tuple:
    w = proj["weights"]
    q = q_regions if q_idx is None else q_regions[q_idx]
    r = r_regions
    qp = ((q.reshape(-1, q.shape[-1]) - proj["query_mean"]) @ proj["query_basis"]).reshape(
        q.shape[0], q.shape[1], -1)
    rp = ((r.reshape(-1, r.shape[-1]) - proj["reference_mean"]) @ proj["reference_basis"]).reshape(
        r.shape[0], r.shape[1], -1)
    qv = _aggregate(qp @ R, w)
    rv = _aggregate(rp, w)
    return qv, rv


# --------------------------------------------------------------------------- #
# Optional 600-step retrieval-oriented Cayley optimisation (no labels)
# --------------------------------------------------------------------------- #
def _cayley(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I - 0.5 * A, I + 0.5 * A)


class OrthogonalRotation(torch.nn.Module):
    def __init__(self, R0: torch.Tensor):
        super().__init__()
        self.register_buffer("R0", R0.detach().clone())
        self.P = torch.nn.Parameter(torch.zeros_like(R0))

    def R(self) -> torch.Tensor:
        return self.R0 @ _cayley(self.P - self.P.T)

    def orth_err(self) -> float:
        with torch.no_grad():
            R = self.R()
            return float((R.T @ R - torch.eye(R.shape[0], device=R.device)).abs().max())


def _project_regions(x: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return ((x.reshape(-1, x.shape[-1]) - mean) @ basis).reshape(x.shape[0], x.shape[1], -1)


def _margin_loss(qv, rv, pos, neg, neg_w, m=0.10, tau=0.05):
    s_p = (qv * rv[pos]).sum(1)
    s_n = (qv[:, None, :] * rv[neg]).sum(-1)
    per = F.softplus((s_n - s_p[:, None] + m) / tau)
    per = (per * neg_w).sum(1) / neg_w.sum(1).clamp_min(1e-6)
    return per.mean()


def _op_residual(qa, ra, pos, R, w=None):
    a = F.normalize(qa, dim=1)
    b = F.normalize(ra[pos], dim=1)
    r = (1 - (a @ R * b).sum(1)).clamp_min(0)
    if w is not None:
        return (r * w).sum() / w.sum().clamp_min(1e-6)
    return r.mean()


def optimize_rank_unlabeled(proj: dict, q_regions: torch.Tensor, r_regions: torch.Tensor,
                            R0: torch.Tensor, pairs: dict,
                            steps: int = 600, lr: float = 3e-4,
                            batch: int = 2048, kneg: int = 8, margin: float = 0.10,
                            tau: float = 0.05, align: float = 0.5,
                            seed: int = 0, log_every: int = 100) -> dict:
    """Cayley-parameterised ranking objective, with no evaluator or labels.

    The historical `v5rank.optimize_rank` called a label-reading evaluator
    every 200 steps even though it always finalised the 600-step rotation.
    This implementation removes that call entirely.
    """
    dev = R0.device
    fv = fit_view(proj)
    _, corr0 = sims(fv, R0, k=100)
    pos = corr0.argmax(1)
    top_excl = corr0.topk(min(6, corr0.shape[1]), dim=1).indices
    qp = _project_regions(q_regions, proj["query_mean"], proj["query_basis"])
    rp = _project_regions(r_regions, proj["reference_mean"], proj["reference_basis"])
    rd = _aggregate(rp, proj["weights"])
    rot = OrthogonalRotation(R0).to(dev)
    opt = torch.optim.Adam([rot.P], lr=lr)
    g = torch.Generator(device=dev).manual_seed(seed + 7)
    losses = []
    n = qp.shape[0]
    for step in range(int(steps)):
        sel = torch.randint(n, (batch,), generator=g, device=dev)
        qv = _aggregate(qp[sel] @ rot.R(), proj["weights"])
        s = qv @ rd.T
        p = pos[sel]
        s_p = s.gather(1, p[:, None]).squeeze(1)
        masked = s.clone()
        masked.scatter_(1, top_excl[sel], -2.0)
        neg = masked.topk(kneg, dim=1).indices
        s_n = s.gather(1, neg)
        neg_w = torch.sigmoid((s_p[:, None] - s_n) / tau)
        loss = _margin_loss(qv, rd, p, neg, neg_w, m=margin, tau=tau)
        if align > 0:
            n_pair = len(pairs["qi"])
            ai = torch.randint(n_pair, (min(4096, n_pair),), generator=g, device=dev)
            pw = pairs.get("w")
            if pw is None:
                pw = torch.ones(n_pair, device=dev)
            loss = loss + align * _op_residual(
                proj["qa"][pairs["qi"][ai]], proj["ra"], pairs["rj"][ai],
                rot.R(), w=pw[ai])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            losses.append({"step": step + 1, "loss": float(loss.detach())})
    Rfin = rot.R().detach()
    return {"R": Rfin, "R0": R0, "losses": losses, "orth_err": rot.orth_err(),
            "steps": int(steps), "align": float(align), "seed": int(seed),
            "evaluator_called_during_fit": False}