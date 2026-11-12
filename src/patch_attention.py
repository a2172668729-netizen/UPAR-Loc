#!/usr/bin/env python3
"""双向 Patch Cross-Attention 重排序模型（输入为冻结的 DINOv3 patch 特征）。

Preserves the parameter layout of the frozen Round 7 checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossBlock(nn.Module):
    def __init__(self, d=256, h=4, dropout=0.1, ffn=4):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn * d), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ffn * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ctx):
        kv = self.ln_kv(ctx)
        x = x + self.drop(self.attn(self.ln_q(x), kv, kv, need_weights=False)[0])
        x = x + self.drop(self.ff(self.ln_ff(x)))
        return x


class PatchCrossAttentionReranker(nn.Module):
    def __init__(self, d=256, h=4, layers=2, dropout=0.1, bidir=True):
        super().__init__()
        self.bidir = bidir
        self.q_proj = nn.Linear(768, d)
        self.g_proj = nn.Linear(768, d)
        self.q_blocks = nn.ModuleList([CrossBlock(d, h, dropout) for _ in range(layers)])
        self.g_blocks = nn.ModuleList([CrossBlock(d, h, dropout) for _ in range(layers)]) \
            if bidir else nn.ModuleList()
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d if bidir else d),
            nn.Linear(2 * d if bidir else d, d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, 1))

    @staticmethod
    def prep(p):
        p = p.float()
        p = p - p.mean(dim=1, keepdim=True)
        return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, qtok, gtok):
        q0 = self.q_proj(self.prep(qtok))
        g0 = self.g_proj(self.prep(gtok))
        q = q0
        for b in self.q_blocks:
            q = b(q, g0)
        pooled = [q.mean(1)]
        if self.bidir:
            g = g0
            for b in self.g_blocks:
                g = b(g, q0)
            pooled.append(g.mean(1))
        return self.head(torch.cat(pooled, dim=-1)).squeeze(-1)


# 历史代码的类名兼容入口，不影响已有 checkpoint 的参数键。
CAReranker = PatchCrossAttentionReranker
