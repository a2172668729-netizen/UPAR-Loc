"""Source Train-Fit 难负例挖掘专用的有标签 PCA/OP。

原 VFM-Loc 的 PCA-256 + 正交 Procrustes 是公开的数学方法。
这里独立实现同一算法协议，不复制无许可证的 VFM-Loc 源文件。
只允许由训练器调用，禁止用于目标域无标签对齐或官方 Test。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _pca(x: torch.Tensor, dimensions: int) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = x.reshape(-1, x.shape[-1]).float()
    mean = vectors.mean(dim=0)
    centered = vectors - mean
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    # eigh 对称矩阵的输出特征值按升序排列，逆序取最大特征值主成分。
    order = torch.argsort(eigenvalues, descending=True)
    return mean, eigenvectors[:, order[:dimensions]]


@torch.no_grad()
def source_train_vectors(
    drone_regions: torch.Tensor,
    satellite_regions: torch.Tensor,
    drone_ids: torch.Tensor,
    satellite_ids: torch.Tensor,
    *, pca_dim: int = 256, rmac_alpha: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """仅拟合源域 Train-Fit 地点，供 3 Hard + 2 Random 样本挖掘。

    输入原始五区域 DINOv3 FP32 描述符；输出 L2 单位向量，
    与原 Round 4 B0.3 的「分别 PCA + 真实源域 ID OP」一致。
    """
    if drone_regions.ndim != 3 or satellite_regions.ndim != 3:
        raise ValueError("输入必须为 [N, 5, 768] 区域特征")
    if drone_regions.shape[1:] != (5, 768) or satellite_regions.shape[1:] != (5, 768):
        raise ValueError("源域 PCA/OP 仅支持 5×768 DINOv3 区域特征")
    if drone_regions.shape[0] != drone_ids.numel() or satellite_regions.shape[0] != satellite_ids.numel():
        raise ValueError("源域特征与地点 ID 数量不匹配")
    if pca_dim < 1 or pca_dim > 768:
        raise ValueError(f"非法 PCA 维度：{pca_dim}")

    drone_regions = drone_regions.float()
    satellite_regions = satellite_regions.float()
    qmean, qb = _pca(drone_regions, pca_dim)
    rmean, rb = _pca(satellite_regions, pca_dim)
    qp = (drone_regions - qmean) @ qb
    rp = (satellite_regions - rmean) @ rb

    # 五区域 [全图, 四个子区域]；Round 4 rmac_alpha=6。
    w = qp.new_tensor([1.0] + [1.0 / (2.0 ** rmac_alpha)] * 4).reshape(1, 5, 1)
    qa = (qp * w).sum(1) / w.sum()
    ra = (rp * w).sum(1) / w.sum()

    # 对源域已知 ID 做地点均值配对，仅此步骤允许源域监督。
    unique = sorted(set(map(int, drone_ids.tolist())) & set(map(int, satellite_ids.tolist())))
    if len(unique) < 2:
        raise ValueError("Train-Fit 没有足够多的跨视角已知同地点配对")
    qanchors, ranchors = [], []
    for label in unique:
        qmask = drone_ids == label
        rmask = satellite_ids == label
        qanchors.append(qa[qmask].mean(0))
        ranchors.append(ra[rmask].mean(0))
    a = F.normalize(torch.stack(qanchors), dim=-1)
    b = F.normalize(torch.stack(ranchors), dim=-1)
    u, _, vh = torch.linalg.svd(a.T @ b, full_matrices=False)
    rotation = u @ vh

    # 原 VFM-Loc 的区域融合规则：逐区域归一化，再按 R-MAC 权重求和。
    qv = F.normalize((F.normalize(qp @ rotation, dim=-1) * w).sum(1) / w.sum(), dim=-1)
    rv = F.normalize((F.normalize(rp, dim=-1) * w).sum(1) / w.sum(), dim=-1)
    if not torch.isfinite(qv).all() or not torch.isfinite(rv).all():
        raise ValueError("源域 PCA/OP 含非有限数值")
    return qv, rv
