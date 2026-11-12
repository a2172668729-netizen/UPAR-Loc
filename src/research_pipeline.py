"""从原始特征进行无标签全局对齐、源域有监督重排序训练及目标域评测。"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.correct_reranking import fuse_reorder_ranked, stable_topk_candidates
from src.op_core import (build_projection, fit_op, fit_view, optimize_rank_unlabeled,
                         retrieval_vectors)
from src.patch_attention import PatchCrossAttentionReranker
from src.raw_pipeline import image_ids, load_arrays


def read_paths(folder: Path) -> list[str]:
    return json.loads((Path(folder) / "image_paths.json").read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("alignment", "training", "reranking")
    for key in required:
        if key not in data:
            raise KeyError(f"缺少配置节：{key}")
    return data


def fit_alignment(query_folder: Path, gallery_folder: Path, output: Path,
                  config: dict, device: str = "cuda") -> dict:
    """PCA、伪配对 OP 和 Cayley 排序优化；本函数不加载任何标签。"""
    set_seed(int(config["alignment"]["seed"]))
    q, _, qpaths = load_arrays(query_folder, device=device)
    g, _, gpaths = load_arrays(gallery_folder, device=device)
    cfg = config["alignment"]
    projection = build_projection(q, g, dim=int(cfg["pca_dim"]),
                                  whiten_q=float(cfg["whiten_query"]),
                                  whiten_r=float(cfg["whiten_gallery"]))
    op = fit_op(fit_view(projection), cap=int(cfg["cap"]),
                iters=int(cfg["op_updates"]), sim_k=int(cfg["csls_k"]))
    if int(cfg["rank_steps"]):
        tuned = optimize_rank_unlabeled(
            projection, q, g, op["R"], op["pairs"],
            steps=int(cfg["rank_steps"]), lr=float(cfg["rank_lr"]),
            batch=int(cfg["rank_batch"]), kneg=int(cfg["rank_negatives"]),
            align=float(cfg["rank_align"]), seed=int(cfg["seed"]),
        )
        rotation = tuned["R"]
    else:
        tuned = None
        rotation = op["R"]
    qv, rv = retrieval_vectors(projection, q, g, rotation)
    if not torch.isfinite(qv).all() or not torch.isfinite(rv).all():
        raise ValueError("无标签拟合得到非有限数值")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "qv": qv.detach().float().cpu(),
        "rv": rv.detach().float().cpu(),
        "cfg": cfg, "labels_read": False,
        "query_count": len(qpaths), "gallery_count": len(gpaths),
        "query_order_sha256": hashlib.sha256(json.dumps(qpaths).encode()).hexdigest(),
        "gallery_order_sha256": hashlib.sha256(json.dumps(gpaths).encode()).hexdigest(),
        "rotation": rotation.detach().float().cpu(),
    }
    torch.save(payload, output)
    report = {
        "query_count": len(qpaths), "gallery_count": len(gpaths),
        "op_solves": op["actual_op_solves"],
        "op_history": op["hist"],
        "orthogonal_error": float((rotation.T @ rotation -
                                   torch.eye(len(rotation), device=rotation.device)).abs().max()),
        "rank_steps": 0 if tuned is None else tuned["steps"],
        "test_labels_read": False,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def split_source_ids(labels: list[int], seed: int = 20261111) -> dict[str, list[int]]:
    unique = np.array(sorted(set(labels)), dtype=np.int64)
    if len(unique) < 3:
        raise ValueError("训练地点不足，无法创建互斥拆分")
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    if len(unique) == 701:
        train_n, val_n = 561, 70
    else:
        train_n = max(1, int(len(unique) * .8))
        val_n = max(1, int(len(unique) * .1))
        if train_n + val_n >= len(unique):
            train_n = len(unique) - 2
            val_n = 1
    return {
        "train": sorted(map(int, unique[:train_n])),
        "selection": sorted(map(int, unique[train_n:train_n + val_n])),
        "confirmation": sorted(map(int, unique[train_n + val_n:])),
    }


def pixel_digest(path: Path) -> str:
    """对解码后的 RGB 像素做哈希，排除不同 ID 的逐像素重复卫星图。"""
    from PIL import Image
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb)
        h = hashlib.sha256()
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
        return h.hexdigest()


def mine_negatives(query_vectors: torch.Tensor, gallery_vectors: torch.Tensor,
                   query_labels: list[int], gallery_labels: list[int],
                   gallery_digest: list[str], train_ids: set[int],
                   *, count: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """为 Train-Fit 的每条查询选 1 正例 + count 个不同视觉组的困难负例。

    gallery 可以含非 Train-Fit ID 的图片，作为负例；不能含与正例相同
    地点或相同视觉组的图片。没有足够合规负例就报错，不用重复候选填充。
    """
    if len(gallery_labels) != len(gallery_digest):
        raise ValueError("gallery 标签与视觉哈希数量不符")
    idx_by_id = {}
    for j, gid in enumerate(gallery_labels):
        if gid in idx_by_id:
            raise ValueError("要求源域卫星图每地点恰有一张；发现重复地点")
        idx_by_id[gid] = j
    indices = [i for i, label in enumerate(query_labels) if label in train_ids]
    if not indices:
        raise ValueError("Train-Fit 查询为空")
    q = F.normalize(query_vectors.float(), dim=1).cpu()
    g = F.normalize(gallery_vectors.float(), dim=1).cpu()
    choices = []
    gallery_ids = np.asarray(gallery_labels)
    for start in range(0, len(indices), 256):
        part = indices[start:start + 256]
        sim = q[part] @ g.T
        # CPU 稳定排序：相同分数时图库 index 升序
        order = torch.argsort(sim, dim=1, descending=True, stable=True).numpy()
        for row, qi in enumerate(part):
            label = query_labels[qi]
            if label not in idx_by_id:
                raise ValueError(f"查询地点 {label} 无正例卫星图")
            positive = idx_by_id[label]
            positive_digest = gallery_digest[positive]
            seen = {positive_digest}
            negatives = []
            for j in order[row]:
                j = int(j)
                if gallery_ids[j] == label or gallery_digest[j] in seen:
                    continue
                negatives.append(j)
                seen.add(gallery_digest[j])
                if len(negatives) == count:
                    break
            if len(negatives) != count:
                raise ValueError(f"查询 {qi} 找不到 {count} 个不同视觉组的负例")
            choices.append([positive, *negatives])
    return np.asarray(indices, dtype=np.int64), np.asarray(choices, dtype=np.int64)


class PatchPairs(Dataset):
    def __init__(self, qpatch_path: Path, gpatch_path: Path,
                 query_indices: np.ndarray, candidates: np.ndarray):
        self.qpath = str(qpatch_path)
        self.gpath = str(gpatch_path)
        self.queries = query_indices
        self.candidates = candidates
        self._q = None
        self._g = None

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        # 工作进程分别打开磁盘映射，避免复制全部特征。
        if self._q is None:
            self._q = np.load(self.qpath, mmap_mode="r", allow_pickle=False)
            self._g = np.load(self.gpath, mmap_mode="r", allow_pickle=False)
        q = np.array(self._q[int(self.queries[index])], dtype=np.float32)
        g = np.array(self._g[self.candidates[index]], dtype=np.float32)
        return torch.from_numpy(q), torch.from_numpy(g)


def metrics_from_order(order: torch.Tensor, qlabels: list[int],
                       glabels: list[int]) -> dict[str, float]:
    from evaluate_retrieval import evaluate
    q = torch.tensor(qlabels, dtype=torch.long)
    g = torch.tensor(glabels, dtype=torch.long)
    # evaluate() 的 scores 不参与计算（已传显式排名），使用空矩阵即可。
    return evaluate(torch.zeros(order.shape, dtype=torch.float32), q, g, order=order)


@torch.inference_mode()
def infer_reranking(model: PatchCrossAttentionReranker, base: torch.Tensor,
                    query_patch: np.ndarray, gallery_patch: np.ndarray,
                    *, device: str, topk: int, fusion_weight: float,
                    query_indices: list[int] | None = None,
                    query_batch: int = 4) -> torch.Tensor:
    """根据局部模型生成完整图库排名，显式处理相同分数。"""
    model.eval()
    if query_indices is None:
        query_indices = list(range(base.shape[0]))
    results = []
    for start in range(0, len(query_indices), query_batch):
        ids = query_indices[start:start + query_batch]
        scores = base[ids].to(device=device, dtype=torch.float32)
        candidates = stable_topk_candidates(scores, min(topk, scores.shape[1]))
        k = candidates.shape[1]
        q = torch.from_numpy(np.array(query_patch[ids], dtype=np.float32)).to(device)
        g = torch.from_numpy(np.array(gallery_patch[candidates.cpu().numpy()], dtype=np.float32)).to(device)
        n = len(ids)
        with torch.autocast("cuda", enabled=device.startswith("cuda")):
            pair = model(
                q[:, None].expand(-1, k, -1, -1).reshape(-1, *q.shape[1:]),
                g.reshape(-1, *g.shape[-2:]),
            ).float().reshape(n, k)
        _, order = fuse_reorder_ranked(scores, candidates, pair, fusion_weight)
        results.append(order.cpu())
    return torch.cat(results)


def train_reranker(dataset_root: Path, source_query: Path, source_gallery: Path,
                   descriptors_file: Path, output: Path,
                   config: dict, device: str = "cuda") -> dict:
    """只读取 source Train 的地点标签，Val-Selection 用于 checkpoint 选择。"""
    dataset_root = Path(dataset_root)
    qpaths = read_paths(source_query)
    gpaths = read_paths(source_gallery)
    qlabels, glabels = image_ids(qpaths), image_ids(gpaths)
    split = split_source_ids(qlabels, seed=int(config["training"]["split_seed"]))
    if set(split["train"]) & set(split["selection"]) or set(split["selection"]) & set(split["confirmation"]):
        raise AssertionError("源域地点拆分出现重叠")
    data = torch.load(descriptors_file, map_location="cpu", weights_only=True)
    if data["labels_read"] is not False:
        raise ValueError("无标签拟合资产的来源标志异常")
    if data["cfg"] != config["alignment"]:
        raise ValueError("源域全局描述符的拟合参数与本次配置不同，请重跑 align")
    if data["query_count"] != len(qpaths) or data["gallery_count"] != len(gpaths):
        raise ValueError("全局描述符与源域图片顺序不匹配")
    if data["query_order_sha256"] != hashlib.sha256(json.dumps(qpaths).encode()).hexdigest():
        raise ValueError("查询图片顺序哈希不匹配")
    if data["gallery_order_sha256"] != hashlib.sha256(json.dumps(gpaths).encode()).hexdigest():
        raise ValueError("图库图片顺序哈希不匹配")
    qv, gv = data["qv"], data["rv"]
    digest = [pixel_digest(dataset_root / p) for p in gpaths]
    tr = config["training"]
    set_seed(int(tr["seed"]))
    qidx, neg = mine_negatives(qv, gv, qlabels, glabels, digest, set(split["train"]),
                               count=int(tr["negatives"]))
    positives = [i for i in range(len(qlabels)) if qlabels[i] in set(split["selection"])]
    confirms = [i for i in range(len(qlabels)) if qlabels[i] in set(split["confirmation"])]
    if not positives or not confirms:
        raise ValueError("源域 Selection 或 Confirmation 为空")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    loader = DataLoader(
        PatchPairs(source_query / "patches.npy", source_gallery / "patches.npy", qidx, neg),
        batch_size=int(tr["batch_size"]), shuffle=True,
        num_workers=int(tr["workers"]), pin_memory=device.startswith("cuda"),
    )
    qpatch = np.load(source_query / "patches.npy", mmap_mode="r", allow_pickle=False)
    gpatch = np.load(source_gallery / "patches.npy", mmap_mode="r", allow_pickle=False)
    model = PatchCrossAttentionReranker().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]),
                                  weight_decay=float(tr["weight_decay"]))
    amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    source_base = qv @ gv.T
    history = []
    best_score = (-float("inf"), -float("inf"))
    best_epoch = 0
    output_ckpt = output / "best.pt"
    for epoch in range(1, int(tr["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for q, g in loader:
            q = q.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)
            b, options = g.shape[:2]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                logits = model(
                    q[:, None].expand(-1, options, -1, -1).reshape(-1, *q.shape[1:]),
                    g.reshape(-1, *g.shape[-2:]),
                ).reshape(b, options)
                loss = F.cross_entropy(logits.float(), torch.zeros(b, dtype=torch.long, device=device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * b
            seen += b
        val_order = infer_reranking(model, source_base, qpatch, gpatch, device=device,
                    topk=int(config["reranking"]["topk"]),
                    fusion_weight=float(config["reranking"]["weight"]),
                    query_indices=positives)
        metrics = metrics_from_order(val_order, [qlabels[i] for i in positives], glabels)
        record = {"epoch": epoch, "train_loss": total_loss / max(1, seen),
                  "selection_r1": metrics["recall@1"], "selection_map": metrics["mAP"]}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        key = (metrics["recall@1"], metrics["mAP"])
        if key > best_score:
            best_score, best_epoch = key, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "seed": int(tr["seed"]), "training_config": tr,
                        "selection": metrics, "split": split,
                        "fusion_weight": float(config["reranking"]["weight"])}, output_ckpt)
    best = torch.load(output_ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(best["state_dict"])
    confirmed_order = infer_reranking(model, source_base, qpatch, gpatch, device=device,
                 topk=int(config["reranking"]["topk"]),
                 fusion_weight=float(config["reranking"]["weight"]),
                 query_indices=confirms)
    confirmed = metrics_from_order(confirmed_order, [qlabels[i] for i in confirms], glabels)
    report = {"best_epoch": best_epoch, "selection": best["selection"],
              "confirmation": confirmed, "history": history,
              "train_queries": len(qidx), "selection_queries": len(positives),
              "confirmation_queries": len(confirms),
              "model": "源域监督训练的双向 Patch Cross-Attention"}
    (output / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    return report


def evaluate_target(target_query: Path, target_gallery: Path, descriptors: Path,
                    checkpoint: Path, output: Path,
                    config: dict, device: str = "cuda") -> dict:
    """只在最终排名全部生成后读取 Test 的地点标签。不得用于重新选型。"""
    qpaths = read_paths(target_query)
    gpaths = read_paths(target_gallery)
    data = torch.load(descriptors, map_location="cpu", weights_only=True)
    if data["cfg"] != config["alignment"] or data["labels_read"] is not False:
        raise ValueError("目标域全局描述符与当前无标签对齐配置不一致")
    if data["query_count"] != len(qpaths) or data["gallery_count"] != len(gpaths):
        raise ValueError("目标域描述符与输入目录数量不符")
    if data["query_order_sha256"] != hashlib.sha256(json.dumps(qpaths).encode()).hexdigest():
        raise ValueError("目标域查询顺序不符")
    if data["gallery_order_sha256"] != hashlib.sha256(json.dumps(gpaths).encode()).hexdigest():
        raise ValueError("目标域图库顺序不符")
    base = data["qv"] @ data["rv"].T
    model = PatchCrossAttentionReranker().to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if float(state["fusion_weight"]) != float(config["reranking"]["weight"]):
        raise ValueError("源域选型时的融合权重与本次评测配置不同")
    model.load_state_dict(state["state_dict"])
    model.eval()
    qpatch = np.load(target_query / "patches.npy", mmap_mode="r", allow_pickle=False)
    gpatch = np.load(target_gallery / "patches.npy", mmap_mode="r", allow_pickle=False)
    global_order = torch.argsort(base, dim=1, descending=True, stable=True)
    joint_order = infer_reranking(model, base, qpatch, gpatch, device=device,
                                  topk=int(config["reranking"]["topk"]),
                                  fusion_weight=float(config["reranking"]["weight"]))
    # 标签只在两种排名均已生成后打开（metadata 中不保存测试标签）。
    qlabels, glabels = image_ids(qpaths), image_ids(gpaths)
    scores = {
        "global_retrieval": metrics_from_order(global_order, qlabels, glabels),
        "patch_reranking": metrics_from_order(joint_order, qlabels, glabels),
    }
    report = {
        "evaluation": "历史已暴露 University-1652 Test 兼容评测（不是盲测）",
        "query_count": len(qlabels), "gallery_count": len(glabels),
        "model_epoch": state["epoch"], "seed": state["seed"],
        "metrics": scores,
        "target_test_labels_used_for_training": False,
        "warning": "此端到端流程不能冒称已独立重现历史 90.2681%，需在真实资产上实测。",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
