"""Round 7 真实训练协议的可移植实现：源域训练、源域选择、目标域零标签使用。

训练不接触 University-1652 Test，不从旧 checkpoint 读取信息。
特征对齐阶段保留于 src.op_core，本文件只负责训练 B2。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.correct_reranking import fuse_reorder_ranked, stable_topk_candidates
from src.patch_attention import PatchCrossAttentionReranker
from src.raw_pipeline import image_ids
from src.research_pipeline import (PatchPairs, metrics_from_order, read_paths,
                                   set_seed, split_source_ids)


def _source_oracle_features(query_dir: Path, gallery_dir: Path,
                            qlabels: list[int], glabels: list[int],
                            qrows: np.ndarray, grows: np.ndarray,
                            upstream: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Round 4 B0.3: 在 *Train-Fit* 数据上拟合原论文监督 PCA-256/OP 来挖难负例。

    仅供源域监督训练阶段使用；与目标域无标签 V5 相互独立。
    """
    upstream = Path(upstream).resolve()
    if not (upstream / "vfm_loc" / "training_free.py").is_file():
        raise FileNotFoundError(f"缺少原论文公开源码：{upstream}")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from vfm_loc.training_free import fit_alignment_params, align_features
    q_all = np.load(query_dir / "regions.npy", mmap_mode="r", allow_pickle=False)
    g_all = np.load(gallery_dir / "regions.npy", mmap_mode="r", allow_pickle=False)
    q = torch.from_numpy(np.array(q_all[qrows], dtype=np.float32))
    g = torch.from_numpy(np.array(g_all[grows], dtype=np.float32))
    ql = torch.tensor([qlabels[i] for i in qrows], dtype=torch.long)
    gl = torch.tensor([glabels[i] for i in grows], dtype=torch.long)
    cfg = {"alignment": {"use_procrustes": True, "pca_dim": 256, "rmac_alpha": 6.0},
           "model": {"rmac_levels": [1, 2]}}
    with torch.no_grad():
        params = fit_alignment_params(q, g, ql, gl, cfg)
        qv, gv = align_features(q, g, ql, gl, cfg, pretrained=params)
    return qv.float().cpu(), gv.float().cpu()


def _patch_visual_groups(gallery_dir: Path) -> list[str]:
    """旧版 Round 7 的视觉组：对冻结的 FP16 Patch bytes 做 SHA-256。"""
    patches = np.load(gallery_dir / "patches.npy", mmap_mode="r", allow_pickle=False)
    return [hashlib.sha256(np.asarray(patches[i]).tobytes()).hexdigest()
            for i in range(len(patches))]


def prepare_pools(query_dir: Path, gallery_dir: Path,
                  qlabels: list[int], glabels: list[int],
                  split: dict, cfg: dict,
                  *, upstream: Path | None,
                  proxy_vectors: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict:
    """训练数据仅使用 Train-Fit 地点。生成固定 hard/random 候选池，逐轮重新取样。"""
    train_ids = set(split["train"])
    qrows = np.flatnonzero(np.isin(qlabels, list(train_ids)))
    grows = np.flatnonzero(np.isin(glabels, list(train_ids)))
    if len(qrows) == 0 or len(grows) < 3:
        raise ValueError("Train-Fit query 或 satellite gallery 不足")
    pos_by_label = {}
    for j, label in enumerate(glabels):
        if label in pos_by_label:
            raise ValueError(f"源域图库地点 ID 重复：{label}")
        pos_by_label[label] = j
    positives = np.asarray([pos_by_label[qlabels[int(i)]] for i in qrows], dtype=np.int64)
    visual_groups = _patch_visual_groups(gallery_dir)
    if cfg.get("negative_mining", "source_oracle") == "source_oracle":
        if upstream is None:
            raise ValueError("原实验使用 source-oracle PCA/OP 挖难负例，必须传入 --upstream")
        qv, gv = _source_oracle_features(query_dir, gallery_dir, qlabels, glabels,
                                         qrows, grows, upstream)
    elif cfg["negative_mining"] == "global_proxy_test_only":
        # 仅供没有真实 DINOv3 权重的 CPU 合成数据测试。不能作为论文复现路径。
        if proxy_vectors is None:
            raise ValueError("合成数据挖负例需要显式提供 proxy_vectors")
        qa, ga = proxy_vectors
        qv = qa[qrows]
        gv = ga[grows]
    else:
        raise ValueError(f"未知负例来源：{cfg['negative_mining']}")

    if len(qv) != len(qrows) or len(gv) != len(grows):
        raise AssertionError("源域负例检索特征的行号不一致")
    qv, gv = F.normalize(qv.float(), dim=-1), F.normalize(gv.float(), dim=-1)
    hard_pool, random_pool = [], []
    rng = np.random.RandomState(int(cfg.get("data_seed", 20261111)) + 777)
    random_pool_size = int(cfg.get("random_pool", 10))
    for start in range(0, len(qrows), 256):
        chunk = qrows[start:start + 256]
        sim = qv[start:start + len(chunk)] @ gv.T
        ranking = torch.argsort(sim, dim=1, descending=True, stable=True).numpy()
        for offset, qi in enumerate(chunk):
            qi = int(qi)
            label, positive = qlabels[qi], int(positives[start + offset])
            pg = visual_groups[positive]
            eligible = [int(g) for g in grows
                        if glabels[int(g)] != label and visual_groups[int(g)] != pg]
            if not eligible:
                raise ValueError(f"query={qi} 没有合规卫星负例")
            seen, hard = {pg}, []
            for local_g in ranking[offset]:
                g = int(grows[int(local_g)])
                if glabels[g] == label or visual_groups[g] in seen:
                    continue
                hard.append(g)
                seen.add(visual_groups[g])
                if len(hard) == 10:
                    break
            # 随机池只取 Train-Fit Gallery；按 visual group 不放回抽取。
            group_to_id = {}
            for g in eligible:
                group_to_id.setdefault(visual_groups[g], []).append(g)
            unique_groups = sorted(group_to_id)
            take = min(random_pool_size, len(unique_groups))
            sampled_groups = rng.choice(len(unique_groups), size=take, replace=False)
            rand = [int(rng.choice(group_to_id[unique_groups[int(z)]]))
                    for z in sampled_groups]
            hard_pool.append(hard)
            random_pool.append(rand)
    return {"query_indices": qrows, "positive": positives, "group": visual_groups,
            "hard_pool": hard_pool, "random_pool": random_pool,
            "train_gallery_count": len(grows),
            "hard_mining": cfg.get("negative_mining", "source_oracle")}


def sample_epoch(pools: dict, epoch: int, cfg: dict) -> np.ndarray:
    """Round 7: 3 hard + 2 random，每轮换负例，按视觉组去重。"""
    data_seed = int(cfg.get("data_seed", 20261111))
    seed = (data_seed * 100003 + int(epoch) * 97 + 1) % (2**31 - 1)
    rng = np.random.RandomState(seed)
    nh = int(cfg.get("hard_negatives", 3))
    nr = int(cfg.get("random_negatives", 2))
    groups = pools["group"]
    rows = []
    for pos, hp, rp in zip(pools["positive"], pools["hard_pool"], pools["random_pool"]):
        seen = {groups[int(pos)]}
        selected = []
        hard = list(hp)
        rand = list(rp)
        rng.shuffle(hard)
        rng.shuffle(rand)
        for g in hard:
            if groups[g] in seen:
                continue
            selected.append(int(g))
            seen.add(groups[g])
            if len(selected) == nh:
                break
        for g in rand:
            if groups[g] in seen:
                continue
            selected.append(int(g))
            seen.add(groups[g])
            if len(selected) == nh + nr:
                break
        if len(selected) != nh + nr:
            raise ValueError("负例视觉组不足，不能通过重复采样凑数")
        rows.append([int(pos), *selected])
    return np.asarray(rows, dtype=np.int64)


@torch.inference_mode()
def score_candidates(model: PatchCrossAttentionReranker, base: torch.Tensor,
                     qpatch: np.ndarray, gpatch: np.ndarray,
                     query_rows: list[int], topk: int, device: str,
                     batch: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """一次前向输出候选分数，供源域验证集搜索不同 lambda，不重复推理模型。"""
    model.eval()
    base_sub = base[query_rows].float().cpu()
    candidates = stable_topk_candidates(base_sub, min(topk, base_sub.shape[1]))
    out = []
    for start in range(0, len(query_rows), batch):
        rows = query_rows[start:start + batch]
        cand = candidates[start:start + len(rows)]
        q = torch.from_numpy(np.array(qpatch[rows], dtype=np.float32)).to(device)
        g = torch.from_numpy(np.array(gpatch[cand.numpy()], dtype=np.float32)).to(device)
        n, k = len(rows), cand.shape[1]
        scores = model(q[:, None].expand(-1, k, -1, -1).reshape(-1, *q.shape[1:]),
                       g.reshape(-1, *g.shape[-2:])).reshape(n, k)
        out.append(scores.detach().float().cpu())
    return base_sub, candidates, torch.cat(out, dim=0)


def choose_lambda(base: torch.Tensor, candidates: torch.Tensor,
                  pair: torch.Tensor, qlabels: list[int], glabels: list[int],
                  lambda_grid: list[float]) -> tuple[float, dict, dict]:
    """只在源域 Val-Selection 选择 lambda；R@1 > mAP > 小 lambda。"""
    curve = {}
    for lam in lambda_grid:
        _, order = fuse_reorder_ranked(base, candidates, pair, float(lam))
        curve[str(lam)] = metrics_from_order(order, qlabels, glabels)
    chosen = max(lambda_grid, key=lambda l: (
        curve[str(l)]["recall@1"], curve[str(l)]["mAP"], -float(l)))
    return float(chosen), curve[str(chosen)], curve


def train_source_reranker(dataset_root: Path, source_query: Path,
                          source_gallery: Path, descriptors_file: Path,
                          output: Path, config: dict, *, device: str,
                          upstream: Path | None = None) -> dict:
    """完全从源域重新训练，复刻 B2 的采样、BCE、Cosine 和源域选型协议。

    Test 图像/标签没有参与任何采样、拟合、checkpoint 或 lambda 选择。
    """
    from src.research_pipeline import read_paths  # 局部导入以兼容旧模块入口
    qpaths = read_paths(source_query)
    gpaths = read_paths(source_gallery)
    qlabels, glabels = image_ids(qpaths), image_ids(gpaths)
    tr = config["training"]
    split = split_source_ids(qlabels, seed=int(tr.get("split_seed", 20261110)),
                             selection_seed=int(tr.get("selection_seed", 20261111)))
    if len(set(split["train"]) & set(split["selection"])) or len(set(split["selection"]) & set(split["confirmation"])):
        raise AssertionError("源域地点拆分重叠")
    data = torch.load(descriptors_file, map_location="cpu", weights_only=True)
    if data["labels_read"] is not False or data["cfg"] != config["alignment"]:
        raise ValueError("源域全局描述符来源与本次无标签拟合配置不符")
    if (len(qpaths), len(gpaths)) != (data["query_count"], data["gallery_count"]):
        raise ValueError("源域特征和文件行数不符")
    if data["query_order_sha256"] != hashlib.sha256(json.dumps(qpaths).encode()).hexdigest():
        raise ValueError("源域无人机图像路径顺序不符")
    if data["gallery_order_sha256"] != hashlib.sha256(json.dumps(gpaths).encode()).hexdigest():
        raise ValueError("源域卫星图像路径顺序不符")
    qv, gv = data["qv"].float(), data["rv"].float()
    pools = prepare_pools(source_query, source_gallery, qlabels, glabels, split,
                          tr, upstream=upstream, proxy_vectors=(qv, gv))
    train_idx = pools["query_indices"]
    selection_ids = set(split["selection"])
    confirmation_ids = set(split["confirmation"])
    val_sel_rows = [i for i, label in enumerate(qlabels) if label in selection_ids]
    val_conf_rows = [i for i, label in enumerate(qlabels) if label in confirmation_ids]
    if not val_sel_rows or not val_conf_rows:
        raise ValueError("源域 Val-Selection 或 Val-Confirmation 为空")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_split.json").write_text(
        json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    qpatch = np.load(source_query / "patches.npy", mmap_mode="r", allow_pickle=False)
    gpatch = np.load(source_gallery / "patches.npy", mmap_mode="r", allow_pickle=False)
    base = qv @ gv.T
    seeds = list(map(int, tr.get("seeds", [tr.get("seed", 33)])))
    lambda_grid = list(map(float, config["reranking"].get(
        "lambda_grid", [config["reranking"].get("weight", 0.6)])))
    all_seed_reports = []
    global_best = (-float("inf"), -float("inf"), -float("inf"), -float("inf"), -float("inf"))
    winning_state = None
    for seed in seeds:
        set_seed(seed)
        model = PatchCrossAttentionReranker().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]),
                                      weight_decay=float(tr["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(tr["epochs"]))
        run_history = []
        for epoch in range(1, int(tr["epochs"]) + 1):
            model.train()
            choices = sample_epoch(pools, epoch, tr)
            rng = np.random.RandomState(int(tr.get("data_seed", 20261111)) + epoch * 1009)
            perm = rng.permutation(len(train_idx))
            loader = DataLoader(
                PatchPairs(source_query / "patches.npy", source_gallery / "patches.npy",
                           train_idx[perm], choices[perm]),
                batch_size=int(tr["batch_size"]), shuffle=False,
                num_workers=int(tr["workers"]), pin_memory=device.startswith("cuda"))
            loss_sum = 0.0
            batches = 0
            for q, imgs in loader:
                q = q.to(device, non_blocking=True)
                imgs = imgs.to(device, non_blocking=True)
                b, nopt = imgs.shape[:2]
                lp = model(q, imgs[:, 0])
                neg = imgs[:, 1:]
                lg = model(q[:, None].expand(-1, nopt - 1, -1, -1).reshape(
                    -1, *q.shape[1:]), neg.reshape(-1, *neg.shape[-2:]))
                logits = torch.cat((lp, lg))
                target = torch.cat((torch.ones_like(lp), torch.zeros_like(lg)))
                loss = F.binary_cross_entropy_with_logits(logits, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"B2 训练 loss 非有限数：seed={seed} epoch={epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               float(tr["grad_clip"]),
                                               error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach())
                batches += 1
            scheduler.step()
            base_sel, cand, pair = score_candidates(
                model, base, qpatch, gpatch, val_sel_rows,
                int(config["reranking"]["topk"]), device)
            lam, metrics, curve = choose_lambda(
                base_sel, cand, pair, [qlabels[i] for i in val_sel_rows],
                glabels, lambda_grid)
            record = {"seed": seed, "epoch": epoch, "train_loss": loss_sum / max(1, batches),
                      "selected_lambda": lam, "selection": metrics, "lambda_curve": curve}
            run_history.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "lambda_curve"},
                             ensure_ascii=False), flush=True)
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # 路径按 seed/epoch 隔离；不存在旧资产回填。
            torch.save({"state_dict": state, "seed": seed, "epoch": epoch,
                        "selected_lambda": lam, "selection": metrics,
                        "training_config": tr}, output / f"seed{seed}_epoch{epoch:02d}.pt")
            candidate = (metrics["recall@1"], metrics["mAP"],
                         -lam, -epoch, -seed)
            if candidate > global_best:
                global_best = candidate
                winning_state = {"state_dict": state, "seed": seed, "epoch": epoch,
                                 "selected_lambda": lam, "selection": metrics,
                                 "training_config": tr, "split": split,
                                 "lambda_grid": lambda_grid}
        all_seed_reports.append({"seed": seed, "history": run_history})
    if winning_state is None:
        raise RuntimeError("没有选出任何 B2 checkpoint")
    torch.save(winning_state, output / "best.pt")
    # Confirmation 只在 checkpoint/seed/lambda 均已冻结后调用一次。
    model = PatchCrossAttentionReranker().to(device)
    model.load_state_dict(winning_state["state_dict"])
    base_conf, c_conf, p_conf = score_candidates(
        model, base, qpatch, gpatch, val_conf_rows,
        int(config["reranking"]["topk"]), device)
    _, order = fuse_reorder_ranked(base_conf, c_conf, p_conf,
                                   float(winning_state["selected_lambda"]))
    confirmed = metrics_from_order(order, [qlabels[i] for i in val_conf_rows], glabels)
    report = {
        "best_epoch": winning_state["epoch"], "seed": winning_state["seed"],
        "selected_lambda": winning_state["selected_lambda"],
        "selection": winning_state["selection"], "confirmation": confirmed,
        "history": all_seed_reports, "train_queries": int(len(train_idx)),
        "selection_queries": len(val_sel_rows),
        "confirmation_queries": len(val_conf_rows),
        "negative_mining": pools["hard_mining"],
        "train_gallery_count": pools["train_gallery_count"],
        "protocol": "Source Train supervision only; target Test never used for selection",
    }
    (output / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
