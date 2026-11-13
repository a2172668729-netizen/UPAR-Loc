"""源域训练 -> checkpoint 保存 -> 目标域评测的 CPU 集成检查。

本测试使用人工数据与短 Patch 序列，以验证真实 PatchCrossAttentionReranker
的训练/推理接口；不代表原始图像 DINOv3 特征提取已经通过 GPU 验收。
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.research_pipeline import train_reranker, evaluate_target


def build_group(data_root, cache, split, view, ids, seed):
    cache.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i, label in enumerate(ids):
        name = f"{split}/{view}/{label:04d}/{i}.png"
        path = data_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2),
                  ((i * 13) % 256, (i * 41) % 256, (i * 89) % 256)).save(path)
        paths.append(name)
    # 网络可接收任意 Patch 序列长度；使用 2 而非 196 可快速检查真实网络的可微训练。
    np.save(cache / "patches.npy", rng.normal(size=(len(ids), 2, 768)).astype(np.float16))
    (cache / "image_paths.json").write_text(json.dumps(paths), encoding="utf8")
    vec = torch.from_numpy(rng.normal(size=(len(ids), 8)).astype(np.float32))
    return paths, torch.nn.functional.normalize(vec, dim=1)


def save_descriptors(path, qp, gp, qv, gv, align):
    torch.save({
        "qv": qv, "rv": gv, "cfg": align, "labels_read": False,
        "query_count": len(qp), "gallery_count": len(gp),
        "query_order_sha256": hashlib.sha256(json.dumps(qp).encode()).hexdigest(),
        "gallery_order_sha256": hashlib.sha256(json.dumps(gp).encode()).hexdigest(),
    }, path)


class ReviewerSmokeTest(unittest.TestCase):
    def test_retrain_and_reload_into_target_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "University-1652"
            features = root / "features"
            ids = list(range(1001, 1009))
            qpaths, qv = build_group(data, features / "source_query", "train", "drone", ids, 1)
            gpaths, gv = build_group(data, features / "source_gallery", "train", "satellite", ids, 2)
            align = {"pca_dim": 8, "seed": 0, "rank_steps": 1}
            cfg = {
                "alignment": align,
                "training": {
                    "split_seed": 20261110, "selection_seed": 20261111,
                    "data_seed": 20261111, "seed": 33,
                    "negative_mining": "global_proxy_test_only",
                    "hard_negatives": 1, "random_negatives": 1, "random_pool": 3,
                    "batch_size": 2, "workers": 0, "lr": 1e-4,
                    "weight_decay": 0.0, "grad_clip": 1.0, "epochs": 1
                },
                "reranking": {"topk": 3, "weight": 0.6}
            }
            save_descriptors(root / "source.pt", qpaths, gpaths, qv, gv, align)
            report = train_reranker(
                data, features / "source_query", features / "source_gallery",
                root / "source.pt", root / "reranker", cfg, device="cpu")
            self.assertEqual(report["best_epoch"], 1)
            self.assertTrue((root / "reranker" / "best.pt").is_file())

            tq, tv = build_group(data, features / "target_query", "test", "query_drone",
                                 list(range(2001, 2004)), 3)
            tg, gv = build_group(data, features / "target_gallery", "test", "gallery_satellite",
                                 list(range(2001, 2005)), 4)
            save_descriptors(root / "target.pt", tq, tg, tv, gv, align)
            result = evaluate_target(
                features / "target_query", features / "target_gallery",
                root / "target.pt", root / "reranker" / "best.pt",
                root / "final_metrics.json", cfg, device="cpu")
            self.assertEqual(result["query_count"], 3)
            self.assertFalse(result["target_test_labels_used_for_training"])
            self.assertIn("recall@1", result["metrics"]["patch_reranking"])
            self.assertIn("mAP", result["metrics"]["global_retrieval"])


if __name__ == "__main__":
    unittest.main()
