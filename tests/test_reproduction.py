"""不需要数据集、DINOv3 权重或 GPU 的最小集成测试。"""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from src.raw_pipeline import image_list, image_ids, _regions_from_tokens
from src.research_pipeline import (infer_reranking, metrics_from_order,
                                   mine_negatives, split_source_ids)


class _ConstantReranker(torch.nn.Module):
    def forward(self, q, g):
        # 用图库 Patch 的首元素模拟可学习匹配得分。
        return g[:, 0, 0]


class FromScratchTests(unittest.TestCase):
    def test_university_layout_and_id_parse(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for label, image in [("0002", "b.jpg"), ("0001", "a.png")]:
                p = root / "train" / "drone" / label / image
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"not decoded in this test")
            paths = image_list(root / "train" / "drone")
            self.assertEqual(image_ids([str(p) for p in paths]), [1, 2])

    def test_source_id_split_is_disjoint_and_stable(self):
        a = split_source_ids(list(range(701)))
        b = split_source_ids(list(range(701)))
        self.assertEqual(a, b)
        self.assertEqual(list(map(len, a.values())), [561, 70, 70])
        self.assertFalse(set(a["train"]) & set(a["selection"]))
        self.assertFalse(set(a["train"]) & set(a["confirmation"]))
        self.assertFalse(set(a["selection"]) & set(a["confirmation"]))

    def test_hard_negatives_exclude_positive_and_visual_duplicates(self):
        # gallery #2 与正例 #0 逐像素相同；必须跳过，且负例视觉组不重复。
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.99, 0.01],
                                [0.7, 0.3], [0.2, 0.8], [0.0, 1.0]])
        ids, neg = mine_negatives(query, gallery, [10], [10, 11, 12, 13, 14, 15],
                                  ["a", "b", "a", "c", "d", "e"], {10}, count=3)
        self.assertEqual(ids.tolist(), [0])
        self.assertEqual(neg.shape, (1, 4))
        self.assertEqual(neg[0, 0], 0)
        self.assertNotIn(2, neg[0, 1:].tolist())
        self.assertEqual(len(set(neg[0].tolist())), 4)

    def test_region_extraction_matches_five_region_layout(self):
        p = torch.arange(196, dtype=torch.float32)[None, :, None].expand(2, -1, 768)
        regions = _regions_from_tokens(p)
        self.assertEqual(regions.shape, (2, 5, 768))
        self.assertTrue(torch.isfinite(regions).all())

    def test_unlabeled_alignment_from_synthetic_features(self):
        from src.research_pipeline import fit_alignment
        with TemporaryDirectory() as td:
            root = Path(td)
            for name, n in (("query", 12), ("gallery", 6)):
                folder = root / name
                folder.mkdir()
                rng = np.random.default_rng(42 + n)
                np.save(folder / "regions.npy",
                        rng.normal(size=(n, 5, 768)).astype(np.float32))
                np.save(folder / "patches.npy",
                        np.zeros((n, 196, 768), dtype=np.float16))
                (folder / "image_paths.json").write_text(
                    json.dumps([f"train/{name}/{i:04d}/a.jpg" for i in range(n)]))
            config = {"alignment": {
                "seed": 0, "pca_dim": 4,
                "whiten_query": 0.2, "whiten_gallery": 0.1,
                "cap": 3, "op_updates": 1, "csls_k": 3,
                "rank_steps": 0
            }}
            result = fit_alignment(root / "query", root / "gallery",
                                   root / "descriptors.pt", config, device="cpu")
            saved = torch.load(root / "descriptors.pt", weights_only=True)
            self.assertFalse(saved["labels_read"])
            self.assertEqual(saved["qv"].shape, (12, 4))
            self.assertEqual(saved["rv"].shape, (6, 4))
            self.assertEqual(result["op_solves"], 2)

    def test_tie_aware_inference_keeps_intended_candidate_order(self):
        # 同分候选 0/1 的分数相同；patch 得分应使候选 1 排到首位。
        base = torch.tensor([[3.0, 3.0, 1.0, 0.0]])
        query = np.zeros((1, 196, 768), dtype=np.float16)
        gallery = np.zeros((4, 196, 768), dtype=np.float16)
        gallery[1, 0, 0] = 5.0
        ranking = infer_reranking(_ConstantReranker(), base, query, gallery,
                                  device="cpu", topk=3, fusion_weight=2.0,
                                  query_indices=[0], query_batch=1)
        self.assertEqual(ranking.tolist(), [[1, 0, 2, 3]])
        metrics = metrics_from_order(ranking, [9], [7, 9, 7, 11])
        self.assertEqual(metrics["recall@1"], 100.0)
        self.assertEqual(metrics["mAP"], 100.0)


if __name__ == "__main__":
    unittest.main()
