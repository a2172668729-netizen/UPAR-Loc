"""数据集与完整 DINOv3 权重之外没有 VFM-Loc 运行时依赖的最小自检。

仅提供测试代码；本次仓库整合阶段未执行。
"""
import unittest
from pathlib import Path

import torch

from src.dinov3_backbone import _import_bundled_architecture
from src.source_oracle import source_train_vectors


class SelfContainedTests(unittest.TestCase):
    def test_bundled_license_and_model_present(self):
        third_party = Path(__file__).resolve().parents[1] / "third_party" / "dinov3"
        self.assertTrue((third_party / "LICENSE.md").is_file())
        self.assertTrue((third_party / "dinov3/models/vision_transformer.py").is_file())

    def test_bundled_tiny_vit_forward(self):
        # 用小型形状检测打包的 Meta 模块是否完整，不下载任何模型权重。
        backbone = _import_bundled_architecture()
        model = backbone(
            img_size=32, patch_size=16, in_chans=3,
            embed_dim=48, depth=1, num_heads=3, ffn_ratio=4.0,
            norm_layer="layernormbf16", pos_embed_rope_dtype="fp32",
            pos_embed_rope_rescale_coords=2.0, n_storage_tokens=4,
            layerscale_init=1e-5, mask_k_bias=True,
        )
        model.init_weights()
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(1, 3, 32, 32), is_training=True)
        self.assertEqual(tuple(out["x_norm_patchtokens"].shape), (1, 4, 48))

    def test_source_oracle_math(self):
        torch.manual_seed(99)
        q = torch.randn(5, 5, 768)
        g = torch.randn(3, 5, 768)
        ql = torch.tensor([1, 1, 2, 2, 3])
        gl = torch.tensor([1, 2, 3])
        qv, gv = source_train_vectors(q, g, ql, gl, pca_dim=4)
        self.assertEqual(tuple(qv.shape), (5, 4))
        self.assertEqual(tuple(gv.shape), (3, 4))
        self.assertTrue(torch.isfinite(qv).all())
        self.assertTrue(torch.isfinite(gv).all())


if __name__ == "__main__":
    unittest.main()
