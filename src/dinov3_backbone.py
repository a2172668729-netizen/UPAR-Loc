"""本仓库内置的 Meta DINOv3 ViT-B/16 网络入口。

架构定义取自 Meta DINOv3 上游（2025-08-14 固定提交），依赖其许可约束，
源码完整放在 third_party/dinov3；这里不依赖 VFM-Loc，也不联网下载模型。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


META_DINO_COMMIT = "d51437b5e494881a204e71ee8e8a17e471fdcada"
EMBED_DIM = 768
PATCHES = 196
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _import_bundled_architecture():
    third_party = Path(__file__).resolve().parents[1] / "third_party" / "dinov3"
    expected = third_party / "dinov3" / "models" / "vision_transformer.py"
    if not expected.is_file():
        raise FileNotFoundError(
            f"缺少仓库随附的 DINOv3 ViT 结构文件：{expected}，请完整克隆仓库")
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from dinov3.models.vision_transformer import DinoVisionTransformer
    return DinoVisionTransformer


def load_dinov3_vitb16(weights: Path, *, device: str = "cpu") -> torch.nn.Module:
    """严格匹配官方 DINOv3 ViT-B/16 LVD1689M 架构并加载本地权重。

    只读取用户提供的 .pth，不下载任何历史权重或官方 VFM-Loc 代码。
    checkpoint 必须是完整的 ViT-B/16；缺少参数时明确报错而非随机初始化。
    """
    weights = Path(weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"缺少 DINOv3 ViT-B/16 权重：{weights}")
    cls = _import_bundled_architecture()
    model = cls(
        img_size=224, patch_size=16, in_chans=3,
        pos_embed_rope_base=100.0,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0,
        pos_embed_rope_dtype="fp32",
        embed_dim=768, depth=12, num_heads=12,
        ffn_ratio=4.0, qkv_bias=True, drop_path_rate=0.0,
        layerscale_init=1.0e-5, norm_layer="layernormbf16",
        ffn_layer="mlp", ffn_bias=True, proj_bias=True,
        n_storage_tokens=4, mask_k_bias=True,
    )
    model.init_weights()

    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError as err:
        raise RuntimeError("需要支持 weights_only=True 的 PyTorch 版本") from err
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint 不包含有效的 PyTorch state_dict：{weights}")
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "DINOv3 权重与内置 ViT-B/16 架构不匹配；请确认使用官方 "
            "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth，而非其他 backbone"
        ) from exc
    model.eval().requires_grad_(False)
    return model.to(device)
