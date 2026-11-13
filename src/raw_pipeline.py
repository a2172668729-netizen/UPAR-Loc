"""从 University-1652 原图提取内置 DINOv3 ViT-B/16 的区域和 Patch 特征。

无运行时 VFM-Loc 依赖：本仓库自带 Meta DINOv3 的许可推理模块。
图像读取与预处理和原始实验保持一致；不加载或保存地点标签。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



def image_list(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"未找到图片目录：{folder}")
    paths = [
        p for sub in sorted(folder.iterdir()) if sub.is_dir()
        for p in sorted(sub.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not paths:
        raise ValueError(f"目录中没有图片：{folder}")
    return paths


def image_ids(paths: list[str]) -> list[int]:
    # 仅训练器、评测器调用。无标签拟合流程不应调用本函数。
    return [int(Path(x).parent.name) for x in paths]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _Images(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import cv2
        image = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"无法读取图片：{self.paths[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"]


def _regions_from_tokens(patch: torch.Tensor, pool: str = "gem",
                         alpha: float = 3.0) -> torch.Tensor:
    """复刻指定版本 vfm_loc.models.DinoV3Embedder.forward 的五区域池化。"""
    b, n, c = patch.shape
    grid = int(n ** 0.5)
    if grid * grid != n:
        raise ValueError(f"DINOv3 Patch 数量必须为平方数，实际是 {n}")
    x = patch.reshape(b, grid, grid, c).permute(0, 3, 1, 2)
    regions = []
    for level in (1, 2):
        win = max(1, grid // level)
        for i in range(level):
            for j in range(level):
                block = x[:, :, i * win:(grid if i == level - 1 else (i + 1) * win),
                          j * win:(grid if j == level - 1 else (j + 1) * win)]
                if pool == "gem":
                    region = block.clamp_min(1e-6).pow(alpha).mean((-2, -1)).pow(1.0 / alpha)
                else:
                    region = block.mean((-2, -1))
                regions.append(region)
    return torch.stack(regions, dim=1)


@torch.inference_mode()
def extract_folder(folder: Path, output: Path, weights: Path,
                   *, batch_size: int = 16, device: str = "cuda", workers: int = 2,
                   overwrite: bool = False, image_size: int = 224) -> dict:
    """执行冻结特征提取，并把大型 Patch 数组保存为磁盘映射 NPY。"""
    folder, output, weights = map(Path, (folder, output, weights))
    if not weights.is_file():
        raise FileNotFoundError(f"缺少 DINOv3 权重：{weights}")
    if image_size != 224:
        raise ValueError("现阶段仅支持已核对的 224×224 / 196 Patch 协议")
    paths = image_list(folder)
    result_paths = {
        "regions": output / "regions.npy",
        "patches": output / "patches.npy",
        "order": output / "image_paths.json",
        "metadata": output / "metadata.json",
    }
    if not overwrite and any(p.exists() for p in result_paths.values()):
        raise FileExistsError(f"{output} 已存在输出，请确认来源后加 --overwrite 明确重提取")
    output.mkdir(parents=True, exist_ok=True)
    from albumentations import Compose, Normalize, Resize
    from albumentations.pytorch import ToTensorV2
    from src.dinov3_backbone import load_dinov3_vitb16, MEAN, STD, META_DINO_COMMIT
    import cv2

    # 与 VFM-Loc 的 University-1652 D2S eval 预处理一致：无裁剪，
    # 224x224、INTER_LINEAR_EXACT、ImageNet mean/std。
    transform = Compose([
        Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR_EXACT),
        Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ])
    model = load_dinov3_vitb16(weights, device=device)
    loader = DataLoader(_Images(paths, transform), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=device.startswith("cuda"))
    n = len(paths)
    regions = np.lib.format.open_memmap(result_paths["regions"], mode="w+", dtype=np.float32,
                                        shape=(n, 5, 768))
    patches = np.lib.format.open_memmap(result_paths["patches"], mode="w+", dtype=np.float16,
                                        shape=(n, 196, 768))
    offset = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        # Round 7 原始实验使用两次独立前向：区域特征来自 FP32，
        # Patch token 来自 CUDA autocast FP16。混用 Patch 的 FP16 输出生成
        # 区域特征会改变 PCA/OP 输入，不能宣称与历史 V5 相同。
        # 原始 Round 7 分别使用 FP32 区域前向与 FP16 autocast Patch 前向。
        # 区域 GeM 必须在 float32 下执行，不能使用量化后的 Patch token。
        fp32 = model(batch, is_training=True)
        reg = _regions_from_tokens(fp32["x_norm_patchtokens"].float())
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            outputs = model(batch, is_training=True)
            pt = outputs["x_norm_patchtokens"]
        stop = offset + len(batch)
        if pt.shape[1:] != (196, 768):
            raise ValueError(f"异常 Patch 维度：{tuple(pt.shape)}")
        regions[offset:stop] = reg.float().cpu().numpy()
        patches[offset:stop] = pt.float().cpu().numpy().astype(np.float16)
        offset = stop
    regions.flush()
    patches.flush()
    order = [str(p.relative_to(folder.parent.parent)) for p in paths]
    result_paths["order"].write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "count": n, "image_size": image_size, "patch_shape": [196, 768],
        "region_shape": [5, 768], "model": "dinov3_vitb16",
        "checkpoint_sha256": sha256(weights),
        "dinov3_source_commit": META_DINO_COMMIT,
        "input_directory": str(folder.resolve()),
        "region_forward": "fp32 DINOv3 patch tokens + 5-region GeM",
        "patch_forward": "autocast fp16 DINOv3 x_norm_patchtokens",
        "note": "不包含 ID 标签；图像路径供训练和最终评测阶段单独解析",
    }
    result_paths["metadata"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_arrays(folder: Path, device: str = "cuda") -> tuple[torch.Tensor, np.memmap, list[str]]:
    folder = Path(folder)
    regions = np.load(folder / "regions.npy", mmap_mode="r", allow_pickle=False)
    patches = np.load(folder / "patches.npy", mmap_mode="r", allow_pickle=False)
    paths = json.loads((folder / "image_paths.json").read_text(encoding="utf-8"))
    if regions.ndim != 3 or tuple(regions.shape[1:]) != (5, 768):
        raise ValueError(f"区域特征维度错误：{regions.shape}")
    if tuple(patches.shape) != (len(paths), 196, 768) or len(regions) != len(paths):
        raise ValueError("区域特征、Patch 和文件顺序不匹配")
    return torch.from_numpy(np.array(regions)).to(device), patches, paths
