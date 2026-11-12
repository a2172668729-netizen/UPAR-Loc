"""从 University-1652 原图生成与 VFM-Loc 一致的区域及 Patch 特征。

依赖指定版本的官方 VFM-Loc 源码及其 DINOv3 子目录；不包含上游权重。
本模块不读取标签文件，也不写出标签，只保存可追溯的输入路径顺序。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UPSTREAM_COMMIT = "162d885d25f0215025d7fe9f0a1c70f64e7ba119"


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


def _load_upstream(upstream: Path):
    if not (upstream / "vfm_loc" / "models.py").is_file():
        raise FileNotFoundError(f"缺少 VFM-Loc 官方源码：{upstream}")
    if not (upstream / "third_party" / "dinov3" / "dinov3").is_dir():
        raise FileNotFoundError("上游仓库没有 third_party/dinov3，请使用完整仓库副本")
    root = str(upstream.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from vfm_loc.models import DinoV3Embedder
    from vfm_loc.transforms import build_eval_transforms
    return DinoV3Embedder, build_eval_transforms


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
def extract_folder(folder: Path, output: Path, upstream: Path, weights: Path,
                   *, batch_size: int = 16, device: str = "cuda", workers: int = 2,
                   overwrite: bool = False, image_size: int = 224) -> dict:
    """执行冻结特征提取，并把大型 Patch 数组保存为磁盘映射 NPY。"""
    folder, output, upstream, weights = map(Path, (folder, output, upstream, weights))
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
    DinoV3Embedder, build_transforms = _load_upstream(upstream)
    model = DinoV3Embedder(
        model_name="dinov3_vitb16",
        weights=str(weights.resolve()),
        img_size=image_size, pool="gem", use_rmac=True,
        rmac_levels=[1, 2], device=device,
    ).to(device).eval()
    preprocess = model.get_preprocess_config()
    ref_transform, query_transform = build_transforms(
        (image_size, image_size), (image_size, image_size),
        preprocess["mean"], preprocess["std"], ground_cutting=0,
    )
    transform = query_transform if folder.name in ("drone", "query_drone") else ref_transform
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
        with torch.autocast("cuda", enabled=device.startswith("cuda")):
            # 同一次 DINOv3 前向同时生成五区域特征和 Patch，避免两次前向不一致。
            outputs = model.model(batch, is_training=True)
            pt = outputs["x_norm_patchtokens"]
            reg = _regions_from_tokens(pt)
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
        "upstream_expected_commit": UPSTREAM_COMMIT,
        "input_directory": str(folder.resolve()),
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
