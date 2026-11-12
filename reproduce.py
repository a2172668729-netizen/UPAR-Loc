#!/usr/bin/env python3
"""University-1652 + DINOv3 从原始图像重新生成特征、训练并评测。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.raw_pipeline import extract_folder
from src.research_pipeline import (evaluate_target, fit_alignment, load_config,
                                   train_reranker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "align", "train", "evaluate", "all"),
                        help="运行阶段；all 依次运行全部阶段")
    parser.add_argument("--dataset", required=True, type=Path, help="University-1652 根目录")
    parser.add_argument("--weights", type=Path, help="官方 DINOv3 ViT-B/16 预训练权重")
    parser.add_argument("--upstream", type=Path, help="完整 VFM-Loc 官方仓库目录")
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--config", type=Path, default=Path("configs/from_scratch.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="特征提取 batch；可按显存调小")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="明确允许覆盖旧的特征提取结果")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("要求 CUDA，但当前 PyTorch 无法使用 GPU")
    root = args.dataset.resolve()
    out = args.output.resolve()
    paths = {
        "source_query": out / "features" / "source_query",
        "source_gallery": out / "features" / "source_gallery",
        "target_query": out / "features" / "target_query",
        "target_gallery": out / "features" / "target_gallery",
    }
    stages = ("extract", "align", "train", "evaluate") if args.stage == "all" else (args.stage,)
    results = {}
    for stage in stages:
        print(f"\n===== {stage.upper()} =====", flush=True)
        if stage == "extract":
            if args.weights is None or args.upstream is None:
                parser.error("extract / all 必须提供 --weights 和 --upstream")
            inputs = {
                "source_query": root / "train" / "drone",
                "source_gallery": root / "train" / "satellite",
                "target_query": root / "test" / "query_drone",
                "target_gallery": root / "test" / "gallery_satellite",
            }
            results[stage] = {}
            for key, folder in inputs.items():
                print(f"[特征提取] {key}: {folder}", flush=True)
                results[stage][key] = extract_folder(
                    folder, paths[key], args.upstream, args.weights,
                    batch_size=args.batch_size, device=args.device,
                    workers=args.workers, overwrite=args.overwrite,
                )
        elif stage == "align":
            results[stage] = {}
            for domain in ("source", "target"):
                results[stage][domain] = fit_alignment(
                    paths[f"{domain}_query"], paths[f"{domain}_gallery"],
                    out / f"{domain}_descriptors.pt", cfg, device=args.device)
        elif stage == "train":
            results[stage] = train_reranker(
                root, paths["source_query"], paths["source_gallery"],
                out / "source_descriptors.pt", out / "reranker", cfg,
                device=args.device)
        elif stage == "evaluate":
            results[stage] = evaluate_target(
                paths["target_query"], paths["target_gallery"],
                out / "target_descriptors.pt", out / "reranker" / "best.pt",
                out / "final_metrics.json", cfg, device=args.device)
        print(json.dumps(results[stage], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
