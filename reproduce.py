#!/usr/bin/env python3
"""University-1652 + DINOv3: 从空目录复现冻结对齐与源域监督重排序。"""
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
    parser.add_argument(
        "stage",
        choices=("extract", "align", "train", "evaluate", "all",
                 "extract_source", "extract_target", "align_source", "align_target"),
        help="all 先完成并冻结源域训练，再接触目标域 Test 图片")
    parser.add_argument("--dataset", required=True, type=Path, help="University-1652 根目录")
    parser.add_argument("--weights", type=Path, help="官方 DINOv3 ViT-B/16 预训练权重")
    parser.add_argument("--upstream", type=Path, help="固定版本 VFM-Loc 官方仓库")
    parser.add_argument("--output", type=Path, default=Path("outputs/reproduction"))
    parser.add_argument("--config", type=Path, default=Path("configs/from_scratch.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="明确允许重新覆盖已提取的特征")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("PyTorch 无法使用 CUDA，不能按所请求设备运行")
    root = args.dataset.expanduser().resolve()
    out = args.output.expanduser().resolve()
    paths = {
        "source_query": out / "features" / "source_query",
        "source_gallery": out / "features" / "source_gallery",
        "target_query": out / "features" / "target_query",
        "target_gallery": out / "features" / "target_gallery",
    }
    image_folders = {
        "source_query": root / "train" / "drone",
        "source_gallery": root / "train" / "satellite",
        "target_query": root / "test" / "query_drone",
        "target_gallery": root / "test" / "gallery_satellite",
    }

    # 全流程严格先训练并选择源域 checkpoint，再读取 Test 图像与特征。
    # 不以目标域真值调整 epoch、seed、lambda 或全局 OP 配置。
    stage_order = (
        "extract_source", "align_source", "train",
        "extract_target", "align_target", "evaluate",
    ) if args.stage == "all" else (
        ("extract_source", "extract_target") if args.stage == "extract" else
        ("align_source", "align_target") if args.stage == "align" else
        (args.stage,)
    )
    results = {}
    for stage in stage_order:
        print(f"\n===== {stage.upper()} =====", flush=True)
        if stage.startswith("extract"):
            if args.weights is None or args.upstream is None:
                parser.error("提取 DINOv3 特征必须提供 --weights 和 --upstream")
            keys = (("source_query", "source_gallery") if stage.endswith("source")
                    else ("target_query", "target_gallery"))
            results[stage] = {}
            for key in keys:
                folder = image_folders[key]
                print(f"[提取] {key}: {folder}", flush=True)
                results[stage][key] = extract_folder(
                    folder, paths[key], args.upstream, args.weights,
                    batch_size=args.batch_size, device=args.device,
                    workers=args.workers, overwrite=args.overwrite,
                )
        elif stage.startswith("align"):
            domain = "source" if stage.endswith("source") else "target"
            results[stage] = fit_alignment(
                paths[f"{domain}_query"], paths[f"{domain}_gallery"],
                out / f"{domain}_descriptors.pt", cfg, device=args.device)
        elif stage == "train":
            if args.upstream is None and cfg["training"].get("negative_mining", "source_oracle") == "source_oracle":
                parser.error("Round 7 的 source-oracle 负例挖掘必须提供 --upstream")
            results[stage] = train_reranker(
                root, paths["source_query"], paths["source_gallery"],
                out / "source_descriptors.pt", out / "reranker", cfg,
                device=args.device, upstream=args.upstream)
        elif stage == "evaluate":
            results[stage] = evaluate_target(
                paths["target_query"], paths["target_gallery"],
                out / "target_descriptors.pt", out / "reranker" / "best.pt",
                out / "final_metrics.json", cfg, device=args.device)
        print(json.dumps(results[stage], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
