"""Evaluate frozen Round 7 J2 against the University-1652 D2S reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.b2_model import CAReranker
from src.correct_reranking import fuse_reorder


FROZEN = Path(__file__).resolve().parent / "configs" / "round7_j2.json"
RANKS = (1, 5, 10, 20, 50)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_paths(config_path: Path, keys: list[str], expected: dict) -> dict[str, Path]:
    config = read_json(config_path)
    paths = {}
    for key in keys:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing asset path '{key}' in {config_path}")
        path = (config_path.parent / Path(value).expanduser()).resolve(strict=True)
        print(f"Checking {key}: {path.name}", file=sys.stderr, flush=True)
        if sha256(path) != expected[key]:
            raise ValueError(f"SHA-256 mismatch for {key}: {path}")
        paths[key] = path
    return paths


def evaluate(scores: torch.Tensor, query_labels: torch.Tensor,
             gallery_labels: torch.Tensor) -> dict[str, float]:
    scores = scores.float().cpu()
    order = torch.argsort(scores, dim=1, descending=True)
    matches = gallery_labels[order] == query_labels[:, None]
    rank = torch.arange(1, scores.shape[1] + 1, dtype=torch.float32)
    hits = matches.cumsum(1).float()
    average_precision = ((hits / rank) * matches).sum(1) / matches.sum(1).clamp_min(1)
    metrics = {f"recall@{k}": matches[:, :k].any(1).float().mean().item() * 100
               for k in RANKS if k <= scores.shape[1]}
    metrics["mAP"] = average_precision.mean().item() * 100
    return metrics


def score_candidates(checkpoint: Path, query_file: Path, gallery_file: Path,
                     candidates: torch.Tensor, frozen: dict, device: torch.device) -> torch.Tensor:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["epoch"] != frozen["b2"]["epoch"]:
        raise ValueError("Checkpoint epoch differs from frozen configuration")
    model = CAReranker().to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    query = np.load(query_file, mmap_mode="r", allow_pickle=False)
    gallery = np.load(gallery_file, mmap_mode="r", allow_pickle=False)
    patch_shape = tuple(frozen["patch_shape"])
    for patches, count in ((query, frozen["query_count"]), (gallery, frozen["gallery_count"])):
        if patches.shape != (count, *patch_shape) or patches.dtype != np.float16:
            raise ValueError("Patch tensor shape or dtype differs from frozen configuration")

    gallery_gpu = torch.from_numpy(np.array(gallery)).to(device).float()
    result = []
    count, k = candidates.shape
    for start in range(0, count, 4):
        stop = min(start + 4, count)
        q = torch.from_numpy(np.array(query[start:stop])).to(device).float()
        g = gallery_gpu[candidates[start:stop]]
        q = q[:, None].expand(-1, k, -1, -1).reshape(-1, *patch_shape)
        g = g.reshape(-1, *patch_shape)
        result.append(model(q, g).float().cpu().reshape(stop - start, k))
        if stop % 2500 == 0 or stop == count:
            print(f"B2 queries: {stop}/{count}", file=sys.stderr, flush=True)
    return torch.cat(result)


def cached_scores(path: Path, base: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.equal(candidates.cpu(), cache["cand0"]):
        raise ValueError("This environment generated a different candidate order. Use fresh inference.")
    base_digest = hashlib.sha256(base.cpu().contiguous().numpy().tobytes()).hexdigest()
    if base_digest != cache["dependencies"]["j0_similarity_sha256"]:
        raise ValueError("This environment generated different base scores. Use fresh inference.")
    return cache["sc0"]


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    frozen = read_json(FROZEN)
    query_count = frozen["query_count"]
    gallery_count = frozen["gallery_count"]
    count = query_count if args.queries is None else args.queries
    if not 1 <= count <= query_count:
        raise ValueError(f"--queries must be between 1 and {query_count}")
    if args.mode == "cached" and count != query_count:
        raise ValueError("Cached mode requires the full query set")
    if args.require_exact and count != query_count:
        raise ValueError("--require-exact requires a full evaluation")

    keys = ["v5_descriptors"]
    keys += ["b2_scores"] if args.mode == "cached" else ["b2_checkpoint", "query_patches", "gallery_patches"]
    if count == query_count:
        keys.append("evaluation_bundle")
    paths = verified_paths(args.assets, keys, frozen["sha256"])
    if args.check_assets:
        return {"status": "asset_checks_passed", "mode": args.mode, "verified": keys}

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Frozen evaluation requires a CUDA-enabled PyTorch installation and NVIDIA GPU")

    data = torch.load(paths["v5_descriptors"], map_location="cpu", weights_only=True)
    if data["cfg"] != frozen["v5"] or data["labels_read"] is not False:
        raise ValueError("Unexpected V5 configuration or provenance")
    dim = frozen["v5"]["pca_dim"]
    if data["qv"].shape != (query_count, dim) or data["rv"].shape != (gallery_count, dim):
        raise ValueError("Unexpected V5 descriptor shapes")
    base = data["qv"][:count].to(device=device, dtype=torch.float32) @ data["rv"].to(device=device, dtype=torch.float32).T
    candidates = base.argsort(dim=1, descending=True)[:, :frozen["candidate_k"]]

    if args.mode == "cached":
        pair_scores = cached_scores(paths["b2_scores"], base, candidates)
    else:
        pair_scores = score_candidates(paths["b2_checkpoint"], paths["query_patches"],
                                       paths["gallery_patches"], candidates, frozen, device)
    if not torch.isfinite(pair_scores).all():
        raise ValueError("Non-finite B2 scores")
    joint = fuse_reorder(base, candidates, pair_scores, frozen["lambda_j2"])
    if not torch.isfinite(base).all() or not torch.isfinite(joint).all():
        raise ValueError("Non-finite retrieval scores")

    report = {
        "mode": args.mode,
        "queries": count,
        "gallery": gallery_count,
        "pairs": pair_scores.numel(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    if count != query_count:
        report["evaluation"] = "Unlabeled inference smoke test"
        return report

    labels = torch.load(paths["evaluation_bundle"], map_location="cpu", weights_only=True)
    ql, gl = labels["query_labels"].long(), labels["reference_labels"].long()
    if ql.shape != (query_count,) or gl.shape != (gallery_count,):
        raise ValueError("Unexpected evaluation label shapes")
    metrics = {"J0": evaluate(base, ql, gl), "J2": evaluate(joint, ql, gl)}
    differences = {
        name: {metric: value - frozen["expected_metrics"][name][metric]
               for metric, value in actual.items()}
        for name, actual in metrics.items()
    }
    report.update({
        "evaluation": frozen["evaluation"],
        "metrics": metrics,
        "differences_pp": differences,
        "matches_reference_exactly": all(value == 0 for system in differences.values() for value in system.values()),
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="JSON mapping from asset names to file paths")
    parser.add_argument("--mode", choices=("fresh", "cached"), default="fresh")
    parser.add_argument("--queries", type=int, help="Use fewer queries for a smoke test without labels")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check-assets", action="store_true", help="Verify required input hashes and exit")
    parser.add_argument("--require-exact", action="store_true", help="Exit with status 1 if full metrics differ")
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return int(args.require_exact and not args.check_assets and not result["matches_reference_exactly"])


if __name__ == "__main__":
    sys.exit(main())
