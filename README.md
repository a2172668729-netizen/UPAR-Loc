# Round 7 J2 reproduction · University-1652 D2S

J2 retrieves satellite images for drone queries using frozen V5 global descriptors and a source-supervised bidirectional B2 Patch Cross-Attention reranker over the top 20 candidates. The University-1652 D2S run uses 37,855 queries and 951 gallery images.

| Frozen system | R@1 | mAP |
| --- | ---: | ---: |
| J0 · V5 | 81.7541% | 88.7828% |
| J2 · V5 + B2 | **90.2681%** | **94.0511%** |

The benchmark is a **historically exposed Test compatibility evaluation**. V5 fits unlabeled target features and B2 uses source location supervision. See [protocol and limits](docs/REPRODUCTION.md).

## Environment

The reference environment is Ubuntu 22.04 under WSL2, Python 3.10.12, PyTorch 2.7.1+cu128, NumPy 2.2.6 and an NVIDIA RTX 5060 with 8 GB VRAM. Run the following in Linux or a WSL terminal from the repository directory. Set up an isolated environment if one is unavailable.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
```

## Run

1. Obtain the frozen assets from the experiment owner through an authorized channel. [The asset guide](docs/REPRODUCTION.md#external-assets) lists the files. Keep them outside the repository.
2. Copy `configs/assets.example.json` to `configs/assets.local.json` and set the paths. Relative paths are interpreted from the JSON file's directory. Absolute paths and paths with spaces are supported.
3. Verify the assets and run the evaluation.

```bash
.venv/bin/python run_j2.py --assets configs/assets.local.json --check-assets
.venv/bin/python run_j2.py --assets configs/assets.local.json --queries 4
.venv/bin/python run_j2.py --assets configs/assets.local.json --require-exact
```

The last command reruns all 757,100 B2 pairs and compares all six metrics for each system. It prints JSON to stdout and progress to stderr. Use `--mode cached` to validate and replay frozen B2 logits. Cached replay additionally requires the V5 score digest and the candidate order to match exactly. CUDA and sorting differences across environments may affect tied rankings. See [the reference manifest](configs/round7_j2.json) for full-precision values and SHA-256 checks.

The [2026-11-13 reference run](docs/reference_run.json) used this entry point from a separate working directory with an external asset map. The fresh run completed in 7 min 16 s on the reference environment and matched all J0 and J2 metrics exactly.

## Tests

```bash
.venv/bin/python tests/test_reranking.py
.venv/bin/python -m unittest discover -s tests -p test_portability.py
```

The repository contains the frozen evaluation entry point, B2 and reranking source, V5 alignment core, asset mapping, and reference metrics. External assets, original images, and a source-training pipeline are required for an independent reproduction from raw data.
