# Reproduction details

## External assets

The asset paths are supplied by a local JSON configuration. SHA-256 checks use the frozen original files before model or label loading. The following names are examples. Any file name is accepted if the contents match the manifest.

| Config key | Frozen artifact | Required for |
| --- | --- | --- |
| `v5_descriptors` | `test_v5.pt`, with label-free `qv` and `rv` | Both modes |
| `b2_checkpoint` | `best.pt` from the bidirectional B2 seed 33, epoch 7 run | Fresh inference |
| `query_patches` | `query_patch.npy`, float16, 37,855 × 196 × 768 | Fresh inference |
| `gallery_patches` | `ref_patch.npy`, float16, 951 × 196 × 768 | Fresh inference |
| `evaluation_bundle` | `u1652_d2s_dinov3_gem_rmac_features_fp32.pt` with `query_labels` and `reference_labels` | Full evaluation |
| `b2_scores` | `test_b2_scores.pt` with J0 candidate indices and B2 logits | Cached replay |

The fresh-evaluation assets occupy approximately 12 GiB. Cached replay requires only the frozen V5 descriptors, B2 score cache and evaluation bundle. A fresh smoke test with fewer than 37,855 queries uses the checkpoint and patches and skips the evaluation bundle.

The public repository is prepared for authorized access to these external files. A recipient needs a copy matching the checksums or an authorized source for the same artifacts. Official dataset access and the backbone checkpoint alone are insufficient to recover the frozen B2 checkpoint and caches.

## Frozen protocol

- The query and gallery orders correspond to the frozen official University-1652 D2S Test inputs. The label vectors use the same row order.
- V5 uses frozen DINOv3 ViT-B/16, five-region GeM/R-MAC features, per-view PCA-512, whitening exponents 0.55 and 0.10, CSLS K=100, a 55-pair cap per Gallery image, four orthogonal Procrustes updates, and 600 Cayley ranking steps. The full evaluation consumes the saved V5 descriptors.
- B2 uses frozen 196 × 768 patches and a bidirectional Cross-Attention model trained with source identities. The reference checkpoint is seed 33, epoch 7. It reranks 20 candidates using fusion weight 1.0.
- Test labels are accessed for scoring after fusion. Round 7's frozen run uses a historically exposed Test split. Cross-dataset and independent blind generalization are outside this reproduction. Historical V5 cap selection had prior target Test exposure.
- Candidate reordering preserves candidate score values and all scores outside the selected candidates. Exact score ties can affect the CPU evaluator's rank ordering or Top-20 membership.

## Scope and provenance

This repository verifies evaluation from frozen assets. The V5 alignment core is included for code review. An image-to-result reproduction also requires the original feature extraction, source B2 training and V5 fitting procedures, dataset access, and separate acceptance testing.

[VFM-Loc](https://github.com/DingLei14/VFM-Loc) and DINOv3 are upstream components of the frozen feature pipeline. Their upstream source trees, weights, and the University-1652 data are external to this repository. This snapshot includes the local J2 evaluation and method code. The project owner determines the repository license and the authorized distribution channels for external assets before publication.
