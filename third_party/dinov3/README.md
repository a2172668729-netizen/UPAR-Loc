# DINOv3 ViT-B/16 source snapshot

This folder includes the minimal **inference** dependency modules from Meta's [DINOv3](https://github.com/facebookresearch/dinov3), taken unmodified from Git commit [d51437b5e494881a204e71ee8e8a17e471fdcada](https://github.com/facebookresearch/dinov3/commit/d51437b5e494881a204e71ee8e8a17e471fdcada) (14 August 2025).

Copyright (c) Meta Platforms, Inc. and affiliates. These modules and their redistribution are governed by the accompanying [DINOv3 License](LICENSE.md); retain this license for redistribution. No pretrained model weights are included.

The CVGL feature pooling, PCA/OP, and reranking logic are implemented separately in this project. VFM-Loc is credited in the research documentation but its source is not duplicated here because the public upstream repository does not provide an explicit code license.
