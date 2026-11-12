import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from run_j2 import evaluate, verified_paths


class PortableEvaluationTests(unittest.TestCase):
    def test_relative_asset_path_resolves_from_config_directory(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config_dir = root / "local config"
            config_dir.mkdir()
            asset = root / "external asset.pt"
            payload = b"asset integrity test"
            asset.write_bytes(payload)
            config = config_dir / "assets.local.json"
            config.write_text(json.dumps({"weights": "../external asset.pt"}), encoding="utf-8-sig")
            expected = {"weights": hashlib.sha256(payload).hexdigest()}

            self.assertEqual(verified_paths(config, ["weights"], expected), {"weights": asset.resolve()})
            asset.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verified_paths(config, ["weights"], expected)

    def test_evaluate_uses_the_full_gallery_and_all_positives(self):
        scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
        query_labels = torch.tensor([7, 9])
        gallery_labels = torch.tensor([7, 9, 7])
        result = evaluate(scores, query_labels, gallery_labels)
        self.assertEqual(result["recall@1"], 50.0)
        self.assertAlmostEqual(result["mAP"], 100 * ((1 + 2 / 3) / 2 + 1 / 2) / 2, places=4)


if __name__ == "__main__":
    unittest.main()
