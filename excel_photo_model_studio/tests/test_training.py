from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from excel_photo_model_studio.training import embed_description_checkpoint, train_model


class TrainingTests(unittest.TestCase):
    def test_visual_model_starts_from_yaml_without_pretrained_weights(self):
        calls = {}

        class FakeYOLO:
            def __init__(self, architecture):
                calls["architecture"] = architecture

            def train(self, **kwargs):
                calls["train"] = kwargs
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                (save_dir / "weights").mkdir(parents=True)
                (save_dir / "weights" / "best.pt").write_bytes(b"scratch-model")
                return SimpleNamespace(save_dir=str(save_dir))

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "data.yaml").write_text("path: .\n", encoding="utf-8")
            (dataset / "dataset_manifest.json").write_text(json.dumps({"class_names": ["Tcr"]}), encoding="utf-8")
            output = root / "model"
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                info = train_model(
                    dataset, output, architecture="yolo11n-seg.yaml",
                    epochs=1, patience=1, device="cpu",
                )

        self.assertEqual("yolo11n-seg.yaml", calls["architecture"])
        self.assertFalse(calls["train"]["pretrained"])
        self.assertEqual("random_weights", info["initialization"])
        self.assertFalse(info["pretrained"])

    def test_embeds_column_22_checkpoint_and_contract_into_best_pt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            visual = root / "best.pt"
            description = root / "description_best.pt"
            torch.save({"model": "segmentation"}, visual)
            torch.save({"schema": "excel-photo-description-v2", "target_column": 22}, description)
            contract = {"facies_classes": ["Tcr"], "target_column": 22}

            embed_description_checkpoint(visual, description, contract)
            checkpoint = torch.load(visual, map_location="cpu", weights_only=False)

        self.assertEqual("kern-unified-best-v2", checkpoint["core_model_schema"])
        self.assertEqual(22, checkpoint["core_description_checkpoint"]["target_column"])
        self.assertEqual(["Tcr"], checkpoint["core_model_contract"]["facies_classes"])


if __name__ == "__main__":
    unittest.main()
