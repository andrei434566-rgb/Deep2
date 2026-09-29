from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from excel_photo_model_studio.training import DEFAULT_WEIGHTS, train_model


def _dataset(root: Path) -> Path:
    dataset = root / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    (dataset / "train.txt").write_text("images/train/a.jpg\n", encoding="utf-8")
    (dataset / "val.txt").write_text("images/val/b.jpg\n", encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(json.dumps({
        "class_names": ["Dch"], "photo_count": 2, "annotation_count": 4,
    }), encoding="utf-8")
    (dataset / "class_metadata.json").write_text(json.dumps({
        "classes": [{"class_id": 0, "facies_index": "Dch", "facies_name": "Channels"}],
        "target_headers": {},
    }), encoding="utf-8")
    return dataset


class TrainingTests(unittest.TestCase):
    def test_finetunes_yolo11_seg_on_cuda_and_publishes_standard_best_pt(self):
        calls = {}

        class FakeYOLO:
            def __init__(self, weights):
                calls["weights"] = weights
                self.task = "segment"
                self.model = SimpleNamespace(yaml={"yaml_file": "yolo11s-seg.yaml"})

            def train(self, **kwargs):
                calls["train"] = kwargs
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                (save_dir / "weights").mkdir(parents=True)
                (save_dir / "weights" / "best.pt").write_bytes(b"yolo11-checkpoint")
                return SimpleNamespace(save_dir=str(save_dir))

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True, set_device=lambda index: calls.setdefault("gpu", index),
            get_device_name=lambda index: "Test CUDA GPU",
        )
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            output = root / "model"
            with patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultralytics}):
                info = train_model(dataset, output)

            self.assertEqual(b"yolo11-checkpoint", (output / "best.pt").read_bytes())
            self.assertEqual("Dch", json.loads((output / "class_metadata.json").read_text())["classes"][0]["facies_index"])

        self.assertEqual(DEFAULT_WEIGHTS, calls["weights"])
        self.assertEqual(0, calls["gpu"])
        self.assertEqual(0, calls["train"]["device"])
        self.assertTrue(calls["train"]["pretrained"])
        self.assertTrue(calls["train"]["amp"])
        self.assertEqual(1024, calls["train"]["imgsz"])
        self.assertEqual(2, calls["train"]["batch"])
        self.assertEqual(300, calls["train"]["epochs"])
        self.assertEqual("cuda:0", info["device"])
        self.assertEqual("Test CUDA GPU", info["gpu_name"])

    def test_cuda_is_required_and_cpu_is_never_used_as_fallback(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = lambda _weights: self.fail("model should not load without CUDA")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            with patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultralytics}):
                with self.assertRaisesRegex(RuntimeError, "без перехода на CPU"):
                    train_model(dataset, root / "model")

    def test_legacy_yolov8_checkpoint_is_rejected_before_training(self):
        class FakeYOLO:
            def __init__(self, weights):
                self.task = "segment"
                self.model = SimpleNamespace(yaml={"yaml_file": "yolov8n-seg.yaml"})

            def train(self, **_kwargs):
                self.fail("legacy checkpoint must not start training")

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True, set_device=lambda _index: None,
            get_device_name=lambda _index: "Test CUDA GPU",
        )
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            with patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultralytics}):
                with self.assertRaisesRegex(ValueError, "не YOLO11-веса"):
                    train_model(dataset, root / "model", weights="best.pt")


if __name__ == "__main__":
    unittest.main()
