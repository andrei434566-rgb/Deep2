from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from excel_photo_model_studio.training import (
    DEFAULT_WEIGHTS, _publish_model_package, _resolve_pretrained_weights,
    train_model, training_data_warnings,
)


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


def _vlm_inputs(root: Path, dataset: Path, train_count: int, val_count: int) -> tuple[Path, Path]:
    (dataset / "crops").mkdir(exist_ok=True)
    rows = []
    for index in range(train_count + val_count):
        crop = dataset / "crops" / f"{index}.jpg"
        crop.touch()
        rows.append({
            "split": "train" if index < train_count else "val",
            "crop": crop.relative_to(dataset).as_posix(),
            "target_text": "Описание керна.",
        })
    (dataset / "caption_dataset.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    base = root / "vlm-base"
    base.mkdir()
    (base / "config.json").write_text("{}", encoding="utf-8")
    (base / "model.safetensors").touch()
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").touch()
    return base, adapter


class TrainingTests(unittest.TestCase):
    def test_model_package_copy_fallback_when_windows_denies_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged, output = root / "staged", root / "model"
            staged.mkdir()
            (staged / "best.pt").write_bytes(b"weights")
            messages = []
            with patch.object(Path, "rename", side_effect=PermissionError("denied")):
                _publish_model_package(staged, output, messages.append)
            self.assertEqual(b"weights", (output / "best.pt").read_bytes())
            self.assertTrue(any("копирую" in message for message in messages))

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
            legacy_weights = root / "best.pt"
            legacy_weights.write_bytes(b"legacy checkpoint placeholder")
            with patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultralytics}):
                with self.assertRaisesRegex(ValueError, "не YOLO11-веса"):
                    train_model(dataset, root / "model", weights=legacy_weights)

    def test_offline_pretrained_download_error_is_actionable_and_leaves_no_empty_run(self):
        class OfflineYOLO:
            def __init__(self, _weights):
                raise ConnectionError("Download failure; Environment may be offline")

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True, set_device=lambda _index: None,
            get_device_name=lambda _index: "Test CUDA GPU",
        )
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = OfflineYOLO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            output = root / "model"
            with patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultralytics}):
                with self.assertRaisesRegex(RuntimeError, "Локальный .pt"):
                    train_model(dataset, output)
            self.assertFalse(output.exists())
            self.assertFalse((root / "model_runs").exists())

    def test_missing_explicit_checkpoint_fails_before_attempting_network_download(self):
        with self.assertRaisesRegex(FileNotFoundError, "Файл весов не найден"):
            _resolve_pretrained_weights("weights/not-installed.pt")

    def test_missing_vlm_dependencies_fail_before_long_yolo_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            (dataset / "crops").mkdir()
            rows = []
            for index in range(23):
                crop = dataset / "crops" / f"{index}.jpg"
                crop.touch()
                rows.append({
                    "split": "train" if index < 20 else "val",
                    "crop": crop.relative_to(dataset).as_posix(),
                    "target_text": "Описание керна.",
                })
            (dataset / "caption_dataset.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            base = root / "vlm-base"
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            (base / "model.safetensors").touch()
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").touch()
            output = root / "candidate"
            with patch("excel_photo_model_studio.training.importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "до запуска длительного обучения YOLO"):
                    train_model(
                        dataset, output, vlm_base_model_dir=base, vlm_initial_adapter=adapter,
                    )
            self.assertFalse(output.exists())
            self.assertFalse((root / "candidate_runs").exists())

    def test_joint_training_requires_vlm_inputs_before_yolo_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            with self.assertRaisesRegex(ValueError, "Совместное обучение требует"):
                train_model(dataset, root / "model", require_vlm=True)
            self.assertFalse((root / "model_runs").exists())

    def test_joint_training_requires_enough_caption_examples_before_yolo_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            base, adapter = _vlm_inputs(root, dataset, 1, 1)
            with self.assertRaisesRegex(ValueError, "минимум 20 train и 3"):
                train_model(
                    dataset, root / "model", vlm_base_model_dir=base,
                    vlm_initial_adapter=adapter, require_vlm=True,
                )
            self.assertFalse((root / "model_runs").exists())

    def test_joint_training_does_not_publish_partial_model_after_vlm_failure(self):
        calls = {"yolo_trained": False}

        class FakeYOLO:
            def __init__(self, _weights):
                self.task = "segment"
                self.model = SimpleNamespace(yaml={"yaml_file": "yolo11s-seg.yaml"})

            def train(self, **kwargs):
                calls["yolo_trained"] = True
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                (save_dir / "weights").mkdir(parents=True)
                (save_dir / "weights" / "best.pt").write_bytes(b"recoverable-yolo")
                return SimpleNamespace(save_dir=str(save_dir))

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True, set_device=lambda _index: None,
            get_device_name=lambda _index: "Test CUDA GPU", empty_cache=lambda: None,
        )
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            base, adapter = _vlm_inputs(root, dataset, 20, 3)
            output = root / "joint_model"
            with (
                patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultralytics}),
                patch("excel_photo_model_studio.training.importlib.util.find_spec", return_value=object()),
                patch("excel_photo_model_studio.vlm_description.train_vlm_description_model",
                      side_effect=RuntimeError("VLM test failure")),
            ):
                with self.assertRaisesRegex(RuntimeError, "совместная модель не опубликована"):
                    train_model(
                        dataset, output, vlm_base_model_dir=base,
                        vlm_initial_adapter=adapter, require_vlm=True,
                    )
            self.assertTrue(calls["yolo_trained"])
            self.assertFalse(output.exists())
            self.assertEqual(1, len(list(root.glob("joint_model_runs/**/best.pt"))))

    def test_joint_training_packages_yolo_and_vlm_together(self):
        class FakeYOLO:
            def __init__(self, _weights):
                self.task = "segment"
                self.model = SimpleNamespace(yaml={"yaml_file": "yolo11s-seg.yaml"})

            def train(self, **kwargs):
                save_dir = Path(kwargs["project"]) / kwargs["name"]
                (save_dir / "weights").mkdir(parents=True)
                (save_dir / "weights" / "best.pt").write_bytes(b"joint-yolo")
                return SimpleNamespace(save_dir=str(save_dir))

        def fake_train_vlm(_dataset, output_dir, **kwargs):
            output_dir.mkdir(parents=True)
            (output_dir / "adapter_model.safetensors").write_bytes(b"joint-vlm")
            (output_dir / "description_training_info.json").write_text("{}", encoding="utf-8")
            return {
                "status": "trained_candidate_requires_review",
                "base_model_dir": str(kwargs["base_model_dir"]),
            }

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True, set_device=lambda _index: None,
            get_device_name=lambda _index: "Test CUDA GPU", empty_cache=lambda: None,
        )
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = FakeYOLO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = _dataset(root)
            base, adapter = _vlm_inputs(root, dataset, 20, 3)
            output = root / "joint_model"
            with (
                patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultralytics}),
                patch("excel_photo_model_studio.training.importlib.util.find_spec", return_value=object()),
                patch("excel_photo_model_studio.vlm_description.train_vlm_description_model",
                      side_effect=fake_train_vlm),
            ):
                info = train_model(
                    dataset, output, vlm_base_model_dir=base,
                    vlm_initial_adapter=adapter, require_vlm=True,
                )
            self.assertEqual(b"joint-yolo", (output / "best.pt").read_bytes())
            self.assertEqual(
                b"joint-vlm", (output / "description_vlm" / "adapter_model.safetensors").read_bytes(),
            )
            contract = json.loads((output / "model_contract.json").read_text(encoding="utf-8"))
            self.assertEqual("qwen3_vl_lora", contract["description_model_type"])
            self.assertEqual("trained_candidate_requires_review", info["description_model_status"])

    def test_training_quality_warnings_identify_tiny_and_photo_level_splits(self):
        warnings = training_data_warnings({
            "photo_count": 3,
            "annotation_count": 13,
            "facies_count": 4,
            "class_counts": {"SYN-01": 4, "SYN-02": 3, "SYN-03": 3, "SYN-04": 3},
            "train_photo_count": 2,
            "val_photo_count": 1,
            "split_strategy": "photo_fallback",
            "samples": [{"well": "TEST-001"}],
        })
        self.assertTrue(any("пробный набор" in warning for warning in warnings))
        self.assertTrue(any("маленькое" in warning.casefold() for warning in warnings))
        self.assertTrue(any("по фото, а не по скважинам" in warning for warning in warnings))
        self.assertTrue(any("синтетический тестовый набор" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
