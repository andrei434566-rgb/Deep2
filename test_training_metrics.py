from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from app.infrastructure.ml.fine_tune_worker import load_training_manifest, read_completed_epochs, read_training_metrics, runtime_dataset_yaml, validation_report


class TrainingMetricsTests(unittest.TestCase):
    def test_mask_metrics_are_primary_and_absent_classes_are_not_evaluated(self):
        mask = SimpleNamespace(ap_class_index=[1], class_result=lambda _: (0.8, 0.4, 0.6, 0.3), mean_results=lambda: (0.8, 0.4, 0.6, 0.3))
        boxes = SimpleNamespace(ap_class_index=[0, 1], class_result=lambda _: (1, 1, 1, 1), mean_results=lambda: (1, 1, 1, 1))
        report = validation_report(SimpleNamespace(seg=mask, box=boxes), ["A@1", "B@2"], {"A@1": 0, "B@2": 2})
        self.assertEqual("segmentation", report["task"])
        self.assertEqual("not_evaluated", report["per_class"][0]["status"])
        self.assertNotIn("mAP50", report["per_class"][0])
        self.assertEqual(0.3, report["metrics"]["mAP50-95"])
        self.assertAlmostEqual(0.533333, report["metrics"]["macro_f1"])

    def test_manifest_rejects_leaked_source_and_incompatible_class_order(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / "data.yaml"
            data.write_text(yaml.safe_dump({"path": ".", "names": ["A@1", "B@2"]}), encoding="utf-8")
            manifest = {"validation_independent": True, "val_photo_count": 1, "class_names": ["A@1", "B@2"], "samples": [{"split": "train", "source_sha256": "same"}, {"split": "val", "source_sha256": "same"}]}
            destination = folder / "dataset_manifest.json"
            destination.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "копии"):
                load_training_manifest(data)
            manifest["samples"][1]["source_sha256"] = "other"
            destination.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(manifest, load_training_manifest(data))
            manifest["class_names"].reverse()
            destination.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Классы"):
                load_training_manifest(data)

    def test_portable_yaml_is_resolved_without_modifying_shipped_file(self):
        import yaml
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, target = folder / "data.yaml", folder / "runtime.yaml"
            source.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [A]\n", encoding="utf-8")
            runtime_dataset_yaml(source, target)
            self.assertEqual(str(folder.resolve()), yaml.safe_load(target.read_text(encoding="utf-8"))["path"])
            self.assertEqual(".", yaml.safe_load(source.read_text(encoding="utf-8"))["path"])

    def test_reads_best_map_row_from_ultralytics_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
                "1,0.5,0.4,0.6,0.3\n2,0.7,0.8,0.85,0.65\n",
                encoding="utf-8",
            )
            metrics = read_training_metrics(Path(directory))
            completed = read_completed_epochs(Path(directory))
        self.assertEqual(0.65, metrics["mAP50-95"])
        self.assertEqual(0.7, metrics["precision"])
        self.assertEqual(0.8, metrics["recall"])
        self.assertEqual(2, completed)


if __name__ == "__main__":
    unittest.main()
