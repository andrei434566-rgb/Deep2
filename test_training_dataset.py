"""Safety and portability checks for facies training export."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.facies_catalog import FACIES_MODEL_CLASSES
from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.ml.fine_tune_worker import load_training_manifest
from app.infrastructure.training_dataset import export_training_dataset, inspect_training_dataset


_APP = QApplication.instance() or QApplication([])


def _record(identifier: str, label: str, index: str, *, color: str, well: str = "W-1") -> PhotoRecord:
    pixmap = QPixmap(60, 80)
    pixmap.fill(QColor(color))
    detection = FaciesDetection(
        label, 1.0,
        [QPointF(5, 5), QPointF(50, 5), QPointF(50, 70), QPointF(5, 70)],
        attributes={"Индекс фации": index, "Название породы": "Песчаник"},
        depth_from=1.0, depth_to=2.0, training_ready=True,
    )
    return PhotoRecord(identifier, f"{identifier}.jpg", pixmap, [detection], well_name=well)


class TrainingDatasetTests(unittest.TestCase):
    def test_strict_export_has_fixed_registry_and_independent_pixels(self):
        colors = ["#111111", "#223344", "#334455", "#445566", "#556677"]
        records = [_record(str(i), "DWCh", "80", color=color) for i, color in enumerate(colors)]
        report = inspect_training_dataset(records)
        self.assertTrue(report["can_export"])
        self.assertEqual(92, len(report["class_names"]))
        self.assertEqual([item["model_label"] for item in FACIES_MODEL_CLASSES], report["class_names"])
        self.assertEqual(4, report["train_photo_count"])
        self.assertEqual(1, report["val_photo_count"])
        with tempfile.TemporaryDirectory() as temporary:
            result = export_training_dataset(records, Path(temporary) / "dataset", balance=False)
            manifest = load_training_manifest(Path(result["data_yaml"]))
            registry = json.loads(Path(result["class_registry"]).read_text(encoding="utf-8"))
        self.assertEqual(report["class_names"], manifest["class_names"])
        self.assertEqual(92, len(registry["classes"]))
        self.assertIn("DWCh@80", result["val_class_counts"])

    def test_copied_pixels_are_deduplicated_even_across_well_names(self):
        records = [
            _record("original", "DWCh", "80", color="#123456", well="W-1"),
            _record("copy", "DWCh", "80", color="#123456", well="W-2"),
        ]
        report = inspect_training_dataset(records)
        self.assertEqual(1, report["sample_count"])
        self.assertEqual(1, report["duplicates_removed"])
        self.assertFalse(report["can_export"])
        self.assertEqual(1, report["photo_count"])

    def test_ambiguous_legacy_code_blocks_export_until_index_is_selected(self):
        record = _record("ambiguous", "Dch", "", color="#abcdef")
        report = inspect_training_dataset([record])
        self.assertFalse(report["can_export"])
        self.assertEqual(1, len(report["unknown_labels"]))
        self.assertIn("индекс", report["unknown_labels"][0]["reason"])

    def test_imbalance_warns_but_is_not_itself_a_blocker(self):
        colors = [f"#{i + 16:02x}{i + 32:02x}{i + 48:02x}" for i in range(12)]
        records = [_record(str(i), "DWCh" if i < 11 else "w-DMB", "80" if i < 11 else "63", color=color)
                   for i, color in enumerate(colors)]
        report = inspect_training_dataset(records)
        self.assertTrue(any("Дисбаланс" in warning for warning in report["warnings"]))
        self.assertFalse(any("дисбаланс" in reason.casefold() for reason in report["blocking_reasons"]))

    def test_excel_source_metadata_survives_reviewed_dataset_export(self):
        records = [
            _record(str(index), "DWCh", "80", color=f"#{index + 20:02x}{index + 30:02x}{index + 40:02x}")
            for index in range(5)
        ]
        records[0].detections[0].attributes.update({
            "Месторождение": "Северное",
            "№ скважины": "Р-17",
            "Интервал керна": "2500–2505 м",
            "Интервал фации": "2500–2501.25 м",
            "Толщина фации": "1.25 м",
            "Литологическое описание (16 параметров)": "Песчаник серый",
            "__annotation_source": "excel_depth_projection",
        })
        with tempfile.TemporaryDirectory() as temporary:
            result = export_training_dataset(records, Path(temporary) / "dataset", balance=False)
            rows = [
                json.loads(line)
                for line in Path(result["attributes_jsonl"]).read_text(encoding="utf-8").splitlines()
            ]

        source = next(row["source_metadata"] for row in rows if row["source_metadata"])
        self.assertEqual("Северное", source["Месторождение"])
        self.assertEqual("1.25 м", source["Толщина фации"])
        self.assertEqual("excel_depth_projection", source["__annotation_source"])


if __name__ == "__main__":
    unittest.main()
