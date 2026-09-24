from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from excel_photo_model_studio.autopipeline import run_automatic_training


class AutomaticPipelineTests(unittest.TestCase):
    def _inputs(self, root: Path, count: int = 2) -> tuple[Path, list[dict[str, str]]]:
        wells = []
        for index in range(count):
            excel = root / f"well_{index}.xlsx"
            excel.write_bytes(b"excel")
            photos = root / f"photos_{index}"
            photos.mkdir()
            wells.append({"excel": str(excel), "photos": str(photos)})
        manifest = root / "queue.json"
        manifest.write_text(json.dumps({"wells": wells}), encoding="utf-8")
        return manifest, wells

    def test_clean_wells_are_auto_approved_then_dataset_and_bundle_are_built(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, wells = self._inputs(root)
            projects = []

            def create(excel, photos, project, use_ocr):
                project.mkdir(parents=True)
                projects.append(project)
                return {"blocking_errors": 0, "photos": 10, "annotations": 20, "issues": []}

            with patch.dict(os.environ, {"LOCALAPPDATA": str(root / "appdata")}):
                with patch("excel_photo_model_studio.autopipeline.create_project", side_effect=create), \
                        patch("excel_photo_model_studio.autopipeline.register_project"), \
                        patch("excel_photo_model_studio.autopipeline.load_annotations", return_value=[
                            {"annotation_id": "mask-1"}, {"annotation_id": "mask-2"},
                        ]), \
                        patch("excel_photo_model_studio.autopipeline.set_annotation_approvals") as approve, \
                        patch("excel_photo_model_studio.autopipeline.build_dataset", return_value={
                            "photo_count": 20, "annotation_count": 40, "class_names": ["Sand"],
                        }) as build, \
                        patch("excel_photo_model_studio.autopipeline.train_bundle", return_value={
                            "output_dir": str(root / "model"),
                        }) as train:
                    approve.side_effect = lambda _project, values: {"approved_annotations": len(values)}
                    result = run_automatic_training(
                        manifest, root / "dataset", root / "model", epochs=2,
                    )

            self.assertEqual("success", result["status"])
            self.assertEqual(2, len(projects))
            self.assertEqual(2, approve.call_count)
            self.assertEqual(1, build.call_count)
            self.assertEqual(1, train.call_count)
            self.assertTrue((root / "model_automatic_training_report.json").is_file())

    def test_any_incomplete_well_blocks_model_without_auto_approving_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, _wells = self._inputs(root)
            created = []

            def create(excel, photos, project, use_ocr):
                project.mkdir(parents=True)
                created.append(project)
                return {
                    "blocking_errors": 1 if len(created) == 1 else 0,
                    "photos": 10, "annotations": 20,
                    "issues": [{"severity": "error", "message": "Фото без интервала"}]
                    if len(created) == 1 else [],
                }

            with patch.dict(os.environ, {"LOCALAPPDATA": str(root / "appdata")}):
                with patch("excel_photo_model_studio.autopipeline.create_project", side_effect=create), \
                        patch("excel_photo_model_studio.autopipeline.register_project"), \
                        patch("excel_photo_model_studio.autopipeline.load_annotations", return_value=[
                            {"annotation_id": "mask-1"},
                        ]), \
                        patch("excel_photo_model_studio.autopipeline.set_annotation_approvals") as approve, \
                        patch("excel_photo_model_studio.autopipeline.build_dataset") as build, \
                        patch("excel_photo_model_studio.autopipeline.train_bundle") as train:
                    approve.side_effect = lambda _project, values: {"approved_annotations": len(values)}
                    with self.assertRaises(ValueError):
                        run_automatic_training(manifest, root / "dataset", root / "model")

            self.assertEqual(2, len(created), "обработать нужно все входные наборы и сообщить о проблемах")
            self.assertEqual(1, approve.call_count, "проект с ошибкой нельзя автоматически утверждать")
            build.assert_not_called()
            train.assert_not_called()
            report = json.loads((root / "model_automatic_training_report.json").read_text(encoding="utf-8"))
            self.assertEqual("blocked", report["status"])
            self.assertEqual("needs_data_fix", report["wells"][0]["status"])


if __name__ == "__main__":
    unittest.main()
