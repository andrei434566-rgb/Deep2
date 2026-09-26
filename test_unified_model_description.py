"""The main Kern Analyzer consumes the description head embedded by Studio."""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PySide6.QtCore import QPointF

sys.path.insert(0, str(Path(__file__).resolve().parent / "excel_photo_model_studio" / "src"))

from app.domain.models import FaciesDetection
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES
from app.infrastructure.ml.yolo_model_service import YoloModelService


class UnifiedDescriptionTests(unittest.TestCase):
    def test_predict_uses_embedded_network_once_only_for_classified_core(self):
        class FakeYOLO:
            ckpt = {"core_description_checkpoint": {"schema": "test"}}

            def __init__(self, _):
                pass

            def __call__(self, *args, **kwargs):
                box = SimpleNamespace(cls=np.array(0), conf=np.array(.95), xyxy=np.array([[10, 10, 50, 50]]))
                return [SimpleNamespace(
                    orig_shape=(100, 100), names={0: "DWCh"}, boxes=[box],
                    masks=SimpleNamespace(xy=[np.array([[10, 10], [50, 10], [50, 50], [10, 50]])]),
                )]

        fake = types.ModuleType("ultralytics")
        fake.YOLO = FakeYOLO
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "core.jpg"
            image_path.write_bytes(cv2.imencode(".jpg", np.full((100, 100, 3), 120, np.uint8))[1].tobytes())
            with (
                patch.dict(sys.modules, {"ultralytics": fake}),
                patch("excel_photo_model_studio.description_model.DescriptionGenerator") as generator,
                patch.object(YoloModelService, "_best_device", return_value=("cpu", "CPU")),
                patch.object(YoloModelService, "_predict_structure_intervals", return_value=[]),
            ):
                generator.return_value.checkpoint = {"facies_names": ["DWCh"]}
                generator.return_value.generate.return_value = "Песчаник серый, слоистый."
                service = YoloModelService(root / "best.pt")
                columns = [{"left": 10, "top": 10, "right": 50, "bottom": 90}]
                detections = service.predict(str(image_path), (100, 100), columns)
                service.predict(str(image_path), (100, 100), columns)
                generator.assert_called_once_with(FakeYOLO.ckpt["core_description_checkpoint"])
                self.assertEqual(2, generator.return_value.generate.call_count)
                self.assertEqual("DWCh", generator.return_value.generate.call_args.kwargs["facies"])
                self.assertEqual((40, 40, 3), generator.return_value.generate.call_args.args[0].shape)
            known = [item for item in detections if item.label != UNRECOGNIZED_FACIES]
            unknown = [item for item in detections if item.label == UNRECOGNIZED_FACIES]
            self.assertEqual(1, len(known))
            self.assertEqual("DWCh@80", known[0].attributes["Класс модели"])
            self.assertEqual("DWCh", known[0].attributes["__source_model_class"])
            self.assertIn("Песчаник", known[0].attributes["Краткое описание"])
            self.assertIn("черновик", known[0].attributes["Статус описания"])
            self.assertFalse(known[0].training_ready)
            self.assertTrue(unknown)
            self.assertTrue(all("Краткое описание" not in item.attributes for item in unknown))

    def test_display_coordinates_are_scaled_to_source_and_unknown_text_class_is_not_guessed(self):
        service = object.__new__(YoloModelService)
        service._description_generator = Mock()
        service._description_generator.generate.return_value = "Описание."
        service._description_facies = {"Tcr"}
        polygon = [QPointF(5, 5), QPointF(25, 5), QPointF(25, 25), QPointF(5, 25)]
        known = FaciesDetection("Tcr", .9, polygon, attributes={"__source_model_class": "Tcr"})
        other = FaciesDetection("Other", .9, polygon, attributes={"__source_model_class": "Other"})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "core.jpg"
            source.write_bytes(cv2.imencode(".jpg", np.full((100, 100, 3), 120, np.uint8))[1].tobytes())
            service._attach_generated_descriptions(
                str(source), (50, 50), [{"left": 5, "top": 5, "right": 25, "bottom": 45}], [known, other],
            )
        service._description_generator.generate.assert_called_once()
        self.assertEqual((40, 40, 3), service._description_generator.generate.call_args.args[0].shape)
        self.assertNotIn("Краткое описание", other.attributes)
        self.assertIn("нет обученного", other.attributes["Статус описания"])


if __name__ == "__main__":
    unittest.main()
