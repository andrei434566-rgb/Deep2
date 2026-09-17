from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.domain.facies_catalog import FACIES_MODEL_CLASSES
from app.infrastructure.model_blueprint import check_model_blueprint, class_registry, export_model_blueprint
from prepare_model import main


class ModelBlueprintTests(unittest.TestCase):
    def test_export_without_weights_round_trips_full_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "blueprint"
            result = export_model_blueprint(output)
            self.assertEqual(92, result["class_count"])
            self.assertEqual("compatible", check_model_blueprint(output)["status"])
            self.assertFalse(list(output.rglob("*.pt")))
            attributes = json.loads((output / "lithology_attributes.json").read_text(encoding="utf-8"))
            self.assertEqual(16, attributes["field_count"])
            self.assertIsNone(attributes["missing_value"])
            self.assertTrue((output / "START_HERE.md").is_file())

    def test_never_overwrites_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            sentinel = target / "keep.txt"
            sentinel.write_text("user data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_model_blueprint(target)
            self.assertEqual("user data", sentinel.read_text(encoding="utf-8"))

    def test_reordered_or_changed_registry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "blueprint"
            export_model_blueprint(output)
            registry_path = output / "class_registry.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["class_names"].reverse()
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaises(ValueError):
                check_model_blueprint(output)

    def test_registry_is_defensive_copy_and_preserves_exact_identity(self):
        registry = class_registry()
        self.assertIn("Dch@47", registry["class_names"])
        self.assertIn("Dch@92", registry["class_names"])
        self.assertIn("DWCH@79", registry["class_names"])
        self.assertIn("DWCh@80", registry["class_names"])
        original = FACIES_MODEL_CLASSES[0]["metadata"]["Название фации"]
        registry["classes"][0]["metadata"]["Название фации"] = "changed"
        self.assertEqual(original, FACIES_MODEL_CLASSES[0]["metadata"]["Название фации"])

    def test_train_rejects_missing_weights_before_starting_worker(self):
        with tempfile.TemporaryDirectory() as temporary, patch("sys.stderr"):
            result = main(["train", "--model", str(Path(temporary) / "missing.pt"),
                           "--dataset", temporary, "--output", str(Path(temporary) / "out")])
            self.assertEqual(1, result)

    def test_launcher_allows_preparation_without_bundled_weights(self):
        import launch_latest

        with tempfile.TemporaryDirectory() as temporary, patch.object(launch_latest, "ROOT", Path(temporary)):
            launch_latest.self_check()


if __name__ == "__main__":
    unittest.main()
