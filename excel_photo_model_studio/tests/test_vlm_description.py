from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from excel_photo_model_studio.vlm_description import (
    MIN_TRAIN_SAMPLES, MIN_VAL_SAMPLES, build_description_prompt,
    clean_generated_description, normalize_adapter_path, train_vlm_description_model,
    validate_base_model_path,
)


class VLMDescriptionTests(unittest.TestCase):
    def test_prompt_limits_vlm_to_description_not_depth_or_facies_prediction(self):
        prompt = build_description_prompt(
            facies_index="Dch", facies_name="Каналы",
            interval_top=4100.25, interval_base=4107.25, interval_m=7.0,
        )
        self.assertIn("4100.25–4107.25 м", prompt)
        self.assertIn("Не выводи глубины, интервалы", prompt)
        self.assertIn("не подлежат изменению", prompt)

    def test_adapter_file_path_and_local_base_weights_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            (base / "model.safetensors").touch()
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            weights = adapter / "adapter_model.safetensors"
            weights.touch()

            self.assertEqual(adapter.resolve(), normalize_adapter_path(weights))
            self.assertEqual(base, validate_base_model_path(base))

    def test_insufficient_reviewed_pairs_skip_vlm_without_importing_training_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            base = root / "base"
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            (base / "model.safetensors").touch()
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").touch()
            (dataset / "caption_dataset.jsonl").write_text(
                json.dumps({"split": "train", "crop": "missing.jpg", "target_text": "Описание"}) + "\n",
                encoding="utf-8",
            )
            result = train_vlm_description_model(
                dataset, root / "out", base_model_dir=base, initial_adapter=adapter,
                progress=lambda _message: None,
            )

            self.assertEqual("not_trained", result["status"])
            self.assertEqual(0, result["train_samples"])
            self.assertEqual(0, result["val_samples"])
            self.assertIn(str(MIN_TRAIN_SAMPLES), result["reason"])
            self.assertIn(str(MIN_VAL_SAMPLES), result["reason"])

    def test_generated_markdown_and_whitespace_are_cleaned(self):
        self.assertEqual(
            "Серый тонкослоистый керн.",
            clean_generated_description("```text\nОписание: Серый   тонкослоистый керн.\n```"),
        )


if __name__ == "__main__":
    unittest.main()
