from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from excel_photo_model_studio.description_model import (
    DescriptionGenerator, train_description_model,
)


class DescriptionModelTests(unittest.TestCase):
    def test_insufficient_caption_samples_are_reported_instead_of_fabricating_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "dataset", root / "model"
            dataset.mkdir()
            output.mkdir()
            (dataset / "caption_dataset.jsonl").write_text(
                json.dumps({"split": "train", "target_text": "Песчаник"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            info = train_description_model(dataset, output, device="cpu", progress=lambda _message: None)
            self.assertEqual("not_trained", info["status"])
            self.assertIn("независимых проверочных", info["reason"])
            self.assertFalse((output / "description_best.pt").exists())

    def test_trains_interval_conditioned_text_checkpoint_and_loads_generator(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "dataset", root / "model"
            (dataset / "crops" / "train").mkdir(parents=True)
            (dataset / "crops" / "val").mkdir(parents=True)
            output.mkdir()
            rows = []
            for index in range(24):
                split = "train" if index < 20 else "val"
                relative = Path("crops") / split / f"{index}.jpg"
                image = np.full((40, 24, 3), 70 + index * 7, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                (dataset / relative).write_bytes(encoded.tobytes())
                rows.append({
                    "split": split, "crop": relative.as_posix(), "facies_index": "Dch",
                    "facies_name": "Каналы", "interval_m": 0.8 + index / 10,
                    "target_text": "Песчаник серый, слоистый.",
                })
            (dataset / "caption_dataset.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
            )
            (dataset / "class_metadata.json").write_text(json.dumps({"classes": []}), encoding="utf-8")

            info = train_description_model(
                dataset, output, epochs=1, patience=1, image_size=32,
                hidden_size=32, batch_size=5, device="cpu", progress=lambda _message: None,
            )

            checkpoint = torch.load(output / "description_best.pt", map_location="cpu", weights_only=False)
            generator = DescriptionGenerator(output / "description_best.pt")
            generated = generator.generate(
                np.full((24, 40, 3), 150, dtype=np.uint8), "Dch", "Каналы", 1.2,
            )
            self.assertEqual("trained_candidate", info["status"])
            self.assertEqual(
                ["interval_image_crop", "facies_index", "facies_name", "interval_thickness_m"],
                info["conditioning"],
            )
            self.assertEqual("excel-photo-description-v5", checkpoint["schema"])
            self.assertEqual([{"facies_index": "Dch", "facies_name": "Каналы"}], checkpoint["facies_conditions"])
            self.assertIsInstance(generated, str)


if __name__ == "__main__":
    unittest.main()
