from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from excel_photo_model_studio.description_model import DescriptionGenerator, train_description_model


class DescriptionModelTests(unittest.TestCase):
    def test_trains_checkpoint_for_column_22(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "model"
            (dataset / "crops" / "train").mkdir(parents=True)
            (dataset / "crops" / "val").mkdir(parents=True)
            output.mkdir()
            rows = []
            for index in range(6):
                split = "train" if index < 5 else "val"
                relative = Path("crops") / split / f"{index}.jpg"
                image = np.full((32, 32, 3), 50 + index * 20, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", image)
                self.assertTrue(ok)
                (dataset / relative).write_bytes(encoded.tobytes())
                rows.append({
                    "split": split, "crop": relative.as_posix(),
                    "facies": "Tcr",
                    "target_text": "Песчаник серый, слоистый.",
                })
            (dataset / "caption_dataset.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )

            info = train_description_model(
                dataset, output, epochs=1, patience=1, max_text_length=48,
                image_size=32, hidden_size=32, batch_size=2, progress=lambda _: None,
                device="cpu",
            )
            checkpoint = torch.load(output / "description_best.pt", map_location="cpu", weights_only=False)
            combined = output / "best.pt"
            torch.save({"core_description_checkpoint": checkpoint}, combined)
            original_load = torch.load
            with patch("torch.load", wraps=original_load) as load:
                generator = DescriptionGenerator(combined)
                first = generator.generate(np.full((32, 32, 3), 130, dtype=np.uint8), "Tcr")
                second = generator.generate(dataset / rows[0]["crop"], "Tcr")
                self.assertEqual(1, load.call_count)
                self.assertIsInstance(first, str)
                self.assertIsInstance(second, str)
                self.assertNotIn("<unk>", first)

            with self.assertRaisesRegex(ValueError, "обрезано"):
                train_description_model(dataset, output, max_text_length=8)

        self.assertEqual(22, checkpoint["target_column"])
        self.assertEqual("excel-photo-description-v2", checkpoint["schema"])
        self.assertEqual(["Tcr"], checkpoint["facies_names"])
        self.assertEqual(["interval_image", "facies_class"], checkpoint["conditioning"])
        self.assertEqual(5, info["train_samples"])
        self.assertEqual(1, info["val_samples"])
        self.assertEqual("cpu", info["device"])


if __name__ == "__main__":
    unittest.main()
