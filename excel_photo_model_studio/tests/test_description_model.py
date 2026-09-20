from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from excel_photo_model_studio.description_model import train_description_model


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
                    "target_text": "Песчаник серый, слоистый.",
                })
            (dataset / "caption_dataset.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )

            info = train_description_model(
                dataset, output, epochs=1, patience=1, max_text_length=48,
                image_size=32, hidden_size=32, batch_size=2, progress=lambda _: None,
            )
            checkpoint = torch.load(output / "description_best.pt", map_location="cpu", weights_only=False)

        self.assertEqual(22, checkpoint["target_column"])
        self.assertEqual(5, info["train_samples"])
        self.assertEqual(1, info["val_samples"])


if __name__ == "__main__":
    unittest.main()
