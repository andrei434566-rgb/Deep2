from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.infrastructure.ml.fine_tune_worker import read_training_metrics


class TrainingMetricsTests(unittest.TestCase):
    def test_reads_best_map_row_from_ultralytics_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
                "1,0.5,0.4,0.6,0.3\n2,0.7,0.8,0.85,0.65\n",
                encoding="utf-8",
            )
            metrics = read_training_metrics(Path(directory))
        self.assertEqual(0.65, metrics["mAP50-95"])
        self.assertEqual(0.7, metrics["precision"])
        self.assertEqual(0.8, metrics["recall"])


if __name__ == "__main__":
    unittest.main()
