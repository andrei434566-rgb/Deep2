from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from excel_photo_model_studio.autodiscovery import discover_well_pairs


class AutomaticDiscoveryTests(unittest.TestCase):
    def test_pairs_workbooks_with_photo_child_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "Well A"
            first.mkdir()
            (first / "Well A.xlsx").write_bytes(b"excel")
            (first / "Фото").mkdir()
            (first / "Фото" / "001.jpg").write_bytes(b"photo")
            second = root / "Well B"
            second.mkdir()
            (second / "Well B.xlsx").write_bytes(b"excel")
            (second / "image.png").write_bytes(b"photo")

            result = discover_well_pairs(root)

            self.assertEqual(2, result["workbooks_found"])
            self.assertEqual(2, result["photos_found"])
            self.assertEqual(2, len(result["pairs"]))
            self.assertEqual([], result["unmatched"])
            self.assertEqual(
                {"Фото", "Well B"},
                {Path(item["photos"]).name for item in result["pairs"]},
            )

    def test_does_not_guess_when_workbooks_share_the_same_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "well-a.xlsx").write_bytes(b"excel")
            (root / "well-b.xlsx").write_bytes(b"excel")
            (root / "001.jpg").write_bytes(b"photo")

            result = discover_well_pairs(root)

            self.assertEqual([], result["pairs"])
            self.assertEqual(2, len(result["unmatched"]))
            self.assertTrue(all("нескольким Excel" in row["reason"] for row in result["unmatched"]))


if __name__ == "__main__":
    unittest.main()
