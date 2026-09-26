from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from excel_photo_model_studio.dataset import _report_blockers, _split_sources, _validate_annotation, _verify_validation_snapshot
from excel_photo_model_studio.models import Annotation, DescriptionRow, PhotoRecord
from excel_photo_model_studio.project import (
    _annotation_revision, _facies_mask_gaps, _file_stamp, _projection_issues,
    _uncovered_excel_core_intervals,
)


class IntegrityTests(unittest.TestCase):
    def annotation(self, **values):
        defaults = dict(
            annotation_id="a", photo_path=Path("a.jpg"), well="W", photo_top=100., photo_base=102.,
            depth_top=100., depth_base=101., label="Sand",
            polygon=((10., 10.), (29., 10.), (29., 110.), (10., 110.)),
            image_width=100, image_height=150, source_sheet="Data", source_row=2,
            facies_top=100., facies_base=102., target_text="Sandstone.",
        )
        defaults.update(values)
        return Annotation(**defaults)

    def test_table_coverage_does_not_excuse_an_unmapped_physical_column(self):
        photo = PhotoRecord(Path("a.jpg"), "W", 100., 101., "manual", True)
        first, second = (10, 10, 30, 110), (50, 10, 70, 110)
        issues = _projection_issues([photo], [self.annotation()], {photo.path: [first, second]}, {
            photo.path: [(first, 100., 101.)],
        })
        self.assertEqual(1, len(issues))
        self.assertIn("не всем физическим", issues[0].message)

    def test_all_columns_and_every_centimetre_must_have_masks(self):
        photo = PhotoRecord(Path("a.jpg"), "W", 100., 102., "manual", True)
        first, second = (10, 10, 30, 110), (50, 10, 70, 110)
        columns = {photo.path: [first, second]}
        depths = {photo.path: [(first, 100., 101.), (second, 101., 102.)]}
        first_mask = self.annotation()
        self.assertTrue(_projection_issues([photo], [first_mask], columns, depths))
        second_mask = self.annotation(annotation_id="b", depth_top=101., depth_base=102.,
                                      polygon=((50., 10.), (69., 10.), (69., 110.), (50., 110.)))
        self.assertEqual([], _projection_issues([photo], [first_mask, second_mask], columns, depths))
        self.assertTrue(_projection_issues([photo], [first_mask, second_mask, replace(second_mask, annotation_id="c")], columns, depths))

    def test_partial_excel_facies_is_not_treated_as_fully_matched(self):
        row = DescriptionRow("W", 100., 102., "Sand", "Data", 2, target_text="Sandstone.")
        gaps = _facies_mask_gaps([row], [self.annotation()])
        self.assertEqual([("Data!2", [(101., 102.)])], gaps)

    def test_gis_facies_coverage_compares_same_row_not_different_depth_systems(self):
        row = DescriptionRow("W", 100., 102., "Sand", "Data", 2, target_text="Sandstone.", gis_top=99.9, gis_base=101.9)
        annotation = self.annotation(depth_top=99.9, depth_base=101.9, facies_top=99.9, facies_base=101.9)
        self.assertEqual([], _facies_mask_gaps([row], [annotation]))
        photo = PhotoRecord(Path("a.jpg"), "W", 99.9, 101.9, "manual", True, depth_basis="gis")
        row = replace(row, core_top=100., core_base=102.)
        self.assertEqual([], _uncovered_excel_core_intervals([row], [photo]))

    def test_changed_text_or_polygon_requires_new_review(self):
        annotation = self.annotation()
        self.assertNotEqual(_annotation_revision(annotation), _annotation_revision(replace(annotation, target_text="Changed.")))
        self.assertNotEqual(_annotation_revision(annotation), _annotation_revision(replace(annotation, polygon=((10., 20.), (29., 20.), (29., 110.), (10., 110.)))))

    def test_copies_cannot_leak_into_validation(self):
        photos = {name: [{"label": "A", "well": name}] for name in ("a", "copy", "b")}
        split, _ = _split_sources(photos, {"a": "same", "copy": "same", "b": "different"})
        self.assertEqual(split["a"], split["copy"])
        self.assertNotEqual(split["a"], split["b"])
        split, _ = _split_sources({key: photos[key] for key in ("a", "copy")}, {"a": "same", "copy": "same"})
        self.assertNotIn("val", split.values())

    def test_changed_inputs_and_new_photos_invalidate_verified_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            photos.mkdir()
            photo = photos / "a.jpg"
            photo.write_bytes(b"initial")
            snapshot = {"files": {str(photo): _file_stamp(photo)}, "photos_dir": str(photos), "photos": [str(photo.resolve())]}
            _verify_validation_snapshot(root, {"validation_snapshot": snapshot})
            photo.write_bytes(b"changed size")
            with self.assertRaises(ValueError):
                _verify_validation_snapshot(root, {"validation_snapshot": snapshot})
            snapshot["files"][str(photo)] = _file_stamp(photo)
            (photos / "new.jpg").write_bytes(b"new")
            with self.assertRaises(ValueError):
                _verify_validation_snapshot(root, {"validation_snapshot": snapshot})

    def test_projection_and_unmatched_facies_block_training_even_with_zero_legacy_errors(self):
        self.assertTrue(_report_blockers({"projection_errors": 1}))
        self.assertTrue(_report_blockers({"facies_rows_with_incomplete_masks": 1}))
        self.assertTrue(_report_blockers({"excel_facies_rows_without_photo_match": 1}))

    def test_zero_area_mask_is_invalid(self):
        with self.assertRaises(ValueError):
            _validate_annotation({"label": "A", "target_text": "Text", "image_width": "100", "image_height": "100",
                                  "depth_top": "100", "depth_base": "101", "polygon_json": json.dumps([[5, 5], [5, 10], [5, 20]])})


if __name__ == "__main__":
    unittest.main()
