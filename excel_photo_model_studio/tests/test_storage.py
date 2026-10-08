from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from excel_photo_model_studio.storage import app_data_dir


class StorageTests(unittest.TestCase):
    def test_source_run_uses_stable_workspace_data_for_per_run_sandbox_localappdata(self):
        sandbox_local = r"C:\Users\Я\AppData\Local\Packages\sandbox.{test-run-id}\AC"
        with patch.dict(os.environ, {"LOCALAPPDATA": sandbox_local}, clear=False):
            with patch.object(sys, "frozen", False, create=True):
                actual = app_data_dir()

        workspace = Path(__file__).resolve().parents[2]
        self.assertEqual(
            workspace / "outputs" / "appdata" / "ExcelPhotoModelStudio",
            actual,
        )

    def test_explicit_storage_override_takes_precedence(self):
        override = Path("C:/temporary/excel-photo-storage")
        with patch.dict(os.environ, {"EXCEL_PHOTO_MODEL_STUDIO_DATA_DIR": str(override)}):
            self.assertEqual(override, app_data_dir())


if __name__ == "__main__":
    unittest.main()
