from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools import build_release, preflight_check
from tools.core.version import (
    DISPLAY_VERSION,
    FILE_VERSION,
    PRODUCT_NAME,
    get_source_url,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]


class ReleaseMetadataTests(TestCase):
    def test_version_is_centralized(self):
        self.assertEqual(PRODUCT_NAME, "YOLO 数据标注工具箱")
        self.assertEqual(DISPLAY_VERSION, "0.9.0-beta.1")
        self.assertEqual(FILE_VERSION, (0, 9, 0, 1))

    def test_required_source_url_rejects_empty_value(self):
        self.assertEqual(get_source_url("https://example.invalid/source"), "https://example.invalid/source")
        with self.assertRaises(ValueError):
            get_source_url("", required=True)

    def test_license_and_third_party_notices_are_complete(self):
        license_text = (PROJECT_DIR / "LICENSE").read_text(encoding="utf-8")
        notices = (PROJECT_DIR / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

        self.assertIn("GNU AFFERO GENERAL PUBLIC LICENSE", license_text)
        self.assertIn("Version 3, 19 November 2007", license_text)
        for name in (
            "PyQt5",
            "Qt",
            "Ultralytics",
            "PyTorch",
            "TorchVision",
            "OpenCV",
            "TensorBoard",
        ):
            self.assertIn(name, notices)

    def test_release_collects_metadata_and_preflight_requires_it(self):
        expected = {
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "CHANGELOG.md",
            "packaging/installer.iss",
            "packaging/languages/ChineseSimplified.isl",
            "tools/build_installer.py",
            "tools/core/diagnostics.py",
            "tools/core/version.py",
        }
        sources = {
            path.as_posix()
            for path in build_release.collect_sources(include_models=False)
        }

        self.assertTrue(expected <= sources)
        self.assertTrue(expected <= set(preflight_check.REQUIRED_SOURCE_FILES))
