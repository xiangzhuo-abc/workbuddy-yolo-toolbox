import os
import sys
from pathlib import Path
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from PyQt5.QtWidgets import QApplication

from yolo_ui_widgets import PathBar, SectionHeader, StatusBadge, ToolButton


class UiWidgetTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_path_bar_keeps_read_only_alias_and_emits_browse(self):
        bar = PathBar("数据集", "D:/dataset")
        triggered = []
        bar.browse_clicked.connect(lambda: triggered.append(True))

        self.assertTrue(bar.line_edit.isReadOnly())
        self.assertEqual(bar.path(), "D:/dataset")

        bar.set_path("D:/new-dataset")
        bar.browse_button.click()

        self.assertEqual(bar.path(), "D:/new-dataset")
        self.assertEqual(bar.line_edit.cursorPosition(), 0)
        self.assertEqual(triggered, [True])

    def test_section_header_exposes_title_and_optional_subtitle(self):
        header = SectionHeader("运行记录", "当前任务与日志")

        self.assertEqual(header.title_label.text(), "运行记录")
        self.assertEqual(header.subtitle_label.text(), "当前任务与日志")
        self.assertFalse(header.subtitle_label.isHidden())

    def test_status_and_tool_components_have_stable_roles(self):
        badge = StatusBadge("正常", "success")
        button = ToolButton("打开", "打开目录")

        self.assertEqual(badge.property("tone"), "success")
        self.assertEqual(button.toolTip(), "打开目录")
        self.assertGreaterEqual(button.minimumHeight(), 32)
