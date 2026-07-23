import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from tools.core.config_store import ConfigStore
from tools.core.issues import IssueSeverity


class ConfigStoreTests(TestCase):
    def test_load_merges_defaults_and_file_values_take_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool_config.json"
            path.write_text('{"dataset_dir": "D:/data"}', encoding="utf-8")
            store = ConfigStore(
                path,
                {"dataset_dir": "", "runs_dir": "C:/runs"},
            )

            self.assertEqual(
                store.load(),
                {"dataset_dir": "D:/data", "runs_dir": "C:/runs"},
            )
            self.assertIsNone(store.last_issue)

    def test_load_recovers_from_invalid_json_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool_config.json"
            path.write_text("{broken", encoding="utf-8")
            store = ConfigStore(path, {"dataset_dir": "default"})

            self.assertEqual(store.load(), {"dataset_dir": "default"})
            self.assertEqual(path.read_text(encoding="utf-8"), "{broken")
            self.assertEqual(store.last_issue.code, "config.invalid_json")
            self.assertEqual(store.last_issue.severity, IssueSeverity.ERROR)
            self.assertEqual(store.last_issue.path, path)

    def test_load_rejects_non_dict_root_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool_config.json"
            source = '["不是对象"]'
            path.write_text(source, encoding="utf-8")
            store = ConfigStore(path, {"dataset_dir": "default"})

            self.assertEqual(store.load(), {"dataset_dir": "default"})
            self.assertEqual(path.read_text(encoding="utf-8"), source)
            self.assertEqual(store.last_issue.code, "config.invalid_json")
            self.assertEqual(store.last_issue.severity, IssueSeverity.ERROR)

    def test_save_supports_chinese_path_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "中文配置" / "工具配置.json"
            store = ConfigStore(path)

            store.save({"数据目录": "D:/训练数据", "说明": "中文内容"})

            text = path.read_text(encoding="utf-8")
            self.assertIn('"数据目录": "D:/训练数据"', text)
            self.assertNotIn("\\u", text)
            self.assertEqual(
                json.loads(text),
                {"数据目录": "D:/训练数据", "说明": "中文内容"},
            )

    def test_save_replaces_file_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool_config.json"
            temp_path = path.with_suffix(".json.tmp")
            store = ConfigStore(path)

            store.save({"dataset_dir": "D:/数据"})

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{\n  "dataset_dir": "D:/数据"\n}',
            )
            self.assertFalse(temp_path.exists())

    def test_save_cleans_temp_file_and_reraises_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tool_config.json"
            temp_path = path.with_suffix(".json.tmp")
            store = ConfigStore(path)
            error = OSError("模拟替换失败")

            with patch.object(Path, "replace", side_effect=error):
                with self.assertRaises(OSError) as raised:
                    store.save({"dataset_dir": "D:/数据"})

            self.assertIs(raised.exception, error)
            self.assertFalse(temp_path.exists())
            self.assertFalse(path.exists())