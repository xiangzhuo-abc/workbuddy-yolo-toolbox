import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import split_dataset


class SplitDatasetCliTests(TestCase):
    def _plan(self, *, executable=True):
        return SimpleNamespace(
            is_executable=executable,
            plan_id="a" * 64,
            moves=(object(), object()),
            target_counts=(("train", 8), ("val", 2), ("test", 0)),
            planned_counts=(("train", 8), ("val", 2), ("test", 0)),
            risks=(),
            blocking_issues=() if executable else (SimpleNamespace(message="阻断"),),
        )

    def test_defaults_to_preview_without_apply(self):
        plan = self._plan()
        with patch.object(
            split_dataset.tools,
            "build_split_plan",
            return_value=plan,
        ), patch.object(split_dataset.tools, "apply_split_plan") as apply_plan:
            exit_code = split_dataset.main(["--dataset-dir", "D:/dataset"])

        self.assertEqual(exit_code, 0)
        apply_plan.assert_not_called()

    def test_apply_requires_explicit_flag(self):
        plan = self._plan()
        result = SimpleNamespace(success=True, backup_dir=Path("D:/backup"))
        with patch.object(
            split_dataset.tools,
            "build_split_plan",
            return_value=plan,
        ), patch.object(
            split_dataset.tools,
            "apply_split_plan",
            return_value=result,
        ) as apply_plan:
            exit_code = split_dataset.main(
                ["--dataset-dir", "D:/dataset", "--apply"]
            )

        self.assertEqual(exit_code, 0)
        apply_plan.assert_called_once_with(plan, emit=split_dataset._print_emit)

    def test_blocking_plan_returns_failure(self):
        plan = self._plan(executable=False)
        with patch.object(
            split_dataset.tools,
            "build_split_plan",
            return_value=plan,
        ):
            exit_code = split_dataset.main(["--dataset-dir", "D:/dataset"])

        self.assertEqual(exit_code, 1)
