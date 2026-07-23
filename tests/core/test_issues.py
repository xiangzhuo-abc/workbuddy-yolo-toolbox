from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest import TestCase

from tools.core.issues import Issue, IssueSeverity


class IssueTests(TestCase):
    def test_serializes_stable_fields(self):
        issue = Issue(
            code="dataset.split_overlap",
            severity=IssueSeverity.ERROR,
            message="子集重叠",
            path=Path("D:/dataset/images/train/a.png"),
            suggested_action="重新划分",
        )

        self.assertEqual(
            issue.to_dict(),
            {
                "code": "dataset.split_overlap",
                "severity": "error",
                "message": "子集重叠",
                "path": str(Path("D:/dataset/images/train/a.png")),
                "suggested_action": "重新划分",
            },
        )

    def test_defines_expected_severity_values(self):
        self.assertEqual(IssueSeverity.INFO.value, "info")
        self.assertEqual(IssueSeverity.WARNING.value, "warning")
        self.assertEqual(IssueSeverity.ERROR.value, "error")

    def test_serializes_missing_path_as_none(self):
        issue = Issue(
            code="dataset.ready",
            severity=IssueSeverity.INFO,
            message="数据集可用",
        )

        self.assertIsNone(issue.to_dict()["path"])
        self.assertEqual(issue.to_dict()["suggested_action"], "")

    def test_is_immutable(self):
        issue = Issue(
            code="dataset.ready",
            severity=IssueSeverity.INFO,
            message="数据集可用",
        )

        with self.assertRaises(FrozenInstanceError):
            issue.message = "已修改"