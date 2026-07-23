from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from PyQt5.QtWidgets import QApplication

from core.model_evaluation import (
    ClassEvaluationMetrics,
    ClassMetricDelta,
    EvaluationDatasetSnapshot,
    EvaluationMetrics,
    EvaluationSession,
    MetricDelta,
    ModelComparison,
    ModelEvaluationReport,
)
from yolo_evaluation_dialog import ModelEvaluationDialog


class ModelEvaluationDialogTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.data_yaml = self.dataset / "data.yaml"
        self.data_yaml.write_text("path: .\nnames: [按钮, 图标]\n", encoding="utf-8")
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir()
        self.dialog = ModelEvaluationDialog(self.data_yaml, self.runs_dir)
        self.app.processEvents()

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.app.processEvents()
        self.temp_dir.cleanup()

    def _snapshot(self, image_count=20, target_count=30):
        return EvaluationDatasetSnapshot(
            data_yaml=self.data_yaml,
            dataset_dir=self.dataset,
            split="test",
            class_names=("按钮", "图标"),
            image_paths=(),
            label_paths=(),
            image_count=image_count,
            target_count=target_count,
            fingerprint="a" * 64,
        )

    def _report(self, model_name, value, snapshot=None):
        snapshot = snapshot or self._snapshot()
        metrics = EvaluationMetrics(
            precision=value,
            recall=value + 0.01,
            map50=value + 0.02,
            map50_95=value - 0.01,
            image_count=snapshot.image_count,
            target_count=snapshot.target_count,
            classes=(
                ClassEvaluationMetrics(
                    class_id=0,
                    name="按钮",
                    instances=20,
                    precision=value,
                    recall=value,
                    map50=value,
                    map50_95=value,
                ),
                ClassEvaluationMetrics(
                    class_id=1,
                    name="图标",
                    instances=3,
                    precision=value - 0.10,
                    recall=value - 0.10,
                    map50=value - 0.10,
                    map50_95=value - 0.10,
                ),
            ),
        )
        return ModelEvaluationReport(
            model_path=self.root / model_name,
            data_yaml=self.data_yaml,
            split="test",
            imgsz=640,
            device="cpu",
            dataset=snapshot,
            metrics=metrics,
            output_dir=self.runs_dir / model_name,
        )

    def _session(self, low_sample=False):
        snapshot = self._snapshot(10 if low_sample else 20)
        candidate = self._report("candidate.pt", 0.70, snapshot)
        baseline = self._report("baseline.pt", 0.72, snapshot)
        comparison = ModelComparison(
            comparable=True,
            verdict="保持基准",
            reason="mAP50-95 或召回率明显下降",
            low_sample=low_sample,
            delta=MetricDelta(-0.02, -0.01, -0.01, -0.02),
            class_deltas=(
                ClassMetricDelta(0, "按钮", 20, -0.01, -0.02, -0.02, -0.03),
                ClassMetricDelta(1, "图标", 10, -0.10, -0.10, -0.10, -0.10),
            ),
        )
        return EvaluationSession(candidate, baseline, comparison)

    def test_dialog_builds_controls_and_defaults(self):
        self.assertEqual(self.dialog.objectName(), "ModelEvaluationDialog")
        for name in (
            "candidate_edit",
            "baseline_edit",
            "data_edit",
            "split_combo",
            "imgsz_spin",
            "device_combo",
            "start_button",
            "cancel_button",
            "summary_badge",
            "metrics_table",
            "open_output_button",
        ):
            self.assertTrue(hasattr(self.dialog, name), name)
        self.assertEqual(self.dialog.data_edit.text(), str(self.data_yaml))
        self.assertEqual(self.dialog.split_combo.currentText(), "test")
        self.assertEqual(self.dialog.imgsz_spin.value(), 640)
        self.assertFalse(self.dialog.cancel_button.isEnabled())

    def test_device_options_use_shared_device_detection(self):
        with patch(
            "yolo_evaluation_dialog.detect_devices",
            return_value=[
                ("GPU 0: Test GPU（推荐）", "0"),
                ("自动选择（由 Ultralytics 决定）", ""),
                ("CPU（兼容模式，速度较慢）", "cpu"),
            ],
        ):
            dialog = ModelEvaluationDialog(self.data_yaml, self.runs_dir)
        try:
            self.assertEqual(dialog.device_combo.count(), 3)
            self.assertEqual(dialog.device_combo.itemData(0), "0")
            self.assertEqual(dialog.device_combo.itemData(1), "")
            self.assertEqual(dialog.device_combo.itemData(2), "cpu")
        finally:
            dialog.close()
            dialog.deleteLater()

    def test_apply_session_fills_metrics_and_sorts_regressions_first(self):
        session = self._session()
        session.candidate.output_dir.mkdir(parents=True)

        self.dialog.apply_session(session)

        self.assertIn("保持基准", self.dialog.summary_badge.text())
        self.assertEqual(self.dialog.metrics_table.rowCount(), 2)
        self.assertEqual(
            self.dialog.metrics_table.item(0, 1).text(),
            "图标（低样本）",
        )
        self.assertEqual(
            self.dialog.metrics_table.item(1, 1).text(),
            "按钮",
        )
        self.assertIn("0.70", self.dialog.precision_value.text())
        self.assertIn("0.69", self.dialog.map50_95_value.text())
        self.assertTrue(self.dialog.open_output_button.isEnabled())


    def test_low_sample_session_marks_reference_only(self):
        self.dialog.apply_session(self._session(low_sample=True))

        self.assertIn("仅供参考", self.dialog.summary_badge.text())
        self.assertIn("低样本", self.dialog.sample_badge.text())

    def test_running_state_locks_configuration(self):
        self.dialog._set_running(True)

        self.assertFalse(self.dialog.candidate_edit.isEnabled())
        self.assertFalse(self.dialog.baseline_edit.isEnabled())
        self.assertFalse(self.dialog.data_edit.isEnabled())
        self.assertFalse(self.dialog.start_button.isEnabled())
        self.assertTrue(self.dialog.cancel_button.isEnabled())

        self.dialog._set_running(False)
        self.assertTrue(self.dialog.candidate_edit.isEnabled())
        self.assertTrue(self.dialog.start_button.isEnabled())
        self.assertFalse(self.dialog.cancel_button.isEnabled())

    def test_zero_sample_class_displays_missing_metrics(self):
        session = self._session()
        metrics = session.candidate.metrics
        zero_sample = ClassEvaluationMetrics(
            class_id=2,
            name="无样本类别",
            instances=0,
            precision=0.0,
            recall=0.0,
            map50=0.0,
            map50_95=0.0,
        )
        candidate = replace(
            session.candidate,
            metrics=replace(metrics, classes=metrics.classes + (zero_sample,)),
        )

        self.dialog.apply_session(replace(session, candidate=candidate))

        row = next(
            index
            for index in range(self.dialog.metrics_table.rowCount())
            if self.dialog.metrics_table.item(index, 0).text() == "2"
        )
        self.assertEqual(self.dialog.metrics_table.item(row, 1).text(), "无样本类别（无样本）")
        for column in range(3, 8):
            self.assertEqual(self.dialog.metrics_table.item(row, column).text(), "-")

    def test_evaluation_does_not_start_worker_without_runtime(self):
        candidate = self.root / "candidate.pt"
        candidate.write_bytes(b"model")
        self.dialog.candidate_edit.setText(str(candidate))
        with patch(
            "yolo_evaluation_dialog.ensure_ml_runtime",
            return_value=False,
        ) as ensure, patch("yolo_evaluation_dialog.QProcess") as process_class:
            self.dialog.start_evaluation()

        ensure.assert_called_once_with(self.dialog, "模型评估")
        process_class.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
