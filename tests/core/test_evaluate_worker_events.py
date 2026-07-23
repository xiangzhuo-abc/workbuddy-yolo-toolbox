from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import yolo_evaluate_worker
from core.model_evaluation import (
    ClassEvaluationMetrics,
    EvaluationDatasetSnapshot,
    EvaluationMetrics,
    EvaluationSession,
    ModelEvaluationReport,
    scan_evaluation_dataset,
)
from core.task_protocol import TaskEventType, decode_task_event


class EvaluationWorkerEventTests(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "images" / "test").mkdir(parents=True)
        (self.dataset / "labels" / "test").mkdir(parents=True)
        (self.dataset / "classes.txt").write_text("按钮\n", encoding="utf-8")
        self.data_yaml = self.dataset / "data.yaml"
        self.data_yaml.write_text(
            "path: .\n"
            "test: images/test\n"
            "names: [按钮]\n",
            encoding="utf-8",
        )
        self.image = self.dataset / "images" / "test" / "a.png"
        self.image.write_bytes(b"image")
        self.label = self.dataset / "labels" / "test" / "a.txt"
        self.label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _snapshot(self):
        return scan_evaluation_dataset(self.data_yaml, "test")

    @staticmethod
    def _report(model_path, snapshot, value=0.5):
        metrics = EvaluationMetrics(
            precision=value,
            recall=value,
            map50=value,
            map50_95=value,
            image_count=snapshot.image_count,
            target_count=snapshot.target_count,
            classes=(
                ClassEvaluationMetrics(
                    class_id=0,
                    name="按钮",
                    instances=snapshot.target_count,
                    precision=value,
                    recall=value,
                    map50=value,
                    map50_95=value,
                ),
            ),
        )
        return ModelEvaluationReport(
            model_path=Path(model_path),
            data_yaml=snapshot.data_yaml,
            split=snapshot.split,
            imgsz=640,
            device="cpu",
            dataset=snapshot,
            metrics=metrics,
            output_dir=Path(snapshot.dataset_dir) / "evaluation-output",
        )

    def _events(self, stream):
        return [
            event
            for line in stream.getvalue().splitlines()
            if (event := decode_task_event(line)) is not None
        ]

    def test_single_model_success_saves_session_and_result(self):
        stream = io.StringIO()
        snapshot = self._snapshot()
        output = self.root / "evaluations" / "single" / "evaluation.json"

        def evaluate_func(**kwargs):
            self.assertEqual(kwargs["split"], "test")
            self.assertEqual(kwargs["imgsz"], 640)
            return self._report("candidate.pt", kwargs["snapshot"], 0.6)

        exit_code = yolo_evaluate_worker.main(
            [
                "--task-id", "eval-1",
                "--model", "candidate.pt",
                "--data", str(self.data_yaml),
                "--split", "test",
                "--device", "cpu",
                "--output", str(output),
            ],
            evaluate_func=evaluate_func,
            snapshot_func=lambda data, split: snapshot,
            stream=stream,
        )

        events = self._events(stream)
        self.assertEqual(exit_code, 0)
        self.assertEqual(events[0].type, TaskEventType.STARTED)
        self.assertIn(TaskEventType.PROGRESS, [event.type for event in events])
        self.assertEqual(events[-1].type, TaskEventType.RESULT)
        self.assertEqual(len([event for event in events if event.is_final]), 1)
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["candidate"]["metrics"]["map50"], 0.6)
        self.assertIsNone(payload["baseline"])

    def test_candidate_and_baseline_are_compared_in_one_session(self):
        stream = io.StringIO()
        snapshot = self._snapshot()
        output = self.root / "evaluation.json"
        received = []

        def evaluate_func(**kwargs):
            received.append(kwargs["model_path"])
            value = 0.7 if str(kwargs["model_path"]) == "candidate.pt" else 0.5
            return self._report(kwargs["model_path"], kwargs["snapshot"], value)

        exit_code = yolo_evaluate_worker.main(
            [
                "--task-id", "eval-2",
                "--model", "candidate.pt",
                "--baseline", "baseline.pt",
                "--data", str(self.data_yaml),
                "--split", "test",
                "--device", "cpu",
                "--output", str(output),
            ],
            evaluate_func=evaluate_func,
            snapshot_func=lambda data, split: snapshot,
            stream=stream,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(received, [Path("candidate.pt"), Path("baseline.pt")])
        session = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(session["comparison"]["verdict"], "仅供参考")

    def test_backend_exception_emits_failed_final_event(self):
        stream = io.StringIO()
        output = self.root / "evaluation.json"

        def fail(**kwargs):
            raise RuntimeError("evaluation exploded")

        exit_code = yolo_evaluate_worker.main(
            [
                "--model", "candidate.pt",
                "--data", str(self.data_yaml),
                "--output", str(output),
            ],
            evaluate_func=fail,
            snapshot_func=lambda data, split: self._snapshot(),
            stream=stream,
        )

        events = self._events(stream)
        self.assertEqual(exit_code, 1)
        self.assertEqual(events[-1].type, TaskEventType.FAILED)
        self.assertIn("evaluation exploded", events[-1].message)
        self.assertEqual(len([event for event in events if event.is_final]), 1)

    def test_keyboard_interrupt_emits_cancelled(self):
        stream = io.StringIO()
        output = self.root / "evaluation.json"

        def cancel(**kwargs):
            raise KeyboardInterrupt

        exit_code = yolo_evaluate_worker.main(
            [
                "--model", "candidate.pt",
                "--data", str(self.data_yaml),
                "--output", str(output),
            ],
            evaluate_func=cancel,
            snapshot_func=lambda data, split: self._snapshot(),
            stream=stream,
        )

        events = self._events(stream)
        self.assertEqual(exit_code, 130)
        self.assertEqual(events[-1].type, TaskEventType.CANCELLED)

    def test_snapshot_change_fails_before_evaluation(self):
        stream = io.StringIO()
        output = self.root / "evaluation.json"
        snapshot = self._snapshot()
        called = []

        def evaluate_func(**kwargs):
            called.append(True)
            return self._report("candidate.pt", kwargs["snapshot"])

        def changed_snapshot(data, split):
            self.image.write_bytes(b"changed")
            return snapshot

        exit_code = yolo_evaluate_worker.main(
            [
                "--model", "candidate.pt",
                "--data", str(self.data_yaml),
                "--output", str(output),
            ],
            evaluate_func=evaluate_func,
            snapshot_func=changed_snapshot,
            stream=stream,
        )

        events = self._events(stream)
        self.assertEqual(exit_code, 1)
        self.assertTrue(called)
        self.assertEqual(events[-1].type, TaskEventType.FAILED)
        self.assertIn("数据指纹", events[-1].message)
        self.assertFalse(output.exists())

    def test_default_backend_standardizes_box_metrics_without_writing_cache(self):
        snapshot = self._snapshot()
        captured = {}

        class FakeBox:
            mp = 0.71
            mr = 0.62
            map50 = 0.73
            map = 0.58
            p = [0.71]
            r = [0.62]
            ap50 = [0.73]
            ap = [0.58]
            ap_class_index = [0]

        class FakeMetrics:
            box = FakeBox()
            results_dict = {}

        class FakeModel:
            def val(self, **kwargs):
                captured.update(kwargs)
                return FakeMetrics()

        report = yolo_evaluate_worker.evaluate_model(
            model_path=self.root / "candidate.pt",
            data_yaml=self.data_yaml,
            split="test",
            imgsz=640,
            device="cpu",
            output_dir=self.root / "evaluation",
            snapshot=snapshot,
            model_factory=lambda path: FakeModel(),
        )

        self.assertAlmostEqual(report.metrics.precision, 0.71)
        self.assertAlmostEqual(report.metrics.recall, 0.62)
        self.assertAlmostEqual(report.metrics.map50, 0.73)
        self.assertAlmostEqual(report.metrics.map50_95, 0.58)
        self.assertEqual(report.metrics.classes[0].instances, 1)
        self.assertEqual(captured["split"], "test")
        self.assertEqual(captured["save_json"], False)

    def test_class_metrics_follow_ultralytics_class_index_mapping(self):
        snapshot = EvaluationDatasetSnapshot(
            data_yaml=self.data_yaml,
            dataset_dir=self.dataset,
            split="test",
            class_names=("类别0", "无样本类别", "类别2"),
            image_paths=(self.image,),
            label_paths=(self.label,),
            image_count=1,
            target_count=2,
            fingerprint="f" * 64,
            class_target_counts=(1, 0, 1),
        )

        class FakeBox:
            mp = 0.75
            mr = 0.65
            map50 = 0.70
            map = 0.55
            p = [0.80, 0.70]
            r = [0.60, 0.70]
            ap50 = [0.75, 0.65]
            ap = [0.50, 0.60]
            ap_class_index = [0, 2]

        class FakeMetrics:
            box = FakeBox()
            results_dict = {}

        standardized = yolo_evaluate_worker._standardize_metrics(
            FakeMetrics(),
            snapshot,
        )

        self.assertAlmostEqual(standardized.classes[0].precision, 0.80)
        self.assertEqual(standardized.classes[1].instances, 0)
        self.assertEqual(standardized.classes[1].map50_95, 0.0)
        self.assertAlmostEqual(standardized.classes[2].precision, 0.70)
        self.assertAlmostEqual(standardized.classes[2].map50_95, 0.60)

    def test_cache_blocker_covers_dataset_module_alias_and_restores_it(self):
        from ultralytics.data import dataset, utils

        original_dataset = dataset.save_dataset_cache_file
        original_utils = utils.save_dataset_cache_file
        with yolo_evaluate_worker._disable_ultralytics_cache():
            self.assertIsNot(dataset.save_dataset_cache_file, original_dataset)
            self.assertIsNot(utils.save_dataset_cache_file, original_utils)
        self.assertIs(dataset.save_dataset_cache_file, original_dataset)
        self.assertIs(utils.save_dataset_cache_file, original_utils)


if __name__ == "__main__":
    import unittest

    unittest.main()
